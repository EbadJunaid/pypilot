//! pixi projects: manifest parsing and environment layout.
//!
//! pixi (<https://pixi.prefix.dev>) manages conda-forge packages plus PyPI
//! packages (through uv) per project. A project is a `pixi.toml`, or a
//! `pyproject.toml` with a `[tool.pixi.workspace]` (older: `[tool.pixi.project]`)
//! table. Its environments are conda prefixes under `<root>/.pixi/envs/<name>`:
//! no `pyvenv.cfg`, the interpreter sits at the prefix root on Windows, and the
//! installed Python version is recorded in `conda-meta/`.
//!
//! Everything here is read off disk and side-effect free, except the two
//! fallbacks that ask the `pixi` binary itself ([`info_envs`], [`pixi_version`]).

use std::path::{Path, PathBuf};

use crate::core::project::{self, normalize_name, Requirement};
use crate::pypi::pyversion::{Clause, PyVersion};

/// The name pixi gives the environment built from the top-level tables.
pub const DEFAULT_ENV: &str = "default";

/// Which file declares the pixi project.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFormat {
    /// `pixi.toml`: tables live at the top level.
    PixiToml,
    /// `pyproject.toml`: tables live under `[tool.pixi]`.
    Pyproject,
}

/// A conda dependency as declared: a package name and its raw match spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondaDep {
    pub name: String,
    /// The conda match spec as written (`"3.12.*"`, `">=1.26,<2"`, `"*"`).
    /// For table values this is the `version` key, or `"*"` when absent.
    pub spec: String,
}

/// The dependencies a `[feature.<name>]` adds on top of the default set.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PixiFeature {
    pub name: String,
    /// Raw conda spec for `python`, when the feature pins it.
    pub python: Option<String>,
    pub conda_deps: Vec<CondaDep>,
    pub pypi_deps: Vec<Requirement>,
    /// `[feature.<name>.system-requirements] cuda`.
    pub cuda: Option<String>,
}

/// What a pixi manifest declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixiProject {
    /// The directory holding the manifest (and `.pixi/`).
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub format: ManifestFormat,
    /// Raw conda spec for `python` in the default environment. In pyproject
    /// mode `[project].requires-python` stands in when pixi's own tables do
    /// not name python.
    pub python: Option<String>,
    /// Default-environment conda dependencies, `python` excluded.
    pub conda_deps: Vec<CondaDep>,
    /// Default-environment PyPI dependencies that resolve from an index.
    pub pypi_deps: Vec<Requirement>,
    /// PyPI dependencies given as `path`, `git` or `url`: not checkable
    /// against an index, so they are counted rather than listed.
    pub skipped_pypi_deps: usize,
    /// Environment names; always starts with `"default"`.
    pub environments: Vec<String>,
    /// Target platforms (`linux-64`, `win-64`, ...).
    pub platforms: Vec<String>,
    /// CUDA version the project targets, from a platform entry
    /// (`{ platform = "linux-64", cuda = "12.0" }`) or the legacy
    /// `[system-requirements] cuda`.
    pub cuda: Option<String>,
    /// Non-default features, in name order.
    pub features: Vec<PixiFeature>,
}

/// Find the pixi project rooted at `workspace`.
///
/// `pixi.toml` wins over `pyproject.toml`, as it does for pixi itself. A
/// pyproject counts only with a `[tool.pixi.workspace]` or `[tool.pixi.project]`
/// table. Unreadable or malformed manifests yield `None`.
pub fn detect(workspace: &Path) -> Option<PixiProject> {
    let pixi_toml = workspace.join("pixi.toml");
    if pixi_toml.is_file() {
        let text = std::fs::read_to_string(&pixi_toml).ok()?;
        return parse_manifest(&text, ManifestFormat::PixiToml, workspace, pixi_toml);
    }
    let pyproject = workspace.join("pyproject.toml");
    if pyproject.is_file() {
        let text = std::fs::read_to_string(&pyproject).ok()?;
        return parse_manifest(&text, ManifestFormat::Pyproject, workspace, pyproject);
    }
    None
}

/// Parse manifest text. `None` when it is malformed or, for a pyproject, has
/// no pixi workspace table.
pub fn parse_manifest(
    text: &str,
    format: ManifestFormat,
    root: &Path,
    manifest: PathBuf,
) -> Option<PixiProject> {
    let doc = toml::from_str::<toml::Value>(text.trim_start_matches('\u{feff}')).ok()?;
    let doc = doc.as_table()?;

    let (pixi, pep621) = match format {
        ManifestFormat::PixiToml => (doc, None),
        ManifestFormat::Pyproject => {
            let pixi = doc.get("tool")?.get("pixi")?.as_table()?;
            if !pixi.contains_key("workspace") && !pixi.contains_key("project") {
                return None;
            }
            (pixi, doc.get("project").and_then(|p| p.as_table()))
        }
    };

    let mut default = parse_dep_tables(pixi);
    let mut skipped = default.skipped;

    if let Some(project) = pep621 {
        if let Some(deps) = project.get("dependencies").and_then(|d| d.as_array()) {
            for line in deps.iter().filter_map(|d| d.as_str()) {
                match project::parse_requirement(line) {
                    Some(r) if !is_direct_reference(line) => default.pypi.push(r),
                    _ => skipped += 1,
                }
            }
        }
        if default.python.is_none() {
            default.python = project
                .get("requires-python")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
    }

    // `[workspace]` today, `[project]` in older manifests.
    let workspace = pixi
        .get("workspace")
        .or_else(|| pixi.get("project"))
        .and_then(|w| w.as_table());
    let mut platforms = Vec::new();
    let mut cuda = None;
    if let Some(list) = workspace
        .and_then(|w| w.get("platforms"))
        .and_then(|p| p.as_array())
    {
        for entry in list {
            match entry {
                toml::Value::String(s) => platforms.push(s.clone()),
                toml::Value::Table(t) => {
                    if let Some(p) = t.get("platform").and_then(|p| p.as_str()) {
                        platforms.push(p.to_string());
                    }
                    if cuda.is_none() {
                        cuda = t.get("cuda").and_then(scalar_string);
                    }
                }
                _ => {}
            }
        }
    }
    if cuda.is_none() {
        cuda = system_cuda(pixi);
    }

    let mut features = Vec::new();
    if let Some(table) = pixi.get("feature").and_then(|f| f.as_table()) {
        for (name, body) in table {
            let Some(body) = body.as_table() else {
                continue;
            };
            let deps = parse_dep_tables(body);
            features.push(PixiFeature {
                name: name.clone(),
                python: deps.python,
                conda_deps: deps.conda,
                pypi_deps: dedupe(deps.pypi),
                cuda: system_cuda(body),
            });
        }
    }

    let mut environments = vec![DEFAULT_ENV.to_string()];
    if let Some(envs) = pixi.get("environments").and_then(|e| e.as_table()) {
        for name in envs.keys() {
            if name != DEFAULT_ENV {
                environments.push(name.clone());
            }
        }
    }

    Some(PixiProject {
        root: root.to_path_buf(),
        manifest,
        format,
        python: default.python,
        conda_deps: default.conda,
        pypi_deps: dedupe(default.pypi),
        skipped_pypi_deps: skipped,
        environments,
        platforms,
        cuda,
        features,
    })
}

/// Dependencies read from one level (top-level or a feature body).
#[derive(Default)]
struct DepTables {
    python: Option<String>,
    conda: Vec<CondaDep>,
    pypi: Vec<Requirement>,
    skipped: usize,
}

fn parse_dep_tables(level: &toml::Table) -> DepTables {
    let mut out = DepTables::default();
    if let Some(deps) = level.get("dependencies").and_then(|d| d.as_table()) {
        for (name, value) in deps {
            let spec = conda_spec(value);
            if name.eq_ignore_ascii_case("python") {
                out.python = Some(spec);
            } else {
                out.conda.push(CondaDep {
                    name: name.clone(),
                    spec,
                });
            }
        }
    }
    if let Some(deps) = level.get("pypi-dependencies").and_then(|d| d.as_table()) {
        for (name, value) in deps {
            match pypi_spec(value) {
                Some(spec) => out.pypi.push(Requirement {
                    name: normalize_name(name),
                    spec,
                }),
                None => out.skipped += 1,
            }
        }
    }
    out
}

/// The raw conda spec of a `[dependencies]` value.
fn conda_spec(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.trim().to_string(),
        toml::Value::Table(t) => t
            .get("version")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "*".to_string()),
        _ => "*".to_string(),
    }
}

/// The PEP 440 specifier of a `[pypi-dependencies]` value, empty when
/// unconstrained. `None` for path, git and url sources, which no index serves.
fn pypi_spec(value: &toml::Value) -> Option<String> {
    let raw = match value {
        toml::Value::String(s) => s.as_str(),
        toml::Value::Table(t) => {
            if ["path", "git", "url"].iter().any(|k| t.contains_key(*k)) {
                return None;
            }
            t.get("version").and_then(|v| v.as_str()).unwrap_or("")
        }
        _ => "",
    };
    let spec: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    Some(if spec == "*" { String::new() } else { spec })
}

/// `name @ url` in `[project].dependencies`.
fn is_direct_reference(line: &str) -> bool {
    line.split(';').next().unwrap_or(line).contains('@')
}

fn system_cuda(level: &toml::Table) -> Option<String> {
    level
        .get("system-requirements")?
        .get("cuda")
        .and_then(scalar_string)
}

/// A TOML string or number as text (`cuda = 12` is accepted by pixi too).
fn scalar_string(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(s) => Some(s.trim().to_string()).filter(|s| !s.is_empty()),
        toml::Value::Integer(i) => Some(i.to_string()),
        toml::Value::Float(f) => Some(f.to_string()),
        _ => None,
    }
}

/// Keep one entry per name, preferring one that carries a constraint.
fn dedupe(reqs: Vec<Requirement>) -> Vec<Requirement> {
    let mut out: Vec<Requirement> = Vec::new();
    for r in reqs {
        match out.iter_mut().find(|o| o.name == r.name) {
            Some(existing) => {
                if !existing.is_pinned() && r.is_pinned() {
                    *existing = r;
                }
            }
            None => out.push(r),
        }
    }
    out
}

/// The Python minor the default environment will most likely get.
///
/// Exact and prefix specs (`3.12.*`, `3.12`, `==3.11.9`) give their minor;
/// ranges give the highest minor they allow (`>=3.10,<3.13` -> 3.12), which is
/// what the solver picks. `None` for unbounded or unparseable specs.
pub fn python_minor_hint(project: &PixiProject) -> Option<PyVersion> {
    python_spec_minor(project.python.as_deref()?)
}

/// [`python_minor_hint`] for a raw conda (or PEP 440) spec.
pub fn python_spec_minor(spec: &str) -> Option<PyVersion> {
    // `|` separates alternatives; the solver takes the newest across them.
    spec.split('|').filter_map(alternative_max_minor).max()
}

/// Highest minor any 3.x Python can have and still count as bounded.
const MINOR_CEILING: u8 = 99;

fn alternative_max_minor(spec: &str) -> Option<PyVersion> {
    // A build string can follow the version after whitespace (`3.12.* *_cp`),
    // but spaces inside the version (`>=3.10, <3.13`, `>= 3.10`) are not one.
    let version = join_version_spaces(spec);
    let version = version.split_whitespace().next()?;
    if version == "*" {
        return None;
    }
    let clauses: Vec<Clause> = version
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(|c| Clause::parse(&conda_clause_to_pep440(c)))
        .collect::<Option<_>>()?;
    if clauses.is_empty() {
        return None;
    }
    let allows = |minor: u8| {
        (0..=MINOR_CEILING as u64).any(|patch| {
            let candidate = [3, minor as u64, patch];
            clauses.iter().all(|c| c.matches(&candidate))
        })
    };
    if allows(MINOR_CEILING) {
        return None;
    }
    (0..MINOR_CEILING)
        .rev()
        .find(|&m| allows(m))
        .map(PyVersion::py3)
}

/// Drop whitespace that sits after an operator or around a comma, so only the
/// gap before a build string remains.
fn join_version_spaces(spec: &str) -> String {
    let mut out = String::with_capacity(spec.len());
    let mut chars = spec.trim().chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            let prev_glues = out.ends_with(['=', '<', '>', '!', '~', ',']);
            while chars.peek().is_some_and(|n| n.is_whitespace()) {
                chars.next();
            }
            let next_glues = chars.peek() == Some(&',');
            if !prev_glues && !next_glues {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Rewrite one conda comparator into PEP 440. A bare version and a single `=`
/// are prefix matches in conda (`3.12` allows 3.12.7), which PEP 440 spells
/// `==3.12.*`.
fn conda_clause_to_pep440(clause: &str) -> String {
    let starts_with_op = clause.starts_with(['<', '>', '!', '~']) || clause.starts_with("==");
    if starts_with_op {
        return clause.to_string();
    }
    let version = clause.trim_start_matches('=').trim();
    if version.ends_with(".*") || version.ends_with('*') {
        format!("=={}", version.trim_end_matches('*').trim_end_matches('.')) + ".*"
    } else {
        format!("=={version}.*")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pypilot-pixi-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn req(name: &str, spec: &str) -> Requirement {
        Requirement {
            name: name.into(),
            spec: spec.into(),
        }
    }

    #[test]
    fn pixi_toml_project() {
        let dir = scratch("pixi-toml");
        std::fs::write(
            dir.join("pixi.toml"),
            r#"
[workspace]
name = "demo"
channels = ["conda-forge"]
platforms = ["win-64", "linux-64", "osx-arm64"]

[dependencies]
python = "3.12.*"
numpy = ">=1.26,<2"
pandas = "*"
pytorch = { version = "2.5.*", channel = "pytorch" }

[pypi-dependencies]
requests = "*"
httpx = ">= 0.27"
rich = { version = ">=13", extras = ["jupyter"] }
mylib = { path = ".", editable = true }
other = { git = "https://github.com/a/b" }
wheel-thing = { url = "https://example.com/x.whl" }
private = { version = "==1.0", index = "https://example.com/simple" }
"#,
        )
        .unwrap();
        // pixi.toml wins even when a pyproject is present.
        std::fs::write(dir.join("pyproject.toml"), "[project]\nname = \"x\"\n").unwrap();

        let p = detect(&dir).unwrap();
        assert_eq!(p.format, ManifestFormat::PixiToml);
        assert_eq!(p.manifest, dir.join("pixi.toml"));
        assert_eq!(p.root, dir);
        assert_eq!(p.python.as_deref(), Some("3.12.*"));
        assert_eq!(p.platforms, ["win-64", "linux-64", "osx-arm64"]);
        assert_eq!(p.environments, ["default"]);
        assert_eq!(p.cuda, None);
        assert!(p.conda_deps.contains(&CondaDep {
            name: "numpy".into(),
            spec: ">=1.26,<2".into()
        }));
        assert!(p.conda_deps.contains(&CondaDep {
            name: "pytorch".into(),
            spec: "2.5.*".into()
        }));
        assert!(!p.conda_deps.iter().any(|d| d.name == "python"));
        assert!(p.pypi_deps.contains(&req("requests", "")));
        assert!(p.pypi_deps.contains(&req("httpx", ">=0.27")));
        assert!(p.pypi_deps.contains(&req("rich", ">=13")));
        assert!(p.pypi_deps.contains(&req("private", "==1.0")));
        assert_eq!(p.pypi_deps.len(), 4);
        assert_eq!(p.skipped_pypi_deps, 3);
        assert_eq!(python_minor_hint(&p), Some(PyVersion::py3(12)));
    }

    #[test]
    fn older_project_table_is_read() {
        let dir = scratch("legacy-project");
        std::fs::write(
            dir.join("pixi.toml"),
            "[project]\nname = \"old\"\nplatforms = [\"linux-64\"]\n",
        )
        .unwrap();
        let p = detect(&dir).unwrap();
        assert_eq!(p.platforms, ["linux-64"]);
        assert_eq!(p.python, None);
    }

    #[test]
    fn pyproject_workspace_project() {
        let dir = scratch("pyproject");
        std::fs::write(
            dir.join("pyproject.toml"),
            r#"
[project]
name = "demo"
requires-python = ">=3.10,<3.13"
dependencies = ["requests>=2", "pkg @ https://example.com/pkg.whl", "Typer[all]"]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.dependencies]
numpy = "*"

[tool.pixi.pypi-dependencies]
demo = { path = ".", editable = true }
"#,
        )
        .unwrap();
        let p = detect(&dir).unwrap();
        assert_eq!(p.format, ManifestFormat::Pyproject);
        assert_eq!(p.manifest, dir.join("pyproject.toml"));
        // requires-python stands in for the python conda dep.
        assert_eq!(p.python.as_deref(), Some(">=3.10,<3.13"));
        assert_eq!(python_minor_hint(&p), Some(PyVersion::py3(12)));
        assert_eq!(
            p.conda_deps,
            [CondaDep {
                name: "numpy".into(),
                spec: "*".into()
            }]
        );
        assert_eq!(p.pypi_deps, [req("requests", ">=2"), req("typer", "")]);
        // The editable self-install and the direct URL.
        assert_eq!(p.skipped_pypi_deps, 2);
    }

    #[test]
    fn pixi_python_beats_requires_python() {
        let text = r#"
[project]
requires-python = ">=3.9"
[tool.pixi.project]
platforms = ["win-64"]
[tool.pixi.dependencies]
python = "3.11.*"
"#;
        let p = parse_manifest(
            text,
            ManifestFormat::Pyproject,
            Path::new("."),
            "pyproject.toml".into(),
        )
        .unwrap();
        assert_eq!(p.python.as_deref(), Some("3.11.*"));
    }

    #[test]
    fn plain_pyproject_is_not_pixi() {
        let dir = scratch("plain");
        std::fs::write(
            dir.join("pyproject.toml"),
            "[project]\nname = \"x\"\ndependencies = [\"numpy\"]\n\n[tool.ruff]\nline-length = 100\n",
        )
        .unwrap();
        assert!(detect(&dir).is_none());

        // `[tool.pixi]` alone (say, only a `[tool.pixi.dependencies]` table)
        // is not a workspace either.
        std::fs::write(
            dir.join("pyproject.toml"),
            "[tool.pixi.dependencies]\nnumpy = \"*\"\n",
        )
        .unwrap();
        assert!(detect(&dir).is_none());

        let empty = scratch("empty");
        assert!(detect(&empty).is_none());
    }

    #[test]
    fn malformed_toml_is_none() {
        let dir = scratch("malformed");
        std::fs::write(dir.join("pixi.toml"), "[workspace\nname = ").unwrap();
        assert!(detect(&dir).is_none());

        let dir = scratch("malformed-pyproject");
        std::fs::write(dir.join("pyproject.toml"), "[tool.pixi.workspace]\n= = =").unwrap();
        assert!(detect(&dir).is_none());

        // Wrong types where tables are expected are tolerated, not fatal.
        let p = parse_manifest(
            "dependencies = 3\npypi-dependencies = \"x\"\nenvironments = []\n[workspace]\nplatforms = 7\n",
            ManifestFormat::PixiToml,
            Path::new("."),
            "pixi.toml".into(),
        )
        .unwrap();
        assert!(p.conda_deps.is_empty() && p.platforms.is_empty());
        assert_eq!(p.environments, ["default"]);
    }

    #[test]
    fn features_and_environments() {
        let text = r#"
[workspace]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"

[feature.test.dependencies]
pytest = "*"

[feature.test.pypi-dependencies]
hypothesis = ">=6"

[feature.py313.dependencies]
python = "3.13.*"

[feature.gpu.system-requirements]
cuda = "12"

[environments]
test = ["test"]
gpu = { features = ["gpu"], solve-group = "default" }
py313 = ["py313", "test"]
"#;
        let p = parse_manifest(
            text,
            ManifestFormat::PixiToml,
            Path::new("."),
            "pixi.toml".into(),
        )
        .unwrap();
        assert_eq!(p.environments[0], "default");
        let mut rest = p.environments[1..].to_vec();
        rest.sort();
        assert_eq!(rest, ["gpu", "py313", "test"]);
        // Feature deps do not leak into the default environment.
        assert!(p.conda_deps.is_empty());
        assert!(p.pypi_deps.is_empty());
        assert_eq!(p.cuda, None);

        let feature = |n: &str| p.features.iter().find(|f| f.name == n).unwrap();
        assert_eq!(feature("test").conda_deps[0].name, "pytest");
        assert_eq!(feature("test").pypi_deps, [req("hypothesis", ">=6")]);
        assert_eq!(feature("py313").python.as_deref(), Some("3.13.*"));
        assert_eq!(feature("gpu").cuda.as_deref(), Some("12"));
    }

    #[test]
    fn cuda_from_platform_entry() {
        let text = r#"
[workspace]
platforms = ["win-64", { platform = "linux-64", cuda = "12.0" }]
"#;
        let p = parse_manifest(
            text,
            ManifestFormat::PixiToml,
            Path::new("."),
            "pixi.toml".into(),
        )
        .unwrap();
        assert_eq!(p.platforms, ["win-64", "linux-64"]);
        assert_eq!(p.cuda.as_deref(), Some("12.0"));
    }

    #[test]
    fn cuda_from_legacy_system_requirements() {
        let text =
            "[workspace]\nplatforms = [\"linux-64\"]\n\n[system-requirements]\ncuda = \"12\"\n";
        let p = parse_manifest(
            text,
            ManifestFormat::PixiToml,
            Path::new("."),
            "pixi.toml".into(),
        )
        .unwrap();
        assert_eq!(p.cuda.as_deref(), Some("12"));

        let text =
            "[tool.pixi.workspace]\nplatforms = []\n[tool.pixi.system-requirements]\ncuda = 11\n";
        let p = parse_manifest(
            text,
            ManifestFormat::Pyproject,
            Path::new("."),
            "pyproject.toml".into(),
        )
        .unwrap();
        assert_eq!(p.cuda.as_deref(), Some("11"));
    }

    #[test]
    fn python_spec_table() {
        let cases: &[(&str, Option<u8>)] = &[
            ("3.12.*", Some(12)),
            ("3.12", Some(12)),
            ("=3.12", Some(12)),
            ("3.12.7", Some(12)),
            ("==3.11.9", Some(11)),
            ("==3.11.*", Some(11)),
            (">=3.10,<3.13", Some(12)),
            (">=3.10, <3.13", Some(12)),
            (">= 3.10 , < 3.13", Some(12)),
            (">=3.10,<=3.12.4", Some(12)),
            (">=3.9,<3.12.0", Some(11)),
            ("~=3.10.0", Some(10)),
            ("<3.12", Some(11)),
            ("3.10.*|3.11.*", Some(11)),
            ("3.12.* *_cpython", Some(12)),
            ("3.15.*", Some(15)),
            (">=3.10", None),
            ("~=3.10", None),
            ("*", None),
            ("", None),
            ("banana", None),
        ];
        for (spec, want) in cases {
            assert_eq!(
                python_spec_minor(spec),
                want.map(PyVersion::py3),
                "spec {spec:?}"
            );
        }
    }

    /// `pixi init` + `pixi add python=3.12 numpy` + `pixi add --pypi requests`,
    /// as written by pixi 0.81.0.
    #[test]
    fn real_pixi_toml_from_pixi_init() {
        let text = include_str!("../../tests/fixtures/pixi/pixi-0.81-init.pixi.toml");
        let p = parse_manifest(
            text,
            ManifestFormat::PixiToml,
            Path::new("projA"),
            "projA/pixi.toml".into(),
        )
        .unwrap();
        assert_eq!(p.python.as_deref(), Some("3.12.*"));
        assert_eq!(python_minor_hint(&p), Some(PyVersion::py3(12)));
        assert_eq!(
            p.conda_deps,
            [CondaDep {
                name: "numpy".into(),
                spec: ">=2.5.4,<3".into()
            }]
        );
        // pixi writes `">=2.34.2, <3"`, with a space.
        assert_eq!(p.pypi_deps, [req("requests", ">=2.34.2,<3")]);
        assert_eq!(p.platforms, ["win-64"]);
        assert_eq!(p.environments, ["default"]);
    }

    /// The same with `pixi init --format pyproject`: `--pypi` deps land in
    /// `[project].dependencies`, and the project installs itself editable.
    #[test]
    fn real_pyproject_from_pixi_init() {
        let text = include_str!("../../tests/fixtures/pixi/pixi-0.81-init.pyproject.toml");
        let p = parse_manifest(
            text,
            ManifestFormat::Pyproject,
            Path::new("projB"),
            "projB/pyproject.toml".into(),
        )
        .unwrap();
        // pixi's own python dep wins over init's `requires-python = ">= 3.11"`.
        assert_eq!(p.python.as_deref(), Some("3.12.*"));
        assert_eq!(p.pypi_deps, [req("requests", ">=2.34.2,<3")]);
        assert_eq!(p.skipped_pypi_deps, 1);
        assert!(p.conda_deps.is_empty());
        assert_eq!(python_spec_minor(">= 3.11"), None);
    }
}
