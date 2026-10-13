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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::core::command;
use crate::core::installed;
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

/// An installed pixi environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixiEnv {
    pub name: String,
    pub prefix: PathBuf,
    pub python_exe: PathBuf,
    /// From `conda-meta/python-X.Y.Z-<build>.json`, when Python is installed.
    pub python_version: Option<PyVersion>,
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

/// The installed environment `name` of a project, if its prefix exists under
/// `<root>/.pixi/envs/`. Detached environments live elsewhere; find those with
/// [`info_envs`] and build them with [`env_at`].
pub fn env(project: &PixiProject, name: &str) -> Option<PixiEnv> {
    let prefix = project.root.join(".pixi").join("envs").join(name);
    prefix.is_dir().then(|| env_at(name, &prefix))
}

/// Describe the environment at `prefix`, wherever it lives.
pub fn env_at(name: &str, prefix: &Path) -> PixiEnv {
    PixiEnv {
        name: name.to_string(),
        prefix: prefix.to_path_buf(),
        python_exe: interpreter_path(prefix),
        python_version: python_version_from_conda_meta(prefix),
    }
}

/// The interpreter of a conda prefix on this OS.
pub fn interpreter_path(prefix: &Path) -> PathBuf {
    interpreter_path_for(prefix, cfg!(windows))
}

fn interpreter_path_for(prefix: &Path, windows: bool) -> PathBuf {
    if windows {
        prefix.join("python.exe")
    } else {
        prefix.join("bin").join("python")
    }
}

/// The Python minor installed in a conda prefix, read from the
/// `conda-meta/python-X.Y.Z-<build>.json` record conda writes per package.
pub fn python_version_from_conda_meta(prefix: &Path) -> Option<PyVersion> {
    let entries = std::fs::read_dir(prefix.join("conda-meta")).ok()?;
    entries
        .flatten()
        .filter_map(|e| conda_meta_python(&e.file_name().to_string_lossy()))
        .max()
}

/// `python-3.12.7-h1234_0.json` -> 3.12. Other packages whose names start with
/// `python` (`python-dateutil-...`, `python_abi-...`) are rejected because the
/// field after `python-` is not a version.
fn conda_meta_python(file: &str) -> Option<PyVersion> {
    let rest = file.strip_suffix(".json")?.strip_prefix("python-")?;
    let version = rest.split('-').next()?;
    if !version.starts_with(|c: char| c.is_ascii_digit()) || !version.contains('.') {
        return None;
    }
    PyVersion::parse(version)
}

/// `site-packages` directories of a conda prefix: `Lib/site-packages` on
/// Windows, `lib/pythonX.Y/site-packages` elsewhere. Same layout as a venv.
pub fn site_packages_dirs(prefix: &Path) -> Vec<PathBuf> {
    installed::site_packages_dirs(prefix)
}

/// The `conda-meta/pixi` record pixi writes into every environment it
/// installs: which manifest and environment the prefix was built for.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EnvMarker {
    pub manifest_path: PathBuf,
    pub environment_name: String,
    #[serde(default)]
    pub pixi_version: Option<String>,
}

/// Read a prefix's [`EnvMarker`]. `None` for prefixes pixi did not install
/// (or installed with a release that predates the record).
pub fn env_marker(prefix: &Path) -> Option<EnvMarker> {
    parse_env_marker(&std::fs::read_to_string(prefix.join("conda-meta").join("pixi")).ok()?)
}

/// Parse the JSON of a `conda-meta/pixi` record.
pub fn parse_env_marker(text: &str) -> Option<EnvMarker> {
    serde_json::from_str(text).ok()
}

/// Which installer put a distribution into a pixi environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistOrigin {
    /// A conda package. conda-forge Python packages carry a `.dist-info` too,
    /// with `conda` in its `INSTALLER` file.
    Conda,
    /// Installed from PyPI by pixi's uv (`INSTALLER` is `uv-pixi`) or any
    /// other Python installer.
    Pypi,
    /// No `INSTALLER` file to tell.
    Unknown,
}

/// Every distribution in a prefix's `site-packages`, normalized per PEP 503,
/// with the installer that put it there.
pub fn dist_origins(prefix: &Path) -> BTreeMap<String, DistOrigin> {
    let mut out = BTreeMap::new();
    for dir in site_packages_dirs(prefix) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file = entry.file_name();
            let Some(name) = installed::distribution_name(&file.to_string_lossy()) else {
                continue;
            };
            out.insert(name, dist_origin(&entry.path()));
        }
    }
    out
}

/// Classify one `.dist-info` directory by its `INSTALLER` file.
pub fn dist_origin(dist_info: &Path) -> DistOrigin {
    match std::fs::read_to_string(dist_info.join("INSTALLER")) {
        Ok(text) if text.trim().eq_ignore_ascii_case("conda") => DistOrigin::Conda,
        Ok(text) if !text.trim().is_empty() => DistOrigin::Pypi,
        _ => DistOrigin::Unknown,
    }
}

#[derive(Deserialize, Default)]
struct InfoJson {
    #[serde(default)]
    environments_info: Vec<InfoEnv>,
}

#[derive(Deserialize)]
struct InfoEnv {
    #[serde(default)]
    name: String,
    #[serde(default)]
    prefix: Option<PathBuf>,
}

/// `(name, prefix)` for every environment in `pixi info --json` output.
/// Malformed input, or entries without a name or prefix, are skipped.
pub fn parse_info_json(text: &str) -> Vec<(String, PathBuf)> {
    let info: InfoJson = serde_json::from_str(text).unwrap_or_default();
    info.environments_info
        .into_iter()
        .filter_map(|e| {
            let prefix = e.prefix.filter(|p| !p.as_os_str().is_empty())?;
            (!e.name.is_empty()).then_some((e.name, prefix))
        })
        .collect()
}

/// How long `pixi info` may take before we give up on it.
const INFO_TIMEOUT: Duration = Duration::from_secs(10);

/// Ask pixi where a project's environments live. A fallback for detached
/// environments: the default layout is read straight off disk by [`env`].
/// Empty when pixi is missing, slow or fails.
pub async fn info_envs(root: &Path) -> Vec<(String, PathBuf)> {
    let Some((pixi, _)) = find_pixi().await else {
        return Vec::new();
    };
    let root_arg = root.to_string_lossy().into_owned();
    let args = ["info", "--json", "--manifest-path", root_arg.as_str()];
    match command::run_with_timeout(&pixi, &args, Some(root), INFO_TIMEOUT).await {
        Ok(out) if out.success() => parse_info_json(&out.stdout),
        _ => Vec::new(),
    }
}

/// The installed pixi version (`0.40.3`), if pixi can be found.
pub async fn pixi_version() -> Option<String> {
    find_pixi().await.map(|(_, version)| version)
}

/// Locate a working pixi: PATH first, then `$PIXI_HOME/bin` and
/// `~/.pixi/bin`, where the official installer puts it (an editor started
/// before the install may not have it on PATH yet).
async fn find_pixi() -> Option<(PathBuf, String)> {
    if let Some(v) = command::probe_version("pixi", &["--version"]).await {
        return Some((PathBuf::from("pixi"), clean_version(&v)));
    }
    let bin = if cfg!(windows) { "pixi.exe" } else { "pixi" };
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("PIXI_HOME") {
        candidates.push(PathBuf::from(home).join("bin").join(bin));
    }
    if let Some(base) = directories::BaseDirs::new() {
        candidates.push(base.home_dir().join(".pixi").join("bin").join(bin));
    }
    for candidate in candidates {
        if candidate.is_file() {
            if let Some(v) = command::probe_version(&candidate, &["--version"]).await {
                return Some((candidate, clean_version(&v)));
            }
        }
    }
    None
}

/// `pixi 0.40.3` -> `0.40.3`.
fn clean_version(line: &str) -> String {
    line.trim()
        .strip_prefix("pixi")
        .map(str::trim)
        .unwrap_or(line.trim())
        .to_string()
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

    #[test]
    fn interpreter_path_per_os() {
        let prefix = Path::new("proj").join(".pixi").join("envs").join("default");
        assert_eq!(
            interpreter_path_for(&prefix, true),
            prefix.join("python.exe")
        );
        assert_eq!(
            interpreter_path_for(&prefix, false),
            prefix.join("bin").join("python")
        );
    }

    fn fake_project(root: &Path) -> PixiProject {
        std::fs::write(root.join("pixi.toml"), "[workspace]\nplatforms = []\n").unwrap();
        detect(root).unwrap()
    }

    fn write_meta(prefix: &Path) {
        let meta = prefix.join("conda-meta");
        std::fs::create_dir_all(&meta).unwrap();
        for f in [
            "python-3.12.7-h1234_0.json",
            "python-dateutil-2.9.0-pyhd8ed1ab_0.json",
            "python_abi-3.12-9_cp312.json",
            "libpython-3.13.1-hcc31f7b_1_cpython.json",
            "pixi_env_prefix",
            "pip-24.2-pyh8b19718_1.json",
            "history",
        ] {
            std::fs::write(meta.join(f), "{}").unwrap();
        }
    }

    #[test]
    fn windows_env_layout() {
        let root = scratch("env-windows");
        let project = fake_project(&root);
        assert!(env(&project, "default").is_none());

        let prefix = root.join(".pixi").join("envs").join("default");
        let sp = prefix.join("Lib").join("site-packages");
        std::fs::create_dir_all(&sp).unwrap();
        std::fs::write(prefix.join("python.exe"), "").unwrap();
        write_meta(&prefix);

        let e = env(&project, "default").unwrap();
        assert_eq!(e.name, "default");
        assert_eq!(e.prefix, prefix);
        assert_eq!(e.python_exe, interpreter_path(&prefix));
        assert_eq!(e.python_version, Some(PyVersion::py3(12)));
        assert_eq!(site_packages_dirs(&prefix), [sp]);
        assert!(env(&project, "test").is_none());
    }

    #[test]
    fn posix_env_layout() {
        let root = scratch("env-posix");
        let project = fake_project(&root);
        let prefix = root.join(".pixi").join("envs").join("test");
        let sp = prefix.join("lib").join("python3.12").join("site-packages");
        std::fs::create_dir_all(&sp).unwrap();
        std::fs::create_dir_all(prefix.join("bin")).unwrap();
        std::fs::write(prefix.join("bin").join("python"), "").unwrap();
        write_meta(&prefix);

        let e = env(&project, "test").unwrap();
        assert_eq!(e.python_version, Some(PyVersion::py3(12)));
        assert_eq!(site_packages_dirs(&prefix), [sp]);
    }

    #[test]
    fn conda_meta_without_python() {
        let root = scratch("no-python");
        std::fs::create_dir_all(root.join("conda-meta")).unwrap();
        std::fs::write(
            root.join("conda-meta").join("numpy-2.1.0-py312_0.json"),
            "{}",
        )
        .unwrap();
        assert_eq!(python_version_from_conda_meta(&root), None);
        assert_eq!(python_version_from_conda_meta(&root.join("missing")), None);
        assert_eq!(
            conda_meta_python("python-3.9.18-h1_0_cpython.json"),
            Some(PyVersion::py3(9))
        );
        assert_eq!(conda_meta_python("python-3.12.7-h1234_0.txt"), None);
    }

    #[test]
    fn info_json_parsing() {
        let text = r#"{
  "platform": "win-64",
  "virtual_packages": ["__win=0=0"],
  "version": "0.40.3",
  "project_info": { "name": "demo", "manifest_path": "C:\\demo\\pixi.toml" },
  "environments_info": [
    { "name": "default", "features": ["default"], "prefix": "C:\\demo\\.pixi\\envs\\default", "dependencies": ["python"] },
    { "name": "gpu", "prefix": "/elsewhere/envs/demo-gpu" },
    { "name": "", "prefix": "/nameless" },
    { "name": "noprefix" }
  ]
}"#;
        assert_eq!(
            parse_info_json(text),
            [
                (
                    "default".to_string(),
                    PathBuf::from("C:\\demo\\.pixi\\envs\\default")
                ),
                ("gpu".to_string(), PathBuf::from("/elsewhere/envs/demo-gpu")),
            ]
        );
        assert!(parse_info_json("not json").is_empty());
        assert!(parse_info_json("{}").is_empty());
        assert!(parse_info_json(r#"{"environments_info": null}"#).is_empty());
    }

    #[test]
    fn version_line_is_cleaned() {
        assert_eq!(clean_version("pixi 0.40.3\n"), "0.40.3");
        assert_eq!(clean_version("0.40.3"), "0.40.3");
    }

    /// `pixi info --json` from pixi 0.81.0, home directory redacted.
    #[test]
    fn real_info_json() {
        let text = include_str!("../../tests/fixtures/pixi/pixi-0.81-info.json");
        assert_eq!(
            parse_info_json(text),
            [(
                "default".to_string(),
                PathBuf::from(
                    "<HOME>\\AppData\\Local\\Temp\\pixi-portable\\projA\\.pixi\\envs\\default"
                )
            )]
        );
    }

    #[test]
    fn env_marker_names_the_project() {
        let text = r#"{
  "manifest_path": "C:\\work\\projA\\pixi.toml",
  "environment_name": "default",
  "pixi_version": "0.81.0",
  "environment_lock_file_hash": "697b9a994ccfa7c9",
  "resolved_platform": { "subdir": "win-64", "virtual_packages": ["__win=10.0"] }
}"#;
        let root = scratch("marker");
        std::fs::create_dir_all(root.join("conda-meta")).unwrap();
        assert_eq!(env_marker(&root), None);
        std::fs::write(root.join("conda-meta").join("pixi"), text).unwrap();
        assert_eq!(
            env_marker(&root),
            Some(EnvMarker {
                manifest_path: PathBuf::from("C:\\work\\projA\\pixi.toml"),
                environment_name: "default".into(),
                pixi_version: Some("0.81.0".into()),
            })
        );
        assert_eq!(parse_env_marker("{}"), None);
        assert_eq!(parse_env_marker("garbage"), None);
    }

    #[test]
    fn dist_origin_from_installer_file() {
        let prefix = scratch("origins");
        let sp = prefix.join("Lib").join("site-packages");
        for (dist, installer) in [
            ("numpy-2.5.4.dist-info", Some("conda\n")),
            ("charset_normalizer-3.5.2.dist-info", Some("uv-pixi\n")),
            ("projb-0.1.0.dist-info", Some("uv")),
            ("mystery-1.0.dist-info", None),
        ] {
            std::fs::create_dir_all(sp.join(dist)).unwrap();
            if let Some(text) = installer {
                std::fs::write(sp.join(dist).join("INSTALLER"), text).unwrap();
            }
        }
        std::fs::create_dir_all(sp.join("numpy")).unwrap();
        let origins = dist_origins(&prefix);
        assert_eq!(origins.len(), 4);
        assert_eq!(origins["numpy"], DistOrigin::Conda);
        assert_eq!(origins["charset-normalizer"], DistOrigin::Pypi);
        assert_eq!(origins["projb"], DistOrigin::Pypi);
        assert_eq!(origins["mystery"], DistOrigin::Unknown);
        assert!(dist_origins(&prefix.join("missing")).is_empty());
    }
}
