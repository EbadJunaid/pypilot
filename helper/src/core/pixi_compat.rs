//! pixi failures, explained: which Python would work, and the command to get it.
//!
//! When `pixi add --pypi` or `pixi install` fails, pixi relays uv's resolver
//! verdict ("has no wheels with a matching Python ABI tag (e.g., `cp312`)")
//! without saying which Python the package *does* support or how to switch
//! (prefix-dev/pixi#5230, #5752, #5266). [`explain_pixi_error`] turns that
//! stderr into one paragraph with a next step.

use crate::core::pixi::{self, ManifestFormat, PixiError, PixiProject};
use crate::core::project::normalize_name;
use crate::matrix;
use crate::pypi::{CompatReport, PyVersion, PyVersionSet};

/// Explain the stderr of a failed `pixi add` / `pixi install` in one paragraph
/// that ends with what to do next.
pub fn explain_pixi_error(stderr: &str, project: Option<&PixiProject>) -> String {
    explain_pixi_error_with(stderr, project, None)
}

/// [`explain_pixi_error`] with a PyPI analysis of the project's dependencies
/// at hand, which supplies the supported Python range when pixi's own hint
/// does not list one.
pub fn explain_pixi_error_with(
    stderr: &str,
    project: Option<&PixiProject>,
    compat: Option<&CompatReport>,
) -> String {
    let text = pixi::unwrap_report(stderr);
    let platform = target_platform(&text);
    match pixi::classify_pixi_error(stderr) {
        PixiError::NoMatchingWheel { package, abi } => explain_no_wheel(
            &text,
            &package,
            abi.as_deref(),
            platform.as_deref(),
            project,
            compat,
        ),
        PixiError::NotFound { package } => explain_not_found(&package, project),
        PixiError::Unsolvable => explain_unsolvable(&text, platform.as_deref(), project),
        PixiError::Other => explain_other(&text),
    }
}

/// `win-64` from "for platform 'win-64'". miette may wrap inside the
/// quotes (`'win- 64'`), so whitespace in the name is dropped.
fn target_platform(text: &str) -> Option<String> {
    let rest = text.split_once("for platform '")?.1;
    let name: String = rest
        .split_once('\'')?
        .0
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    (!name.is_empty()).then_some(name)
}

fn on_platform(platform: Option<&str>) -> String {
    platform.map(|p| format!(" on {p}")).unwrap_or_default()
}

/// The text between `start` and the next `end` after it.
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let rest = text.split_once(start)?.1;
    Some(rest.split_once(end).map_or(rest, |(a, _)| a))
}

fn explain_no_wheel(
    text: &str,
    package: &str,
    abi: Option<&str>,
    platform: Option<&str>,
    project: Option<&PixiProject>,
    compat: Option<&CompatReport>,
) -> String {
    // uv's hint: "You require CPython 3.12 (`cp312`), but we only found wheels
    // for `tensorflow` (v2.15.0) with the following Python ABI tags: `cp39`, ..."
    let python = between(text, "You require CPython ", " ")
        .and_then(PyVersion::parse)
        .or_else(|| abi.and_then(PyVersion::parse))
        .or_else(|| project.and_then(pixi::python_minor_hint));
    let version = between(text, "but we only found wheels for", ")")
        .and_then(|s| s.split_once("(v"))
        .map(|(_, v)| v.trim().to_string());
    let hinted: PyVersionSet = text
        .split_once("Python ABI tags:")
        .map(|(_, tags)| {
            PyVersionSet::from_versions(tags.split('`').skip(1).step_by(2).filter_map(abi_python))
        })
        .unwrap_or_default();

    let analysed = compat.and_then(|c| {
        c.per_package
            .iter()
            .find(|p| normalize_name(&p.name) == normalize_name(package))
    });
    let supported = match analysed {
        Some(p) if !p.supported.is_empty() => Some(p.supported.clone()),
        _ => (!hinted.is_empty()).then_some(hinted),
    };
    // The newest Python every dependency supports, when the whole project was
    // analysed; else the newest this package supports.
    let target = compat
        .map(|c| &c.intersection)
        .filter(|i| !i.is_empty())
        .and_then(PyVersionSet::max)
        .or_else(|| supported.as_ref().and_then(PyVersionSet::max));

    let who = match &version {
        Some(v) => format!("{package} {v}"),
        None => package.to_string(),
    };
    let tag = python
        .map(|p| p.cp_tag())
        .or_else(|| abi.map(str::to_string));
    let which = match (python, &tag) {
        (Some(p), Some(t)) => format!("Python {p} (`{t}`)"),
        _ => "this environment's Python".to_string(),
    };
    let mut out = format!("{who} has no wheel for {which}{}", on_platform(platform));
    match (&supported, target) {
        (Some(s), Some(t)) => out.push_str(&format!(
            "; it only ships wheels for Python {}. Run `pixi add python={t}` to move the environment to Python {t}",
            s.to_range_string()
        )),
        _ => out.push_str(&format!(
            ". Check which Python versions its wheels cover at https://pypi.org/project/{package}/#files, then run `pixi add python=3.X` with one of them"
        )),
    }
    match tag {
        Some(t) => out.push_str(&format!(
            ", or require a {package} release that publishes `{t}` wheels."
        )),
        None => out.push('.'),
    }
    out
}

/// `cp311` -> 3.11; `abi3`, `none` and PyPy tags -> none.
fn abi_python(tag: &str) -> Option<PyVersion> {
    tag.strip_prefix("cp")?;
    PyVersion::parse(tag)
}

fn explain_not_found(package: &str, project: Option<&PixiProject>) -> String {
    let index_table = match project.map(|p| p.format) {
        Some(ManifestFormat::Pyproject) => "`[tool.pixi.pypi-options]` in pyproject.toml",
        _ => "`[pypi-options]` in pixi.toml",
    };
    let import_hint = if matrix::is_mapped(package) {
        let pypi_name = matrix::resolve_package(package);
        format!(
            " `{package}` is an import name: the PyPI package is `{pypi_name}`, so run `pixi add --pypi {pypi_name}`."
        )
    } else {
        " Check the spelling; PyPI names often differ from import names (`scikit-learn`, not `sklearn`).".to_string()
    };
    format!(
        "`{package}` was not found on PyPI or any configured index.{import_hint} If it is a conda package, add it without `--pypi`; if it lives on a private index, add that index under `extra-index-urls` in {index_table}."
    )
}

/// How many unavailable requirements an Unsolvable explanation lists.
const MAX_LEAVES: usize = 3;

fn explain_unsolvable(text: &str, platform: Option<&str>, project: Option<&PixiProject>) -> String {
    let on = on_platform(platform);
    if let Some(root) = between(text, "Cannot solve the request because of:", ":") {
        // Conda: "tensorflow * cannot be installed because there are no viable options"
        let root = root
            .split(" because there are no viable options")
            .next()
            .unwrap_or(root)
            .trim();
        let package = root.split_whitespace().next().unwrap_or(root);
        let leaves = unavailable_requirements(text);
        let mut out = format!("pixi could not solve the conda dependencies{on}: {root}.");
        if !leaves.is_empty() {
            out.push_str(&format!(
                " Every candidate needs something that is not available: {}.",
                leaves.join("; ")
            ));
        }
        if leaves.iter().any(|l| l.starts_with("python ")) {
            let current = project
                .and_then(pixi::python_minor_hint)
                .map(|p| format!(" (this environment asks for Python {p})"))
                .unwrap_or_default();
            out.push_str(&format!(
                " Its builds need a different Python{current}: change the `python` pin with `pixi add python=3.X`, or pin a {package} version built for your Python; `pixi search {package}` lists the builds for this platform."
            ));
        } else {
            out.push_str(&format!(
                " Loosen one of the pins named above; `pixi search {package}` lists the builds for this platform."
            ));
        }
        return out;
    }
    if let Some(reason) = between(text, "Because ", ", we can conclude") {
        return format!(
            "pixi could not solve the PyPI dependencies{on}: {}. Loosen or remove one of the pins named here.",
            reason.trim()
        );
    }
    format!(
        "pixi could not solve the environment{on}: {}. Loosen the version pins it names.",
        summary(text)
    )
}

/// The requirements a conda solve found no candidates for, in order of
/// appearance, without repeats: `numpy >=1.16.1,<2.0.0a0`, `python 3.6.*`.
fn unavailable_requirements(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = text;
    while let Some((before, after)) = rest.split_once(", for which no candidates were found") {
        if let Some((_, leaf)) = before.rsplit_once(" would require ") {
            let leaf = leaf.trim().to_string();
            if !out.contains(&leaf) {
                out.push(leaf);
            }
        }
        if out.len() == MAX_LEAVES {
            break;
        }
        rest = after;
    }
    out
}

/// How much of an unrecognised error is quoted back.
const SUMMARY_CHARS: usize = 300;

/// The error text without pixi's `Error: ×` lead-in, cut to [`SUMMARY_CHARS`].
fn summary(text: &str) -> String {
    let text = text.trim();
    let text = text.strip_prefix("Error:").unwrap_or(text);
    let text = text.trim_start_matches(|c: char| c.is_whitespace() || c == '×');
    let mut out: String = text.chars().take(SUMMARY_CHARS).collect();
    if text.chars().count() > SUMMARY_CHARS {
        out.push('…');
    }
    out
}

fn explain_other(text: &str) -> String {
    let summary = summary(text);
    if summary.is_empty() {
        "pixi failed without an error message. Re-run the command with `-v` to see what it was doing.".to_string()
    } else {
        format!("pixi failed: {summary}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pixi::{parse_manifest, ManifestFormat};
    use crate::pypi::metadata::PackageAnalysis;
    use std::path::Path;

    const NO_WHEEL: &str = include_str!("../../tests/fixtures/pixi/pixi-0.81-err-no-wheel.txt");
    const NOT_FOUND: &str = include_str!("../../tests/fixtures/pixi/pixi-0.81-err-not-found.txt");
    const CONDA_CONFLICT: &str =
        include_str!("../../tests/fixtures/pixi/pixi-0.81-err-conda-conflict.txt");

    fn project(manifest: &str) -> PixiProject {
        parse_manifest(
            manifest,
            ManifestFormat::PixiToml,
            Path::new("/proj"),
            Path::new("/proj/pixi.toml").to_path_buf(),
        )
        .unwrap()
    }

    #[test]
    fn explains_missing_wheel_from_uv_hint() {
        let msg = explain_pixi_error(NO_WHEEL, None);
        assert_eq!(
            msg,
            "tensorflow 2.15.0 has no wheel for Python 3.12 (`cp312`) on win-64; it only ships wheels for Python 3.9–3.11. Run `pixi add python=3.11` to move the environment to Python 3.11, or require a tensorflow release that publishes `cp312` wheels."
        );
        assert!(!msg.contains('\u{1b}'));
    }

    #[test]
    fn missing_wheel_prefers_the_project_analysis() {
        let compat = CompatReport {
            platform: crate::core::platform::Platform::current(),
            per_package: vec![PackageAnalysis {
                name: "tensorflow".into(),
                version: "2.15.0".into(),
                supported: PyVersionSet::from_versions([PyVersion::py3(10), PyVersion::py3(11)]),
                has_platform_wheels: true,
                sdist_only: false,
                has_foreign_wheels: false,
                requires_python: Some(">=3.9".into()),
            }],
            unresolved: Vec::new(),
            intersection: PyVersionSet::from_versions([PyVersion::py3(10)]),
            conflicts: Vec::new(),
            sdist_warnings: Vec::new(),
        };
        let msg = explain_pixi_error_with(NO_WHEEL, None, Some(&compat));
        assert!(
            msg.contains("it only ships wheels for Python 3.10–3.11"),
            "{msg}"
        );
        assert!(msg.contains("`pixi add python=3.10`"), "{msg}");
    }

    #[test]
    fn missing_wheel_without_hint_stays_generic() {
        let stderr = "Error:   × failed to solve the pypi requirements of environment 'default' for\n  │ platform 'linux-64'\n  ╰─▶ Because foo==1.0 has no wheels with a matching Python ABI tag (e.g., `cp313`) and you require foo==1.0, we can conclude that your requirements are unsatisfiable.\n";
        assert_eq!(
            explain_pixi_error(stderr, None),
            "foo has no wheel for Python 3.13 (`cp313`) on linux-64. Check which Python versions its wheels cover at https://pypi.org/project/foo/#files, then run `pixi add python=3.X` with one of them, or require a foo release that publishes `cp313` wheels."
        );
    }

    #[test]
    fn explains_unknown_package() {
        let msg = explain_pixi_error(NOT_FOUND, None);
        assert_eq!(
            msg,
            "`definitely-not-a-real-package-xyz` was not found on PyPI or any configured index. Check the spelling; PyPI names often differ from import names (`scikit-learn`, not `sklearn`). If it is a conda package, add it without `--pypi`; if it lives on a private index, add that index under `extra-index-urls` in `[pypi-options]` in pixi.toml."
        );
    }

    #[test]
    fn unknown_import_name_points_at_the_pypi_name() {
        let stderr = NOT_FOUND.replace("definitely-not-a-real-package-xyz", "sklearn");
        let msg = explain_pixi_error(&stderr, None);
        assert!(
            msg.contains(
                "the PyPI package is `scikit-learn`, so run `pixi add --pypi scikit-learn`"
            ),
            "{msg}"
        );
    }

    #[test]
    fn explains_conda_conflict() {
        let proj = project("[workspace]\nname = \"p\"\n[dependencies]\npython = \"3.14.*\"\n");
        let msg = explain_pixi_error(CONDA_CONFLICT, Some(&proj));
        assert_eq!(
            msg,
            "pixi could not solve the conda dependencies on win-64: tensorflow * cannot be installed. Every candidate needs something that is not available: numpy >=1.16.1,<2.0.0a0; python 3.6.*; python >=3.6,<3.7.0a0. Its builds need a different Python (this environment asks for Python 3.14): change the `python` pin with `pixi add python=3.X`, or pin a tensorflow version built for your Python; `pixi search tensorflow` lists the builds for this platform."
        );
    }

    #[test]
    fn explains_pypi_conflict() {
        let stderr = "Error:   × failed to solve the pypi requirements of environment 'default' for\n  │ platform 'win-64'\n  ╰─▶ Because a==1.0 depends on b<2 and you require b>=2, we can conclude that your requirements are unsatisfiable.\n";
        assert_eq!(
            explain_pixi_error(stderr, None),
            "pixi could not solve the PyPI dependencies on win-64: a==1.0 depends on b<2 and you require b>=2. Loosen or remove one of the pins named here."
        );
    }

    #[test]
    fn other_errors_are_quoted() {
        assert_eq!(
            explain_pixi_error(
                "Error:   × could not reach https://conda.anaconda.org\n",
                None
            ),
            "pixi failed: could not reach https://conda.anaconda.org"
        );
        assert!(explain_pixi_error("", None).starts_with("pixi failed without an error message"));
    }
}
