//! What is already installed in a project's virtualenv.
//!
//! Read straight off disk from the `.dist-info` directories rather than by
//! running `pip list`. F4 consults this on every debounce, and a subprocess per
//! keystroke pause would blow the latency budget. It also works when the venv
//! has no pip in it, which is the normal case for environments uv creates.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::core::project::normalize_name;
use crate::pypi::pyversion::PyVersion;

/// Packages present in a virtualenv, normalized per PEP 503.
#[derive(Debug, Clone, Default)]
pub struct Installed {
    pub packages: BTreeSet<String>,
    /// Installed version per package, when the metadata directory records one.
    pub versions: BTreeMap<String, String>,
}

impl Installed {
    /// Is this distribution installed? `name` is normalized before comparison.
    pub fn contains(&self, name: &str) -> bool {
        self.packages.contains(&normalize_name(name))
    }

    pub fn count(&self) -> usize {
        self.packages.len()
    }

    /// The installed version of a distribution, if known.
    pub fn version_of(&self, name: &str) -> Option<String> {
        self.versions.get(&normalize_name(name)).cloned()
    }
}

/// The requirement PyPilot installs so notebooks can use the venv as a kernel.
///
/// Pinned to the 6.x line on purpose, because each half of the range avoids a
/// known failure:
///
/// - `<7`: on ipykernel 7.x (reported on 7.1.0 and 7.2.0) VS Code notebooks
///   can hang with cells stuck "pending", while 6.31.0 runs them; no 7.x
///   release is confirmed to fix it, and the maintainers' advice is to pin
///   below 7. <https://github.com/microsoft/vscode-jupyter/issues/17228>
/// - `>=6.31.0`: the newest 6.x, released after Python 3.14 and tested on it.
///   The Python 3.14 "kernel won't connect" failure was specific to 7.0.0
///   (its new shell-channel thread overrode `Thread._context`, which 3.14
///   uses); 6.30.1+ is unaffected and 7.0.1 fixed it.
///   <https://github.com/ipython/ipykernel/issues/1445>
///
/// The floor also matters because a bare `ipykernel` could resolve to a cached
/// or pinned release that installs fine and still leaves the notebook unable to
/// start.
pub const IPYKERNEL_REQUIREMENT: &str = "ipykernel>=6.31.0,<7";

/// Can Jupyter / VS Code use this venv as a notebook kernel?
///
/// True when `ipykernel` is installed, except that on Python 3.14+ the
/// releases known not to connect there count as missing, so setup replaces
/// them: 7.0.0 and its pre-releases (fixed in 7.0.1, see
/// [`IPYKERNEL_REQUIREMENT`]) and anything before 6.30.1, the oldest release
/// reported to work on 3.14.
///
/// A later 7.x is deliberately left alone even though VS Code may hang on it:
/// that hang is setup-dependent, many users run 7.x fine, and the version may
/// be required by the project, so downgrading it behind the user's back could
/// break more than it fixes. An unknown version or unknown Python is given the
/// benefit of the doubt, so PyPilot never nags about something it cannot
/// verify.
pub fn has_usable_ipykernel(installed: &Installed, python: Option<PyVersion>) -> bool {
    if !installed.contains("ipykernel") {
        return false;
    }
    if !python.is_some_and(|p| p >= PyVersion::py3(14)) {
        return true;
    }
    match installed.version_of("ipykernel") {
        // Pre-release suffixes are cut, so `7.0.0rc1` lands in the bad band.
        Some(v) => {
            let release = release_tuple(&v);
            release >= (7, 0, 1) || ((6, 30, 1)..(7, 0, 0)).contains(&release)
        }
        None => true,
    }
}

/// The leading `major.minor.patch` of a version string, missing parts as 0.
/// Pre-release suffixes are cut, so `7.0.0rc1` reads as `(7, 0, 0)`.
fn release_tuple(version: &str) -> (u64, u64, u64) {
    let mut parts = version.split('.').map(|p| {
        let digits: String = p.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u64>().unwrap_or(0)
    });
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// Read the installed distributions from a venv directory.
///
/// Returns an empty set if the venv is missing or unreadable, which callers
/// treat the same as "nothing installed".
pub fn scan(venv: &Path) -> Installed {
    let mut packages = BTreeSet::new();
    let mut versions = BTreeMap::new();
    for dir in site_packages_dirs(venv) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(dist) = distribution_name(&name) {
                if let Some(v) = distribution_version(&name) {
                    versions.insert(dist.clone(), v);
                }
                packages.insert(dist);
            }
        }
    }
    Installed { packages, versions }
}

/// The version out of a `.dist-info` directory name, if it has one.
pub(crate) fn distribution_version(entry: &str) -> Option<String> {
    let stem = entry
        .strip_suffix(".dist-info")
        .or_else(|| entry.strip_suffix(".egg-info"))?;
    let (_, version) = stem.split_once('-')?;
    if version.is_empty() {
        return None;
    }
    Some(version.to_string())
}

/// Candidate `site-packages` locations for a venv, across platforms.
///
/// Windows uses `Lib/site-packages`; POSIX nests under a version directory that
/// we discover rather than guess, since the venv's Python version is exactly
/// what F4 may be about to change.
pub fn site_packages_dirs(venv: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    let windows = venv.join("Lib").join("site-packages");
    if windows.is_dir() {
        dirs.push(windows);
    }

    let lib = venv.join("lib");
    if let Ok(entries) = std::fs::read_dir(&lib) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("site-packages");
            if candidate.is_dir() {
                dirs.push(candidate);
            }
        }
    }
    dirs
}

/// Pull the distribution name out of a `.dist-info` or `.egg-info` directory.
///
/// `charset_normalizer-3.3.2.dist-info` -> `charset-normalizer`.
pub(crate) fn distribution_name(entry: &str) -> Option<String> {
    let stem = entry
        .strip_suffix(".dist-info")
        .or_else(|| entry.strip_suffix(".egg-info"))?;
    // Everything before the first `-` is the name; the rest is the version.
    let name = stem.split('-').next()?;
    if name.is_empty() {
        return None;
    }
    Some(normalize_name(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dist_info_names() {
        assert_eq!(
            distribution_name("charset_normalizer-3.3.2.dist-info").as_deref(),
            Some("charset-normalizer")
        );
        assert_eq!(
            distribution_name("opencv_python-5.0.0.93.dist-info").as_deref(),
            Some("opencv-python")
        );
        assert_eq!(
            distribution_name("numpy-2.0.1.egg-info").as_deref(),
            Some("numpy")
        );
    }

    #[test]
    fn ignores_non_metadata_entries() {
        assert!(distribution_name("numpy").is_none());
        assert!(distribution_name("__pycache__").is_none());
        assert!(distribution_name("numpy.libs").is_none());
    }

    #[test]
    fn lookup_normalizes_the_query() {
        let installed = Installed {
            packages: ["opencv-python".to_string()].into_iter().collect(),
            ..Default::default()
        };
        assert!(installed.contains("opencv_python"));
        assert!(installed.contains("OpenCV-Python"));
        assert!(!installed.contains("opencv"));
    }

    fn with_ipykernel(version: Option<&str>) -> Installed {
        let mut installed = Installed {
            packages: ["ipykernel".to_string()].into_iter().collect(),
            ..Default::default()
        };
        if let Some(v) = version {
            installed.versions.insert("ipykernel".into(), v.into());
        }
        installed
    }

    #[test]
    fn broken_ipykernel_is_missing_only_on_python_314_and_later() {
        let ancient = with_ipykernel(Some("6.29.5"));
        let seven_oh = with_ipykernel(Some("7.0.0"));
        let py313 = Some(PyVersion::py3(13));
        let py314 = Some(PyVersion::py3(14));

        // Before 3.14 any installed ipykernel is accepted.
        assert!(has_usable_ipykernel(&ancient, py313));
        assert!(has_usable_ipykernel(&seven_oh, py313));
        // On 3.14: too old, and the 7.0.0 line, cannot connect.
        assert!(!has_usable_ipykernel(&ancient, py314));
        assert!(!has_usable_ipykernel(&seven_oh, py314));
        assert!(!has_usable_ipykernel(
            &with_ipykernel(Some("7.0.0rc1")),
            py314
        ));
        // 6.30.1+ on the 6.x line works on 3.14, as does 7.0.1+.
        assert!(has_usable_ipykernel(&with_ipykernel(Some("6.30.1")), py314));
        assert!(has_usable_ipykernel(&with_ipykernel(Some("6.31.0")), py314));
        assert!(has_usable_ipykernel(&with_ipykernel(Some("7.0.1")), py314));
        assert!(has_usable_ipykernel(&with_ipykernel(Some("7.10")), py314));
        // Unverifiable: don't nag.
        assert!(has_usable_ipykernel(&ancient, None));
        assert!(has_usable_ipykernel(&with_ipykernel(None), py314));
        // Not installed at all.
        assert!(!has_usable_ipykernel(&Installed::default(), py313));
    }

    #[test]
    fn installed_requirement_accepts_only_a_usable_release() {
        // Whatever the requirement installs must pass the usability check on
        // every Python, or setup would reinstall it on every run.
        assert_eq!(IPYKERNEL_REQUIREMENT, "ipykernel>=6.31.0,<7");
        let floor = with_ipykernel(Some("6.31.0"));
        for minor in [9, 13, 14, 15] {
            assert!(has_usable_ipykernel(&floor, Some(PyVersion::py3(minor))));
        }
    }

    #[test]
    fn missing_venv_scans_empty() {
        let empty = scan(Path::new("definitely/not/a/real/venv"));
        assert_eq!(empty.count(), 0);
    }
}
