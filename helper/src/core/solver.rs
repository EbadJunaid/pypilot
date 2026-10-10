//! The constraint solver — turns probes + project + PyPI metadata into findings.
//!
//! [`assess`] is the single read-only entry point used by both `doctor` and F5.
//! The decision logic lives in the pure [`synthesize`] function so it can be unit
//! tested without spawning any processes. Every finding states the *why* — which
//! package constrained the Python choice.

use std::path::Path;

use crate::core::platform::{Os, Platform};
use crate::core::probe::Probes;
use crate::core::project::ProjectDeps;
#[cfg(test)]
use crate::core::project::Requirement;
use crate::core::toolchain::{self, Toolchain};
use crate::core::uv;
use crate::core::{Assessment, Finding, FixKind, Severity};
use crate::matrix;
use crate::pypi::metadata::{PackageAnalysis, SdistKind};
use crate::pypi::pyversion::PyVersion;
use crate::pypi::{self, CompatReport, MetadataSource};
use crate::settings::{PackageManager, Settings};

/// Flattened, process-free view of the environment for [`synthesize`].
#[derive(Debug, Clone, Default)]
pub struct EnvSummary {
    pub has_venv: bool,
    pub venv_python: Option<PyVersion>,
    pub uv_present: bool,
    pub interpreters: Vec<PyVersion>,
    /// The workspace contains at least one `.ipynb`.
    pub has_notebooks: bool,
    /// The venv has an `ipykernel` that works on its Python, so Jupyter and
    /// VS Code can use it as a kernel.
    pub venv_has_ipykernel: bool,
    /// Build tools on this machine, probed only when some dependency would
    /// compile from source. `None` means not probed: advise everything.
    pub toolchain: Option<Toolchain>,
}

impl EnvSummary {
    fn from_probes(p: &Probes, workspace: &Path) -> EnvSummary {
        EnvSummary {
            has_venv: p.venv.is_some(),
            venv_python: p.venv.as_ref().and_then(|v| v.python),
            uv_present: p.uv.is_some(),
            interpreters: p.interpreters.iter().map(|i| i.version).collect(),
            has_notebooks: crate::core::project::has_notebooks(workspace),
            venv_has_ipykernel: p.venv.as_ref().is_some_and(|v| {
                crate::core::installed::has_usable_ipykernel(
                    &crate::core::installed::scan(&v.path),
                    v.python,
                )
            }),
            toolchain: None,
        }
    }
}

/// Full read-only assessment of a workspace (F5 / doctor).
pub async fn assess<S: MetadataSource>(
    workspace: &Path,
    settings: &Settings,
    source: &S,
) -> Assessment {
    let probes = crate::core::probe::run(workspace, settings).await;
    let project = crate::core::project::scan(workspace);

    let compat = if project.packages.is_empty() {
        None
    } else {
        Some(pypi::analyze(source, Platform::current(), &project.packages).await)
    };

    // F2 seam (empty in Phase 1).
    let hardware = matrix::check(workspace, settings, &project.packages).await;

    let mut env = EnvSummary::from_probes(&probes, workspace);
    // Probing compilers spawns processes, so only do it when a dependency
    // would actually build from source.
    if compat
        .as_ref()
        .is_some_and(|c| needs_source_build(c, env.venv_python))
    {
        env.toolchain = Some(toolchain::detect(Platform::current().os).await);
    }
    let (target_python, mut findings) = synthesize(&env, &project, compat.as_ref());
    findings.extend(hardware.findings);

    if let Some(venv) = &probes.venv {
        // A CPU torch on an NVIDIA machine, however it got there: checked
        // against what is installed, not what is declared, since a later
        // install can swap the CUDA build out without touching any manifest.
        findings.extend(crate::core::torch_gpu::static_check(&venv.path, settings).await);

        // Installed packages whose metadata rules out an installed dependency,
        // above all extensions built for NumPy 1.x sitting next to NumPy 2.
        // Read off disk, so it costs file reads rather than a resolve.
        findings.extend(crate::core::depcheck::check(&venv.path, venv.python));
    }

    // F3's hybrid confirmation: metadata explains which package would be the
    // blocker; a real `uv --dry-run` resolve confirms the answer is actually
    // installable, catching version conflicts wheel-tag analysis can't see
    // (yanked releases, transitive pins, sdist build requirements). Only
    // meaningful once the target interpreter exists on the machine, so this
    // is opportunistic rather than blocking — most "no venv yet" scans skip it.
    if let Some(finding) =
        confirm_with_uv(&probes, settings, target_python, &project, workspace).await
    {
        findings.push(finding);
    }

    Assessment {
        probes,
        project,
        compat,
        target_python,
        findings,
    }
}

/// Run `uv pip install --dry-run` for `target` against the project's declared
/// dependencies, and surface a finding only when it disagrees with what the
/// metadata-based intersection already concluded.
async fn confirm_with_uv(
    probes: &Probes,
    settings: &Settings,
    target: Option<PyVersion>,
    project: &ProjectDeps,
    workspace: &Path,
) -> Option<Finding> {
    if settings.package_manager != PackageManager::Uv {
        return None;
    }
    let uv_info = probes.uv.as_ref()?;
    let target = target?;
    if project.packages.is_empty() {
        return None;
    }
    // uv needs the interpreter to actually exist to resolve against it.
    probes.interpreters.iter().find(|i| i.version == target)?;

    let names = project.names();
    let out = uv::dry_run_resolve(uv_info, target, &names, workspace)
        .await
        .ok()?;
    if out.success() {
        return None; // Confirms the metadata answer; nothing new to say.
    }

    let reason = out.stderr.trim();
    if reason.is_empty() {
        return None;
    }
    Some(Finding {
        severity: Severity::Warning,
        title: format!("uv could not resolve dependencies for Python {target}"),
        detail: format!(
            "Package metadata suggested Python {target} would work, but `uv pip install --dry-run` disagrees, which usually means a version conflict metadata alone can't see: {reason}"
        ),
        fix: FixKind::Manual,
    })
}

/// Pure decision logic: given a flattened environment, the project, and the
/// compatibility report, produce the target interpreter and the findings.
pub fn synthesize(
    env: &EnvSummary,
    project: &ProjectDeps,
    compat: Option<&CompatReport>,
) -> (Option<PyVersion>, Vec<Finding>) {
    let mut findings = Vec::new();

    // Notebooks run on whatever kernel Jupyter can find, and an environment
    // without ipykernel is not one. Checked before the Python-project gate: a
    // `.venv` next to notebooks is reason enough, even with no manifest or
    // `.py` file. With no venv at all, the "no environment" finding covers it.
    if env.has_venv && env.has_notebooks && !env.venv_has_ipykernel {
        findings.push(Finding {
            severity: Severity::Warning,
            title: "Notebooks can't use .venv yet".to_string(),
            detail: "This project has Jupyter notebooks, but its environment has no usable ipykernel (on Python 3.14, ipykernel 7.0.0 and releases before 6.30.1 cannot connect), so VS Code and Jupyter can't offer it as a kernel and fall back to another Python where the project's packages are missing. Set up the environment to install ipykernel.".to_string(),
            fix: FixKind::SetupEnvironment,
        });
    }

    // Not a Python project at all → stay silent (bar the notebook kernel above).
    if !project.is_python_project() {
        return (None, findings);
    }

    // Conda projects: detect + offer migration, don't fight conda in v1.
    if project.has_conda_env {
        findings.push(Finding {
            severity: Severity::Info,
            title: "Conda environment detected".to_string(),
            detail: "environment.yml found. Run `pypilot migrate-conda` to generate a pyproject.toml from it; PyPilot won't manage conda directly.".to_string(),
            fix: FixKind::MigrateConda,
        });
    }

    let Some(compat) = compat else {
        // Python files but no declared dependencies to check.
        if !env.has_venv {
            findings.push(Finding {
                severity: Severity::Warning,
                title: "No virtual environment".to_string(),
                detail: "This project has Python files but no .venv. Create one to isolate its dependencies.".to_string(),
                fix: FixKind::SetupEnvironment,
            });
        }
        return (None, findings);
    };

    // Surface unresolved dependency names (typos, private packages).
    for (name, err) in &compat.unresolved {
        findings.push(Finding {
            severity: Severity::Warning,
            title: format!("Could not resolve `{name}`"),
            detail: format!(
                "PyPI lookup failed for `{name}` ({err}). It may be a typo or a private package."
            ),
            fix: FixKind::Manual,
        });
    }

    // sdist-only compile warnings, with the toolchain advice for this OS.
    for p in &compat.per_package {
        if let Some(kind) = p.sdist_kind() {
            findings.push(sdist_finding(p, kind, compat, env.toolchain));
        }
    }

    // Empty intersection → hard conflict; can't be auto-fixed by choosing a version.
    if compat.intersection.is_empty() {
        let detail = compat
            .conflicts
            .first()
            .map(|c| c.explanation.clone())
            .unwrap_or_else(|| {
                "The project's dependencies have no common supported Python version.".to_string()
            });
        findings.push(Finding {
            severity: Severity::Error,
            title: "Dependencies conflict on Python version".to_string(),
            detail,
            fix: FixKind::Manual,
        });
        return (None, findings);
    }

    let target = compat.suggested_python();

    // Determine the interpreter currently in effect: the venv's, else nothing.
    match env.venv_python {
        // Venv exists and is compatible → all good (stay silent).
        Some(current) if compat.is_compatible(current) => {}

        // Venv exists but its Python is unsupported → recreate on the right version.
        Some(current) => {
            let mut why = incompat_reason(compat, current, target);
            why.push_str(&wheel_gap_note(compat, current, target, env.toolchain));
            findings.push(Finding {
                severity: Severity::Error,
                title: format!("Python {current} is not supported by this project"),
                detail: why,
                fix: target
                    .map(FixKind::RecreateWithPython)
                    .unwrap_or(FixKind::Manual),
            });
        }

        // No venv (or unknown version) → set one up on the correct interpreter.
        None => {
            let range = compat.intersection.to_range_string();
            let why = match target {
                Some(t) => format!(
                    "Dependencies support Python {range}. Recommended: {t}{}.",
                    capping_clause(compat)
                ),
                None => format!("Dependencies support Python {range}."),
            };
            findings.push(Finding {
                severity: Severity::Warning,
                title: "No environment set up yet".to_string(),
                detail: why,
                fix: FixKind::SetupEnvironment,
            });
        }
    }

    (target, findings)
}

/// Would installing the project compile anything from source? True for any
/// sdist-only dependency, and for one whose wheels skip the venv's Python.
fn needs_source_build(compat: &CompatReport, venv_python: Option<PyVersion>) -> bool {
    compat
        .per_package
        .iter()
        .any(|p| p.sdist_only || venv_python.is_some_and(|py| p.lacks_wheel_for(py)))
}

fn os_name(os: Os) -> &'static str {
    match os {
        Os::Windows => "Windows",
        Os::Macos => "macOS",
        Os::Linux => "Linux",
    }
}

/// The "Will compile from source" finding for one sdist-only dependency.
///
/// No other Python helps here (no wheel exists for this platform at all), so
/// the fix is the toolchain. When the needed tools are already installed the
/// finding drops to Info: the build should just work.
fn sdist_finding(
    p: &PackageAnalysis,
    kind: SdistKind,
    compat: &CompatReport,
    toolchain: Option<Toolchain>,
) -> Finding {
    let os = compat.platform.os;
    let advice = toolchain::build_advice(&p.name, os, toolchain);
    let what = match kind {
        SdistKind::Native => format!(
            "{} {} ships prebuilt wheels for other platforms but none for {}, so installing it compiles native code from source.",
            p.name,
            p.version,
            os_name(os)
        ),
        SdistKind::Unknown => format!(
            "{} {} publishes only a source distribution, so it builds from source; if it has C/C++/Rust extensions that needs a compiler.",
            p.name, p.version
        ),
    };
    let (severity, detail) = match advice {
        Some(a) => (
            Severity::Warning,
            format!("{what} If the build fails (\"Failed building wheel\", \"Microsoft Visual C++ 14.0 or greater is required\"): {a}."),
        ),
        None => (
            Severity::Info,
            format!("{what} The build tools it needs are installed."),
        ),
    };
    Finding {
        severity,
        title: "Will compile from source".to_string(),
        detail,
        fix: FixKind::Manual,
    }
}

/// Extra explanation when the venv's Python is excluded only because some
/// dependency has no wheel for it: pip would fall back to a source build that
/// needs a compiler, and switching to `target` (which has wheels) avoids it.
fn wheel_gap_note(
    compat: &CompatReport,
    current: PyVersion,
    target: Option<PyVersion>,
    toolchain: Option<Toolchain>,
) -> String {
    let gaps: Vec<&PackageAnalysis> = compat
        .per_package
        .iter()
        .filter(|p| p.lacks_wheel_for(current))
        .collect();
    if gaps.is_empty() {
        return String::new();
    }
    let names: Vec<&str> = gaps.iter().map(|p| p.name.as_str()).collect();
    let mut note = format!(
        " {} {} no prebuilt wheel for Python {current} on {}, so installing on it means compiling from source",
        names.join(", "),
        if gaps.len() == 1 { "has" } else { "have" },
        os_name(compat.platform.os)
    );
    if let Some(t) = target {
        note.push_str(&format!("; Python {t} has wheels and needs no compiler"));
    }
    note.push('.');
    // Advice for staying on this Python anyway.
    let advice: Vec<String> = gaps
        .iter()
        .filter_map(|p| toolchain::build_advice(&p.name, compat.platform.os, toolchain))
        .collect();
    if let Some(a) = advice.iter().max_by_key(|a| a.len()) {
        note.push_str(&format!(" To keep Python {current}: {a}."));
    }
    note
}

/// Explain why the current interpreter is unsupported, naming the blockers.
fn incompat_reason(compat: &CompatReport, current: PyVersion, target: Option<PyVersion>) -> String {
    let blockers = compat.blockers_for(current);
    let names: Vec<String> = blockers
        .iter()
        .map(|p| format!("{} (supports {})", p.name, p.supported.to_range_string()))
        .collect();
    let who = if names.is_empty() {
        "some dependencies".to_string()
    } else {
        names.join(", ")
    };
    match target {
        Some(t) => format!(
            "{who} do not support Python {current}. Recreate the environment with Python {t}."
        ),
        None => format!("{who} do not support Python {current}."),
    }
}

/// "(constrained by X, which tops out at 3.Y)" for the recommendation message.
fn capping_clause(compat: &CompatReport) -> String {
    let Some(cap) = compat.intersection.max() else {
        return String::new();
    };
    // The package whose upper bound equals the intersection's cap is the limiter.
    let limiter = compat
        .per_package
        .iter()
        .filter(|p| p.supported.max() == Some(cap))
        .min_by_key(|p| p.supported.len());
    match limiter {
        Some(p) => format!(
            " (upper bound set by {}, which supports {})",
            p.name,
            p.supported.to_range_string()
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pypi::pyversion::PyVersionSet;

    fn analysis(name: &str, minors: &[u8]) -> PackageAnalysis {
        PackageAnalysis {
            name: name.into(),
            version: "1.0".into(),
            supported: PyVersionSet::from_versions(minors.iter().map(|&m| PyVersion::py3(m))),
            has_platform_wheels: true,
            sdist_only: false,
            has_foreign_wheels: false,
            requires_python: None,
        }
    }

    fn report(pkgs: Vec<PackageAnalysis>) -> CompatReport {
        let intersection = pkgs
            .iter()
            .map(|p| p.supported.clone())
            .reduce(|a, b| a.intersect(&b))
            .unwrap_or_else(PyVersionSet::universe);
        CompatReport {
            platform: Platform::current(),
            per_package: pkgs,
            unresolved: vec![],
            intersection,
            conflicts: vec![],
            sdist_warnings: vec![],
        }
    }

    fn python_project() -> ProjectDeps {
        ProjectDeps {
            sources: vec!["requirements.txt".into()],
            packages: vec![Requirement::any("mediapipe")],
            ..Default::default()
        }
    }

    #[test]
    fn mediapipe_on_313_recommends_312() {
        // mediapipe supports 3.9–3.12; venv is 3.13.
        let compat = report(vec![analysis("mediapipe", &[9, 10, 11, 12])]);
        let env = EnvSummary {
            has_venv: true,
            venv_python: Some(PyVersion::py3(13)),
            uv_present: true,
            interpreters: vec![PyVersion::py3(13)],
            ..Default::default()
        };
        let (target, findings) = synthesize(&env, &python_project(), Some(&compat));
        assert_eq!(target, Some(PyVersion::py3(12)));
        let f = findings
            .iter()
            .find(|f| f.severity == Severity::Error)
            .unwrap();
        assert_eq!(f.fix, FixKind::RecreateWithPython(PyVersion::py3(12)));
        assert!(f.detail.contains("mediapipe"));
        assert!(f.detail.contains("3.13"));
    }

    #[test]
    fn compatible_venv_stays_silent() {
        let compat = report(vec![analysis("numpy", &[9, 10, 11, 12, 13])]);
        let env = EnvSummary {
            has_venv: true,
            venv_python: Some(PyVersion::py3(12)),
            uv_present: true,
            interpreters: vec![PyVersion::py3(12)],
            ..Default::default()
        };
        let (_t, findings) = synthesize(&env, &python_project(), Some(&compat));
        assert!(findings.iter().all(|f| f.severity == Severity::Info));
    }

    #[test]
    fn no_venv_suggests_setup_with_reason() {
        let compat = report(vec![analysis("mediapipe", &[9, 10, 11, 12])]);
        let env = EnvSummary::default();
        let (target, findings) = synthesize(&env, &python_project(), Some(&compat));
        assert_eq!(target, Some(PyVersion::py3(12)));
        let f = &findings[0];
        assert_eq!(f.fix, FixKind::SetupEnvironment);
        assert!(f.detail.contains("mediapipe"));
    }

    #[test]
    fn notebooks_without_ipykernel_suggest_setup() {
        let compat = report(vec![analysis("numpy", &[9, 10, 11, 12, 13])]);
        let env = EnvSummary {
            has_venv: true,
            venv_python: Some(PyVersion::py3(12)),
            has_notebooks: true,
            venv_has_ipykernel: false,
            ..Default::default()
        };
        let (_t, findings) = synthesize(&env, &python_project(), Some(&compat));
        let f = findings
            .iter()
            .find(|f| f.title == "Notebooks can't use .venv yet")
            .unwrap();
        assert_eq!(f.severity, Severity::Warning);
        assert_eq!(f.fix, FixKind::SetupEnvironment);
        assert!(f.detail.contains("ipykernel"));

        // Already a kernel: silent again.
        let ready = EnvSummary {
            venv_has_ipykernel: true,
            ..env.clone()
        };
        let (_t, findings) = synthesize(&ready, &python_project(), Some(&compat));
        assert!(findings.iter().all(|f| f.severity == Severity::Info));

        // No venv: only the existing "no environment" finding, not this one.
        let none = EnvSummary {
            has_notebooks: true,
            ..Default::default()
        };
        let (_t, findings) = synthesize(&none, &python_project(), Some(&compat));
        assert!(findings
            .iter()
            .all(|f| f.title != "Notebooks can't use .venv yet"));
        assert!(findings
            .iter()
            .any(|f| f.title == "No environment set up yet"));
    }

    fn report_on(os: Os, pkgs: Vec<PackageAnalysis>) -> CompatReport {
        CompatReport {
            platform: Platform {
                os,
                arch: crate::core::platform::Arch::X86_64,
            },
            ..report(pkgs)
        }
    }

    fn sdist_pkg(name: &str, foreign_wheels: bool) -> PackageAnalysis {
        PackageAnalysis {
            has_platform_wheels: false,
            sdist_only: true,
            has_foreign_wheels: foreign_wheels,
            ..analysis(name, &[9, 10, 11, 12, 13])
        }
    }

    fn compat_venv(py: u8) -> EnvSummary {
        EnvSummary {
            has_venv: true,
            venv_python: Some(PyVersion::py3(py)),
            uv_present: true,
            interpreters: vec![PyVersion::py3(py)],
            ..Default::default()
        }
    }

    fn sdist_findings(findings: &[Finding]) -> Vec<&Finding> {
        findings
            .iter()
            .filter(|f| f.title == "Will compile from source")
            .collect()
    }

    #[test]
    fn native_sdist_on_windows_names_build_tools() {
        let compat = report_on(Os::Windows, vec![sdist_pkg("nativelib", true)]);
        let (_t, findings) = synthesize(&compat_venv(12), &python_project(), Some(&compat));
        let f = sdist_findings(&findings)[0];
        assert_eq!(f.severity, Severity::Warning);
        assert!(f
            .detail
            .contains("wheels for other platforms but none for Windows"));
        assert!(f.detail.contains("Microsoft C++ Build Tools"));
        assert!(f.detail.contains("Desktop development with C++"));
        assert_eq!(f.fix, FixKind::Manual);
    }

    #[test]
    fn unknown_sdist_hedges_and_uses_os_advice() {
        let compat = report_on(Os::Macos, vec![sdist_pkg("oldlib", false)]);
        let (_t, findings) = synthesize(&compat_venv(12), &python_project(), Some(&compat));
        let f = sdist_findings(&findings)[0];
        assert!(f.detail.contains("only a source distribution"));
        assert!(f.detail.contains("xcode-select --install"));

        let compat = report_on(Os::Linux, vec![sdist_pkg("tokenizers", true)]);
        let (_t, findings) = synthesize(&compat_venv(12), &python_project(), Some(&compat));
        let f = sdist_findings(&findings)[0];
        assert!(f.detail.contains("build-essential"));
        assert!(f.detail.contains("rustup"));
    }

    #[test]
    fn installed_toolchain_downgrades_sdist_finding() {
        let compat = report_on(Os::Windows, vec![sdist_pkg("nativelib", true)]);
        let env = EnvSummary {
            toolchain: Some(Toolchain {
                c_compiler: true,
                rust: false,
            }),
            ..compat_venv(12)
        };
        let (_t, findings) = synthesize(&env, &python_project(), Some(&compat));
        let f = sdist_findings(&findings)[0];
        assert_eq!(f.severity, Severity::Info);
        assert!(!f.detail.contains("Build Tools"));
        assert!(findings.iter().all(|f| f.severity == Severity::Info));
    }

    #[test]
    fn wheel_gap_recommends_python_with_wheels() {
        // Wheels for 3.9-3.12 only, requires_python allows 3.13: on a 3.13
        // venv pip would compile it, so recommend 3.12 and explain why.
        let gap = PackageAnalysis {
            requires_python: Some(">=3.9".into()),
            ..analysis("mediapipe", &[9, 10, 11, 12])
        };
        let compat = report_on(Os::Windows, vec![gap]);
        let (target, findings) = synthesize(&compat_venv(13), &python_project(), Some(&compat));
        assert_eq!(target, Some(PyVersion::py3(12)));
        let f = findings
            .iter()
            .find(|f| f.severity == Severity::Error)
            .unwrap();
        assert_eq!(f.fix, FixKind::RecreateWithPython(PyVersion::py3(12)));
        assert!(f
            .detail
            .contains("no prebuilt wheel for Python 3.13 on Windows"));
        assert!(f.detail.contains("Python 3.12 has wheels"));
        assert!(f.detail.contains("Microsoft C++ Build Tools"));
        assert!(needs_source_build(&compat, Some(PyVersion::py3(13))));
        assert!(!needs_source_build(&compat, Some(PyVersion::py3(12))));
    }

    #[test]
    fn requires_python_exclusion_is_not_a_wheel_gap() {
        let capped = PackageAnalysis {
            requires_python: Some(">=3.9,<3.13".into()),
            ..analysis("capped", &[9, 10, 11, 12])
        };
        let compat = report_on(Os::Linux, vec![capped]);
        let (_t, findings) = synthesize(&compat_venv(13), &python_project(), Some(&compat));
        let f = findings
            .iter()
            .find(|f| f.severity == Severity::Error)
            .unwrap();
        assert!(!f.detail.contains("compiling from source"));
    }

    #[test]
    fn not_a_python_project_is_silent() {
        let (target, findings) = synthesize(&EnvSummary::default(), &ProjectDeps::default(), None);
        assert!(target.is_none());
        assert!(findings.is_empty());
    }
}
