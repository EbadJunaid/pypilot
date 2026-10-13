//! Turning guardian findings into editor diagnostics and quick fixes.
//!
//! Each diagnostic carries its own fix in the LSP `data` field, so when the
//! editor asks for code actions we rebuild them from the diagnostic under the
//! cursor instead of re-analyzing the buffer.
//!
//! Selecting several imports adds one action that installs them all in a single
//! command, and a whole-file (or whole-notebook) action covers what the
//! selection cannot reach.
//!
//! Every install action names the distribution it will install. When an import
//! resolves to a different name (`cv2` to `opencv-python`) the title says both,
//! because the user is approving a name they never typed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tower_lsp::lsp_types::*;

use crate::core::guardian::{Finding, Problem};
use crate::core::platform::{Os, Platform};
use crate::core::project::normalize_name;
use crate::matrix::install_caveat;

/// Command IDs. The install commands take arguments; the F5 ones do not.
///
/// CMD_INSTALL takes `[package, ...]` and CMD_RECREATE `[python, package, ...]`.
/// One package is how both editors have always called them.
pub const CMD_INSTALL: &str = "pypilot.installPackage";
pub const CMD_RECREATE: &str = "pypilot.recreateWithPython";

/// The fix attached to a diagnostic, round-tripped through `Diagnostic.data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixData {
    pub package: String,
    pub action: FixAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FixAction {
    /// Install into the environment that already exists.
    Install,
    /// Build the project's first environment on `python`, then install.
    Create { python: String },
    /// Replace the existing environment with one on `python`, then install.
    Recreate {
        python: String,
        reinstall_count: usize,
    },
    /// Nothing safe to automate.
    None,
}

/// Build the diagnostic for one finding.
pub fn to_diagnostic(finding: &Finding) -> Diagnostic {
    let provides = if finding.renamed {
        format!(" (provides `{}`)", finding.import.module)
    } else {
        String::new()
    };

    let (severity, message, action) = match &finding.problem {
        Problem::NotInstalled => (
            DiagnosticSeverity::WARNING,
            format!("`{}`{provides} is not installed.", finding.package),
            FixAction::Install,
        ),

        Problem::IncompatibleInterpreter {
            current,
            supported,
            target,
            reinstall_count,
        } => {
            let message = format!(
                "`{}`{provides} has no wheels for Python {current}. It supports {}.",
                finding.package,
                supported.to_range_string()
            );
            let action = match target {
                Some(t) => FixAction::Recreate {
                    python: t.to_string(),
                    reinstall_count: *reinstall_count,
                },
                None => FixAction::None,
            };
            (DiagnosticSeverity::ERROR, message, action)
        }

        Problem::NoEnvironment { target } => {
            let message = format!(
                "`{}`{provides} is not installed, and this project has no virtual environment yet.",
                finding.package
            );
            let action = match target {
                Some(t) => FixAction::Create {
                    python: t.to_string(),
                },
                None => FixAction::None,
            };
            (DiagnosticSeverity::WARNING, message, action)
        }

        // The installed release simply does not have this attribute. Nothing to
        // automate: which version to pin depends on which API the user wants,
        // so the message names what is there instead and lets them choose.
        Problem::AttributeMissing {
            attribute,
            installed_version,
            available,
        } => {
            let version = installed_version
                .as_deref()
                .map(|v| format!(" {v}"))
                .unwrap_or_default();

            // Listing everything would be noise; a handful points the way.
            const SHOWN: usize = 8;
            let mut names: Vec<String> = available.iter().take(SHOWN).cloned().collect();
            if available.len() > SHOWN {
                names.push(format!("and {} more", available.len() - SHOWN));
            }
            let provides = if names.is_empty() {
                String::new()
            } else {
                format!(" It provides: {}.", names.join(", "))
            };

            (
                DiagnosticSeverity::ERROR,
                format!(
                    "The installed `{}`{version} has no `{attribute}`.{provides}",
                    finding.package
                ),
                FixAction::None,
            )
        }

        // No install offer: a plain install is exactly what would fail or
        // replace torch. The message says what to run or use instead.
        Problem::CudaExtension { blocking, message } => (
            if *blocking {
                DiagnosticSeverity::ERROR
            } else {
                DiagnosticSeverity::WARNING
            },
            message.clone(),
            FixAction::None,
        ),

        // No install offer and no interpreter switch: neither would help.
        Problem::NoFilesForPlatform { message } => (
            DiagnosticSeverity::ERROR,
            format!("`{}`{provides}: {message}", finding.package),
            FixAction::None,
        ),

        // Deliberately actionless. Offering to install a name PyPI does not know
        // is how a typo becomes a supply chain incident.
        Problem::NotOnPyPi => (
            DiagnosticSeverity::WARNING,
            format!(
                "PyPI has no project named `{}`. Check the spelling, or it may come from a private index.",
                finding.package
            ),
            FixAction::None,
        ),
    };

    let data = serde_json::to_value(FixData {
        package: finding.package.clone(),
        action,
    })
    .ok();

    Diagnostic {
        range: Range {
            start: Position {
                line: finding.import.line,
                character: finding.import.start,
            },
            end: Position {
                line: finding.import.line,
                character: finding.import.end,
            },
        },
        severity: Some(severity),
        source: Some("pypilot".to_string()),
        message,
        data,
        ..Default::default()
    }
}

/// How far a whole-document action reaches. Only its title depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    File,
    Notebook,
}

/// What decides whether two fixes can run as one command: the same command on
/// the same interpreter. Mixing kinds would let one click rebuild an
/// environment that another fix only meant to add to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FixKey {
    Install,
    Create {
        python: String,
    },
    Recreate {
        python: String,
        reinstall_count: usize,
    },
}

impl FixKey {
    fn of(action: &FixAction) -> Option<Self> {
        match action {
            FixAction::Install => Some(FixKey::Install),
            FixAction::Create { python } => Some(FixKey::Create {
                python: python.clone(),
            }),
            FixAction::Recreate {
                python,
                reinstall_count,
            } => Some(FixKey::Recreate {
                python: python.clone(),
                reinstall_count: *reinstall_count,
            }),
            FixAction::None => None,
        }
    }

    /// The action title for installing `what`, which is either one quoted name
    /// or a phrase like "3 packages: numpy, pandas, requests".
    fn title(&self, what: &str) -> String {
        match self {
            FixKey::Install => format!("Install {what}"),
            FixKey::Create { python } => {
                format!("Create environment on Python {python} and install {what}")
            }
            FixKey::Recreate {
                python,
                reinstall_count,
            } => {
                // Say how much gets rebuilt. "3 packages will be reinstalled" is
                // the difference between a trusted fix and a surprise.
                let scope = match reinstall_count {
                    0 => String::new(),
                    1 => " (1 existing package will be reinstalled)".to_string(),
                    n => format!(" ({n} existing packages will be reinstalled)"),
                };
                format!("Rebuild environment on Python {python} and install {what}{scope}")
            }
        }
    }

    /// CMD_INSTALL takes `[pkg, ...]`, CMD_RECREATE `[python, pkg, ...]`.
    fn command(&self, packages: &[String]) -> Command {
        let names = packages.iter().cloned().map(serde_json::Value::String);
        let (title, command, arguments) = match self {
            FixKey::Install => (
                format!("Install {}", packages.join(", ")),
                CMD_INSTALL,
                names.collect(),
            ),
            FixKey::Create { python } => (
                "Create environment".to_string(),
                CMD_RECREATE,
                std::iter::once(serde_json::Value::String(python.clone()))
                    .chain(names)
                    .collect(),
            ),
            FixKey::Recreate { python, .. } => (
                "Rebuild environment".to_string(),
                CMD_RECREATE,
                std::iter::once(serde_json::Value::String(python.clone()))
                    .chain(names)
                    .collect(),
            ),
        };
        Command {
            title,
            command: command.to_string(),
            arguments: Some(arguments),
        }
    }
}

/// Automatable fixes sharing one [`FixKey`], each package listed once.
struct Group<'a> {
    key: FixKey,
    packages: Vec<String>,
    diagnostics: Vec<&'a Diagnostic>,
}

impl Group<'_> {
    fn same_packages(&self, other: &Group) -> bool {
        let set = |g: &Group| -> HashSet<String> {
            g.packages.iter().map(|p| normalize_name(p)).collect()
        };
        self.key == other.key && set(self) == set(other)
    }
}

/// Our diagnostics with their decoded fixes. Only ours carry a FixData payload.
fn fixes(diagnostics: &[Diagnostic]) -> impl Iterator<Item = (&Diagnostic, FixData)> {
    diagnostics.iter().filter_map(|diagnostic| {
        if diagnostic.source.as_deref() != Some("pypilot") {
            return None;
        }
        let fix = serde_json::from_value::<FixData>(diagnostic.data.clone()?).ok()?;
        Some((diagnostic, fix))
    })
}

/// Group automatable fixes by kind, in the order they first appear. The same
/// import used twice is two diagnostics but one package to install.
fn groups(diagnostics: &[Diagnostic]) -> Vec<Group<'_>> {
    let mut groups: Vec<Group> = Vec::new();
    for (diagnostic, fix) in fixes(diagnostics) {
        let Some(key) = FixKey::of(&fix.action) else {
            continue;
        };
        let index = match groups.iter().position(|g| g.key == key) {
            Some(i) => i,
            None => {
                groups.push(Group {
                    key,
                    packages: Vec::new(),
                    diagnostics: Vec::new(),
                });
                groups.len() - 1
            }
        };
        let group = &mut groups[index];
        group.diagnostics.push(diagnostic);
        let name = normalize_name(&fix.package);
        if !group.packages.iter().any(|p| normalize_name(p) == name) {
            group.packages.push(fix.package);
        }
    }
    groups
}

/// Package names for a title. Past a handful the list stops informing and
/// starts pushing the rest of the menu off screen.
fn name_list(packages: &[String]) -> String {
    const SHOWN: usize = 4;
    if packages.len() <= SHOWN {
        packages.join(", ")
    } else {
        let head = &packages[..SHOWN - 1];
        format!(
            "{} and {} more",
            head.join(", "),
            packages.len() - head.len()
        )
    }
}

/// Append any platform caveat for `packages` to an action title, so a fix
/// that installs tensorflow on Windows does not read as a GPU setup.
fn caveated(title: String, packages: &[String], os: Os) -> String {
    let notes: Vec<String> = packages
        .iter()
        .filter_map(|p| install_caveat(p, os).map(|note| (p, note)))
        .map(|(p, note)| {
            if packages.len() == 1 {
                note.to_string()
            } else {
                format!("{p}: {note}")
            }
        })
        .collect();
    if notes.is_empty() {
        title
    } else {
        format!("{title} ({})", notes.join("; "))
    }
}

fn quick_fix(
    title: String,
    command: Command,
    diagnostics: Option<Vec<Diagnostic>>,
    preferred: bool,
) -> CodeActionOrCommand {
    CodeActionOrCommand::CodeAction(CodeAction {
        title,
        kind: Some(CodeActionKind::QUICKFIX),
        diagnostics,
        command: Some(command),
        is_preferred: preferred.then_some(true),
        ..Default::default()
    })
}

/// Quick fixes for a selection alone, with nothing else known about the file.
#[cfg(test)]
pub fn code_actions(diagnostics: &[Diagnostic]) -> Vec<CodeActionOrCommand> {
    code_actions_in(diagnostics, &[], Scope::File)
}

/// Quick fixes for `selection`, plus one action covering `whole`, the
/// diagnostics last published for the document (or every open cell of its
/// notebook).
///
/// Order: a combined action per fix kind with 2+ packages selected, then one
/// action per package, then the whole-document action. The combined action is
/// the preferred one when it exists, so "apply preferred fix" installs the
/// selection in one go instead of picking a single package. The whole-document
/// action exists because notebook cells cannot be selected together, so a
/// selection never reaches past the cell it is in.
pub fn code_actions_in(
    selection: &[Diagnostic],
    whole: &[Diagnostic],
    scope: Scope,
) -> Vec<CodeActionOrCommand> {
    let mut actions = Vec::new();
    let selected = groups(selection);
    let os = Platform::current().os;

    for group in selected.iter().filter(|g| g.packages.len() >= 2) {
        let what = format!(
            "{} packages: {}",
            group.packages.len(),
            name_list(&group.packages)
        );
        actions.push(quick_fix(
            caveated(group.key.title(&what), &group.packages, os),
            group.key.command(&group.packages),
            Some(group.diagnostics.iter().map(|d| (*d).clone()).collect()),
            actions.is_empty(),
        ));
    }
    let combined = !actions.is_empty();

    let mut offered: Vec<(FixKey, String)> = Vec::new();
    for (diagnostic, fix) in fixes(selection) {
        let Some(key) = FixKey::of(&fix.action) else {
            continue;
        };
        let name = normalize_name(&fix.package);
        if offered.iter().any(|(k, n)| *k == key && *n == name) {
            continue;
        }
        offered.push((key.clone(), name));
        actions.push(quick_fix(
            caveated(
                key.title(&format!("`{}`", fix.package)),
                std::slice::from_ref(&fix.package),
                os,
            ),
            key.command(std::slice::from_ref(&fix.package)),
            Some(vec![diagnostic.clone()]),
            !combined,
        ));
    }

    let place = match scope {
        Scope::File => "file",
        Scope::Notebook => "notebook",
    };
    for group in groups(whole).iter().filter(|g| g.packages.len() >= 2) {
        if selected.iter().any(|s| s.same_packages(group)) {
            continue;
        }
        let what = format!(
            "all {} missing packages in this {place}",
            group.packages.len()
        );
        actions.push(quick_fix(
            caveated(group.key.title(&what), &group.packages, os),
            group.key.command(&group.packages),
            None,
            false,
        ));
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::imports::ImportRef;
    use crate::pypi::pyversion::{PyVersion, PyVersionSet};

    fn finding(problem: Problem, package: &str, renamed: bool) -> Finding {
        Finding {
            import: ImportRef {
                module: "cv2".into(),
                binding: Some("cv2".into()),
                line: 2,
                start: 7,
                end: 10,
            },
            package: package.into(),
            renamed,
            problem,
        }
    }

    fn supported(minors: &[u8]) -> PyVersionSet {
        PyVersionSet::from_versions(minors.iter().map(|&m| PyVersion::py3(m)))
    }

    #[test]
    fn missing_package_is_a_warning_with_an_install_fix() {
        let d = to_diagnostic(&finding(Problem::NotInstalled, "mediapipe", false));
        assert_eq!(d.severity, Some(DiagnosticSeverity::WARNING));
        assert!(d.message.contains("mediapipe"));

        let actions = code_actions(&[d]);
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn tensorflow_install_titles_say_cpu_only_on_windows() {
        let one = vec!["tensorflow".to_string()];
        assert_eq!(
            caveated("Install `tensorflow`".into(), &one, Os::Windows),
            "Install `tensorflow` (CPU-only on native Windows)"
        );
        assert_eq!(
            caveated("Install `tensorflow`".into(), &one, Os::Linux),
            "Install `tensorflow`"
        );

        let several = vec!["numpy".to_string(), "tensorflow".to_string()];
        assert_eq!(
            caveated("Install 2 packages".into(), &several, Os::Windows),
            "Install 2 packages (tensorflow: CPU-only on native Windows)"
        );
        let plain = vec!["numpy".to_string()];
        assert_eq!(
            caveated("Install `numpy`".into(), &plain, Os::Windows),
            "Install `numpy`"
        );
    }

    #[test]
    fn renamed_imports_show_both_names() {
        let d = to_diagnostic(&finding(Problem::NotInstalled, "opencv-python", true));
        assert!(d.message.contains("opencv-python"));
        assert!(d.message.contains("cv2"), "must reveal the resolved rename");

        let CodeActionOrCommand::CodeAction(action) = &code_actions(&[d])[0] else {
            panic!("expected a code action");
        };
        assert!(action.title.contains("opencv-python"));
    }

    #[test]
    fn platformless_release_is_an_error_without_a_fix() {
        let d = to_diagnostic(&finding(
            Problem::NoFilesForPlatform {
                message: "tensorflow 2.15.0.post1 publishes no files for Windows x64 (only Linux x64 wheels, no source distribution), so it cannot be installed here.".into(),
            },
            "tensorflow",
            false,
        ));
        assert_eq!(d.severity, Some(DiagnosticSeverity::ERROR));
        assert!(d.message.contains("publishes no files for Windows x64"));
        assert!(code_actions(&[d]).is_empty());
    }

    #[test]
    fn interpreter_conflict_is_an_error_naming_the_range() {
        let d = to_diagnostic(&finding(
            Problem::IncompatibleInterpreter {
                current: PyVersion::py3(13),
                supported: supported(&[9, 10, 11, 12]),
                target: Some(PyVersion::py3(12)),
                reinstall_count: 3,
            },
            "mediapipe",
            false,
        ));
        assert_eq!(d.severity, Some(DiagnosticSeverity::ERROR));
        assert!(d.message.contains("3.13"));
        assert!(d.message.contains("3.9–3.12"));

        let CodeActionOrCommand::CodeAction(action) = &code_actions(&[d])[0] else {
            panic!("expected a code action");
        };
        assert!(action.title.contains("Python 3.12"));
        assert!(
            action
                .title
                .contains("3 existing packages will be reinstalled"),
            "the fix must state its blast radius, got: {}",
            action.title
        );
    }

    #[test]
    fn fresh_project_creates_rather_than_rebuilds() {
        let d = to_diagnostic(&finding(
            Problem::NoEnvironment {
                target: Some(PyVersion::py3(12)),
            },
            "mediapipe",
            false,
        ));
        let CodeActionOrCommand::CodeAction(action) = &code_actions(&[d])[0] else {
            panic!("expected a code action");
        };
        // There is nothing to rebuild or reinstall when no venv exists.
        assert!(
            action.title.starts_with("Create environment"),
            "{}",
            action.title
        );
        assert!(!action.title.contains("reinstalled"));
    }

    #[test]
    fn unknown_projects_get_no_install_action() {
        let d = to_diagnostic(&finding(Problem::NotOnPyPi, "mediapip", false));
        assert!(
            code_actions(&[d]).is_empty(),
            "never offer to install a name PyPI does not have"
        );
    }

    #[test]
    fn cuda_extensions_explain_instead_of_offering_an_install() {
        let warn = to_diagnostic(&finding(
            Problem::CudaExtension {
                blocking: false,
                message: "flash-attn is published on PyPI only as source.".into(),
            },
            "flash-attn",
            false,
        ));
        assert_eq!(warn.severity, Some(DiagnosticSeverity::WARNING));
        assert!(warn.message.contains("only as source"));
        assert!(code_actions(&[warn]).is_empty());

        let blocking = to_diagnostic(&finding(
            Problem::CudaExtension {
                blocking: true,
                message: "macOS has no CUDA.".into(),
            },
            "flash-attn",
            false,
        ));
        assert_eq!(blocking.severity, Some(DiagnosticSeverity::ERROR));
    }

    #[test]
    fn foreign_diagnostics_are_ignored() {
        let other = Diagnostic {
            source: Some("ruff".to_string()),
            message: "unused import".to_string(),
            ..Default::default()
        };
        assert!(code_actions(&[other]).is_empty());
    }

    fn missing(package: &str) -> Diagnostic {
        to_diagnostic(&finding(Problem::NotInstalled, package, false))
    }

    fn rebuild(package: &str, python: u8) -> Diagnostic {
        to_diagnostic(&finding(
            Problem::IncompatibleInterpreter {
                current: PyVersion::py3(13),
                supported: supported(&[9, 10, 11, python]),
                target: Some(PyVersion::py3(python)),
                reinstall_count: 3,
            },
            package,
            false,
        ))
    }

    fn unwrap(action: &CodeActionOrCommand) -> &CodeAction {
        match action {
            CodeActionOrCommand::CodeAction(a) => a,
            CodeActionOrCommand::Command(_) => panic!("expected a code action"),
        }
    }

    fn args(action: &CodeAction) -> Vec<String> {
        let command = action.command.as_ref().expect("runs a command");
        command
            .arguments
            .iter()
            .flatten()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn selecting_several_imports_offers_one_install_for_all() {
        let selection = [missing("numpy"), missing("pandas"), missing("requests")];
        let actions = code_actions(&selection);
        assert_eq!(
            actions.len(),
            4,
            "one combined action, then one per package"
        );

        let combined = unwrap(&actions[0]);
        assert_eq!(
            combined.title,
            "Install 3 packages: numpy, pandas, requests"
        );
        assert_eq!(combined.is_preferred, Some(true));
        assert_eq!(combined.command.as_ref().unwrap().command, CMD_INSTALL);
        assert_eq!(args(combined), vec!["numpy", "pandas", "requests"]);
        assert_eq!(combined.diagnostics.as_ref().map(Vec::len), Some(3));

        // Only one preferred action, so "apply preferred fix" is unambiguous.
        for single in &actions[1..] {
            let single = unwrap(single);
            assert!(single.title.starts_with("Install `"), "{}", single.title);
            assert_eq!(single.is_preferred, None);
            assert_eq!(args(single).len(), 1);
        }
    }

    #[test]
    fn a_single_selection_is_unchanged() {
        let actions = code_actions(&[missing("numpy")]);
        assert_eq!(actions.len(), 1);
        let only = unwrap(&actions[0]);
        assert_eq!(only.title, "Install `numpy`");
        assert_eq!(only.is_preferred, Some(true));
        assert_eq!(args(only), vec!["numpy"]);
    }

    #[test]
    fn the_same_package_twice_counts_once() {
        let selection = [missing("numpy"), missing("numpy"), missing("pandas")];
        let actions = code_actions(&selection);
        let combined = unwrap(&actions[0]);
        assert_eq!(combined.title, "Install 2 packages: numpy, pandas");
        assert_eq!(args(combined), vec!["numpy", "pandas"]);
        assert_eq!(actions.len(), 3, "no duplicate per-package actions");

        // Two diagnostics for one package is still a single install.
        assert_eq!(code_actions(&[missing("numpy"), missing("numpy")]).len(), 1);
    }

    #[test]
    fn mixed_fix_kinds_are_not_combined() {
        let actions = code_actions(&[missing("numpy"), rebuild("mediapipe", 12)]);
        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .all(|a| !unwrap(a).title.contains("packages:")));

        // Rebuilds aimed at different interpreters are different fixes too.
        let actions = code_actions(&[rebuild("mediapipe", 12), rebuild("tflite", 11)]);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn rebuilds_on_one_interpreter_combine() {
        let actions = code_actions(&[rebuild("mediapipe", 12), rebuild("tflite", 12)]);
        assert_eq!(actions.len(), 3);
        let combined = unwrap(&actions[0]);
        assert_eq!(
            combined.title,
            "Rebuild environment on Python 3.12 and install 2 packages: mediapipe, tflite \
             (3 existing packages will be reinstalled)"
        );
        assert_eq!(combined.command.as_ref().unwrap().command, CMD_RECREATE);
        assert_eq!(args(combined), vec!["3.12", "mediapipe", "tflite"]);
    }

    #[test]
    fn long_package_lists_are_truncated_in_the_title() {
        let names = ["a1", "b2", "c3", "d4", "e5", "f6"];
        let selection: Vec<Diagnostic> = names.iter().map(|n| missing(n)).collect();
        let combined = code_actions(&selection).remove(0);
        let combined = unwrap(&combined);
        assert_eq!(combined.title, "Install 6 packages: a1, b2, c3 and 3 more");
        assert_eq!(args(combined).len(), 6, "the command still installs all");
    }

    #[test]
    fn whole_notebook_action_reaches_past_the_selection() {
        let whole = [missing("numpy"), missing("pandas"), missing("requests")];
        let actions = code_actions_in(&[missing("numpy")], &whole, Scope::Notebook);
        assert_eq!(actions.len(), 2);
        assert_eq!(unwrap(&actions[0]).is_preferred, Some(true));

        let all = unwrap(&actions[1]);
        assert_eq!(all.title, "Install all 3 missing packages in this notebook");
        assert_eq!(all.is_preferred, None);
        assert_eq!(args(all), vec!["numpy", "pandas", "requests"]);
    }

    #[test]
    fn whole_file_action_is_offered_without_a_selection() {
        let whole = [missing("numpy"), missing("pandas")];
        let actions = code_actions_in(&[], &whole, Scope::File);
        assert_eq!(actions.len(), 1);
        assert_eq!(
            unwrap(&actions[0]).title,
            "Install all 2 missing packages in this file"
        );
    }

    #[test]
    fn whole_file_action_is_skipped_when_the_selection_already_covers_it() {
        let whole = [missing("numpy"), missing("pandas")];
        let selection = [missing("pandas"), missing("numpy")];
        let actions = code_actions_in(&selection, &whole, Scope::File);
        assert_eq!(actions.len(), 3);
        assert!(actions
            .iter()
            .all(|a| !unwrap(a).title.contains("in this file")));
    }

    #[test]
    fn whole_file_action_needs_two_installable_packages() {
        let whole = [
            missing("numpy"),
            to_diagnostic(&finding(Problem::NotOnPyPi, "nmupy", false)),
        ];
        assert!(code_actions_in(&[], &whole, Scope::File).is_empty());
    }
}
