//! `pypilot doctor` — read-only probe + compatibility report. Executes nothing.

use std::path::Path;

use colored::Colorize;

use crate::core::project::ProjectDeps;
use crate::core::solver;
use crate::core::Severity;
use crate::pypi::MetadataSource;
use crate::settings::Settings;

pub async fn run<S: MetadataSource>(
    workspace: &Path,
    settings: &Settings,
    source: &S,
) -> crate::Result<()> {
    let a = solver::assess(workspace, settings, source).await;

    println!(
        "{} — {}",
        "PyPilot doctor".bold().cyan(),
        workspace.display()
    );
    println!(
        "  {:<16}: {}",
        "package manager",
        format!("{:?}", settings.package_manager).bold()
    );
    println!();

    // --- Environment probes ---
    println!(
        "{}",
        "── Environment ──────────────────────────────────────────"
            .bold()
            .cyan()
    );
    match &a.probes.uv {
        Some(uv) => println!(
            "  {:<16}: {} ({}, {})",
            "uv",
            "present".green().bold(),
            format!("v{}", uv.version).cyan().bold(),
            if uv.managed { "managed" } else { "on PATH" }
        ),
        None => println!("  {:<16}: {}", "uv", "not detected".yellow()),
    }
    match &a.probes.venv {
        Some(v) => println!(
            "  {:<16}: {} ({})",
            if v.kind.is_pixi() {
                "pixi env"
            } else {
                "virtualenv"
            },
            v.path.display(),
            v.python
                .map(|p| format!("Python {p}").green().bold().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        None => println!("  {:<16}: {}", "virtualenv", "none".yellow()),
    }
    if let Some(pixi) = crate::core::pixi::detect(workspace) {
        println!("  {:<16}: {}", "pixi project", pixi.manifest.display());
    }
    if a.probes.interpreters.is_empty() {
        println!("  {:<16}: {}", "interpreters", "none found".yellow());
    } else {
        let list = a
            .probes
            .interpreters
            .iter()
            .map(|i| format!("{} ({})", i.version.to_string().cyan().bold(), i.command))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {:<16}: {list}", "interpreters");
    }
    println!();

    // --- Project ---
    println!(
        "{}",
        "── Project ──────────────────────────────────────────────"
            .bold()
            .cyan()
    );
    if a.project.sources.is_empty() {
        println!("  (no Python project files detected)");
    } else {
        let files = a
            .project
            .sources
            .iter()
            .map(|p| {
                p.file_name()
                    .map(|f| f.to_string_lossy().bold().to_string())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {:<16}: {files}", "files");
        // pixi installs the Python its manifest asks for, so show that spec
        // as written rather than its PEP 440 translation.
        if let Some(spec) = a
            .project
            .pixi_python
            .as_ref()
            .filter(|_| a.project.is_pixi())
        {
            println!(
                "  {:<16}: {}",
                "pixi python",
                format!("python {spec}").green().bold()
            );
        } else if let Some(rp) = &a.project.declared_requires_python {
            println!(
                "  {:<16}: {}",
                "requires-python",
                rp.to_string().green().bold()
            );
        }
        println!(
            "  {:<16}: {}",
            "dependencies",
            dependency_list(&a.project).bold()
        );
    }
    println!();

    // --- Compatibility ---
    if let Some(compat) = &a.compat {
        println!(
            "{}",
            "── Compatibility ────────────────────────────────────────"
                .bold()
                .cyan()
        );
        println!(
            "  {:<16}: {}",
            "supported Python",
            compat.intersection.to_range_string().green().bold()
        );
        if let Some(text) = solver::recommendation(&a.project, compat, a.target_python) {
            println!("  {:<16}: {}", "recommended", text.cyan().bold());
        }
        for p in &compat.per_package {
            let sdist_note = if p.sdist_only {
                " (sdist only)".yellow().to_string()
            } else {
                "".to_string()
            };
            println!(
                "    {} {:<20} {}{}",
                "•".cyan(),
                p.name.bold(),
                p.supported.to_range_string().green(),
                sdist_note
            );
        }
        for (name, err) in &compat.unresolved {
            println!(
                "    {} {:<20} {}",
                "•".cyan(),
                name.bold(),
                format!("unresolved ({err})").red().bold()
            );
        }
        println!();
    }

    // --- Findings ---
    println!(
        "{}",
        "── Findings ─────────────────────────────────────────────"
            .bold()
            .cyan()
    );
    if a.findings.is_empty() {
        println!("  {} Everything looks good.", "✓".green().bold());
    } else {
        for f in &a.findings {
            let tag = match f.severity {
                Severity::Error => "[ERROR]".red().bold(),
                Severity::Warning => "[WARN ]".yellow().bold(),
                Severity::Info => "[INFO ]".cyan().bold(),
            };
            println!("  {tag} {}", f.title.bold());
            println!("          {}", f.detail);
        }
    }

    // If an LSP instance is watching this workspace (e.g. the command was run
    // from a Zed task), have it surface the same result as a toast.
    crate::core::rescan::notify(workspace, crate::core::rescan::Kind::Doctor);

    Ok(())
}

/// The project's dependencies for the report. A pixi project has two kinds,
/// installed by different solvers, so each list is labelled:
/// `conda: numpy>=1.26 · pypi: requests>=2.31`.
fn dependency_list(project: &ProjectDeps) -> String {
    let pypi = project
        .packages
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if !project.is_pixi() {
        return pypi;
    }
    let conda = project
        .conda_deps
        .iter()
        .map(|d| match d.spec.trim() {
            "" | "*" => d.name.clone(),
            spec if spec.starts_with(['<', '>', '=', '!', '~']) => format!("{}{spec}", d.name),
            spec => format!("{} {spec}", d.name),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut parts = Vec::new();
    if !conda.is_empty() {
        parts.push(format!("conda: {conda}"));
    }
    if !pypi.is_empty() {
        parts.push(format!("pypi: {pypi}"));
    }
    parts.join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pixi::CondaDep;
    use crate::core::project::Requirement;

    #[test]
    fn pixi_dependencies_are_labelled_by_source() {
        let conda = |name: &str, spec: &str| CondaDep {
            name: name.into(),
            spec: spec.into(),
        };
        let mut project = ProjectDeps {
            packages: vec![Requirement {
                name: "requests".into(),
                spec: ">=2.31".into(),
            }],
            ..Default::default()
        };
        assert_eq!(dependency_list(&project), "requests>=2.31");

        project.pixi_manifest = Some("pixi.toml".into());
        project.conda_deps = vec![
            conda("numpy", ">=1.26"),
            conda("scipy", "*"),
            conda("pandas", "2.2.*"),
        ];
        assert_eq!(
            dependency_list(&project),
            "conda: numpy>=1.26, scipy, pandas 2.2.* · pypi: requests>=2.31"
        );
        project.packages.clear();
        assert_eq!(
            dependency_list(&project),
            "conda: numpy>=1.26, scipy, pandas 2.2.*"
        );
    }
}
