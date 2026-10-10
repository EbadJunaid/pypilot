//! CUDA extension packages: flash-attn, xformers, bitsandbytes, mamba-ssm,
//! causal-conv1d.
//!
//! These are the dependencies that most often turn a working `pip install`
//! into an hour-long compile or a hard failure, and PyPI metadata alone does
//! not say why. flash-attn ships only an sdist whose build script imports
//! torch, so under pip's default build isolation it dies with
//! "No module named 'torch'" before compiling anything, and on Windows there
//! is no prebuilt binary to fall back on. xformers wheels are compiled against
//! one exact torch release, so `pip install xformers` quietly replaces the
//! torch already in the environment. bitsandbytes gained Windows and Apple
//! Silicon builds only in later releases.
//!
//! The table (`data/cuda_extensions.json`) records per release where PyPI has
//! a binary wheel and which torch the wheel was built for. [`assess_project`]
//! checks declared dependencies (for the assessment); [`check_import`] checks
//! one not-yet-installed import (for the editor guardian). Both are pure, so
//! every branch is testable without a network, a GPU or a venv.

use serde::Deserialize;

use crate::core::platform::{Arch, Os, Platform};
use crate::core::project::{normalize_name, Requirement};
use crate::core::{Finding, FixKind, Severity};
use crate::pypi::version::{PackageVersion, VersionSpec};

#[derive(Debug, Deserialize)]
struct ExtensionsFile {
    packages: Vec<Extension>,
}

/// One CUDA extension package and what is known about installing it.
#[derive(Debug, Clone, Deserialize)]
pub struct Extension {
    /// PyPI distribution name.
    pub name: String,
    /// Useless without an NVIDIA GPU, so macOS can never run it.
    pub requires_cuda: bool,
    /// The source build imports torch, so it needs `--no-build-isolation`
    /// with torch already installed.
    pub build_needs_torch: bool,
    /// Extra context on how its source build behaves.
    #[serde(default)]
    pub build_note: Option<String>,
    /// What to do instead when it will not install here.
    pub alternative: String,
    /// Rows apply from their version up to the next row.
    releases: Vec<ReleaseRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseRow {
    version: String,
    /// `os-arch` keys with a binary wheel on PyPI, see [`platform_key`].
    platforms: Vec<String>,
    /// The torch requirement this release's wheels declare.
    #[serde(default)]
    torch: Option<String>,
}

/// The parsed `cuda_extensions.json`.
#[derive(Debug, Clone)]
pub struct ExtensionTable {
    packages: Vec<Extension>,
}

/// Something worth saying before the user installs a CUDA extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    pub package: String,
    /// It cannot work at all here, not merely install badly.
    pub blocking: bool,
    pub title: String,
    /// Full sentences, naming the package; used as the editor diagnostic.
    pub detail: String,
}

impl Issue {
    pub fn to_finding(&self) -> Finding {
        Finding {
            severity: if self.blocking {
                Severity::Error
            } else {
                Severity::Warning
            },
            title: self.title.clone(),
            detail: self.detail.clone(),
            fix: FixKind::Manual,
        }
    }
}

impl ExtensionTable {
    pub fn load(json: &str) -> Option<ExtensionTable> {
        let file: ExtensionsFile = serde_json::from_str(json).ok()?;
        Some(ExtensionTable {
            packages: file.packages,
        })
    }

    /// A table that knows no packages, so every check stays silent.
    pub fn empty() -> ExtensionTable {
        ExtensionTable {
            packages: Vec::new(),
        }
    }

    /// Look a dependency up by name, PEP 503 normalized, so `flash_attn` and
    /// `Flash-Attn` both find `flash-attn`.
    pub fn get(&self, name: &str) -> Option<&Extension> {
        let wanted = normalize_name(name);
        self.packages
            .iter()
            .find(|e| normalize_name(&e.name) == wanted)
    }
}

impl Extension {
    /// The row describing the release `spec` would install. An exact pin uses
    /// its own row when one exists (post-releases matter for xformers, whose
    /// `.postN` builds target different torch versions); anything else uses
    /// the newest release the spec admits.
    fn row_for(&self, spec: &str) -> Option<&ReleaseRow> {
        if let Some(pinned) = exact_pin(spec) {
            if let Some(row) = self.releases.iter().find(|r| r.version == pinned) {
                return Some(row);
            }
            return self.row_at(&PackageVersion::parse(pinned)?);
        }
        let picked = VersionSpec::parse(spec)
            .select_newest(self.releases.iter().map(|r| r.version.as_str()))?;
        self.releases.iter().rev().find(|r| r.version == picked.raw)
    }

    /// The newest row at or below `version`.
    fn row_at(&self, version: &PackageVersion) -> Option<&ReleaseRow> {
        self.releases
            .iter()
            .rev()
            .find(|r| PackageVersion::parse(&r.version).is_some_and(|v| v <= *version))
    }

    /// The newest release whose wheels were built for `torch`.
    fn release_for_torch(&self, torch: &PackageVersion) -> Option<&str> {
        self.releases
            .iter()
            .rev()
            .find(|r| {
                r.torch
                    .as_deref()
                    .is_some_and(|t| VersionSpec::parse(t).matches(torch))
            })
            .map(|r| r.version.as_str())
    }
}

/// `==X` with no wildcard and no other clause, as written.
fn exact_pin(spec: &str) -> Option<&str> {
    let rest = spec.trim().strip_prefix("==")?.trim();
    if rest.is_empty() || rest.contains([',', '*', ';']) {
        return None;
    }
    Some(rest)
}

/// The key the table uses for a platform, e.g. `windows-x86_64`.
pub fn platform_key(platform: Platform) -> &'static str {
    match (platform.os, platform.arch) {
        (Os::Linux, Arch::X86_64) => "linux-x86_64",
        (Os::Linux, Arch::Aarch64) => "linux-aarch64",
        (Os::Windows, Arch::X86_64) => "windows-x86_64",
        (Os::Windows, Arch::Aarch64) => "windows-aarch64",
        (Os::Macos, Arch::X86_64) => "macos-x86_64",
        (Os::Macos, Arch::Aarch64) => "macos-aarch64",
    }
}

fn platform_name(platform: Platform) -> &'static str {
    match (platform.os, platform.arch) {
        (Os::Linux, Arch::X86_64) => "Linux x86-64",
        (Os::Linux, Arch::Aarch64) => "Linux ARM64",
        (Os::Windows, Arch::X86_64) => "Windows x64",
        (Os::Windows, Arch::Aarch64) => "Windows ARM64",
        (Os::Macos, Arch::X86_64) => "Intel macOS",
        (Os::Macos, Arch::Aarch64) => "Apple Silicon macOS",
    }
}

/// `xformers 0.0.30` for an exact pin, `xformers>=0.0.28` for a range, and
/// the bare name when unpinned.
fn describe(ext: &Extension, spec: &str) -> String {
    match exact_pin(spec) {
        Some(v) => format!("{} {v}", ext.name),
        None => format!("{}{}", ext.name, spec.trim()),
    }
}

/// No binary wheel for this platform: say what pip will do instead.
fn wheel_issue(ext: &Extension, spec: &str, platform: Platform) -> Option<Issue> {
    let row = ext.row_for(spec)?;
    if row.platforms.iter().any(|p| p == platform_key(platform)) {
        return None;
    }
    let what = describe(ext, spec);
    let name = &ext.name;
    let alternative = &ext.alternative;

    if ext.requires_cuda && platform.os == Os::Macos {
        return Some(Issue {
            package: name.clone(),
            blocking: true,
            title: format!("{name} cannot run on macOS"),
            detail: format!(
                "{what} is a CUDA extension and macOS has no CUDA, so there is no wheel and its source build fails. {alternative}"
            ),
        });
    }

    let mut detail = if row.platforms.is_empty() {
        format!(
            "{what} is published on PyPI only as source, so installing it compiles CUDA kernels: that needs the CUDA toolkit (nvcc) matching torch's CUDA version and a C++ compiler, and can take an hour or fail."
        )
    } else {
        format!(
            "{what} has no {} wheel on PyPI, so installing it either compiles from source (CUDA toolkit and a C++ compiler) or gets a build whose native library does not load here.",
            platform_name(platform)
        )
    };
    if platform.os == Os::Linux {
        if let Some(note) = &ext.build_note {
            detail.push(' ');
            detail.push_str(note);
        }
    } else if row.platforms.is_empty() {
        detail.push_str(&format!(
            " {name} provides no {} builds.",
            platform_name(platform)
        ));
    }
    if ext.build_needs_torch {
        detail.push_str(&format!(
            " Its build imports torch, so with pip's default build isolation it fails with \"No module named 'torch'\": install torch first, then `pip install {name} --no-build-isolation` (or `uv pip install --no-build-isolation {name}`)."
        ));
    }
    detail.push(' ');
    detail.push_str(alternative);

    Some(Issue {
        package: name.clone(),
        blocking: false,
        title: format!(
            "{name} will build from source on {}",
            platform_name(platform)
        ),
        detail,
    })
}

/// Check a project's declared dependencies. `torch_versions` are the torch
/// releases worth trying when deciding whether two torch requirements can
/// both hold (the bundled framework table's torch list is the natural input).
pub fn assess_project(
    table: &ExtensionTable,
    packages: &[Requirement],
    platform: Platform,
    torch_versions: &[String],
) -> Vec<Issue> {
    let torch = packages.iter().find(|r| normalize_name(&r.name) == "torch");
    let mut issues = Vec::new();

    for req in packages {
        let Some(ext) = table.get(&req.name) else {
            continue;
        };
        issues.extend(wheel_issue(ext, &req.spec, platform));

        // Only a pinned extension against a pinned torch can conflict: an
        // unpinned one lets the resolver pick the release matching torch.
        let (Some(torch), true) = (torch, req.is_pinned()) else {
            continue;
        };
        if !torch.is_pinned() {
            continue;
        }
        if let Some(issue) = torch_conflict(ext, &req.spec, &torch.spec, torch_versions) {
            issues.push(issue);
        }
    }
    issues
}

/// The pinned extension's wheels need a torch the project's torch pin rules out.
fn torch_conflict(
    ext: &Extension,
    spec: &str,
    torch_spec: &str,
    torch_versions: &[String],
) -> Option<Issue> {
    let row = ext.row_for(spec)?;
    let needs = row.torch.as_deref()?;
    let needed = VersionSpec::parse(needs);
    let declared = VersionSpec::parse(torch_spec);

    // Candidates: known torch releases plus the versions both specs name, so
    // an exact pin outside the bundled list is still tried.
    let candidates: Vec<PackageVersion> = torch_versions
        .iter()
        .map(String::as_str)
        .chain([strip_op(needs), strip_op(torch_spec)])
        .filter_map(PackageVersion::parse)
        .collect();
    if candidates
        .iter()
        .any(|v| needed.matches(v) && declared.matches(v))
    {
        return None;
    }

    let what = describe(ext, spec);
    let name = &ext.name;
    let suggestion = declared
        .select_newest(candidates.iter().map(|v| v.raw.as_str()))
        .and_then(|t| ext.release_for_torch(&t).map(|x| (t, x)))
        .map(|(t, x)| format!(" {name}=={x} is the release built for torch {}.", t.raw))
        .unwrap_or_default();
    Some(Issue {
        package: name.clone(),
        blocking: true,
        title: format!("{what} needs torch{needs}"),
        detail: format!(
            "{what} wheels are compiled against torch{needs}, but the project asks for torch{}. The install will fail to resolve or pull a different torch.{suggestion} Change one pin so they match.",
            torch_spec.trim()
        ),
    })
}

fn strip_op(spec: &str) -> &str {
    spec.trim()
        .trim_start_matches(['=', '~', '>', '<', '!', ' '])
}

/// Check one import that is not installed yet, as `pip install <package>`
/// would see it: the newest release, next to whatever torch the venv has.
pub fn check_import(
    table: &ExtensionTable,
    package: &str,
    platform: Platform,
    installed_torch: Option<&str>,
) -> Option<Issue> {
    let ext = table.get(package)?;
    if let Some(issue) = wheel_issue(ext, "", platform) {
        return Some(issue);
    }

    // The latest release wants a different torch than the venv has, so
    // installing it replaces torch (often a CUDA build with a CPU one).
    let torch = PackageVersion::parse(installed_torch?)?;
    let latest = ext.row_for("")?;
    let needs = latest.torch.as_deref()?;
    if VersionSpec::parse(needs).matches(&torch) {
        return None;
    }
    let name = &ext.name;
    let fix = match ext.release_for_torch(&torch) {
        Some(v) => format!(
            "Install `{name}=={v}` instead, which is built for torch {}.",
            torch.raw
        ),
        None => format!(
            "No {name} release in PyPilot's data targets torch {}.",
            torch.raw
        ),
    };
    Some(Issue {
        package: name.clone(),
        blocking: false,
        title: format!("{name} would replace torch {}", torch.raw),
        detail: format!(
            "The latest {name} ({}) is built for torch{needs}, so installing it replaces the installed torch {}, often with a build for a different CUDA. {fix}",
            latest.version, torch.raw
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> ExtensionTable {
        ExtensionTable::load(crate::matrix::CUDA_EXTENSIONS_JSON).expect("bundled table parses")
    }

    const WINDOWS: Platform = Platform {
        os: Os::Windows,
        arch: Arch::X86_64,
    };
    const LINUX: Platform = Platform {
        os: Os::Linux,
        arch: Arch::X86_64,
    };
    const MAC: Platform = Platform {
        os: Os::Macos,
        arch: Arch::Aarch64,
    };

    fn req(name: &str, spec: &str) -> Requirement {
        Requirement {
            name: name.into(),
            spec: spec.into(),
        }
    }

    fn torch_versions() -> Vec<String> {
        crate::matrix::frameworks::FrameworkTable::load(crate::matrix::FRAMEWORKS_JSON)
            .unwrap()
            .all_versions(crate::matrix::frameworks::Framework::Torch)
    }

    #[test]
    fn lookup_is_name_normalized() {
        let t = table();
        assert!(t.get("flash_attn").is_some());
        assert!(t.get("Flash-Attn").is_some());
        assert!(t.get("numpy").is_none());
    }

    #[test]
    fn flash_attn_on_windows_warns_with_sdpa_and_no_build_isolation() {
        let issues = assess_project(&table(), &[req("flash-attn", "")], WINDOWS, &[]);
        assert_eq!(issues.len(), 1);
        let i = &issues[0];
        assert!(!i.blocking);
        assert!(i.title.contains("Windows"), "{}", i.title);
        assert!(i.detail.contains("no Windows x64 builds"), "{}", i.detail);
        assert!(i.detail.contains("--no-build-isolation"));
        assert!(i.detail.contains("No module named 'torch'"));
        assert!(i.detail.contains("attn_implementation=\"sdpa\""));
        assert!(i.detail.contains("WSL2"));
    }

    #[test]
    fn flash_attn_on_linux_still_needs_torch_at_build_time() {
        let issues = assess_project(&table(), &[req("flash-attn", "==2.8.3")], LINUX, &[]);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].detail.starts_with("flash-attn 2.8.3 "));
        assert!(issues[0]
            .detail
            .contains("prebuilt wheel from the flash-attention"));
        assert!(issues[0].detail.contains("--no-build-isolation"));
    }

    #[test]
    fn cuda_extensions_are_blocking_on_macos() {
        let issues = assess_project(&table(), &[req("xformers", "")], MAC, &[]);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].blocking);
        assert_eq!(issues[0].to_finding().severity, Severity::Error);
    }

    #[test]
    fn xformers_wheel_gaps_follow_the_exact_release() {
        let t = table();
        // 0.0.28 to 0.0.29.post1 shipped no Windows wheel; .post2 brought it back.
        assert_eq!(
            assess_project(&t, &[req("xformers", "==0.0.29.post1")], WINDOWS, &[]).len(),
            1
        );
        assert!(assess_project(&t, &[req("xformers", "==0.0.29.post2")], WINDOWS, &[]).is_empty());
        assert!(assess_project(&t, &[req("xformers", "==0.0.29.post1")], LINUX, &[]).is_empty());
        assert!(assess_project(&t, &[req("xformers", "")], WINDOWS, &[]).is_empty());
    }

    #[test]
    fn xformers_pinned_against_a_different_torch_is_an_error() {
        let packages = [req("torch", "==2.8.0"), req("xformers", "==0.0.30")];
        let issues = assess_project(&table(), &packages, LINUX, &torch_versions());
        assert_eq!(issues.len(), 1, "{issues:?}");
        let i = &issues[0];
        assert!(i.blocking);
        assert_eq!(i.title, "xformers 0.0.30 needs torch==2.7.0");
        assert!(i.detail.contains("torch==2.8.0"));
        assert!(
            i.detail
                .contains("xformers==0.0.32.post2 is the release built for torch 2.8.0"),
            "{}",
            i.detail
        );
    }

    #[test]
    fn matching_or_unpinned_torch_is_silent() {
        let t = table();
        let v = torch_versions();
        let ok = [req("torch", "==2.7.0"), req("xformers", "==0.0.30")];
        assert!(assess_project(&t, &ok, LINUX, &v).is_empty());
        let range = [req("torch", ">=2.6,<2.8"), req("xformers", "==0.0.30")];
        assert!(assess_project(&t, &range, LINUX, &v).is_empty());
        let unpinned_torch = [req("torch", ""), req("xformers", "==0.0.30")];
        assert!(assess_project(&t, &unpinned_torch, LINUX, &v).is_empty());
        let unpinned_xformers = [req("torch", "==2.4.0"), req("xformers", "")];
        assert!(assess_project(&t, &unpinned_xformers, LINUX, &v).is_empty());
        let open_ended = [req("torch", "==2.11.0"), req("xformers", "==0.0.35")];
        assert!(assess_project(&t, &open_ended, LINUX, &v).is_empty());
    }

    #[test]
    fn bitsandbytes_platform_support_depends_on_release() {
        let t = table();
        assert_eq!(
            assess_project(&t, &[req("bitsandbytes", "==0.42.0")], WINDOWS, &[]).len(),
            1
        );
        assert_eq!(
            assess_project(&t, &[req("bitsandbytes", "<0.43")], WINDOWS, &[]).len(),
            1
        );
        assert!(assess_project(&t, &[req("bitsandbytes", "==0.43.0")], WINDOWS, &[]).is_empty());
        assert!(assess_project(&t, &[req("bitsandbytes", "")], MAC, &[]).is_empty());
        let old_on_mac = assess_project(&t, &[req("bitsandbytes", "==0.45.5")], MAC, &[]);
        assert_eq!(old_on_mac.len(), 1);
        assert!(!old_on_mac[0].blocking, "bitsandbytes is not CUDA-only");
    }

    #[test]
    fn other_packages_are_ignored() {
        let issues = assess_project(&table(), &[req("numpy", "==2.0")], WINDOWS, &[]);
        assert!(issues.is_empty());
    }

    #[test]
    fn importing_xformers_next_to_an_older_torch_names_the_matching_release() {
        let issue = check_import(&table(), "xformers", LINUX, Some("2.7.0")).unwrap();
        assert!(!issue.blocking);
        assert!(issue.detail.contains("replaces the installed torch 2.7.0"));
        assert!(
            issue.detail.contains("`xformers==0.0.30`"),
            "{}",
            issue.detail
        );

        // Already the torch the latest release wants: nothing to say.
        assert!(check_import(&table(), "xformers", LINUX, Some("2.11.0")).is_none());
        assert!(check_import(&table(), "xformers", LINUX, None).is_none());
    }

    #[test]
    fn importing_flash_attn_on_windows_warns() {
        let issue = check_import(&table(), "flash_attn", WINDOWS, Some("2.7.0")).unwrap();
        assert!(issue.detail.contains("sdpa"));
        assert!(check_import(&table(), "requests", WINDOWS, None).is_none());
    }
}
