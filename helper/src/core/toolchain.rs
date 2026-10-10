//! Build-toolchain detection and per-OS advice for source builds.
//!
//! When a dependency has no wheel for this machine, pip/uv compile its sdist,
//! and the usual failure is a missing compiler ("Microsoft Visual C++ 14.0 or
//! greater is required", "Failed building wheel"). This module checks, cheaply
//! and only when a source build is actually coming, whether a C toolchain and
//! Rust are installed, and words the install advice for the current OS.

use std::path::PathBuf;

use crate::core::command;
use crate::core::platform::Os;

/// What build tooling the machine has. Probed only when needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Toolchain {
    /// A working C/C++ compiler (MSVC Build Tools, Xcode CLT, or `cc`).
    pub c_compiler: bool,
    /// `rustc` is on PATH.
    pub rust: bool,
}

/// Packages whose sdists build with Rust (maturin / setuptools-rust). PyPI's
/// JSON API doesn't expose build requirements, so this is a known-name list of
/// the popular ones that most often reach users as source builds.
const RUST_PACKAGES: &[&str] = &[
    "bcrypt",
    "cryptography",
    "jiter",
    "maturin",
    "orjson",
    "polars",
    "pydantic-core",
    "pyo3",
    "rpds-py",
    "ruff",
    "safetensors",
    "setuptools-rust",
    "tiktoken",
    "tokenizers",
    "watchfiles",
];

/// Does this package's sdist need a Rust toolchain to build?
pub fn needs_rust(name: &str) -> bool {
    let norm = name.to_ascii_lowercase().replace(['_', '.'], "-");
    RUST_PACKAGES.contains(&norm.as_str())
}

/// How to get a C/C++ compiler on `os`.
pub fn compiler_advice(os: Os) -> &'static str {
    match os {
        Os::Windows => "Install \"Microsoft C++ Build Tools\" (https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the \"Desktop development with C++\" workload",
        Os::Macos => "Install the Xcode Command Line Tools with `xcode-select --install`",
        Os::Linux => "Install a compiler and the Python headers (Debian/Ubuntu: `sudo apt install build-essential python3-dev`; Fedora: `sudo dnf install gcc gcc-c++ python3-devel`)",
    }
}

/// How to get Rust (the same everywhere).
pub const RUST_ADVICE: &str = "Install Rust from https://rustup.rs";

/// The toolchain steps still missing for building `package` on `os`, as one
/// sentence, or `None` when everything needed is already installed.
/// `toolchain` is `None` when it wasn't probed: then give the full advice.
pub fn build_advice(package: &str, os: Os, toolchain: Option<Toolchain>) -> Option<String> {
    let have = toolchain.unwrap_or_default();
    let mut steps = Vec::new();
    // Rust extensions still link through the platform C toolchain (MSVC on
    // Windows), so the compiler advice applies to them too.
    if !have.c_compiler {
        steps.push(compiler_advice(os).to_string());
    }
    if needs_rust(package) && !have.rust {
        steps.push(RUST_ADVICE.to_string());
    }
    if steps.is_empty() {
        None
    } else {
        Some(steps.join("; "))
    }
}

/// Probe for a C compiler and Rust. Spawns a couple of short `--version`
/// style processes, so callers run it only when a source build is coming.
pub async fn detect(os: Os) -> Toolchain {
    let c_compiler = match os {
        Os::Windows => has_msvc().await,
        Os::Macos => succeeds("xcode-select", &["-p"]).await,
        Os::Linux => succeeds("cc", &["--version"]).await,
    };
    let rust = succeeds("rustc", &["--version"]).await;
    Toolchain { c_compiler, rust }
}

/// True when `program args` spawns and exits 0.
async fn succeeds(program: &str, args: &[&str]) -> bool {
    command::run(program, args, None)
        .await
        .is_ok_and(|o| o.success())
}

/// MSVC is installed when `vswhere` finds an instance with the x64 C++ tools.
/// `vswhere.exe` ships with every Visual Studio / Build Tools installer at a
/// fixed location, so no PATH lookup is involved.
async fn has_msvc() -> bool {
    let Some(base) = std::env::var_os("ProgramFiles(x86)") else {
        return false;
    };
    let vswhere: PathBuf = [
        base.as_os_str(),
        "Microsoft Visual Studio".as_ref(),
        "Installer".as_ref(),
        "vswhere.exe".as_ref(),
    ]
    .iter()
    .collect();
    if !vswhere.is_file() {
        return false;
    }
    let args = [
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ];
    command::run(&vswhere, &args, None)
        .await
        .is_ok_and(|o| o.success() && o.first_stdout_line().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_packages_match_normalized_names() {
        assert!(needs_rust("pydantic_core"));
        assert!(needs_rust("Tokenizers"));
        assert!(!needs_rust("numpy"));
    }

    #[test]
    fn advice_is_os_specific() {
        let win = build_advice("lxml", Os::Windows, None).unwrap();
        assert!(win.contains("Microsoft C++ Build Tools"));
        assert!(win.contains("Desktop development with C++"));
        assert!(!win.contains("rustup"));
        let mac = build_advice("lxml", Os::Macos, None).unwrap();
        assert!(mac.contains("xcode-select --install"));
        let linux = build_advice("lxml", Os::Linux, None).unwrap();
        assert!(linux.contains("build-essential"));
        assert!(linux.contains("python3-dev"));
    }

    #[test]
    fn rust_packages_also_get_rustup() {
        let a = build_advice("tokenizers", Os::Linux, None).unwrap();
        assert!(a.contains("build-essential"));
        assert!(a.contains("rustup"));
    }

    #[test]
    fn installed_tools_drop_their_advice() {
        let cc_only = Toolchain {
            c_compiler: true,
            rust: false,
        };
        assert_eq!(build_advice("lxml", Os::Windows, Some(cc_only)), None);
        let a = build_advice("orjson", Os::Windows, Some(cc_only)).unwrap();
        assert!(a.contains("rustup"));
        assert!(!a.contains("Build Tools"));
        let all = Toolchain {
            c_compiler: true,
            rust: true,
        };
        assert_eq!(build_advice("orjson", Os::Macos, Some(all)), None);
    }
}
