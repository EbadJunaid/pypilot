//! Does the torch in `.venv` actually use the NVIDIA GPU?
//!
//! F2 picks the right CUDA index when PyPilot installs torch, but the result
//! does not stay put: torch may have come from the default index first (on
//! Windows PyPI only carries CPU wheels), or a later `pip install` of some
//! other package may have swapped `torch+cu12x` for a CPU build to satisfy its
//! own pin. Either way the user sees `torch.cuda.is_available() == False` on a
//! machine with a perfectly good GPU, and nothing says why.
//!
//! Two checks live here. [`static_check`] reads the installed metadata off
//! disk, so it is cheap enough for every assessment. [`self_test`] imports
//! torch in the venv and asks it directly; it costs a few seconds, so it only
//! runs right after PyPilot itself installed torch.

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::core::gpu::{self, Accelerator};
use crate::core::platform::{Arch, Os, Platform};
use crate::core::project::Requirement;
use crate::core::{command, installed, pip};
use crate::core::{Finding, FixKind, Severity};
use crate::matrix;
use crate::settings::Settings;

/// Title of the static finding, shared with tests.
pub const CPU_TORCH_TITLE: &str =
    "torch in .venv is a CPU build but this machine has an NVIDIA GPU";

/// How long the self-test may take. Importing torch and initializing CUDA is
/// slow on a cold cache, but a minute means something is stuck.
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(60);

/// What kind of build an installed torch is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TorchBuild {
    /// Built against CUDA; the version when known, e.g. "12.1".
    Cuda(Option<String>),
    /// CPU only.
    Cpu,
    /// Some other accelerator build (ROCm, XPU): not this module's business.
    Other,
    /// Cannot tell from what is on disk.
    Unknown,
}

/// The local version label of a version string: `2.5.1+cu121` -> `cu121`.
pub fn local_tag(version: &str) -> Option<&str> {
    version
        .split_once('+')
        .map(|(_, tag)| tag)
        .filter(|t| !t.is_empty())
}

/// The version without its local label: `2.5.1+cu121` -> `2.5.1`.
pub fn public_version(version: &str) -> &str {
    version.split_once('+').map_or(version, |(v, _)| v)
}

/// The `cuda` and `hip` fields of `torch/version.py`, which torch's build
/// writes as `cuda: Optional[str] = '12.1'` (or `= None`). Outer `None` means
/// the file did not have the field at all.
fn version_py_field(text: &str, field: &str) -> Option<Option<String>> {
    text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(field)?;
        // `cuda:` or `cuda =`, never `cuda_home` or similar.
        if !rest.starts_with([':', '=', ' ']) {
            return None;
        }
        let (_, value) = rest.split_once('=')?;
        let value = value.trim().trim_matches(['\'', '"']);
        Some((value != "None" && !value.is_empty()).then(|| value.to_string()))
    })
}

/// Classify an installed torch from its version string and, when readable,
/// the `torch/version.py` it ships.
///
/// `version.py` is authoritative. Without it, the local label decides
/// (`+cpu`, `+cu121`, `+rocm6.1`). An unlabelled version is whatever PyPI's
/// default wheel is on this platform: CUDA on Linux x86_64, CPU on Windows,
/// and unknown elsewhere rather than a guess.
pub fn classify(version: &str, version_py: Option<&str>, os: Os, arch: Arch) -> TorchBuild {
    if let Some(text) = version_py {
        if let Some(hip) = version_py_field(text, "hip") {
            if hip.is_some() {
                return TorchBuild::Other;
            }
        }
        if let Some(cuda) = version_py_field(text, "cuda") {
            return match cuda {
                Some(v) => TorchBuild::Cuda(Some(v)),
                None => TorchBuild::Cpu,
            };
        }
    }
    match local_tag(version) {
        Some("cpu") => TorchBuild::Cpu,
        Some(tag) if tag.starts_with("cu") => TorchBuild::Cuda(None),
        Some(_) => TorchBuild::Other,
        None => match (os, arch) {
            (Os::Linux, Arch::X86_64) => TorchBuild::Cuda(None),
            (Os::Windows, _) => TorchBuild::Cpu,
            _ => TorchBuild::Unknown,
        },
    }
}

/// The warning for a CPU torch on an NVIDIA machine, or `None` when the
/// build is fine (or cannot be judged). `reinstall` is the command that pulls
/// the build matched to the driver, when the matrix knows one.
pub fn cpu_torch_finding(
    build: &TorchBuild,
    version: &str,
    gpu_name: &str,
    reinstall: Option<&str>,
) -> Option<Finding> {
    if *build != TorchBuild::Cpu {
        return None;
    }
    let fix = match reinstall {
        Some(cmd) => format!("Reinstall it from the index matched to your driver: {cmd}"),
        None => "Reinstall it from the CUDA index for your driver (see pytorch.org/get-started/locally).".to_string(),
    };
    Some(Finding {
        severity: Severity::Warning,
        title: CPU_TORCH_TITLE.to_string(),
        detail: format!(
            "{gpu_name} is present, but torch {version} in .venv is a CPU-only build, so torch.cuda.is_available() returns False and everything runs on the CPU. This happens when torch came from the default index (CPU-only on Windows) or a later install replaced the CUDA build. {fix} In a uv project, also pin the index in pyproject.toml so the next sync keeps it."
        ),
        fix: FixKind::Manual,
    })
}

/// The index the matrix matches to this driver for `version`, or `None` when
/// it has no answer (a release outside the bundled data).
async fn matched_index(version: &str, settings: &Settings) -> Option<String> {
    let pin = Requirement {
        name: "torch".to_string(),
        spec: format!("=={}", public_version(version)),
    };
    matrix::solve::solve_framework("torch", Some(&pin), settings)
        .await?
        .index_url
}

/// Read-only check of the torch in `venv`, for the assessment.
///
/// Only disk reads unless the installed torch looks like a CPU build: only
/// then is the (cached) GPU probe consulted, so a machine without torch, or
/// with a CUDA build, pays nothing.
pub async fn static_check(venv: &Path, settings: &Settings) -> Option<Finding> {
    let version = installed::scan(venv).version_of("torch")?;
    let version_py = installed::site_packages_dirs(venv)
        .into_iter()
        .find_map(|dir| std::fs::read_to_string(dir.join("torch").join("version.py")).ok());
    let platform = Platform::current();
    let build = classify(&version, version_py.as_deref(), platform.os, platform.arch);
    if build != TorchBuild::Cpu {
        return None;
    }
    let Accelerator::Nvidia { name, .. } = gpu::detect().await else {
        return None;
    };
    let reinstall = match matched_index(&version, settings).await {
        // The driver is too old for every CUDA build, so CPU is right; F2
        // already says so in its own finding.
        Some(url) if url.trim_end_matches('/').ends_with("/cpu") => return None,
        Some(url) => Some(format!(
            "pip install --force-reinstall torch=={} --index-url {url}",
            public_version(&version)
        )),
        None => None,
    };
    cpu_torch_finding(&build, &version, &name, reinstall.as_deref())
}

/// Run in the venv's Python. Prints one JSON line; never raises, so a broken
/// torch install still produces output the engine can explain.
const SELF_TEST_SCRIPT: &str = r#"import json
r = {"cuda_available": False}
try:
    import torch
    r["torch_version"] = torch.__version__
    r["torch_cuda_version"] = torch.version.cuda
    r["cuda_available"] = bool(torch.cuda.is_available())
    if r["cuda_available"]:
        r["arch_list"] = torch.cuda.get_arch_list()
        r["device_name"] = torch.cuda.get_device_name(0)
        r["device_capability"] = list(torch.cuda.get_device_capability(0))
except Exception as e:
    r["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(r))
"#;

/// What [`SELF_TEST_SCRIPT`] prints.
#[derive(Debug, Deserialize)]
struct SelfTestReport {
    cuda_available: bool,
    torch_version: Option<String>,
    torch_cuda_version: Option<String>,
    #[serde(default)]
    arch_list: Vec<String>,
    device_name: Option<String>,
    device_capability: Option<(u32, u32)>,
    error: Option<String>,
}

/// Turn the self-test's stdout into a step result: `Ok` with a one-line
/// confirmation, or `Err` explaining the mismatch.
pub fn interpret_self_test(stdout: &str, gpu_name: &str) -> Result<String, String> {
    let line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| l.starts_with('{'))
        .ok_or_else(|| "the GPU check printed no result".to_string())?;
    let report: SelfTestReport = serde_json::from_str(line)
        .map_err(|e| format!("could not read the GPU check result: {e}"))?;

    if let Some(error) = report.error {
        return Err(format!("importing torch failed: {error}"));
    }
    let version = report.torch_version.as_deref().unwrap_or("(unknown)");
    let Some(cuda) = report.torch_cuda_version else {
        return Err(format!(
            "torch {version} is a CPU-only build, so it cannot use {gpu_name}. Reinstall torch from the CUDA index matched to your driver (pypilot doctor shows the command)."
        ));
    };
    if !report.cuda_available {
        return Err(format!(
            "torch {version} is built for CUDA {cuda}, but torch.cuda.is_available() is False. The NVIDIA driver may be too old for CUDA {cuda}, or the GPU is not visible to this process; check that nvidia-smi works."
        ));
    }

    let device = report.device_name.as_deref().unwrap_or(gpu_name);
    if let Some((major, minor)) = report.device_capability {
        if !supports_capability(&report.arch_list, major, minor) {
            return Err(format!(
                "{device} (sm_{major}{minor}) is not among the GPU architectures torch {version} was built for ({}), so CUDA kernels will fail to launch. Install a torch build for a newer CUDA release.",
                report.arch_list.join(", ")
            ));
        }
    }
    Ok(format!(
        "CUDA works on {device} (torch {version}, CUDA {cuda})"
    ))
}

/// Can a build compiled for `arch_list` run on a GPU of compute capability
/// `major.minor`? An exact `sm_XY` match, or any `compute_XY` PTX at or below
/// the device's capability (the driver JIT-compiles it). An empty list means
/// torch did not say, which is given the benefit of the doubt.
fn supports_capability(arch_list: &[String], major: u32, minor: u32) -> bool {
    if arch_list.is_empty() {
        return true;
    }
    let device = major * 10 + minor;
    arch_list.iter().any(|arch| {
        if let Some(sm) = arch.strip_prefix("sm_") {
            sm.trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse::<u32>()
                .is_ok_and(|v| v == device)
        } else if let Some(ptx) = arch.strip_prefix("compute_") {
            ptx.trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse::<u32>()
                .is_ok_and(|v| v <= device)
        } else {
            false
        }
    })
}

/// Ask the venv's torch whether it can see the GPU. `None` when there is no
/// NVIDIA GPU to check against.
pub async fn self_test(venv: &Path, workspace: &Path) -> Option<Result<String, String>> {
    let Accelerator::Nvidia { name, .. } = gpu::detect().await else {
        return None;
    };
    let python = pip::venv_python(venv);
    // -I: the workspace must not shadow torch with a local `torch.py`.
    let args = ["-I", "-c", SELF_TEST_SCRIPT];
    let result =
        match command::run_with_timeout(&python, &args, Some(workspace), SELF_TEST_TIMEOUT).await {
            Ok(out) if out.success() => interpret_self_test(&out.stdout, &name),
            Ok(out) => Err(format!("the GPU check failed: {}", out.stderr.trim())),
            Err(e) => Err(e.to_string()),
        };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUDA_VERSION_PY: &str = "__version__ = '2.5.1+cu121'\ndebug = False\ncuda: Optional[str] = '12.1'\ngit_version = 'abc'\nhip: Optional[str] = None\n";
    const CPU_VERSION_PY: &str =
        "__version__ = '2.5.1'\ndebug = False\ncuda: Optional[str] = None\nhip: Optional[str] = None\n";
    const ROCM_VERSION_PY: &str =
        "__version__ = '2.5.1+rocm6.2'\ncuda: Optional[str] = None\nhip: Optional[str] = '6.2.41133'\n";

    #[test]
    fn splits_local_version_labels() {
        assert_eq!(local_tag("2.5.1+cu121"), Some("cu121"));
        assert_eq!(local_tag("2.5.1+cpu"), Some("cpu"));
        assert_eq!(local_tag("2.5.1"), None);
        assert_eq!(local_tag("2.5.1+"), None);
        assert_eq!(public_version("2.5.1+cu121"), "2.5.1");
        assert_eq!(public_version("2.5.1"), "2.5.1");
    }

    #[test]
    fn version_py_is_authoritative() {
        // The label says nothing, the file says CUDA.
        assert_eq!(
            classify("2.5.1", Some(CUDA_VERSION_PY), Os::Windows, Arch::X86_64),
            TorchBuild::Cuda(Some("12.1".into()))
        );
        // The file says CPU even where the PyPI default would be CUDA.
        assert_eq!(
            classify("2.5.1", Some(CPU_VERSION_PY), Os::Linux, Arch::X86_64),
            TorchBuild::Cpu
        );
        assert_eq!(
            classify(
                "2.5.1+rocm6.2",
                Some(ROCM_VERSION_PY),
                Os::Linux,
                Arch::X86_64
            ),
            TorchBuild::Other
        );
        // Older torch wrote `cuda = None` without an annotation.
        assert_eq!(
            classify("1.13.1", Some("cuda = None\n"), Os::Linux, Arch::X86_64),
            TorchBuild::Cpu
        );
    }

    #[test]
    fn local_labels_decide_without_version_py() {
        for os in [Os::Linux, Os::Windows] {
            assert_eq!(
                classify("2.5.1+cpu", None, os, Arch::X86_64),
                TorchBuild::Cpu
            );
            assert_eq!(
                classify("2.5.1+cu124", None, os, Arch::X86_64),
                TorchBuild::Cuda(None)
            );
        }
        assert_eq!(
            classify("2.5.1+rocm6.2", None, Os::Linux, Arch::X86_64),
            TorchBuild::Other
        );
        // Unparseable file: fall back to the label.
        assert_eq!(
            classify("2.5.1+cpu", Some("garbage"), Os::Linux, Arch::X86_64),
            TorchBuild::Cpu
        );
    }

    #[test]
    fn unlabelled_versions_follow_the_platform_default() {
        // PyPI ships CUDA wheels for Linux x86_64 and CPU wheels for Windows.
        assert_eq!(
            classify("2.5.1", None, Os::Linux, Arch::X86_64),
            TorchBuild::Cuda(None)
        );
        assert_eq!(
            classify("2.5.1", None, Os::Windows, Arch::X86_64),
            TorchBuild::Cpu
        );
        assert_eq!(
            classify("2.5.1", None, Os::Linux, Arch::Aarch64),
            TorchBuild::Unknown
        );
    }

    #[test]
    fn only_cpu_builds_produce_the_warning() {
        let gpu = "NVIDIA GeForce RTX 3080";
        let cmd = "pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124";
        let f = cpu_torch_finding(&TorchBuild::Cpu, "2.5.1+cpu", gpu, Some(cmd)).unwrap();
        assert_eq!(f.severity, Severity::Warning);
        assert_eq!(f.title, CPU_TORCH_TITLE);
        assert!(f.detail.contains(gpu));
        assert!(f.detail.contains("2.5.1+cpu"));
        assert!(f.detail.contains(cmd));

        let f = cpu_torch_finding(&TorchBuild::Cpu, "2.5.1", gpu, None).unwrap();
        assert!(f.detail.contains("pytorch.org"));

        for build in [
            TorchBuild::Cuda(None),
            TorchBuild::Other,
            TorchBuild::Unknown,
        ] {
            assert!(cpu_torch_finding(&build, "2.5.1", gpu, Some(cmd)).is_none());
        }
    }

    const GPU: &str = "NVIDIA GeForce RTX 3080";

    #[test]
    fn self_test_reports_a_working_gpu() {
        let out = r#"{"cuda_available": true, "torch_version": "2.5.1+cu124", "torch_cuda_version": "12.4", "arch_list": ["sm_50", "sm_80", "sm_86", "sm_90"], "device_name": "NVIDIA GeForce RTX 3080", "device_capability": [8, 6]}"#;
        let ok = interpret_self_test(out, GPU).unwrap();
        assert!(ok.contains("RTX 3080"));
        assert!(ok.contains("12.4"));
    }

    #[test]
    fn self_test_ignores_noise_before_the_result() {
        let out = "UserWarning: something\n{\"cuda_available\": true, \"torch_version\": \"2.5.1\", \"torch_cuda_version\": \"12.4\"}\n";
        assert!(interpret_self_test(out, GPU).is_ok());
    }

    #[test]
    fn self_test_explains_a_cpu_build() {
        let out = r#"{"cuda_available": false, "torch_version": "2.5.1+cpu", "torch_cuda_version": null}"#;
        let err = interpret_self_test(out, GPU).unwrap_err();
        assert!(err.contains("CPU-only"));
        assert!(err.contains(GPU));
    }

    #[test]
    fn self_test_explains_an_invisible_gpu() {
        let out = r#"{"cuda_available": false, "torch_version": "2.5.1+cu124", "torch_cuda_version": "12.4"}"#;
        let err = interpret_self_test(out, GPU).unwrap_err();
        assert!(err.contains("CUDA 12.4"));
        assert!(err.contains("is_available() is False"));
    }

    #[test]
    fn self_test_catches_a_missing_architecture() {
        // An RTX 50-series card (sm_120) on a build that predates it.
        let out = r#"{"cuda_available": true, "torch_version": "2.5.1+cu124", "torch_cuda_version": "12.4", "arch_list": ["sm_80", "sm_86", "sm_90"], "device_name": "NVIDIA GeForce RTX 5090", "device_capability": [12, 0]}"#;
        let err = interpret_self_test(out, GPU).unwrap_err();
        assert!(err.contains("sm_120"));
        assert!(err.contains("RTX 5090"));
    }

    #[test]
    fn self_test_reports_import_errors_and_garbage() {
        let out = r#"{"cuda_available": false, "error": "ImportError: DLL load failed"}"#;
        assert!(interpret_self_test(out, GPU)
            .unwrap_err()
            .contains("DLL load failed"));
        assert!(interpret_self_test("", GPU).is_err());
        assert!(interpret_self_test("{not json", GPU).is_err());
    }

    #[test]
    fn architecture_matching_accepts_exact_sm_and_older_ptx() {
        let list = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(supports_capability(&list(&["sm_86"]), 8, 6));
        assert!(!supports_capability(&list(&["sm_80"]), 8, 6));
        assert!(supports_capability(&list(&["sm_80", "compute_80"]), 8, 6));
        assert!(!supports_capability(&list(&["compute_90"]), 8, 6));
        assert!(supports_capability(&list(&["sm_90a"]), 9, 0));
        assert!(supports_capability(&[], 12, 0));
    }
}
