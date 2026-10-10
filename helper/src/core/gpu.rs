//! GPU detection, the input side of F2.
//!
//! `nvidia-smi` is the only local, offline signal for "what driver is
//! installed" — there is no packaging metadata for it. The probe result is
//! cached to disk with a 24h TTL, the same convention the PyPI "latest
//! version" cache uses, since a driver upgrade is rare enough that re-running
//! `nvidia-smi` on every buffer edit would be wasted work but a permanent
//! cache could go stale after a real upgrade.
//!
//! The probe also asks for the GPU's compute capability, because a driver
//! new enough for a CUDA runtime is not sufficient on its own: a torch wheel
//! only carries kernels for the architectures it was compiled for, so an RTX
//! 50-series card (sm_120) on a cu121 build fails at the first kernel launch
//! with "no kernel image is available" despite a perfectly current driver.
//!
//! Apple Silicon is detected from the platform, not probed, since there is no
//! `nvidia-smi` equivalent to ask and none is needed: MPS is unconditional on
//! that hardware.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::core::command;
use crate::core::platform::{self, Arch, Os, Platform};

const CACHE_TTL_SECS: u64 = 24 * 60 * 60;

/// Bumped whenever the probe starts collecting something new. A cache entry
/// written by an older probe (no `probe_version` at all, for the original
/// driver-only probe) is treated as stale, so an upgrade re-probes once
/// instead of serving a day-old answer that lacks the compute capability.
const PROBE_VERSION: u32 = 2;

/// The modern query. `compute_cap` arrived in nvidia-smi well after
/// `driver_version`, so drivers that predate it reject the whole query and
/// [`OLD_QUERY`] is tried instead.
const QUERY: &str = "--query-gpu=name,driver_version,compute_cap";
/// The original driver-only query, understood by every nvidia-smi.
const OLD_QUERY: &str = "--query-gpu=driver_version,name";

/// What the machine can run ML frameworks on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Accelerator {
    /// An NVIDIA GPU with a readable driver version, e.g. "535.104.05".
    Nvidia {
        name: String,
        driver_version: String,
        /// Compute capability as nvidia-smi prints it, e.g. "12.0" for an
        /// RTX 5090 or "8.9" for an RTX 4090. `None` on drivers too old to
        /// report it; those drivers also predate every architecture PyPilot
        /// has a CUDA floor for, so nothing is lost by skipping the check.
        compute_cap: Option<String>,
    },
    /// Apple Silicon: MPS is available unconditionally.
    AppleSilicon,
    /// No accelerator this crate knows how to target. CPU wheels are correct.
    None,
}

/// Detect the accelerator, using the disk cache when it is fresh.
pub async fn detect() -> Accelerator {
    let platform = Platform::current();
    if platform.os == Os::Macos && platform.arch == Arch::Aarch64 {
        return Accelerator::AppleSilicon;
    }

    if let Some(cached) = read_cache() {
        return cached;
    }

    let probed = probe_nvidia_smi().await;
    write_cache(&probed);
    probed
}

async fn probe_nvidia_smi() -> Accelerator {
    if let Some(found) = smi_first_line(QUERY)
        .await
        .and_then(|line| parse_smi_line(&line))
    {
        return found;
    }
    // Old drivers reject the unknown `compute_cap` field with a non-zero
    // exit, so a failure above is not proof that there is no GPU.
    smi_first_line(OLD_QUERY)
        .await
        .and_then(|line| parse_old_smi_line(&line))
        .unwrap_or(Accelerator::None)
}

/// First line of `nvidia-smi <query> --format=csv,noheader` (the first
/// reporting GPU), or `None` when the tool is missing or rejects the query.
async fn smi_first_line(query: &str) -> Option<String> {
    let args = [query, "--format=csv,noheader"];
    let out = command::run("nvidia-smi", &args, None).await.ok()?;
    if !out.success() {
        return None;
    }
    out.first_stdout_line().map(str::to_string)
}

/// Parse a [`QUERY`] line: "NVIDIA GeForce RTX 5090, 576.02, 12.0".
///
/// Split from the right, because the two trailing fields are numeric and the
/// marketing name is the only one that could ever contain a comma.
fn parse_smi_line(line: &str) -> Option<Accelerator> {
    let mut fields = line.rsplitn(3, ',');
    let cap = fields.next()?.trim();
    let driver_version = fields.next()?.trim().to_string();
    let name = fields.next()?.trim().to_string();
    if !driver_version.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    // "[N/A]" (or anything else non-numeric) means the driver knows the field
    // but could not read it: an unknown capability, not a failed probe.
    let compute_cap = cap
        .starts_with(|c: char| c.is_ascii_digit())
        .then(|| cap.to_string());
    Some(Accelerator::Nvidia {
        name,
        driver_version,
        compute_cap,
    })
}

/// Parse an [`OLD_QUERY`] line: "535.104.05, NVIDIA GeForce RTX 3080".
fn parse_old_smi_line(line: &str) -> Option<Accelerator> {
    let (driver, name) = line.split_once(',')?;
    let driver_version = driver.trim().to_string();
    if driver_version.is_empty() {
        return None;
    }
    Some(Accelerator::Nvidia {
        name: name.trim().to_string(),
        driver_version,
        compute_cap: None,
    })
}

#[derive(Serialize, Deserialize)]
struct CachedAccelerator {
    fetched_at_unix: u64,
    /// Absent (so 0) in caches written before the compute-capability probe.
    #[serde(default)]
    probe_version: u32,
    #[serde(flatten)]
    kind: CacheKind,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CacheKind {
    Nvidia {
        name: String,
        driver_version: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compute_cap: Option<String>,
    },
    None,
}

fn cache_path() -> PathBuf {
    platform::cache_dir().join("gpu.json")
}

fn read_cache() -> Option<Accelerator> {
    let text = std::fs::read_to_string(cache_path()).ok()?;
    let cached: CachedAccelerator = serde_json::from_str(&text).ok()?;
    if cached.probe_version < PROBE_VERSION {
        return None;
    }
    let age = now_unix().saturating_sub(cached.fetched_at_unix);
    if age > CACHE_TTL_SECS {
        return None;
    }
    Some(match cached.kind {
        CacheKind::Nvidia {
            name,
            driver_version,
            compute_cap,
        } => Accelerator::Nvidia {
            name,
            driver_version,
            compute_cap,
        },
        CacheKind::None => Accelerator::None,
    })
}

fn write_cache(accel: &Accelerator) {
    // Apple Silicon is never cached: it is derived from the platform, not
    // probed, so caching it would just be an extra file for no benefit.
    let kind = match accel {
        Accelerator::Nvidia {
            name,
            driver_version,
            compute_cap,
        } => CacheKind::Nvidia {
            name: name.clone(),
            driver_version: driver_version.clone(),
            compute_cap: compute_cap.clone(),
        },
        Accelerator::None => CacheKind::None,
        Accelerator::AppleSilicon => return,
    };
    let cached = CachedAccelerator {
        fetched_at_unix: now_unix(),
        probe_version: PROBE_VERSION,
        kind,
    };
    if let Some(parent) = cache_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string(&cached) {
        let _ = std::fs::write(cache_path(), text);
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvidia(name: &str, driver: &str, cap: Option<&str>) -> Accelerator {
        Accelerator::Nvidia {
            name: name.into(),
            driver_version: driver.into(),
            compute_cap: cap.map(str::to_string),
        }
    }

    #[test]
    fn ttl_is_24_hours() {
        assert_eq!(CACHE_TTL_SECS, 24 * 60 * 60);
    }

    #[test]
    fn parses_a_blackwell_line() {
        assert_eq!(
            parse_smi_line("NVIDIA GeForce RTX 5090, 576.02, 12.0"),
            Some(nvidia("NVIDIA GeForce RTX 5090", "576.02", Some("12.0")))
        );
    }

    #[test]
    fn parses_an_ada_line() {
        assert_eq!(
            parse_smi_line("NVIDIA GeForce RTX 4090, 550.54.14, 8.9"),
            Some(nvidia("NVIDIA GeForce RTX 4090", "550.54.14", Some("8.9")))
        );
    }

    #[test]
    fn a_name_containing_a_comma_survives() {
        assert_eq!(
            parse_smi_line("Quadro RTX 8000, Rev A, 535.54.03, 7.5"),
            Some(nvidia("Quadro RTX 8000, Rev A", "535.54.03", Some("7.5")))
        );
    }

    #[test]
    fn unreadable_compute_cap_is_unknown_not_a_failed_probe() {
        assert_eq!(
            parse_smi_line("Tesla K80, 470.42.01, [N/A]"),
            Some(nvidia("Tesla K80", "470.42.01", None))
        );
    }

    #[test]
    fn an_old_driver_error_line_does_not_parse_as_a_gpu() {
        // What an nvidia-smi that predates `compute_cap` prints. The probe
        // must fall back to OLD_QUERY, not report a GPU named after the error.
        assert_eq!(
            parse_smi_line("Field \"compute_cap\" is not a valid field to query."),
            None
        );
    }

    #[test]
    fn old_driver_query_still_parses_without_compute_cap() {
        assert_eq!(
            parse_old_smi_line("418.67, Tesla V100-SXM2-16GB"),
            Some(nvidia("Tesla V100-SXM2-16GB", "418.67", None))
        );
    }

    #[test]
    fn a_pre_compute_cap_cache_entry_still_parses_but_is_stale() {
        // The exact shape the driver-only probe used to write.
        let old = r#"{"fetched_at_unix":1,"kind":"nvidia","name":"RTX 3080","driver_version":"535.54.03"}"#;
        let cached: CachedAccelerator = serde_json::from_str(old).unwrap();
        assert!(cached.probe_version < PROBE_VERSION);
        let CacheKind::Nvidia { compute_cap, .. } = cached.kind else {
            panic!("expected an nvidia entry");
        };
        assert_eq!(compute_cap, None);
    }

    #[test]
    fn cache_round_trips_the_compute_cap() {
        let cached = CachedAccelerator {
            fetched_at_unix: 1,
            probe_version: PROBE_VERSION,
            kind: CacheKind::Nvidia {
                name: "RTX 5090".into(),
                driver_version: "576.02".into(),
                compute_cap: Some("12.0".into()),
            },
        };
        let text = serde_json::to_string(&cached).unwrap();
        let back: CachedAccelerator = serde_json::from_str(&text).unwrap();
        assert_eq!(back.probe_version, PROBE_VERSION);
        let CacheKind::Nvidia { compute_cap, .. } = back.kind else {
            panic!("expected an nvidia entry");
        };
        assert_eq!(compute_cap.as_deref(), Some("12.0"));
    }
}
