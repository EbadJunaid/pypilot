//! GPU architecture -> the CUDA runtime range a build must target.
//!
//! [`crate::matrix::nvidia`] answers "is the driver new enough for this CUDA
//! runtime"; this module answers the other half, "does a build for this CUDA
//! runtime contain kernels for this GPU at all". Both must hold. The second is
//! what bites RTX 50-series owners: their drivers are brand new, so a cu121
//! torch passes the driver check and then dies with "no kernel image is
//! available for execution on the device", because sm_120 kernels only exist
//! in builds made with CUDA 12.8 or newer.
//!
//! The data lives in the `architectures` section of `data/nvidia.json`, next
//! to the driver table it complements, with its sources cited there.

use serde::Deserialize;

use crate::matrix::nvidia::CudaVersion;
use crate::pypi::version::PackageVersion;

#[derive(Debug, Deserialize)]
struct ArchFile {
    #[serde(default)]
    architectures: Vec<ArchRow>,
}

#[derive(Debug, Deserialize)]
struct ArchRow {
    compute_cap: String,
    family: String,
    min_cuda: String,
    #[serde(default)]
    max_cuda: Option<String>,
    #[serde(default)]
    min_torch: Option<String>,
}

/// A compute capability, e.g. 12.0 for sm_120.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeCap {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCap {
    /// Parse nvidia-smi's "12.0" form.
    pub fn parse(s: &str) -> Option<ComputeCap> {
        let (major, minor) = s.trim().split_once('.')?;
        Some(ComputeCap {
            major: major.parse().ok()?,
            minor: minor.parse().ok()?,
        })
    }

    /// The `sm_XY` spelling PyTorch's own warnings and arch lists use, so a
    /// message can be matched against the error the user is staring at.
    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }
}

/// What one GPU needs from a CUDA build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchRequirement {
    /// The GPU's own capability (not the table row it matched).
    pub cap: ComputeCap,
    pub family: String,
    /// Oldest CUDA runtime whose builds carry kernels for this GPU.
    pub min_cuda: CudaVersion,
    /// Newest CUDA build that still carries them, for architectures PyTorch
    /// has since dropped. `None` means no known ceiling.
    pub max_cuda: Option<CudaVersion>,
    /// Oldest torch release with a build for this GPU, when that is a
    /// stronger statement than `min_cuda` alone.
    pub min_torch: Option<String>,
}

impl ArchRequirement {
    /// Does a build against `cuda` carry kernels for this GPU?
    pub fn accepts_cuda(&self, cuda: CudaVersion) -> bool {
        cuda >= self.min_cuda && self.max_cuda.map_or(true, |max| cuda <= max)
    }

    /// Is torch `version` new enough for this GPU? Unparseable versions are
    /// given the benefit of the doubt, matching the "never nag about what
    /// cannot be verified" rule elsewhere.
    pub fn accepts_torch(&self, version: &str) -> bool {
        let Some(min) = self.min_torch.as_deref().and_then(PackageVersion::parse) else {
            return true;
        };
        PackageVersion::parse(version).map_or(true, |v| v >= min)
    }

    /// "RTX 5090 (Blackwell RTX 50-series / RTX PRO, sm_120)".
    pub fn describe(&self, gpu_name: &str) -> String {
        format!("{gpu_name} ({}, {})", self.family, self.cap.sm())
    }
}

pub struct ArchTable {
    rows: Vec<(ComputeCap, ArchRow)>,
}

impl ArchTable {
    /// Load the `architectures` section. A refreshed `nvidia.json` written
    /// before that section existed parses as an empty table, so the caller
    /// should fall back to the bundled copy via [`ArchTable::is_empty`].
    pub fn load(json: &str) -> Option<ArchTable> {
        let file: ArchFile = serde_json::from_str(json).ok()?;
        let rows = file
            .architectures
            .into_iter()
            .filter_map(|r| Some((ComputeCap::parse(&r.compute_cap)?, r)))
            .collect();
        Some(ArchTable { rows })
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The requirement for a GPU reporting `compute_cap`.
    ///
    /// Matches NVIDIA's binary-compatibility rule: a cubin runs on the same
    /// major and any equal-or-higher minor, so the row used is the one with
    /// the same major and the highest minor not above the GPU's. `None` for
    /// an architecture the table has never heard of (a new major), where
    /// guessing would be worse than staying silent.
    pub fn requirement_for(&self, compute_cap: &str) -> Option<ArchRequirement> {
        let cap = ComputeCap::parse(compute_cap)?;
        let (_, row) = self
            .rows
            .iter()
            .filter(|(c, _)| c.major == cap.major && c.minor <= cap.minor)
            .max_by_key(|(c, _)| *c)?;
        Some(ArchRequirement {
            cap,
            family: row.family.clone(),
            min_cuda: CudaVersion::parse(&row.min_cuda)?,
            max_cuda: row.max_cuda.as_deref().and_then(CudaVersion::parse),
            min_torch: row.min_torch.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> ArchTable {
        ArchTable::load(crate::matrix::NVIDIA_JSON).expect("bundled table parses")
    }

    fn cuda(major: u32, minor: u32) -> CudaVersion {
        CudaVersion { major, minor }
    }

    #[test]
    fn bundled_table_has_architectures() {
        assert!(!table().is_empty());
    }

    #[test]
    fn rtx_50_series_needs_cuda_12_8_and_torch_2_7() {
        let req = table().requirement_for("12.0").unwrap();
        assert_eq!(req.min_cuda, cuda(12, 8));
        assert_eq!(req.min_torch.as_deref(), Some("2.7.0"));
        assert_eq!(req.cap.sm(), "sm_120");
        assert!(!req.accepts_cuda(cuda(12, 6)));
        assert!(req.accepts_cuda(cuda(12, 8)));
        assert!(!req.accepts_torch("2.6.0"));
        assert!(req.accepts_torch("2.7.1"));
    }

    #[test]
    fn data_center_blackwell_needs_cuda_12_8() {
        let req = table().requirement_for("10.0").unwrap();
        assert_eq!(req.min_cuda, cuda(12, 8));
    }

    #[test]
    fn higher_minor_inherits_its_family_row() {
        // GB10 (DGX Spark) is 12.1 and runs sm_120 cubins; B300 is 10.3.
        let t = table();
        assert_eq!(t.requirement_for("12.1").unwrap().min_cuda, cuda(12, 8));
        assert_eq!(t.requirement_for("10.3").unwrap().cap.sm(), "sm_103");
        // Ampere consumer 8.6 must not pick up Ada's 8.9 row.
        assert_eq!(t.requirement_for("8.6").unwrap().min_cuda, cuda(11, 1));
    }

    #[test]
    fn ada_and_hopper_are_fine_from_cuda_11_8() {
        let t = table();
        for cap in ["8.9", "9.0"] {
            let req = t.requirement_for(cap).unwrap();
            assert_eq!(req.min_cuda, cuda(11, 8));
            assert!(req.max_cuda.is_none());
            assert!(req.accepts_torch("2.4.0"));
        }
    }

    #[test]
    fn pascal_is_capped_at_cuda_12_6() {
        let req = table().requirement_for("6.1").unwrap();
        assert!(req.accepts_cuda(cuda(12, 6)));
        assert!(!req.accepts_cuda(cuda(13, 0)));
    }

    #[test]
    fn unknown_architecture_is_none() {
        let t = table();
        assert!(t.requirement_for("99.0").is_none());
        assert!(t.requirement_for("garbage").is_none());
    }

    #[test]
    fn a_driver_only_nvidia_json_is_an_empty_table() {
        let t = ArchTable::load(r#"{"drivers": []}"#).unwrap();
        assert!(t.is_empty());
    }
}
