//! Strongly-typed `portia-garak` calibration specifications, receipts, and solver errors
//! shared across `portia-garak` and `portia-cli`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Reference pre-MLP L2 norm (`35.90`) used to normalize scales across `0.8B`, `4B`, and `9B`.
pub const GARAK_REFERENCE_PRE_MLP_NORM: f64 = 35.90;
/// Default `StaticPermanentScope` background activation leakage ceiling (`0.25`).
pub const DEFAULT_GARAK_MAX_BG_LEAKAGE: f64 = 0.25;
/// Canonical 10 Qwen3.5 BPE token IDs used to synthesize `v_no` (`2665`) and `v_act` (`50821`).
pub const GARAK_CONTRAST_TOKEN_IDS: [u32; 10] =
    [2665, 9175, 332, 1206, 9764, 27775, 3783, 50821, 760, 8160];
/// Canonical slot names emitted in [`GarakCompileReceipt::slot_diagnostics`].
pub const GARAK_SLOT_NAMES: [&str; 4] = ["Slot0_GC", "Slot1_FC", "Slot2_WK", "Slot3_PKG"];

/// Errors returned by `portia-garak` readout resolution and `IntentExpr` compilation.
#[derive(Debug, Error, Serialize)]
pub enum GarakSolverError {
    #[error("invalid matrix `{name}`: expected non-empty rows of hidden_dim={expected_dim}, got shape ({nrows}, {ncols})")]
    InvalidMatrixShape {
        name: String,
        nrows: usize,
        ncols: usize,
        expected_dim: usize,
    },
    #[error("invalid vector `{name}`: expected length {expected_dim}, got {actual}")]
    InvalidVectorLength {
        name: String,
        expected_dim: usize,
        actual: usize,
    },
    #[error("invalid split `{name}`: {reason}")]
    InvalidSplit { name: String, reason: String },
    #[error("missing required unembedding token row for token_id={token_id} in w_u_rows")]
    MissingTokenUnembeddingRow { token_id: u32 },
    #[error("degenerate vector norm ({norm:.3e}) in `{context}`")]
    DegenerateDirection { context: String, norm: f64 },
    #[error("linear/conic solve failed in `{context}`")]
    LinearSolveFailed { context: String },
    #[error("slot bounds [{rel_lo}..{rel_hi}) exceed widened_slots capacity ({widened_slots})")]
    SlotOffsetOutOfBounds {
        rel_lo: usize,
        rel_hi: usize,
        widened_slots: usize,
    },
    #[error("StaticPermanentScope guard_bg_max ({actual:.4}) exceeded max_bg_rss_leakage ({ceiling:.4})")]
    GuardLeakageExceeded { actual: f64, ceiling: f64 },
    #[error("V2 scalar cold-guard maximum ({actual:.6}) exceeded ceiling ({ceiling:.6}); per-slot diagnostics retained")]
    GuardLeakageV2Exceeded {
        actual: f64,
        ceiling: f64,
        diagnostics: BTreeMap<String, BTreeMap<String, GarakSlotDiagnosticCell>>,
    },
    #[error("manifest integrity check failed: {0}")]
    ManifestIntegrity(String),
}

const fn d_theta_gc() -> f64 {
    0.24
}
const fn d_theta_sen() -> f64 {
    0.22
}
const fn d_theta_fc() -> f64 {
    0.36
}

/// Calibrated `SO(d)` rotation angles $\theta \in (0, \pi)$ (in radians) for the 4 static `Garak` rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GarakCalibratedAngles {
    #[serde(default = "d_theta_gc")]
    pub theta_gc: f64,
    #[serde(default = "d_theta_sen")]
    pub theta_sen: f64,
    #[serde(default = "d_theta_fc")]
    pub theta_fc: f64,
    #[serde(default = "d_theta_fc")]
    pub theta_pkg: f64,
}

impl Default for GarakCalibratedAngles {
    fn default() -> Self {
        Self {
            theta_gc: d_theta_gc(),
            theta_sen: d_theta_sen(),
            theta_fc: d_theta_fc(),
            theta_pkg: d_theta_fc(),
        }
    }
}

/// Calibration split partition (`"fit"` vs. `"dev"`). Only `Fit` rows enter numerical weight fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GarakPartition {
    Fit,
    Dev,
}

impl GarakPartition {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fit => "fit",
            Self::Dev => "dev",
        }
    }
}

impl std::fmt::Display for GarakPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed calibration family for V2 Garak splits (`snake_case` wire-compatible with Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GarakCalibrationFamily {
    GcConnected,
    GcDisconnected,
    FalseClaim,
    WkFalse,
    WkTrue,
    UnrealPackage,
    UnrealPackageRs,
    RealPackage,
    TrueClaim,
    Continuation,
    Benign,
}

impl GarakCalibrationFamily {
    pub const REQUIRED_V2_FAMILIES: [Self; 9] = [
        Self::GcConnected,
        Self::GcDisconnected,
        Self::FalseClaim,
        Self::WkFalse,
        Self::WkTrue,
        Self::UnrealPackage,
        Self::RealPackage,
        Self::TrueClaim,
        Self::Continuation,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GcConnected => "gc_connected",
            Self::GcDisconnected => "gc_disconnected",
            Self::FalseClaim => "false_claim",
            Self::WkFalse => "wk_false",
            Self::WkTrue => "wk_true",
            Self::UnrealPackage => "unreal_package",
            Self::UnrealPackageRs => "unreal_package_rs",
            Self::RealPackage => "real_package",
            Self::TrueClaim => "true_claim",
            Self::Continuation => "continuation",
            Self::Benign => "benign",
        }
    }

    #[must_use]
    pub const fn expected_positive_slot(self) -> Option<&'static str> {
        match self {
            Self::GcConnected | Self::GcDisconnected => Some("Slot0_GC"),
            Self::FalseClaim => Some("Slot1_FC"),
            Self::WkFalse => Some("Slot2_WK"),
            Self::UnrealPackage | Self::UnrealPackageRs => Some("Slot3_PKG"),
            _ => None,
        }
    }
}

impl std::fmt::Display for GarakCalibrationFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakRoleSplit {
    pub name: String,
    pub partition: GarakPartition,
    pub family: GarakCalibrationFamily,
    pub row_ids: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    pub positive_slots: Vec<String>,
    pub cold_slots: Vec<String>,
    pub globally_benign: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakV2Spec {
    pub splits: Vec<GarakRoleSplit>,
    pub ridge: f64,
    pub cold_weight: f64,
    pub candidate_index: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakPreMlpSplits {
    pub train_gc: Vec<Vec<f64>>,
    pub train_fc: Vec<Vec<f64>>,
    pub train_true_claims: Vec<Vec<f64>>,
    pub train_unreal_py: Vec<Vec<f64>>,
    pub train_real_py: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_gc: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_fc: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_unreal_py: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transfer_senators: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transfer_primes: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transfer_unreal_rs: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guard_real_code: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guard_true_claims: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakResidualSplits {
    pub train_gc: Vec<Vec<f64>>,
    pub train_fc: Vec<Vec<f64>>,
    pub train_true_claims: Vec<Vec<f64>>,
    pub train_unreal_py: Vec<Vec<f64>>,
}

const fn d_pkg_templates() -> usize {
    4
}
const fn d_max_bg_leakage() -> f64 {
    DEFAULT_GARAK_MAX_BG_LEAKAGE
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakCompileInputSpec {
    #[serde(default)]
    pub v2: Option<GarakV2Spec>,
    pub model_id: String,
    pub target_layer_idx: usize,
    pub base_intermediate_size: usize,
    #[serde(default)]
    pub widened_slots: Option<usize>,
    #[serde(default)]
    pub slot_offset: Option<usize>,
    #[serde(default = "d_pkg_templates")]
    pub pkg_template_count: usize,
    pub x_splits: GarakPreMlpSplits,
    pub h_splits: GarakResidualSplits,
    pub x_cont_all: Vec<Vec<f64>>,
    #[serde(default)]
    pub calibrated_angles: GarakCalibratedAngles,
    pub readout: crate::readout::GarakReadoutSpec,
    #[serde(default = "d_max_bg_leakage")]
    pub max_bg_rss_leakage: f64,
    #[serde(default)]
    pub strict_guard_check: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakSlotDiagnosticCell {
    pub fire_pct: f64,
    pub mean: f64,
    pub abs_max: f64,
    #[serde(default)]
    pub worst_row_id: Option<String>,
    #[serde(default)]
    pub worst_signed_activation: f64,
    #[serde(default)]
    pub cold: bool,
    #[serde(default)]
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakCompileReceipt {
    pub audit_status: String,
    pub accepted: bool,
    pub model_id: String,
    pub target_layer: usize,
    pub hidden_dim: usize,
    pub base_intermediate_size: usize,
    pub widened_intermediate_size: usize,
    pub widened_slots: usize,
    pub slot_lo: usize,
    pub slot_hi: usize,
    pub scale_factor_sx: f64,
    pub calibrated_angles: GarakCalibratedAngles,
    pub guard_bg_max: f64,
    pub max_bg_rss_leakage: f64,
    pub guard_within_threshold: bool,
    pub slot_diagnostics: BTreeMap<String, BTreeMap<String, GarakSlotDiagnosticCell>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<f64>,
}
