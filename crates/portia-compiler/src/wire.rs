//! Runtime JSON wire format (`wire`) consumed by `scripts/portia_qwen_stdio_worker.py` and `portia-cli`.

use crate::error::V2CompilerError;
use crate::numerics::{dot_f64, silu_f64};
use half::bf16;
use portia_core::{GrammarCartridgeSpec, Mode, MountPolicy, PenOwner};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Default dynamic widened slot bank capacity (`128`).
pub const TOTAL_RESERVED_SLOTS: usize = 128;

/// Parse a raw bank / mount policy wire key into `(normalized_wire_key, MountPolicy, PenOwner)`.
pub fn parse_mount_policy_and_pen(
    raw_bank: &str,
) -> Result<(String, MountPolicy, PenOwner), V2CompilerError> {
    let (mount_policy, pen_owner) = MountPolicy::from_wire_key(raw_bank).ok_or_else(|| {
        V2CompilerError::WireIntegrity(format!(
            "unsupported bank_name {raw_bank:?}; expected \"OrientTrunk\", \"UnifiedStatic\", \"Base\", \"ModePayload:<Mode>\", \"ModePlan:<Mode>\", or \"Prefill:<Seg>\""
        ))
    })?;
    Ok((mount_policy.wire_key().to_string(), mount_policy, pen_owner))
}

/// Deterministic synthetic coordinate readout vector used only when `--allow-synthetic-fixture-targets` is set in tests.
pub fn canonical_mode_readout_vector(mode: Mode, d: usize) -> Result<Vec<f64>, V2CompilerError> {
    if d < 8 {
        return Err(V2CompilerError::DimensionMismatch {
            context: "canonical_mode_readout_vector",
            expected: 8,
            actual: d,
        });
    }
    let mut v = vec![0.0; d];
    let idx = match mode {
        Mode::Probe => 1,
        Mode::Read => 2,
        Mode::Ask => 3,
        Mode::Write => 4,
        Mode::Result => 5,
        Mode::Abstain => 6,
    };
    v[idx] = 1.0;
    Ok(v)
}

/// Quantized `bf16` weights for a single SwiGLU neuron slot (`gate`, `up`, `down` $\in \mathbb{R}^d$).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledSlotBf16 {
    pub slot_offset: usize,
    pub w_gate: Vec<bf16>,
    pub w_up: Vec<bf16>,
    pub w_down: Vec<bf16>,
}

impl CompiledSlotBf16 {
    #[inline]
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.w_gate.len()
    }

    /// Evaluate exact `bf16` pre-SiLU gate and post-SwiGLU scalar activation on `x` (`&[f64]`).
    #[must_use]
    pub fn eval_gate_and_act(&self, x: &[f64]) -> (f64, f64) {
        let wg: Vec<f64> = self.w_gate.iter().map(|v| v.to_f64()).collect();
        let wu: Vec<f64> = self.w_up.iter().map(|v| v.to_f64()).collect();
        let g = dot_f64(&wg, x);
        let u = dot_f64(&wu, x);
        (g, silu_f64(g) * u)
    }
}

/// Wire entry for physical widened MLP range `[slot_lo .. slot_hi)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireSlotEntryJson {
    pub slot_lo: usize,
    pub slot_hi: usize,
    pub w_gate: Vec<Vec<f32>>,
    pub w_up: Vec<Vec<f32>>,
    pub w_down: Vec<Vec<f32>>,
}

impl WireSlotEntryJson {
    /// Width of the physical MLP slot range `[slot_lo .. slot_hi)`.
    #[inline]
    #[must_use]
    pub const fn slot_span(&self) -> usize {
        self.slot_hi.saturating_sub(self.slot_lo)
    }

    /// Verify `(span, d)` dimensions on `w_gate`, `w_up`, and `(d, span)` on `w_down`.
    pub fn verify_dimensions(&self, expected_dim: usize) -> Result<(), V2CompilerError> {
        let span = self.slot_span();
        if span == 0 || self.w_gate.len() != span || self.w_up.len() != span {
            return Err(V2CompilerError::WireIntegrity(format!(
                "invalid slot span [{}, {}): gate_rows={}, up_rows={}",
                self.slot_lo,
                self.slot_hi,
                self.w_gate.len(),
                self.w_up.len()
            )));
        }
        if self.w_down.len() != expected_dim {
            return Err(V2CompilerError::DimensionMismatch {
                context: "WireSlotEntryJson::w_down",
                expected: expected_dim,
                actual: self.w_down.len(),
            });
        }
        for r in self.w_gate.iter().chain(self.w_up.iter()) {
            if r.len() != expected_dim {
                return Err(V2CompilerError::DimensionMismatch {
                    context: "WireSlotEntryJson::w_gate/w_up",
                    expected: expected_dim,
                    actual: r.len(),
                });
            }
        }
        for r in &self.w_down {
            if r.len() != span {
                return Err(V2CompilerError::DimensionMismatch {
                    context: "WireSlotEntryJson::w_down_cols",
                    expected: span,
                    actual: r.len(),
                });
            }
        }
        Ok(())
    }

    /// Pack a contiguous slice of `CompiledSlotBf16` into a single `WireSlotEntryJson` block.
    pub fn from_contiguous_slots(
        slots: &[CompiledSlotBf16],
        base_intermediate_size: usize,
    ) -> Result<Self, V2CompilerError> {
        let Some(first) = slots.first() else {
            return Err(V2CompilerError::WireIntegrity(
                "cannot pack empty CompiledSlotBf16 slice".to_string(),
            ));
        };
        let d = first.hidden_dim();
        let span = slots.len();
        for (i, s) in slots.iter().enumerate() {
            if s.slot_offset != first.slot_offset + i {
                return Err(V2CompilerError::WireIntegrity(format!(
                    "non-contiguous slot offsets: expected {}, got {}",
                    first.slot_offset + i,
                    s.slot_offset
                )));
            }
            if s.hidden_dim() != d || s.w_up.len() != d || s.w_down.len() != d {
                return Err(V2CompilerError::DimensionMismatch {
                    context: "WireSlotEntryJson::from_contiguous_slots",
                    expected: d,
                    actual: s.hidden_dim(),
                });
            }
        }

        let w_gate = slots
            .iter()
            .map(|s| s.w_gate.iter().map(|v| v.to_f32()).collect())
            .collect();
        let w_up = slots
            .iter()
            .map(|s| s.w_up.iter().map(|v| v.to_f32()).collect())
            .collect();
        let w_down = (0..d)
            .map(|row_i| slots.iter().map(|s| s.w_down[row_i].to_f32()).collect())
            .collect();

        Ok(Self {
            slot_lo: base_intermediate_size + first.slot_offset,
            slot_hi: base_intermediate_size + first.slot_offset + span,
            w_gate,
            w_up,
            w_down,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireCartridgeHeader {
    pub model_family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireBisimulationReceipt {
    pub total_leaves_verified: usize,
    pub positive_rows_checked: usize,
    pub negative_rows_checked: usize,
    pub min_positive_gate_pre_silu: f64,
    pub max_negative_gate_pre_silu: f64,
    pub max_negative_swiglu_leak_l2: f64,
    pub min_positive_chordal_cosine: f64,
    pub decoded_mode_matches: usize,
    pub residual_source: String,
    pub bisimulation_holds: bool,
}

fn default_base_inter() -> usize {
    3584
}

/// Complete executable runtime manifest consumed by `portia_qwen_stdio_worker.py` and `portia-cli`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireCartridgeManifestJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<WireCartridgeHeader>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default = "default_base_inter")]
    pub base_intermediate_size: usize,
    pub widened_slots: usize,
    #[serde(default)]
    pub banks: BTreeMap<String, BTreeMap<String, Vec<WireSlotEntryJson>>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub permanent_static: BTreeMap<String, Vec<WireSlotEntryJson>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mode_grammars: BTreeMap<String, GrammarCartridgeSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bisimulation_receipt: Option<WireBisimulationReceipt>,
}

impl WireCartridgeManifestJson {
    /// Resolve canonical model family while rejecting conflicting metadata.
    pub fn model_family(&self) -> Result<Option<&str>, V2CompilerError> {
        let mut family = None;
        for id in [
            self.header.as_ref().map(|h| h.model_family.as_str()),
            self.model_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if id.trim().is_empty() || family.is_some_and(|prev| prev != id) {
                return Err(V2CompilerError::WireIntegrity(
                    "conflicting or empty cartridge model identities".to_string(),
                ));
            }
            family = Some(id);
        }
        Ok(family)
    }

    pub fn to_json_compact(&self) -> Result<String, V2CompilerError> {
        serde_json::to_string(self).map_err(|e| V2CompilerError::WireIntegrity(e.to_string()))
    }

    pub fn to_json_pretty(&self) -> Result<String, V2CompilerError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| V2CompilerError::WireIntegrity(e.to_string()))
    }

    pub fn from_json_str(raw: &str) -> Result<Self, V2CompilerError> {
        let parsed: Self =
            serde_json::from_str(raw).map_err(|e| V2CompilerError::WireIntegrity(e.to_string()))?;
        parsed.verify_schema_integrity()?;
        Ok(parsed)
    }

    /// Verify bounds, non-overlapping slot intervals, and consistent hidden dimension across all banks.
    pub fn verify_schema_integrity(&self) -> Result<(), V2CompilerError> {
        self.model_family()?;
        if self.widened_slots == 0 || self.base_intermediate_size == 0 {
            return Err(V2CompilerError::WireIntegrity(
                "widened_slots and base_intermediate_size must be > 0".to_string(),
            ));
        }
        let base_m = self.base_intermediate_size;
        let max_slot = base_m + self.widened_slots;
        let mut inferred_d: Option<usize> = None;
        let mut static_by_layer: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();

        for (l_key, entries) in &self.permanent_static {
            if l_key.parse::<usize>().is_err() {
                return Err(V2CompilerError::WireIntegrity(format!(
                    "non-numeric layer key {l_key:?}"
                )));
            }
            let used = static_by_layer.entry(l_key.clone()).or_default();
            for e in entries {
                if e.slot_lo < base_m || e.slot_hi > max_slot || e.slot_lo >= e.slot_hi {
                    return Err(V2CompilerError::WireIntegrity(format!(
                        "permanent_static slot [{}, {}) out of [{base_m}, {max_slot})",
                        e.slot_lo, e.slot_hi
                    )));
                }
                let d = *inferred_d.get_or_insert(e.w_down.len());
                e.verify_dimensions(d)?;
                for s in e.slot_lo..e.slot_hi {
                    if !used.insert(s) {
                        return Err(V2CompilerError::WireIntegrity(format!(
                            "overlapping permanent_static slot {s} on layer {l_key}"
                        )));
                    }
                }
            }
        }

        for (bank_key, layers) in &self.banks {
            let _ = parse_mount_policy_and_pen(bank_key)?;
            for (l_key, entries) in layers {
                if l_key.parse::<usize>().is_err() {
                    return Err(V2CompilerError::WireIntegrity(format!(
                        "non-numeric layer key {l_key:?}"
                    )));
                }
                let mut used = BTreeSet::new();
                let st_slots = static_by_layer.get(l_key);
                for e in entries {
                    if e.slot_lo < base_m || e.slot_hi > max_slot || e.slot_lo >= e.slot_hi {
                        return Err(V2CompilerError::WireIntegrity(format!(
                            "bank {bank_key} slot [{}, {}) out of [{base_m}, {max_slot})",
                            e.slot_lo, e.slot_hi
                        )));
                    }
                    let d = *inferred_d.get_or_insert(e.w_down.len());
                    e.verify_dimensions(d)?;
                    for s in e.slot_lo..e.slot_hi {
                        if !used.insert(s) || st_slots.is_some_and(|st| st.contains(&s)) {
                            return Err(V2CompilerError::WireIntegrity(format!(
                                "slot {s} collision in bank {bank_key} layer {l_key}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
