//! Multi-scale `Qwen3.5` architecture & compiled `WireCartridgeManifestJson` slot geometry
//! resolution (`0.8B`, `2B`, `4B`, `9B`, `27B`) for the SQL-OODA TUI.

use crate::model_identity::normalize_model_identifier;
use portia_compiler::wire::{WireCartridgeHeader, TOTAL_RESERVED_SLOTS};
use portia_core::{CartridgeBank, Mode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Resolved physical transformer & widened SwiGLU rack geometry displayed in TUI chrome and
/// `SidebarMode::RawTokenTrace`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiModelScaleSpec {
    /// Canonical HuggingFace family or raw model identifier (e.g. `"Qwen/Qwen3.5-2B-Base"`).
    pub canonical_family: String,
    /// Compact header badge (e.g. `"0.8B-Base"`, `"2B-Base"`, `"4B-Base"`, `"9B-Base"`).
    pub scale_badge: String,
    /// Total transformer blocks (`24` on `0.8B`/`2B`, `32` on `4B`/`9B`, `64` on `27B`).
    pub num_layers: usize,
    /// Residual stream width `d` (`1024`, `2048`, `2560`, `4096`, `5120`).
    pub hidden_dim: usize,
    /// Native SwiGLU intermediate width `M` (`3584`, `6144`, `9216`, `12288`, `17408`).
    pub base_intermediate_size: usize,
    /// Widened dynamic cartridge slot bank capacity (`128` by default, or `0` on `--cartridge none`).
    pub widened_slots: usize,
    /// Hybrid Gated-DeltaNet linear-attention block count (`3/4` of layers on `Qwen3.5`).
    pub deltanet_layers: usize,
    /// Full Grouped-Query Attention (`GQA`) block count (`1/4` of layers on `Qwen3.5`).
    pub gqa_layers: usize,
    /// Primary commitment-band router layer (`num_layers - 2`: `L22` on `0.8B`/`2B`, `L30` on `4B`/`9B`).
    pub router_layer: usize,
    /// Physical `[slot_lo, slot_hi)` range for `OrientTrunk` at `router_layer`.
    pub orient_slot_range: (usize, usize),
    /// Per-layer `(layer_idx, slot_lo, slot_hi)` envelope across `ModePayload:*` banks.
    pub payload_layer_ranges: Vec<(usize, usize, usize)>,
    /// Whether slot bounds were inspected directly from an on-disk `WireCartridgeManifestJson`.
    pub inspected_from_cartridge: bool,
}

#[derive(Deserialize)]
struct SlotSpanProbe {
    slot_lo: usize,
    slot_hi: usize,
    #[serde(default)]
    #[allow(dead_code)]
    w_gate: serde::de::IgnoredAny,
    #[serde(default)]
    #[allow(dead_code)]
    w_up: serde::de::IgnoredAny,
    #[serde(default)]
    #[allow(dead_code)]
    w_down: serde::de::IgnoredAny,
}

#[derive(Deserialize)]
struct CartridgeTopologyProbe {
    #[serde(default)]
    header: Option<WireCartridgeHeader>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    base_intermediate_size: Option<usize>,
    #[serde(default)]
    widened_slots: Option<usize>,
    #[serde(default)]
    banks: BTreeMap<String, BTreeMap<String, Vec<SlotSpanProbe>>>,
}

impl CartridgeTopologyProbe {
    fn model_family(&self) -> Option<&str> {
        self.header
            .as_ref()
            .map(|h| h.model_family.as_str())
            .or(self.model_id.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
}

fn compact_badge_from_family(normalized: &str) -> String {
    if let Some(rest) = normalized.strip_prefix("Qwen/Qwen3.5-") {
        if !rest.trim().is_empty() {
            return rest.to_string();
        }
    }
    let base = Path::new(normalized)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(normalized);
    base.strip_prefix("Qwen3.5-")
        .unwrap_or(base)
        .chars()
        .take(20)
        .collect()
}

fn canonical_architecture_tuple(normalized: &str) -> (usize, usize, usize, usize, usize, usize) {
    let lower = normalized.to_ascii_lowercase();
    if lower.contains("27b") {
        (64, 5120, 17408, 48, 16, 62)
    } else if lower.contains("9b") {
        (32, 4096, 12288, 24, 8, 30)
    } else if lower.contains("4b") {
        (32, 2560, 9216, 24, 8, 30)
    } else if lower.contains("2b") {
        (24, 2048, 6144, 18, 6, 22)
    } else {
        (24, 1024, 3584, 18, 6, 22)
    }
}

fn candidate_cartridge_paths(cartridge_label: &str) -> Vec<PathBuf> {
    let raw = cartridge_label
        .trim()
        .trim_end_matches(" (default)")
        .trim_end_matches(" (env)")
        .trim();
    if raw.is_empty()
        || raw.eq_ignore_ascii_case("none")
        || raw.starts_with("none ")
        || raw.starts_with("unconfigured")
        || !raw.ends_with(".json")
    {
        return Vec::new();
    }

    let direct = PathBuf::from(raw);
    if direct.is_file() {
        return vec![direct];
    }

    let mut out = Vec::new();
    if let Ok(env_home) = std::env::var("PORTIA_HOME") {
        if !env_home.trim().is_empty() {
            out.push(PathBuf::from(env_home.trim()).join("artifacts").join(raw));
        }
    }
    out.push(PathBuf::from("artifacts").join(raw));
    if let Some(ws_root) = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
    {
        out.push(ws_root.join("artifacts").join(raw));
    }
    out.into_iter().filter(|p| p.is_file()).collect()
}

impl TuiModelScaleSpec {
    /// Canonical `Qwen/Qwen3.5-0.8B-Base` scale specification (`24L`, `d=1024`, `M=3584`, `L22`).
    #[must_use]
    pub fn default_08b_base() -> Self {
        Self::from_model_name("Qwen/Qwen3.5-0.8B-Base")
    }

    /// Derive canonical scale geometry purely from a model identifier (`0.8b-base`, `2b-base`,
    /// `4b-base`, `9b-base`, `27b`, or HuggingFace repo / snapshot path).
    #[must_use]
    pub fn from_model_name(raw_model: &str) -> Self {
        let trimmed = raw_model.trim();
        let effective = if trimmed.is_empty() {
            "Qwen/Qwen3.5-0.8B-Base"
        } else {
            trimmed
        };
        let normalized = normalize_model_identifier(effective);
        let scale_badge = compact_badge_from_family(&normalized);
        let (num_layers, hidden_dim, base_inter, dn_layers, gqa_layers, router_layer) =
            canonical_architecture_tuple(&normalized);
        let orient_slot_range = (base_inter, base_inter + 30);
        let payload_layer_ranges = vec![(router_layer, base_inter + 32, base_inter + 70)];

        Self {
            canonical_family: normalized,
            scale_badge,
            num_layers,
            hidden_dim,
            base_intermediate_size: base_inter,
            widened_slots: TOTAL_RESERVED_SLOTS,
            deltanet_layers: dn_layers,
            gqa_layers,
            router_layer,
            orient_slot_range,
            payload_layer_ranges,
            inspected_from_cartridge: false,
        }
    }

    /// Resolve scale geometry from `(model_name, cartridge_label)`, overlaying exact physical
    /// `base_intermediate_size`, `widened_slots`, `OrientTrunk`, and `ModePayload` layer/slot
    /// ranges when a matching `WireCartridgeManifestJson` is available on disk.
    #[must_use]
    pub fn resolve(model_name: &str, cartridge_label: &str) -> Self {
        let mut spec = Self::from_model_name(model_name);
        let trimmed_cart = cartridge_label.trim();
        if trimmed_cart.eq_ignore_ascii_case("none") || trimmed_cart.starts_with("none ") {
            spec.widened_slots = 0;
            return spec;
        }

        for cand in candidate_cartridge_paths(trimmed_cart) {
            let Ok(bytes) = std::fs::read(&cand) else {
                continue;
            };
            let Ok(probe) = serde_json::from_slice::<CartridgeTopologyProbe>(&bytes) else {
                continue;
            };
            if let Some(cart_fam) = probe.model_family() {
                let norm_cart_fam = normalize_model_identifier(cart_fam);
                if !model_name.trim().is_empty() && norm_cart_fam != spec.canonical_family {
                    continue;
                }
                if model_name.trim().is_empty() {
                    spec = Self::from_model_name(&norm_cart_fam);
                }
            }
            spec.apply_topology_probe(&probe);
            break;
        }
        spec
    }

    fn apply_topology_probe(&mut self, probe: &CartridgeTopologyProbe) {
        if let Some(base_m) = probe.base_intermediate_size.filter(|&m| m > 0) {
            self.base_intermediate_size = base_m;
        }
        if let Some(widened) = probe.widened_slots {
            self.widened_slots = widened;
        }

        if let Some(orient_layers) = probe.banks.get("OrientTrunk") {
            let mut best: Option<(usize, usize, usize)> = None;
            for (l_str, entries) in orient_layers {
                let Ok(layer_idx) = l_str.parse::<usize>() else {
                    continue;
                };
                let min_lo = entries.iter().map(|e| e.slot_lo).min();
                let max_hi = entries.iter().map(|e| e.slot_hi).max();
                if let (Some(lo), Some(hi)) = (min_lo, max_hi) {
                    if best.is_none_or(|(prev_l, _, _)| layer_idx >= prev_l) {
                        best = Some((layer_idx, lo, hi));
                    }
                }
            }
            if let Some((layer_idx, lo, hi)) = best {
                self.router_layer = layer_idx;
                self.orient_slot_range = (lo, hi);
            }
        }

        let mut payload_by_layer: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        for (bank_key, layer_map) in &probe.banks {
            if !bank_key.starts_with("ModePayload:") && !bank_key.starts_with("ModePlan:") {
                continue;
            }
            for (l_str, entries) in layer_map {
                let Ok(layer_idx) = l_str.parse::<usize>() else {
                    continue;
                };
                for entry in entries {
                    payload_by_layer
                        .entry(layer_idx)
                        .and_modify(|(lo, hi)| {
                            *lo = (*lo).min(entry.slot_lo);
                            *hi = (*hi).max(entry.slot_hi);
                        })
                        .or_insert((entry.slot_lo, entry.slot_hi));
                }
            }
        }
        if !payload_by_layer.is_empty() {
            self.payload_layer_ranges = payload_by_layer
                .into_iter()
                .map(|(l, (lo, hi))| (l, lo, hi))
                .collect();
        } else if self.orient_slot_range.0 >= self.base_intermediate_size {
            self.payload_layer_ranges = vec![(
                self.router_layer,
                self.base_intermediate_size + 32,
                self.base_intermediate_size + 70,
            )];
        }
        self.inspected_from_cartridge = true;
    }

    /// Total widened MLP intermediate dimension (`M + widened_slots`).
    #[inline]
    #[must_use]
    pub const fn widened_intermediate_size(&self) -> usize {
        self.base_intermediate_size + self.widened_slots
    }

    /// Format `OrientTrunk` physical layer & slot range (e.g. `"L22[6144..6174)"` or `"L30[9216..9246)"`).
    #[must_use]
    pub fn format_orient_slots(&self) -> String {
        if self.widened_slots == 0 {
            return format!("L{}[unpatched W_0]", self.router_layer);
        }
        format!(
            "L{}[{}..{})",
            self.router_layer, self.orient_slot_range.0, self.orient_slot_range.1
        )
    }

    /// Format `ModePayload` physical layer & slot envelope across active layers.
    #[must_use]
    pub fn format_payload_slots(&self) -> String {
        if self.widened_slots == 0 {
            return format!("L{}[unpatched W_0]", self.router_layer);
        }
        if self.payload_layer_ranges.is_empty() {
            return format!(
                "L{}[{}..{})",
                self.router_layer,
                self.base_intermediate_size + 32,
                self.base_intermediate_size + 70
            );
        }
        self.payload_layer_ranges
            .iter()
            .map(|(layer, lo, hi)| format!("L{layer}[{lo}..{hi})"))
            .collect::<Vec<_>>()
            .join(" · ")
    }

    /// Compact one-line architecture summary (`"24L d=2048 M=6144->6272 (18 DeltaNet + 6 GQA)"`).
    #[must_use]
    pub fn format_architecture_summary(&self) -> String {
        if self.widened_slots == 0 {
            format!(
                "{}L d={} M={} (unpatched · {} DeltaNet + {} GQA)",
                self.num_layers,
                self.hidden_dim,
                self.base_intermediate_size,
                self.deltanet_layers,
                self.gqa_layers
            )
        } else {
            format!(
                "{}L d={} M={}->{} ({} DeltaNet + {} GQA)",
                self.num_layers,
                self.hidden_dim,
                self.base_intermediate_size,
                self.widened_intermediate_size(),
                self.deltanet_layers,
                self.gqa_layers
            )
        }
    }
}

/// Format a [`CartridgeBank`] label with scale-accurate physical slot coordinates.
#[must_use]
pub fn format_cartridge_bank_badge_for_scale(
    bank: CartridgeBank,
    scale: &TuiModelScaleSpec,
) -> String {
    match bank {
        CartridgeBank::Base => format!(
            "Base [Un-Steered W_0 · {} SwiGLU + {} DeltaNet + {} GQA · 0 Active Slots]",
            scale.base_intermediate_size, scale.deltanet_layers, scale.gqa_layers
        ),
        CartridgeBank::OrientTrunk => format!(
            "OrientTrunk [{} · Pointer-ISA Stiefel SO(d={}) Router]",
            scale.format_orient_slots(),
            scale.hidden_dim
        ),
        CartridgeBank::ModePayload(mode) => match mode {
            Mode::Probe | Mode::Read | Mode::Write => format!(
                "ModePayload({}) [{} · Pointer-ISA SQL Payload]",
                mode.as_str(),
                scale.format_payload_slots()
            ),
            Mode::Ask => format!(
                "ModePayload(Ask · Structured-GUI [{}]) | ModePayload(Ask · Chitchat -> Base W_0)",
                scale.format_payload_slots()
            ),
            Mode::Result | Mode::Abstain => format!(
                "ModePayload({}) [{} -> Base W_0 Detach]",
                mode.as_str(),
                scale.format_payload_slots()
            ),
        },
    }
}

/// Format a [`CartridgeBank`] label using canonical `0.8B-Base` coordinates (use
/// [`format_cartridge_bank_badge_for_scale`] when [`TuiModelScaleSpec`] is available).
#[must_use]
pub fn format_cartridge_bank_badge(bank: CartridgeBank) -> String {
    format_cartridge_bank_badge_for_scale(bank, &TuiModelScaleSpec::default_08b_base())
}
