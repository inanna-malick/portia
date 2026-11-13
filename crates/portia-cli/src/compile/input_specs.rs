//! Pure-`f64` JSON capture, obligation, and trunk weight-geometry input DTOs shared across
//! `compile`, `compile_v2`, `worker`, and `nursery_cmd::oracle_capture` (`compile::input_specs`).

use anyhow::{bail, Result};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, l2_norm, normalize_unit_vec, StateMatrix};
use portia_core::{BaseIntermediateSize, Mode, PhysicalSlotIndex};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Default base intermediate MLP width $M$ for `Qwen/Qwen3.5-0.8B-Base` (`3584`).
pub const DEFAULT_QWEN35_08B_BASE_INTERMEDIATE_SIZE: usize = 3584;

/// Default residual-to-post-RMSNorm radius scale when raw pre-norm residuals are omitted (`7.25`).
pub const DEFAULT_RESIDUAL_RADIUS_SCALE: f64 = 7.25;

/// JSON-RPC capture response mapping stringified layer indices (e.g. `"22"`) to `d`-dimensional
/// pre-MLP activation rows and optional pre-RMSNorm residual rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorPipeCaptureResponse {
    pub status: String,
    pub activations: BTreeMap<String, Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residuals: Option<BTreeMap<String, Vec<Vec<f64>>>>,
}

impl TensorPipeCaptureResponse {
    #[must_use]
    pub fn new(activations: BTreeMap<String, Vec<Vec<f64>>>) -> Self {
        Self {
            status: "activations".to_string(),
            activations,
            residuals: None,
        }
    }

    #[must_use]
    pub fn with_residuals(mut self, residuals: BTreeMap<String, Vec<Vec<f64>>>) -> Self {
        self.residuals = Some(residuals);
        self
    }

    /// Infer hidden dimension $d$ from the first non-empty layer activation table.
    pub fn infer_hidden_dim(&self) -> Result<usize> {
        for rows in self.activations.values() {
            if let Some(first_row) = rows.first() {
                if first_row.is_empty() {
                    bail!("empty activation row (ncols = 0) in TensorPipeCaptureResponse");
                }
                let d = first_row.len();
                for r in rows {
                    if r.len() != d {
                        bail!(
                            "activation row hidden_dim mismatch: expected {d}, got {}",
                            r.len()
                        );
                    }
                }
                return Ok(d);
            }
        }
        bail!("TensorPipeCaptureResponse contains no non-empty activation rows")
    }

    #[must_use]
    pub fn has_observed_residuals(&self, layer: usize) -> bool {
        self.residuals
            .as_ref()
            .and_then(|m| m.get(&layer.to_string()))
            .is_some_and(|rows| !rows.is_empty())
    }

    pub fn extract_pre_mlp_rows(&self, layer: usize, indices: &[usize]) -> Result<Vec<Vec<f64>>> {
        let key = layer.to_string();
        let rows = self
            .activations
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("layer {layer} not found in activations"))?;
        if indices.is_empty() {
            bail!("cannot extract empty prompt_indices slice on layer {layer}");
        }
        indices
            .iter()
            .map(|&idx| {
                rows.get(idx).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "row index {idx} out of bounds (len={}) on layer {layer}",
                        rows.len()
                    )
                })
            })
            .collect()
    }

    pub fn extract_residual_rows(&self, layer: usize, indices: &[usize]) -> Result<Vec<Vec<f64>>> {
        let key = layer.to_string();
        if let Some(res_rows) = self.residuals.as_ref().and_then(|m| m.get(&key)) {
            if indices.is_empty() {
                bail!("cannot extract empty residual prompt_indices slice on layer {layer}");
            }
            return indices
                .iter()
                .map(|&idx| {
                    res_rows.get(idx).cloned().ok_or_else(|| {
                        anyhow::anyhow!(
                            "residual row index {idx} out of bounds (len={}) on layer {layer}",
                            res_rows.len()
                        )
                    })
                })
                .collect();
        }
        let pre = self.extract_pre_mlp_rows(layer, indices)?;
        Ok(pre
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| v * DEFAULT_RESIDUAL_RADIUS_SCALE)
                    .collect()
            })
            .collect())
    }
}

/// Natural vocabulary BPE token contrast specification (model vocabulary IDs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenReadoutContrastSpec {
    pub promote_token_ids: Vec<u32>,
    pub suppress_token_ids: Vec<u32>,
    pub suppress_weight: f64,
}

/// Slot-range compilation & verification obligation over captured activations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipeSlotObligationSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readout_contrast: Option<TokenReadoutContrastSpec>,
    pub obligation_id: String,
    pub bank_name: String,
    pub layer: usize,
    pub slot_lo: usize,
    pub slot_hi: usize,
    pub positive_prompt_indices: Vec<usize>,
    pub negative_prompt_indices: Vec<usize>,
    #[serde(default)]
    pub background_prompt_indices: Vec<usize>,
    pub v_target: Vec<f64>,
    pub theta_rad: f64,
    pub gate_target: f64,
    pub gate_neg_target: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_mode: Option<Mode>,
}

impl PipeSlotObligationSpec {
    #[inline]
    #[must_use]
    pub const fn slot_span(&self) -> usize {
        self.slot_hi.saturating_sub(self.slot_lo)
    }

    /// Resolve physical `[abs_lo .. abs_hi)` slot indices within `[base .. base + widened_slots)`.
    pub fn resolve_physical_slot_bounds(
        &self,
        base_intermediate_size: usize,
        widened_slots: usize,
    ) -> Result<(usize, usize)> {
        let span = self.slot_span();
        if self.slot_hi <= self.slot_lo || (span != 1 && span != 2) {
            bail!(
                "obligation {}: slot span must be 1 or 2, got [{}..{})",
                self.obligation_id,
                self.slot_lo,
                self.slot_hi
            );
        }
        let base = BaseIntermediateSize::new(base_intermediate_size);
        let (rel_lo, abs_lo) = PhysicalSlotIndex::resolve_flexible_boundary(self.slot_lo, base);
        let rel_hi = rel_lo.get() + span;
        if rel_hi > widened_slots {
            bail!(
                "obligation {}: relative slots [{}..{rel_hi}) exceed widened capacity {widened_slots}",
                self.obligation_id,
                rel_lo.get()
            );
        }
        Ok((abs_lo.get(), abs_lo.get() + span))
    }
}

/// Pre-attention QK routing direction vectors (`SET` and `WHERE`) for an upstream MLP layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct QkLayerVectorsSpec {
    #[serde(default, alias = "set_vector", alias = "v_dir_set", alias = "set")]
    pub v_set: Vec<f64>,
    #[serde(
        default,
        alias = "where_vector",
        alias = "v_dir_where",
        alias = "where"
    )]
    pub v_where: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_pos_gate: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_neg_gate: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_qk_set: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_qk_where: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_q: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_k: Option<Vec<Vec<f64>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "input_layernorm_weight"
    )]
    pub ln_gamma: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_set: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_where: Option<Vec<f64>>,
}

impl QkLayerVectorsSpec {
    pub fn resolve_directions(&self, d: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        if self.v_set.len() == d && self.v_where.len() == d {
            return Ok((self.v_set.clone(), self.v_where.clone()));
        }
        if let (Some(pg), Some(ng), Some(qs), Some(qw)) = (
            &self.v_pos_gate,
            &self.v_neg_gate,
            &self.v_qk_set,
            &self.v_qk_where,
        ) {
            if pg.len() == d && ng.len() == d && qs.len() == d && qw.len() == d {
                let diff: Vec<f64> = pg.iter().zip(ng).map(|(a, b)| a - b).collect();
                let gc_n = l2_norm(&diff).max(1e-12);
                let qs_n = l2_norm(qs).max(1e-12);
                let qw_n = l2_norm(qw).max(1e-12);
                let vs_raw: Vec<f64> = (0..d)
                    .map(|i| 0.55 * (diff[i] / gc_n) + 0.45 * (qs[i] / qs_n))
                    .collect();
                let vw_raw: Vec<f64> = (0..d)
                    .map(|i| 0.55 * (diff[i] / gc_n) + 0.45 * (qw[i] / qw_n))
                    .collect();
                let vs_n = l2_norm(&vs_raw).max(1e-12);
                let vw_n = l2_norm(&vw_raw).max(1e-12);
                return Ok((
                    vs_raw.into_iter().map(|x| x / vs_n).collect(),
                    vw_raw.into_iter().map(|x| x / vw_n).collect(),
                ));
            }
        }
        if let (Some(wq), Some(wk), Some(gamma), Some(ks), Some(kw)) = (
            &self.w_q,
            &self.w_k,
            &self.ln_gamma,
            &self.k_set,
            &self.k_where,
        ) {
            if !wq.is_empty() && !wk.is_empty() && gamma.len() == d {
                let proj = |vec_h: &[f64]| -> Vec<f64> {
                    let mut raw = vec![0.0; d];
                    if vec_h.len() == wk.len() {
                        for (r, row) in wk.iter().enumerate() {
                            for (c, &w) in row.iter().take(d).enumerate() {
                                raw[c] += w * vec_h[r];
                            }
                        }
                    }
                    if vec_h.len() == wq.len() {
                        for (r, row) in wq.iter().enumerate() {
                            for (c, &w) in row.iter().take(d).enumerate() {
                                raw[c] += w * vec_h[r];
                            }
                        }
                    }
                    for (c, val) in raw.iter_mut().enumerate() {
                        *val *= 1.0 + gamma[c];
                    }
                    let nrm = l2_norm(&raw).max(1e-12);
                    raw.into_iter().map(|x| x / nrm).collect()
                };
                return Ok((proj(ks), proj(kw)));
            }
        }
        bail!("QkLayerVectorsSpec missing valid direction vectors of length {d}")
    }

    pub fn resolve_combined_direction(&self, pre_mlp_layer: usize, d: usize) -> Result<Vec<f64>> {
        let (v_set, v_where) = self.resolve_directions(d)?;
        let sum: Vec<f64> = v_set.iter().zip(&v_where).map(|(a, b)| a + b).collect();
        normalize_unit_vec(&sum)
            .or_else(|| normalize_unit_vec(&v_set))
            .ok_or_else(|| anyhow::anyhow!("degenerate QK direction at L{pre_mlp_layer}"))
    }
}

/// Raw DeltaNet decay/update gate projection weights (`in_proj_a`, `in_proj_b`, `input_layernorm`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DeltaNetLayerWeightSpec {
    #[serde(default, alias = "w_a", alias = "in_proj_a_weight")]
    pub in_proj_a: Vec<Vec<f64>>,
    #[serde(default, alias = "w_b", alias = "in_proj_b_weight")]
    pub in_proj_b: Vec<Vec<f64>>,
    #[serde(
        default,
        alias = "ln_gamma",
        alias = "norm_weight",
        alias = "input_layernorm"
    )]
    pub input_layernorm_weight: Vec<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "joint_flush_and_latch_vector"
    )]
    pub v_mmu: Option<Vec<f64>>,
}

/// Optional background activation states (`ByLayer` map or `Flat` batch).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BackgroundActivationsSpec {
    ByLayer(BTreeMap<String, Vec<Vec<f64>>>),
    Flat(Vec<Vec<f64>>),
}

impl BackgroundActivationsSpec {
    #[must_use]
    pub fn rows_for_layer(&self, layer: usize) -> Option<&[Vec<f64>]> {
        match self {
            Self::ByLayer(map) => map.get(&layer.to_string()).map(Vec::as_slice),
            Self::Flat(rows) => Some(rows.as_slice()),
        }
    }
}

fn default_top_k_anti_echo() -> usize {
    16
}

/// Compact downstream (`L_total - 1`) MLP anti-echo slice extracted from the loaded checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownstreamMlpAntiEchoSpec {
    pub w_gate: Vec<Vec<f64>>,
    pub w_up: Vec<Vec<f64>>,
    pub w_down: Vec<Vec<f64>>,
    pub post_attn_norm_weight: Vec<f64>,
    #[serde(default = "default_top_k_anti_echo")]
    pub top_k_anti_echo_neurons: usize,
}

impl DownstreamMlpAntiEchoSpec {
    pub fn build_anti_echo_subspace(
        &self,
        d: usize,
        u_promote_mean: &[f64],
        u_suppress_mean: &[f64],
    ) -> Result<Option<OrthonormalSubspace>> {
        let m = self.w_gate.len();
        if m == 0 || self.w_up.len() != m || self.w_down.len() != d {
            bail!(
                "DownstreamMlpAntiEchoSpec shape mismatch: gate_rows={m}, up_rows={}, down_rows={}, d={d}",
                self.w_up.len(),
                self.w_down.len()
            );
        }
        if self.post_attn_norm_weight.len() != d {
            bail!("DownstreamMlpAntiEchoSpec post_attn_norm_weight length mismatch");
        }
        let oppose: Vec<f64> = u_suppress_mean
            .iter()
            .zip(u_promote_mean)
            .map(|(s, p)| s - p)
            .collect();
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for j in 0..m {
            let mut s = 0.0;
            for (r, (&opp, row)) in oppose.iter().zip(&self.w_down).enumerate() {
                if row.len() != m {
                    bail!("DownstreamMlpAntiEchoSpec w_down row {r} len != {m}");
                }
                s += opp * row[j];
            }
            if s > 0.0 && s.is_finite() {
                scored.push((j, s));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.top_k_anti_echo_neurons.max(1));
        if scored.is_empty() {
            return Ok(None);
        }

        let mut rows = Vec::with_capacity(2 * scored.len());
        for &(j, _) in &scored {
            if self.w_gate[j].len() != d || self.w_up[j].len() != d {
                bail!("DownstreamMlpAntiEchoSpec gate/up row {j} len != {d}");
            }
            let g_scaled: Vec<f64> = (0..d)
                .map(|c| self.w_gate[j][c] * (1.0 + self.post_attn_norm_weight[c]))
                .collect();
            let u_scaled: Vec<f64> = (0..d)
                .map(|c| self.w_up[j][c] * (1.0 + self.post_attn_norm_weight[c]))
                .collect();
            rows.push(g_scaled);
            rows.push(u_scaled);
        }
        let mat = StateMatrix::from_rows("anti_echo_rows", &rows)?;
        Ok(Some(OrthonormalSubspace::from_row_space_svd(
            "anti_echo_basis",
            &mat,
            1e-9,
            d.saturating_sub(1).max(1),
        )?))
    }
}

fn default_mean_natural_token_norm() -> f64 {
    1.0
}

/// Complete weight-space geometry payload extracted from a loaded Qwen3.5 checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilerWeightGeometrySpec {
    #[serde(default)]
    pub vocab_top_k_eigvecs: Vec<Vec<f64>>,
    #[serde(default)]
    pub protected_sql_embeddings: Vec<Vec<f64>>,
    #[serde(alias = "final_norm_weight", alias = "model_norm_weight")]
    pub norm_weight: Vec<f64>,
    #[serde(default = "default_mean_natural_token_norm")]
    pub mean_natural_token_norm: f64,
    #[serde(default)]
    pub qk_layer_vectors: BTreeMap<String, QkLayerVectorsSpec>,
    #[serde(default)]
    pub deltanet_layer_weights: BTreeMap<String, DeltaNetLayerWeightSpec>,
    #[serde(default)]
    pub named_token_unembeddings: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    pub downstream_mlp_anti_echo: Option<DownstreamMlpAntiEchoSpec>,
}

impl CompilerWeightGeometrySpec {
    fn compile_contrast_from_rows(
        &self,
        rows: &[Vec<f64>],
        promote_indices: &[usize],
        suppress_indices: &[usize],
        suppress_weight: f64,
    ) -> Result<(Vec<f64>, Option<OrthonormalSubspace>)> {
        if promote_indices.is_empty() {
            bail!("token contrast requires at least 1 promoted row");
        }
        if !suppress_weight.is_finite() || suppress_weight < 0.0 {
            bail!("suppress_weight ({suppress_weight}) must be finite and >= 0.0");
        }
        let d = self.norm_weight.len();
        if d == 0 {
            bail!("CompilerWeightGeometrySpec norm_weight is empty");
        }
        let eff_row = |idx: usize| -> Vec<f64> {
            rows[idx]
                .iter()
                .zip(&self.norm_weight)
                .map(|(&w, &nw)| w * (1.0 + nw))
                .collect()
        };

        let mut promo_dedup = Vec::new();
        let mut u_promo_mean = vec![0.0; d];
        for &idx in promote_indices {
            if !promo_dedup.contains(&idx) {
                let u = eff_row(idx);
                for c in 0..d {
                    u_promo_mean[c] += u[c];
                }
                promo_dedup.push(idx);
            }
        }
        let inv_p = 1.0 / (promo_dedup.len() as f64);
        for val in &mut u_promo_mean {
            *val *= inv_p;
        }

        let mut sup_dedup = Vec::new();
        let mut u_sup_mean = vec![0.0; d];
        for &idx in suppress_indices {
            if !promo_dedup.contains(&idx) && !sup_dedup.contains(&idx) {
                let u = eff_row(idx);
                for c in 0..d {
                    u_sup_mean[c] += u[c];
                }
                sup_dedup.push(idx);
            }
        }

        let (mut v_dir, q_anti) = if sup_dedup.is_empty() {
            (u_promo_mean, None)
        } else {
            let inv_s = 1.0 / (sup_dedup.len() as f64);
            for val in &mut u_sup_mean {
                *val *= inv_s;
            }
            let v_raw: Vec<f64> = (0..d)
                .map(|c| u_promo_mean[c] - suppress_weight * u_sup_mean[c])
                .collect();
            let q = match &self.downstream_mlp_anti_echo {
                Some(ds) => ds.build_anti_echo_subspace(d, &u_promo_mean, &u_sup_mean)?,
                None => None,
            };
            let proj = match &q {
                Some(sub) => sub.project_out(&v_raw),
                None => v_raw,
            };
            (proj, q)
        };

        v_dir = normalize_unit_vec(&v_dir)
            .ok_or_else(|| anyhow::anyhow!("degenerate readout contrast vector"))?;
        Ok((v_dir, q_anti))
    }

    /// Derive a unit-normalized readout vector for an explicit [`TokenReadoutContrastSpec`].
    pub fn derive_safe_token_readout_vector(
        &self,
        contrast: &TokenReadoutContrastSpec,
    ) -> Result<Vec<f64>> {
        if contrast.promote_token_ids.is_empty() {
            bail!("token contrast needs a promoted token");
        }
        let d = self.norm_weight.len();
        let token_ids: Vec<u32> = contrast
            .promote_token_ids
            .iter()
            .chain(&contrast.suppress_token_ids)
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut rows = Vec::with_capacity(token_ids.len());
        for tid in &token_ids {
            let row = self
                .named_token_unembeddings
                .get(&tid.to_string())
                .ok_or_else(|| {
                    anyhow::anyhow!("token contrast missing unembedding for token {tid}")
                })?;
            if row.len() != d {
                bail!("token {tid} unembedding dimension mismatch");
            }
            rows.push(row.clone());
        }
        let remap = |ids: &[u32]| -> Vec<usize> {
            ids.iter()
                .map(|tid| token_ids.binary_search(tid).expect("collected token"))
                .collect()
        };
        let (unit_v, _) = self.compile_contrast_from_rows(
            &rows,
            &remap(&contrast.promote_token_ids),
            &remap(&contrast.suppress_token_ids),
            contrast.suppress_weight,
        )?;
        Ok(unit_v)
    }

    /// Derive unit-normalized natural BPE mode readout directions orthogonal to `protected_sql_embeddings`.
    pub fn derive_mode_readout_targets(&self) -> Result<BTreeMap<Mode, Vec<f64>>> {
        let d = self.norm_weight.len();
        if d == 0 {
            bail!("CompilerWeightGeometrySpec norm_weight is empty");
        }
        let sql_basis = if self.protected_sql_embeddings.is_empty() {
            None
        } else {
            let mat = StateMatrix::from_rows("sql_embeddings", &self.protected_sql_embeddings)?;
            if mat.hidden_dim() != d {
                bail!("protected_sql_embeddings hidden_dim != {d}");
            }
            Some(OrthonormalSubspace::from_row_space_svd(
                "sql_basis",
                &mat,
                1e-9,
                d.saturating_sub(1).max(1),
            )?)
        };

        let mut ordered_modes: Vec<(Mode, Vec<f64>)> = Vec::new();
        for mode in Mode::ALL {
            let tid_key = mode.qwen35_first_token_id().to_string();
            let candidates = [
                format!("{mode:?}"),
                tid_key,
                mode.as_str().to_string(),
                format!(" {}", mode.as_str()),
            ];
            for key in candidates {
                if let Some(vec) = self.named_token_unembeddings.get(&key) {
                    if vec.len() == d {
                        ordered_modes.push((mode, vec.clone()));
                        break;
                    }
                }
            }
        }
        if ordered_modes.is_empty() {
            return Ok(BTreeMap::new());
        }

        let rows: Vec<Vec<f64>> = ordered_modes.iter().map(|(_, v)| v.clone()).collect();
        let n_m = rows.len();
        let mut out = BTreeMap::new();
        for (idx, &(mode, _)) in ordered_modes.iter().enumerate() {
            let others: Vec<usize> = (0..n_m).filter(|&j| j != idx).collect();
            let (mut unit_v, q_anti) =
                self.compile_contrast_from_rows(&rows, &[idx], &others, 1.0)?;
            if let Some(ref q_sql) = sql_basis {
                let mut proj = q_sql.project_out(&unit_v);
                if let Some(ref qa) = q_anti {
                    proj = qa.project_out(&proj);
                }
                unit_v = normalize_unit_vec(&proj).ok_or_else(|| {
                    anyhow::anyhow!("degenerate mode readout target for {mode:?}")
                })?;
                debug_assert!(self
                    .protected_sql_embeddings
                    .iter()
                    .all(|r| dot_f64(&unit_v, r).abs() < 1e-8));
            }
            out.insert(mode, unit_v);
        }
        Ok(out)
    }
}
