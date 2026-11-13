//! Submerged GQA Pointer Bus (`ker(W_V) ∩ ker(W_U_top)`), Polymorphic Copy-Head Discovery,
//! & Paired Write-Time (`W_K`) / Read-Time (`W_Q`) Hardware `ASID` Symbol-Table Coloring (`gqa_bus`).
//!
//! Exploits `Qwen3.5` / `Qwen3.8`'s `4:1` GQA query-to-KV head ratio:
//! - Preserves value projections with zero training-token OV leakage,
//! - Projects both `W_K` write-tags and `W_Q` read-queries strictly into
//!   $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$ (`GCAD` zero KV-value cache pollution!),
//! - Operates strictly inside the position-invariant RoPE pass-through slice `[rotary_dim .. head_dim)`.

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, l2_norm, normalize_unit_vec, rms_norm_slice, StateMatrix};
use serde::{Deserialize, Serialize};

/// Logical memory-segment Address-Space Identifier (`ASID`) stamped into `W_K` at prefill/write time
/// and queried by `W_Q` at generation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MemorySegmentAsid {
    /// Authoritative external schema / AST / type declarations (`<schema>`, `CREATE TABLE`, trait sigs).
    AuthoritativeSchema = 0,
    /// User natural-language conversation (`<user>`).
    UserConversation = 1,
    /// Explicit verified decomposition plan (`Plan:`).
    VerifiedPlan = 2,
    /// Ephemeral internal `<think>` scratchpad (suppressed during `<sql>`/`<rust>` copy lookups!).
    EphemeralScratchpad = 3,
}

impl MemorySegmentAsid {
    pub const ALL: [Self; 4] = [
        Self::AuthoritativeSchema,
        Self::UserConversation,
        Self::VerifiedPlan,
        Self::EphemeralScratchpad,
    ];
}

/// Geometry of a single `Qwen3_5Attention` GQA sublayer ($\ell + 1$).
#[derive(Debug, Clone)]
pub struct GqaPointerLayer {
    pub layer_idx: usize,
    pub hidden_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub input_norm_scale: Vec<f64>,
    pub q_norm_scale: Vec<f64>,
    pub k_norm_scale: Vec<f64>,
    /// Interleaved `(2 * num_q_heads * head_dim, hidden_dim)` (`head_dim` query rows + `head_dim` gate rows per head).
    pub w_q_proj: StateMatrix,
    /// `(num_kv_heads * head_dim, hidden_dim)`.
    pub w_k_proj: StateMatrix,
    /// `(num_kv_heads * head_dim, hidden_dim)`.
    pub w_v_proj: StateMatrix,
}

impl GqaPointerLayer {
    #[inline]
    #[must_use]
    pub const fn kv_group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    #[inline]
    #[must_use]
    pub const fn passthrough_dim(&self) -> usize {
        self.head_dim.saturating_sub(self.rotary_dim)
    }

    /// Build the submerged forbidden basis $\operatorname{span}(\mathbf{W}_V \cup \mathbf{W}_{U,\text{top}})$
    /// so any vector projected out of this basis causes **zero KV-value cache pollution** and
    /// **zero immediate `stdout` vocabulary echo**.
    pub fn build_submerged_forbidden_basis(
        &self,
        w_u_top_basis: Option<&OrthonormalSubspace>,
        max_wv_rank: usize,
    ) -> Result<OrthonormalSubspace, V2CompilerError> {
        let d = self.hidden_dim;
        let n_v_rows = self.w_v_proj.nrows();
        let mut scaled_v = Vec::with_capacity(n_v_rows * d);
        for r in 0..n_v_rows {
            for (&w, &s) in self.w_v_proj.row(r).iter().zip(&self.input_norm_scale) {
                scaled_v.push(w * s);
            }
        }
        let v_mat = StateMatrix::try_new("gqa_wv_scaled", n_v_rows, d, scaled_v)?;
        let q_wv = OrthonormalSubspace::from_row_space_svd(
            "gqa_wv_basis",
            &v_mat,
            1e-4,
            max_wv_rank.min(d.saturating_sub(64)),
        )?;
        match w_u_top_basis {
            Some(q_wu) => OrthonormalSubspace::union_bases(
                "submerged_wv_wu_union",
                d,
                &[&q_wv, q_wu],
                (q_wv.rank() + q_wu.rank()).min(d.saturating_sub(32)),
            ),
            None => Ok(q_wv),
        }
    }

    /// Pull a RoPE-passthrough head vector `delta_pass` (`len == passthrough_dim()`) back through
    /// `head_idx`'s query rows $(\mathbf{W}_{Q,\text{query}}^{(h)})^\top$ and `input_norm_scale`.
    #[must_use]
    pub fn pullback_query_passthrough(&self, head_idx: usize, delta_pass: &[f64]) -> Vec<f64> {
        debug_assert_eq!(delta_pass.len(), self.passthrough_dim());
        let d = self.hidden_dim;
        let r_start = head_idx * 2 * self.head_dim + self.rotary_dim;
        let mut out = vec![0.0; d];
        for (p_idx, &dp) in delta_pass.iter().enumerate() {
            if dp.abs() <= 1e-15 {
                continue;
            }
            let weight = dp * self.q_norm_scale[self.rotary_dim + p_idx];
            let q_row = self.w_q_proj.row(r_start + p_idx);
            for j in 0..d {
                out[j] += weight * q_row[j] * self.input_norm_scale[j];
            }
        }
        out
    }

    /// Pull a RoPE-passthrough head vector `delta_pass` back through `kv_head_idx`'s key rows
    /// $(\mathbf{W}_K^{(g)})^\top$ and `input_norm_scale`.
    #[must_use]
    pub fn pullback_key_passthrough(&self, kv_head_idx: usize, delta_pass: &[f64]) -> Vec<f64> {
        debug_assert_eq!(delta_pass.len(), self.passthrough_dim());
        let d = self.hidden_dim;
        let r_start = kv_head_idx * self.head_dim + self.rotary_dim;
        let mut out = vec![0.0; d];
        for (p_idx, &dp) in delta_pass.iter().enumerate() {
            if dp.abs() <= 1e-15 {
                continue;
            }
            let weight = dp * self.k_norm_scale[self.rotary_dim + p_idx];
            let k_row = self.w_k_proj.row(r_start + p_idx);
            for j in 0..d {
                out[j] += weight * k_row[j] * self.input_norm_scale[j];
            }
        }
        out
    }

    /// Pull sigmoid head-gate opening (`+1`) vs closing (`-1`) back through `W_{Q,gate}` at `cursor_h`.
    #[must_use]
    pub fn pullback_head_gate(&self, head_idx: usize, cursor_h: &[f64]) -> Vec<f64> {
        let d = self.hidden_dim;
        let c_norm = rms_norm_slice(cursor_h, &self.input_norm_scale, 1e-6);
        let g_start = head_idx * 2 * self.head_dim + self.head_dim;
        let mut out = vec![0.0; d];
        for i in 0..self.head_dim {
            let g_row = self.w_q_proj.row(g_start + i);
            let raw = dot_f64(g_row, &c_norm);
            let sig = 1.0 / (1.0 + (-raw).exp());
            let dsig = sig * (1.0 - sig);
            for j in 0..d {
                out[j] += dsig * g_row[j] * self.input_norm_scale[j];
            }
        }
        out
    }
}

/// Paired Write-Side (`W_K`) and Read-Side (`W_Q`) `ASID` segment-coloring codebook submerged in
/// $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsidSegmentCodebook {
    pub gqa_layer: usize,
    /// Unit residual write-directions `[4][d_model]` stamped during prefill/generation of each segment.
    pub write_key_tags: [Vec<f64>; 4],
    /// Unit residual read-directions `[4][d_model]` selecting a single target `ASID` while suppressing others.
    pub read_query_selectors: [Vec<f64>; 4],
    pub max_wv_leakage: f64,
}

/// Synthesize a 4-segment `ASID` codebook (`Schema`, `Chat`, `Plan`, `Scratchpad`) across `copy_heads`
/// strictly inside `submerged_forbidden^⊥` (`0.000` `W_V` value corruption!).
pub fn synthesize_asid_codebook(
    gqa: &GqaPointerLayer,
    copy_heads: &[usize],
    submerged_forbidden: &OrthonormalSubspace,
) -> Result<AsidSegmentCodebook, V2CompilerError> {
    let pass_d = gqa.passthrough_dim();
    if pass_d < 8 || copy_heads.is_empty() {
        return Err(V2CompilerError::InvalidConfig {
            context: "synthesize_asid_codebook",
            reason: "requires passthrough_dim >= 8 and >= 1 copy head".to_string(),
        });
    }

    // Use 4 disjoint multi-coordinate Walsh-like blocks in the high-index end of `[rotary_dim..head_dim)`
    // so tags are orthogonal in head-space BEFORE pulling back into residual space:
    let block_w = (pass_d / 8).max(2);
    let base_idx = pass_d - 4 * block_w;
    let mut head_tags: Vec<Vec<f64>> = Vec::with_capacity(4);
    for seg in 0..4 {
        let mut t = vec![0.0; pass_d];
        let val = 1.0 / (block_w as f64).sqrt();
        for k in 0..block_w {
            t[base_idx + seg * block_w + k] = val;
        }
        head_tags.push(t);
    }

    let mut write_vecs: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut max_leak = 0.0_f64;

    for (seg, head_tag) in head_tags.iter().enumerate() {
        let mut raw_k = vec![0.0; gqa.hidden_dim];
        let mut seen_kv = Vec::new();
        for &h in copy_heads {
            let kv_h = h / gqa.kv_group_size();
            if !seen_kv.contains(&kv_h) {
                seen_kv.push(kv_h);
                let pb = gqa.pullback_key_passthrough(kv_h, head_tag);
                for (dst, &src) in raw_k.iter_mut().zip(pb.iter()) {
                    *dst += src;
                }
            }
        }
        let submerged_k = submerged_forbidden.project_out(&raw_k);
        let unit_k = normalize_unit_vec(&submerged_k).ok_or_else(|| {
            V2CompilerError::DegenerateProjectedTarget {
                rule: format!("asid_write_{seg}"),
                norm: l2_norm(&submerged_k),
            }
        })?;
        max_leak = max_leak.max(submerged_forbidden.leakage_norm(&unit_k));
        write_vecs.push(unit_k);
    }

    let mut read_vecs: Vec<Vec<f64>> = Vec::with_capacity(4);
    for target_seg in 0..4 {
        // Contrastive read query: `+1.0 * target_seg - 0.50 * other_segs` (especially `-0.75 * Scratchpad`!)
        let mut q_head_comb = vec![0.0; pass_d];
        for (seg, head_tag) in head_tags.iter().enumerate() {
            let coeff = if seg == target_seg {
                1.0
            } else if seg == MemorySegmentAsid::EphemeralScratchpad as usize {
                -0.75
            } else {
                -0.35
            };
            for (dst, &src) in q_head_comb.iter_mut().zip(head_tag.iter()) {
                *dst += coeff * src;
            }
        }
        let mut raw_q = vec![0.0; gqa.hidden_dim];
        for &h in copy_heads {
            let pb = gqa.pullback_query_passthrough(h, &q_head_comb);
            for (dst, &src) in raw_q.iter_mut().zip(pb.iter()) {
                *dst += src;
            }
        }
        let submerged_q = submerged_forbidden.project_out(&raw_q);
        let unit_q = normalize_unit_vec(&submerged_q).ok_or_else(|| {
            V2CompilerError::DegenerateProjectedTarget {
                rule: format!("asid_read_{target_seg}"),
                norm: l2_norm(&submerged_q),
            }
        })?;
        max_leak = max_leak.max(submerged_forbidden.leakage_norm(&unit_q));
        read_vecs.push(unit_q);
    }

    Ok(AsidSegmentCodebook {
        gqa_layer: gqa.layer_idx,
        write_key_tags: write_vecs.try_into().expect("len 4"),
        read_query_selectors: read_vecs.try_into().expect("len 4"),
        max_wv_leakage: max_leak,
    })
}
