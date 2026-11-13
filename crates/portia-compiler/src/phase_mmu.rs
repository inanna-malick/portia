//! Zero-Gate-Tax Deterministic Phase-Multiplexed 4-Color `ASID` MMU (`phase_mmu`).
//!
//! When the Rust state machine (`RelayMachine` + `EpistemicDomainEnv`) knows the active protocol
//! phase deterministically (`Prefill:Schema`, `Prefill:User`, `Prefill:ToolObs`, `ModePlan:<Mode>`,
//! `ModePayload:<Mode>`), each phase mounts into the **exact same 1–2 physical SwiGLU slots** using
//! Homogeneous DC-Carrier Constant-On Diodes (`100%` recall while mounted, `0%` cross-phase leak
//! when unmounted) inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$.

use crate::discovery::gqa_bus::GqaPointerLayer;
use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, l2_norm, normalize_unit_vec};
use faer::Mat;
use serde::{Deserialize, Serialize};

pub mod cursor_slots;
pub mod dc_slots;
pub use cursor_slots::{
    compile_cursor_gated_seven_bank_phase_mmu, compile_intra_phase_cursor_slot,
    CompiledIntraPhaseCursorSlot, IntraPhaseCursorCapture, SevenBankCursorCaptures,
};
pub use dc_slots::{compile_dc_carrier_phase_slot, compile_seven_bank_phase_asid_mmu};

/// 4-Color Hardware Memory-Tagging (`ASID`) Segments Driven by Deterministic Host Phase Switching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseMmuSegment {
    /// `0`: `<schema>` read-only DDL symbol table (`"Prefill:Schema"`).
    AuthoritativeSchema = 0,
    /// `1`: `<user>` untrusted prompt / presupposition (`"Prefill:User"`).
    UserPresupposition = 1,
    /// `2`: `<tool_response>` execution-verified observation rows (`"Prefill:ToolObs"`).
    VerifiedObservation = 2,
    /// `3`: `<think>` / `Plan:` ephemeral reasoning scratchpad (`"ModePlan:<Mode>"`).
    EphemeralScratchpad = 3,
}

impl PhaseMmuSegment {
    pub const ALL: [Self; 4] = [
        Self::AuthoritativeSchema,
        Self::UserPresupposition,
        Self::VerifiedObservation,
        Self::EphemeralScratchpad,
    ];
}

/// Joint-KV 4-Color Submerged `ASID` Codebook (`write_tags[4]` + `read_selectors[4]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointKvPhaseMmuCodebook {
    pub gqa_layer: usize,
    pub write_key_tags: [Vec<f64>; 4],
    pub read_query_selectors: [Vec<f64>; 4],
    pub max_submerged_leakage: f64,
}

/// Configuration for compiling a complete 7-bank Phase-Multiplexed 4-Color `ASID` MMU into 2 slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMmuCompileConfig {
    pub model_id: String,
    pub base_intermediate_size: usize,
    pub upstream_mlp_layer: usize,
    pub copy_heads: Vec<usize>,
    pub distractor_heads: Vec<usize>,
    pub gate_drive: f64,
    pub write_impulse_scale: f64,
    pub read_impulse_scale: f64,
}

/// Synthesize a 4-color `ASID` codebook jointly spanning **all** KV groups of `copy_heads`
/// strictly inside `submerged_forbidden^⊥` (`ker(W_V · diag(γ_in)) ∩ ker(W_U_top)`).
pub fn synthesize_joint_kv_4color_codebook(
    gqa: &GqaPointerLayer,
    copy_heads: &[usize],
    distractor_heads: &[usize],
    cursor_ref_h: &[f64],
    submerged_forbidden: &OrthonormalSubspace,
) -> Result<JointKvPhaseMmuCodebook, V2CompilerError> {
    let d = gqa.hidden_dim;
    let pass_d = gqa.passthrough_dim();
    if pass_d < 16 || copy_heads.is_empty() {
        return Err(V2CompilerError::InvalidConfig {
            context: "synthesize_joint_kv_4color_codebook",
            reason: "requires passthrough_dim >= 16 and >= 1 copy head".to_string(),
        });
    }
    let grp = gqa.kv_group_size();
    let mut kv_groups = Vec::new();
    for &h in copy_heads {
        let g = h / grp;
        if !kv_groups.contains(&g) {
            kv_groups.push(g);
        }
    }

    // Build joint passthrough `W_K` matrix across all active KV groups, projected into `submerged^⊥`:
    let mut joint_rows: Vec<Vec<f64>> = Vec::with_capacity(kv_groups.len() * pass_d);
    for &kv_h in &kv_groups {
        let r0 = kv_h * gqa.head_dim + gqa.rotary_dim;
        for p in 0..pass_d {
            let kn = gqa.k_norm_scale[gqa.rotary_dim + p];
            let k_row = gqa.w_k_proj.row(r0 + p);
            let raw: Vec<f64> = (0..d)
                .map(|j| kn * k_row[j] * gqa.input_norm_scale[j])
                .collect();
            joint_rows.push(submerged_forbidden.project_out(&raw));
        }
    }
    let m_rows = joint_rows.len();
    let mat = Mat::from_fn(m_rows, d, |r, c| joint_rows[r][c]);
    let svd = mat
        .thin_svd()
        .map_err(|_| V2CompilerError::SvdConvergence {
            context: "synthesize_joint_kv_4color_codebook",
        })?;
    let v = svd.V();
    let rank_avail = v.ncols();
    if rank_avail < 16 {
        return Err(V2CompilerError::InvalidConfig {
            context: "synthesize_joint_kv_4color_codebook",
            reason: format!(
                "insufficient submerged W_K rank ({rank_avail} < 16) for 4 disjoint colors"
            ),
        });
    }

    // Partition top-16 right singular vectors into 4 strictly orthogonal 4-mode color bundles!
    let mut write_vecs: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut max_leak = 0.0_f64;
    for seg in 0..4 {
        let mut sum_v = vec![0.0_f64; d];
        for k in 0..4 {
            let col = seg * 4 + k;
            for (j, dst) in sum_v.iter_mut().enumerate() {
                *dst += v[(j, col)];
            }
        }
        let clean = submerged_forbidden.project_out(&sum_v);
        let unit_w =
            normalize_unit_vec(&clean).ok_or(V2CompilerError::DegenerateProjectedTarget {
                rule: format!("phase_mmu_write_{seg}"),
                norm: l2_norm(&clean),
            })?;
        max_leak = max_leak.max(submerged_forbidden.leakage_norm(&unit_w));
        write_vecs.push(unit_w);
    }

    // Pull back contrastive read selectors + `W_Q_gate` copy-head opening / distractor closing:
    let mut gate_delta = vec![0.0_f64; d];
    for &h in copy_heads {
        let pb_g = gqa.pullback_head_gate(h, cursor_ref_h);
        if let Some(ug) = normalize_unit_vec(&pb_g) {
            for (dst, &val) in gate_delta.iter_mut().zip(&ug) {
                *dst += 1.5 * val;
            }
        }
    }
    for &h in distractor_heads {
        let pb_g = gqa.pullback_head_gate(h, cursor_ref_h);
        if let Some(ug) = normalize_unit_vec(&pb_g) {
            for (dst, &val) in gate_delta.iter_mut().zip(&ug) {
                *dst -= 1.5 * val;
            }
        }
    }

    let mut read_vecs: Vec<Vec<f64>> = Vec::with_capacity(4);
    for target_seg in 0..4 {
        let mut raw_q = gate_delta.clone();
        for &h in copy_heads {
            let kv_h = h / grp;
            let r0 = kv_h * gqa.head_dim + gqa.rotary_dim;
            // Compute target passthrough key signature `p_comb` in `[0..pass_d)` for this KV group:
            let mut p_comb = vec![0.0_f64; pass_d];
            for (seg, w_seg) in write_vecs.iter().enumerate() {
                let coeff = if seg == target_seg {
                    1.0
                } else if seg == PhaseMmuSegment::EphemeralScratchpad as usize {
                    -0.85
                } else {
                    -0.55
                };
                let w_scaled: Vec<f64> = w_seg
                    .iter()
                    .zip(&gqa.input_norm_scale)
                    .map(|(&x, &s)| x * s)
                    .collect();
                for (p, dst) in p_comb.iter_mut().enumerate() {
                    let k_val = dot_f64(gqa.w_k_proj.row(r0 + p), &w_scaled)
                        * gqa.k_norm_scale[gqa.rotary_dim + p];
                    *dst += coeff * k_val;
                }
            }
            let pb_q = gqa.pullback_query_passthrough(h, &p_comb);
            if let Some(uq) = normalize_unit_vec(&pb_q) {
                for (dst, &val) in raw_q.iter_mut().zip(&uq) {
                    *dst += 2.5 * val;
                }
            }
        }
        let clean_q = submerged_forbidden.project_out(&raw_q);
        let unit_q =
            normalize_unit_vec(&clean_q).ok_or(V2CompilerError::DegenerateProjectedTarget {
                rule: format!("phase_mmu_read_{target_seg}"),
                norm: l2_norm(&clean_q),
            })?;
        max_leak = max_leak.max(submerged_forbidden.leakage_norm(&unit_q));
        read_vecs.push(unit_q);
    }

    Ok(JointKvPhaseMmuCodebook {
        gqa_layer: gqa.layer_idx,
        write_key_tags: write_vecs.try_into().expect("4 write tags"),
        read_query_selectors: read_vecs.try_into().expect("4 read selectors"),
        max_submerged_leakage: max_leak,
    })
}
