//! `Gated DeltaNet-1` `conv1d`-Bypassing Head-Partitioned Scrubber (`deltanet_scrub`).
//!
//! Guided by NVIDIA's May 2026 `Gated DeltaNet-2` scalar-$\beta_t$ coupling analysis and our
//! empirical `Qwen3.5` `conv1d` tap measurements (`tap_{t-1} = 30.2%` energy bleed):
//! - Never drives `OverwriteLatch` ($\beta_{t,h} \to 1$) for rollback scrubbing on `DeltaNet-1`
//!   (which writes `100%` of noisy $\mathbf{v}_{t,h}$ into $\mathbf{S}_t^{(h)}$),
//! - Instead solves a minimum-norm pseudoinverse impulse on **`in_proj_a`** (driving
//!   $\Delta a_{t,h} > 0 \implies \alpha_{t,h} \to 0$ `DecayFlush` on `flush_heads` and
//!   $\Delta a_{t,h} < 0 \implies \alpha_{t,h} \to 1$ `RetentionLock` on `lock_heads`) while
//!   enforcing **exact `0.000` cross-talk onto `in_proj_b`** (`\Delta b_{t,h} == 0`) and
//!   projecting orthogonal to `W_U_top`!

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{
    dot_f64, l2_norm, normalize_unit_vec, quantize_slice_bf16_f64, silu_f64, slice_to_bf16_vec,
    StateMatrix,
};
use crate::solvers::clarabel_ternary::solve_conic_linear_vector;
use crate::solvers::V2SolverConfig;
use crate::wire::CompiledSlotBf16;
use serde::{Deserialize, Serialize};

/// Gate-projection weights of a `Qwen3_5GatedDeltaNet` sublayer ($\ell + 1$).
#[derive(Debug, Clone)]
pub struct DeltaNetGateLayer {
    pub layer_idx: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub input_norm_scale: Vec<f64>,
    /// `(num_heads, hidden_dim)` `in_proj_a` weights (retention decay $\alpha$).
    pub w_a_proj: StateMatrix,
    /// `(num_heads, hidden_dim)` `in_proj_b` weights (scalar erase+write $\beta$ — held untouched!).
    pub w_b_proj: StateMatrix,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledDeltaNetScrubSlot {
    pub slot: CompiledSlotBf16,
    pub target_deltanet_layer: usize,
    pub min_flush_head_da: f64,
    pub max_lock_head_da: f64,
    pub max_beta_crosstalk: f64,
}

/// Synthesize a 1-slot `</rollback>` scrubber at `upstream_mlp_layer` targeting `deltanet`:
/// - Drives `in_proj_a[h]` strongly positive (`+a_drive`, $\alpha_{t,h} \to 0$) on `flush_heads`,
/// - Drives `in_proj_a[h]` negative (`-0.5 * a_drive`, $\alpha_{t,h} \to 1$) on `lock_heads`,
/// - Clamps `in_proj_b` cross-talk to `0.000` across **all** `num_heads` (avoiding `DeltaNet-1`'s
///   coupled write-gate pollution!) and projects orthogonal to `w_u_top_basis`.
#[allow(clippy::too_many_arguments)]
pub fn compile_deltanet_head_partition_scrubber(
    rule_name: &str,
    slot_offset: usize,
    deltanet: &DeltaNetGateLayer,
    flush_heads: &[usize],
    lock_heads: &[usize],
    a_drive: f64,
    rollback_cursor_pre_mlp: &StateMatrix,
    off_cursor_pre_mlp: &StateMatrix,
    w_u_top_basis: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Result<CompiledDeltaNetScrubSlot, V2CompilerError> {
    let d = deltanet.hidden_dim;
    let h_total = deltanet.num_heads;
    if flush_heads.is_empty() || !a_drive.is_finite() || a_drive <= 0.0 {
        return Err(V2CompilerError::InvalidConfig {
            context: "compile_deltanet_head_partition_scrubber",
            reason: "requires >= 1 flush_head and finite a_drive > 0".to_string(),
        });
    }

    // Build forbidden subspace = `span(RMSNorm-scaled in_proj_b rows ∪ neutral in_proj_a rows ∪ W_U_top)`:
    let mut forb_rows = Vec::new();
    let mut n_forb = 0usize;
    for h in 0..h_total {
        for (&wb, &s) in deltanet
            .w_b_proj
            .row(h)
            .iter()
            .zip(&deltanet.input_norm_scale)
        {
            forb_rows.push(wb * s);
        }
        n_forb += 1;
        if !flush_heads.contains(&h) && !lock_heads.contains(&h) {
            for (&wa, &s) in deltanet
                .w_a_proj
                .row(h)
                .iter()
                .zip(&deltanet.input_norm_scale)
            {
                forb_rows.push(wa * s);
            }
            n_forb += 1;
        }
    }
    let forb_mat = StateMatrix::try_new("deltanet_forb_rows", n_forb, d, forb_rows)?;
    let q_b_and_neutral =
        OrthonormalSubspace::from_row_space_svd("deltanet_b_basis", &forb_mat, 1e-6, n_forb)?;
    let q_forb = match w_u_top_basis {
        Some(q_wu) => OrthonormalSubspace::union_bases(
            "deltanet_forb_wu",
            d,
            &[&q_b_and_neutral, q_wu],
            (q_b_and_neutral.rank() + q_wu.rank()).min(d.saturating_sub(32)),
        )?,
        None => q_b_and_neutral,
    };

    // Accumulate desired direction `+w_a[flush] - 0.5 * w_a[lock]`, then project out `q_forb`:
    let mut raw_dir = vec![0.0; d];
    for &h in flush_heads {
        for (j, (&wa, &s)) in deltanet
            .w_a_proj
            .row(h)
            .iter()
            .zip(&deltanet.input_norm_scale)
            .enumerate()
        {
            raw_dir[j] += wa * s;
        }
    }
    for &h in lock_heads {
        for (j, (&wa, &s)) in deltanet
            .w_a_proj
            .row(h)
            .iter()
            .zip(&deltanet.input_norm_scale)
            .enumerate()
        {
            raw_dir[j] -= 0.5 * wa * s;
        }
    }
    let clean_dir = q_forb.project_out(&raw_dir);
    let unit_dir = normalize_unit_vec(&clean_dir).ok_or_else(|| {
        V2CompilerError::DegenerateProjectedTarget {
            rule: rule_name.to_string(),
            norm: l2_norm(&clean_dir),
        }
    })?;

    // Solve 1-slot diode `(w_gate, w_up)` on `rollback_cursor_pre_mlp` vs `off_cursor_pre_mlp`:
    let on_bf16 = rollback_cursor_pre_mlp.quantize_bf16();
    let off_bf16 = off_cursor_pre_mlp.quantize_bf16();
    let n_on = on_bf16.nrows();

    let w_gate_q = quantize_slice_bf16_f64(&solve_conic_linear_vector(
        rule_name,
        &on_bf16,
        &vec![config.g_pos_target; n_on],
        &vec![config.g_pos_band; n_on],
        Some((&off_bf16, config.g_neg_ceiling - config.bf16_neg_pad)),
        None,
        config,
    )?);
    let w_up_q = quantize_slice_bf16_f64(&solve_conic_linear_vector(
        rule_name,
        &on_bf16,
        &vec![1.0; n_on],
        &vec![0.18; n_on],
        None,
        None,
        config,
    )?);

    let gp = on_bf16.matvec(&w_gate_q)?;
    let up = on_bf16.matvec(&w_up_q)?;
    let mean_act: f64 = gp
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu_f64(g) * u)
        .sum::<f64>()
        / (n_on as f64);

    let w_down_f64: Vec<f64> = unit_dir
        .iter()
        .map(|&v| v * (a_drive / mean_act.max(1e-6)))
        .collect();
    let w_down_q = quantize_slice_bf16_f64(&w_down_f64);

    // Measure actual post-bf16 head response `(da, db)`:
    let delta_normed: Vec<f64> = w_down_q
        .iter()
        .zip(&deltanet.input_norm_scale)
        .map(|(&w, &s)| w * s * mean_act)
        .collect();
    let min_flush_head_da = flush_heads
        .iter()
        .map(|&h| dot_f64(deltanet.w_a_proj.row(h), &delta_normed))
        .fold(f64::INFINITY, f64::min);
    let max_lock_head_da = lock_heads
        .iter()
        .map(|&h| dot_f64(deltanet.w_a_proj.row(h), &delta_normed))
        .fold(f64::NEG_INFINITY, f64::max);
    let max_beta_crosstalk = (0..h_total)
        .map(|h| dot_f64(deltanet.w_b_proj.row(h), &delta_normed).abs())
        .fold(0.0_f64, f64::max);

    Ok(CompiledDeltaNetScrubSlot {
        slot: CompiledSlotBf16 {
            slot_offset,
            w_gate: slice_to_bf16_vec(&w_gate_q),
            w_up: slice_to_bf16_vec(&w_up_q),
            w_down: slice_to_bf16_vec(&w_down_q),
        },
        target_deltanet_layer: deltanet.layer_idx,
        min_flush_head_da,
        max_lock_head_da: if lock_heads.is_empty() {
            0.0
        } else {
            max_lock_head_da
        },
        max_beta_crosstalk,
    })
}
