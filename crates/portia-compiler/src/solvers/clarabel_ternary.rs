//! Convex-Feasible Bilinear `SwiGLU` Ternary/Twin Pair Compiler (`clarabel_ternary`).
//!
//! Solves the convexity trap where an ambiguous `0 / ASK` state sits near the midpoint
//! $\mathbf{x}_0 \approx \frac{1}{2}(\mathbf{x}_+ + \mathbf{x}_-)$ between Branch `+1` (`FORK`)
//! and Branch `-1` (`INLINE`):
//! - `w_gate` fires `+g_pos_target` across **all** on-cursor states `{+1, 0, -1}` and clamps
//!   `<= -8.0` across all off-cursor / continuation / background states;
//! - `w_up_tan` is linear (no `SiLU`), cleanly mapping `+1 -> +1.0`, `0 -> [-tol, +tol]`, `-1 -> -1.0`.

use crate::error::V2CompilerError;
use crate::numerics::sod::{ChordalAngleRad, StiefelChordalFrame};
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{quantize_slice_bf16_f64, silu_f64, slice_to_bf16_vec, StateMatrix};
pub use crate::solvers::conic_qp::solve_conic_linear_vector;
use crate::solvers::V2SolverConfig;
use crate::wire::CompiledSlotBf16;

/// Ternary branch polarity on an active syntactic cursor (`+1`, `0` neutral/ASK, `-1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorPolarity {
    PlusOne,
    ZeroNeutral,
    MinusOne,
}

impl CursorPolarity {
    #[inline]
    #[must_use]
    pub const fn target_tan_value(self) -> f64 {
        match self {
            Self::PlusOne => 1.0,
            Self::ZeroNeutral => 0.0,
            Self::MinusOne => -1.0,
        }
    }
}

/// Output of compiling a 2-slot coupled `SO(d)` bilinear `SwiGLU` pair (`radial_slot`, `tangential_slot`).
#[derive(Debug, Clone)]
pub struct CompiledTernaryPair {
    pub radial_slot: CompiledSlotBf16,
    pub tangential_slot: CompiledSlotBf16,
    pub frame: StiefelChordalFrame,
    pub min_pos_gate_bf16: f64,
    pub max_neg_gate_bf16: f64,
    pub min_signed_active_tan_up: f64,
    pub max_neutral_abs_tan_up: f64,
    pub preserve_leak_gate_bf16: f64,
}

/// Compile a coupled 2-slot `(radial_slot_offset, tangential_slot_offset)` pair supporting single,
/// signed-twin (`+1 / -1`), or full convex-midpoint ternary (`+1 / 0 / -1`) cursor states.
#[allow(clippy::too_many_arguments)]
pub fn compile_coupled_ternary_pair(
    rule_name: &str,
    radial_slot_offset: usize,
    tangential_slot_offset: usize,
    on_cursor_pre_mlp: &StateMatrix,
    polarities: &[CursorPolarity],
    pos_residuals: &StateMatrix,
    off_cursor_pre_mlp: &StateMatrix,
    target_dir: &[f64],
    angle: ChordalAngleRad,
    q_preserve_gate: Option<&OrthonormalSubspace>,
    q_forbidden_down: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Result<CompiledTernaryPair, V2CompilerError> {
    let n_on = on_cursor_pre_mlp.nrows();
    if polarities.len() != n_on {
        return Err(V2CompilerError::DimensionMismatch {
            context: "compile_coupled_ternary_pair(polarities)",
            expected: n_on,
            actual: polarities.len(),
        });
    }

    let frame = StiefelChordalFrame::construct(
        rule_name,
        pos_residuals,
        target_dir,
        angle,
        q_forbidden_down,
    )?;

    let on_bf16 = on_cursor_pre_mlp.quantize_bf16();
    let off_bf16 = off_cursor_pre_mlp.quantize_bf16();

    let gate_targets = vec![config.g_pos_target; n_on];
    let gate_bands = vec![config.g_pos_band; n_on];
    let neg_bound = config.g_neg_ceiling - config.bf16_neg_pad;
    let w_gate_f64 = solve_conic_linear_vector(
        rule_name,
        &on_bf16,
        &gate_targets,
        &gate_bands,
        Some((&off_bf16, neg_bound)),
        q_preserve_gate,
        config,
    )?;
    let w_gate_q = quantize_slice_bf16_f64(&w_gate_f64);

    let gp = on_bf16.matvec(&w_gate_q)?;
    let gn = off_bf16.matvec(&w_gate_q)?;
    let min_gp = gp.iter().copied().fold(f64::INFINITY, f64::min);
    let max_gn = gn.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let req_min_pos = (config.g_pos_target - config.g_pos_band - 0.25).max(0.5);
    if min_gp < req_min_pos || max_gn > config.g_neg_ceiling + 1e-3 {
        return Err(V2CompilerError::InfeasibleDiode {
            rule: rule_name.to_string(),
            status: "post-bf16 verification failed".to_string(),
            min_pos_gate_bf16: min_gp,
            req_min_pos,
            max_neg_gate_bf16: max_gn,
            req_max_neg: config.g_neg_ceiling,
        });
    }

    let rad_targets = vec![1.0_f64; n_on];
    let rad_bands = vec![0.18_f64; n_on];
    let w_up_rad_f64 = solve_conic_linear_vector(
        rule_name,
        &on_bf16,
        &rad_targets,
        &rad_bands,
        None,
        q_preserve_gate,
        config,
    )?;
    let w_up_rad_q = quantize_slice_bf16_f64(&w_up_rad_f64);

    let tan_targets: Vec<f64> = polarities.iter().map(|p| p.target_tan_value()).collect();
    let tan_bands: Vec<f64> = polarities
        .iter()
        .map(|p| match p {
            CursorPolarity::ZeroNeutral => config.neutral_up_tol,
            CursorPolarity::PlusOne | CursorPolarity::MinusOne => 0.18,
        })
        .collect();
    let w_up_tan_f64 = if polarities.iter().all(|&p| p == CursorPolarity::PlusOne) {
        w_up_rad_f64
    } else {
        solve_conic_linear_vector(
            rule_name,
            &on_bf16,
            &tan_targets,
            &tan_bands,
            None,
            q_preserve_gate,
            config,
        )?
    };
    let w_up_tan_q = quantize_slice_bf16_f64(&w_up_tan_f64);

    let ur = on_bf16.matvec(&w_up_rad_q)?;
    let ut = on_bf16.matvec(&w_up_tan_q)?;

    let mut sum_act_rad = 0.0;
    let mut sum_signed_act_tan = 0.0;
    let mut n_active = 0usize;
    let mut min_signed_active_tan_up = f64::INFINITY;
    let mut max_neutral_abs_tan_up = 0.0_f64;

    for i in 0..n_on {
        let sg = silu_f64(gp[i]);
        sum_act_rad += sg * ur[i];
        match polarities[i] {
            CursorPolarity::PlusOne => {
                min_signed_active_tan_up = min_signed_active_tan_up.min(ut[i]);
                sum_signed_act_tan += sg * ut[i];
                n_active += 1;
            }
            CursorPolarity::MinusOne => {
                min_signed_active_tan_up = min_signed_active_tan_up.min(-ut[i]);
                sum_signed_act_tan += sg * (-ut[i]);
                n_active += 1;
            }
            CursorPolarity::ZeroNeutral => {
                max_neutral_abs_tan_up = max_neutral_abs_tan_up.max(ut[i].abs());
            }
        }
    }

    if n_active == 0 || min_signed_active_tan_up <= 0.15 {
        return Err(V2CompilerError::InfeasibleTernaryUp {
            rule: rule_name.to_string(),
            status: "insufficient signed tangential separation".to_string(),
            min_signed_active: min_signed_active_tan_up,
            max_neutral_abs: max_neutral_abs_tan_up,
            neutral_tol: config.neutral_up_tol + 0.05,
        });
    }

    let mean_act_rad = sum_act_rad / (n_on as f64);
    let mean_signed_act_tan = sum_signed_act_tan / (n_active as f64);
    let w_down_rad = frame.scaled_radial_down(mean_act_rad);
    let w_down_tan = frame.scaled_tangential_down(mean_signed_act_tan);

    let preserve_leak_gate_bf16 = q_preserve_gate.map_or(0.0, |q| q.leakage_norm(&w_gate_q));
    let w_gate_bf16 = slice_to_bf16_vec(&w_gate_q);

    Ok(CompiledTernaryPair {
        radial_slot: CompiledSlotBf16 {
            slot_offset: radial_slot_offset,
            w_gate: w_gate_bf16.clone(),
            w_up: slice_to_bf16_vec(&w_up_rad_q),
            w_down: slice_to_bf16_vec(&w_down_rad),
        },
        tangential_slot: CompiledSlotBf16 {
            slot_offset: tangential_slot_offset,
            w_gate: w_gate_bf16,
            w_up: slice_to_bf16_vec(&w_up_tan_q),
            w_down: slice_to_bf16_vec(&w_down_tan),
        },
        frame,
        min_pos_gate_bf16: min_gp,
        max_neg_gate_bf16: max_gn,
        min_signed_active_tan_up,
        max_neutral_abs_tan_up,
        preserve_leak_gate_bf16,
    })
}
