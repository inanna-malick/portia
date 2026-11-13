//! Contextual Secant & Jacobian Anti-Echo Auto-Discovery (`discovery`).
//!
//! Uses signed directional Jacobian / Secant attribution on downstream SwiGLU blocks to isolate
//! **only true suppressor neurons** whose response to a candidate residual write $+\delta\mathbf{h}$
//! actively opposes the target readout ($\Delta\text{logit} < 0$).

pub mod gqa_bus;

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, normalize_unit_vec, rms_norm_slice, silu_f64, StateMatrix};
use serde::{Deserialize, Serialize};

/// Downstream SwiGLU MLP block weights used for contextual secant anti-echo analysis.
#[derive(Debug, Clone)]
pub struct DownstreamSwigluBlock {
    pub layer_idx: usize,
    pub hidden_dim: usize,
    pub intermediate_size: usize,
    /// Effective `post_attention_layernorm` scale `(1 + w_norm)` of length `hidden_dim`.
    pub post_attn_norm_scale: Vec<f64>,
    /// Row-major `(intermediate_size, hidden_dim)` gate projection.
    pub w_gate: StateMatrix,
    /// Row-major `(intermediate_size, hidden_dim)` up projection.
    pub w_up: StateMatrix,
    /// Row-major `(intermediate_size, hidden_dim)` transposed `down_proj^T` (each row `m` is column `m` of `w_down`).
    pub w_down_cols: StateMatrix,
}

impl DownstreamSwigluBlock {
    pub fn try_new(
        layer_idx: usize,
        post_attn_norm_scale: Vec<f64>,
        w_gate: StateMatrix,
        w_up: StateMatrix,
        w_down_cols: StateMatrix,
    ) -> Result<Self, V2CompilerError> {
        let d = post_attn_norm_scale.len();
        let m = w_gate.nrows();
        if w_gate.hidden_dim() != d || w_up.hidden_dim() != d || w_down_cols.hidden_dim() != d {
            return Err(V2CompilerError::DimensionMismatch {
                context: "DownstreamSwigluBlock::hidden_dim",
                expected: d,
                actual: w_gate
                    .hidden_dim()
                    .min(w_up.hidden_dim())
                    .min(w_down_cols.hidden_dim()),
            });
        }
        if w_up.nrows() != m || w_down_cols.nrows() != m {
            return Err(V2CompilerError::DimensionMismatch {
                context: "DownstreamSwigluBlock::intermediate_size",
                expected: m,
                actual: w_up.nrows().min(w_down_cols.nrows()),
            });
        }
        Ok(Self {
            layer_idx,
            hidden_dim: d,
            intermediate_size: m,
            post_attn_norm_scale,
            w_gate,
            w_up,
            w_down_cols,
        })
    }
}

/// Attribution receipt for contextual anti-echo suppressor discovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AntiEchoDiscoveryReceipt {
    pub downstream_layer: usize,
    pub suppressor_neuron_indices: Vec<usize>,
    pub worst_secant_pushback: f64,
    pub nullspace_rank: usize,
    pub target_retained_energy: f64,
}

/// Discover true anti-echo suppressor neurons ($m$ where pushing $+\delta\,\hat{\mathbf{u}}$ along
/// `raw_readout_dir` induces $\Delta\text{MLP}_m \cdot \hat{\mathbf{u}} < 0$ across `pos_residuals`)
/// and return `(projected_unit_target, anti_echo_input_basis, receipt)`.
pub fn discover_and_project_anti_echo(
    rule_name: &str,
    block: &DownstreamSwigluBlock,
    pos_residuals: &StateMatrix,
    raw_readout_dir: &[f64],
    probe_step_frac: f64,
    max_suppressors: usize,
) -> Result<(Vec<f64>, OrthonormalSubspace, AntiEchoDiscoveryReceipt), V2CompilerError> {
    let d = block.hidden_dim;
    if pos_residuals.hidden_dim() != d || raw_readout_dir.len() != d {
        return Err(V2CompilerError::DimensionMismatch {
            context: "discover_and_project_anti_echo",
            expected: d,
            actual: pos_residuals.hidden_dim().min(raw_readout_dir.len()),
        });
    }
    let u_hat = normalize_unit_vec(raw_readout_dir).ok_or_else(|| {
        V2CompilerError::DegenerateProjectedTarget {
            rule: rule_name.to_string(),
            norm: 0.0,
        }
    })?;

    let m_size = block.intermediate_size;
    let n_pos = pos_residuals.nrows();
    let inv_n = 1.0 / (n_pos as f64);
    let step_frac = probe_step_frac.clamp(0.05, 0.50);

    // Precompute `w_down[:, m] · u_hat` for every neuron m:
    let down_dot_u: Vec<f64> = (0..m_size)
        .map(|m| dot_f64(block.w_down_cols.row(m), &u_hat))
        .collect();

    let mut secant_scores = vec![0.0_f64; m_size];
    for r in 0..n_pos {
        let h0 = pos_residuals.row(r);
        let h_rad = dot_f64(h0, h0).sqrt().max(1e-6);
        let h0_hat: Vec<f64> = h0.iter().map(|&v| v / h_rad).collect();

        // Constant-energy SO(d) chordal probe step (`||h1||_2 == ||h0||_2`, zero RMSNorm denominator drift!):
        let dot_hu = dot_f64(&h0_hat, &u_hat);
        let mut u_perp: Vec<f64> = u_hat
            .iter()
            .zip(&h0_hat)
            .map(|(&ui, &hi)| ui - dot_hu * hi)
            .collect();
        if let Some(u_perp_hat) = normalize_unit_vec(&u_perp) {
            u_perp = u_perp_hat;
        } else {
            u_perp.clone_from(&u_hat);
        }
        let cos_t = step_frac.cos();
        let sin_t = step_frac.sin();
        let h1: Vec<f64> = h0_hat
            .iter()
            .zip(&u_perp)
            .map(|(&hi, &vi)| h_rad * (cos_t * hi + sin_t * vi))
            .collect();

        let x0 = rms_norm_slice(h0, &block.post_attn_norm_scale, 1e-6);
        let x1 = rms_norm_slice(&h1, &block.post_attn_norm_scale, 1e-6);

        for m in 0..m_size {
            if down_dot_u[m].abs() < 1e-9 {
                continue;
            }
            let wg = block.w_gate.row(m);
            let wu = block.w_up.row(m);
            let a0 = silu_f64(dot_f64(wg, &x0)) * dot_f64(wu, &x0);
            let a1 = silu_f64(dot_f64(wg, &x1)) * dot_f64(wu, &x1);
            // Negative score means neuron `m`'s output change actively fights `u_hat`!
            secant_scores[m] += (a1 - a0) * down_dot_u[m] * inv_n;
        }
    }

    let mut suppressors: Vec<(usize, f64)> = secant_scores
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, s)| s < -1e-7)
        .collect();
    suppressors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    suppressors.truncate(max_suppressors);

    let worst_secant_pushback = suppressors.first().map_or(0.0, |x| x.1);
    let suppressor_indices: Vec<usize> = suppressors.iter().map(|&(idx, _)| idx).collect();

    if suppressor_indices.is_empty() {
        let receipt = AntiEchoDiscoveryReceipt {
            downstream_layer: block.layer_idx,
            suppressor_neuron_indices: Vec::new(),
            worst_secant_pushback: 0.0,
            nullspace_rank: 0,
            target_retained_energy: 1.0,
        };
        return Ok((u_hat, OrthonormalSubspace::empty(d), receipt));
    }

    // Collect RMSNorm-weighted gate & up input rows of strictly the true suppressor neurons:
    let mut input_rows = Vec::with_capacity(2 * suppressor_indices.len() * d);
    for &m in &suppressor_indices {
        for (&w, &s) in block.w_gate.row(m).iter().zip(&block.post_attn_norm_scale) {
            input_rows.push(w * s);
        }
        for (&w, &s) in block.w_up.row(m).iter().zip(&block.post_attn_norm_scale) {
            input_rows.push(w * s);
        }
    }
    let anti_mat = StateMatrix::try_new(
        "anti_echo_suppressor_rows",
        2 * suppressor_indices.len(),
        d,
        input_rows,
    )?;
    let q_anti = OrthonormalSubspace::from_row_space_svd(
        "anti_echo_basis",
        &anti_mat,
        1e-4,
        2 * suppressor_indices.len(),
    )?;

    let retained = q_anti.retained_energy_fraction(&u_hat);
    let u_proj_raw = q_anti.project_out(&u_hat);
    let u_safe = normalize_unit_vec(&u_proj_raw).ok_or_else(|| {
        V2CompilerError::DegenerateProjectedTarget {
            rule: rule_name.to_string(),
            norm: 0.0,
        }
    })?;

    let receipt = AntiEchoDiscoveryReceipt {
        downstream_layer: block.layer_idx,
        suppressor_neuron_indices: suppressor_indices,
        worst_secant_pushback,
        nullspace_rank: q_anti.rank(),
        target_retained_energy: retained,
    };
    Ok((u_safe, q_anti, receipt))
}
