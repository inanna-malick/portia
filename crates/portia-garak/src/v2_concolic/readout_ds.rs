//! Downstream (`L_total - 1`) MLP anti-echo block lift and `faer` thin-SVD `StrictZeroCone`
//! (`AlphaEdit` ICLR 2025) benign/continuation subspace synthesis.

use crate::readout::GarakReadoutSpec;
use crate::specs::{GarakSolverError, GARAK_CONTRAST_TOKEN_IDS};
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{l2_norm, normalize_unit_vec, StateMatrix};
use portia_compiler::solvers::{NullspacePolicy, V2SolverConfig};
use std::collections::BTreeMap;

pub(crate) fn state_mat(
    name: &str,
    rows: &[Vec<f64>],
    exp_d: Option<usize>,
) -> Result<StateMatrix, GarakSolverError> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let d = exp_d.unwrap_or(ncols);
    if nrows == 0 || d == 0 || ncols != d || rows.iter().any(|r| r.len() != d) {
        return Err(GarakSolverError::InvalidMatrixShape {
            name: name.to_string(),
            nrows,
            ncols,
            expected_dim: d,
        });
    }
    StateMatrix::from_rows("garak_v2_split", rows).map_err(|e| GarakSolverError::InvalidSplit {
        name: name.to_string(),
        reason: e.to_string(),
    })
}

pub(crate) fn centered_rows_vec(mat: &StateMatrix) -> Vec<Vec<f64>> {
    if mat.nrows() < 2 {
        return Vec::new();
    }
    let mu = mat.mean_row();
    (0..mat.nrows())
        .map(|r| mat.row(r).iter().zip(&mu).map(|(&x, &m)| x - m).collect())
        .collect()
}

/// Extract an orthonormal `AlphaEdit` (`StrictZeroCone`) preservation basis from centered
/// benign/continuation states projected orthogonal to the active on/off separation subspace.
pub(crate) fn build_alphaedit_preserve_subspace(
    d: usize,
    on_mat: &StateMatrix,
    off_mat: &StateMatrix,
    benign_centered: &[Vec<f64>],
) -> Result<Option<OrthonormalSubspace>, GarakSolverError> {
    if benign_centered.is_empty() || d < 8 {
        return Ok(None);
    }
    let on_cap = if on_mat.nrows() <= d / 2 {
        on_mat.nrows().max(1)
    } else {
        (d * 2 / 5).max(1)
    };
    let on_tol = if on_mat.nrows() <= d / 2 { 1e-7 } else { 0.02 };
    let q_on = OrthonormalSubspace::from_row_space_svd("garak_q_on", on_mat, on_tol, on_cap)
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    let off_cap = (d * 3 / 4).saturating_sub(q_on.rank()).max(1);
    let q_off = OrthonormalSubspace::from_row_space_svd("garak_q_off", off_mat, 0.025, off_cap)
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    let max_sep = d.saturating_sub(4).max(1);
    let q_sep = OrthonormalSubspace::union_bases("garak_q_sep", d, &[&q_on, &q_off], max_sep)
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;

    let kept: Vec<Vec<f64>> = benign_centered
        .iter()
        .map(|r| q_sep.project_out(r))
        .filter(|p| l2_norm(p) > 1e-6)
        .collect();
    if kept.is_empty() {
        return Ok(None);
    }
    let resid_mat = StateMatrix::from_rows("garak_benign_resid", &kept)
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    let max_pres = (d / 8).clamp(1, 24);
    let q_pres =
        OrthonormalSubspace::from_row_space_svd("garak_alphaedit", &resid_mat, 1e-3, max_pres)
            .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    Ok((q_pres.rank() > 0).then_some(q_pres))
}

pub(crate) type ExtractedDownstreamAndRawPair =
    (Option<DownstreamSwigluBlock>, Option<(Vec<f64>, Vec<f64>)>);

/// When [`GarakReadoutSpec::FullDownstreamMlp`] is supplied, construct a [`DownstreamSwigluBlock`]
/// for contextual chordal-secant anti-echo projection alongside unprojected `(v_no_raw, v_act_raw)`.
pub(crate) fn extract_downstream_mlp_and_raw_contrast(
    layer_idx: usize,
    readout: &GarakReadoutSpec,
    d: usize,
) -> Result<ExtractedDownstreamAndRawPair, GarakSolverError> {
    match readout {
        GarakReadoutSpec::WithTrace { base, .. } => {
            extract_downstream_mlp_and_raw_contrast(layer_idx, base, d)
        }
        GarakReadoutSpec::FullDownstreamMlp {
            w_u_rows,
            final_norm_weight,
            post_attn_norm_weight,
            w_gate,
            w_up,
            w_down,
            ..
        } => {
            let m_f = w_gate.len();
            let wg = state_mat("FullDownstreamMlp.w_gate", w_gate, Some(d))?;
            let wu = state_mat("FullDownstreamMlp.w_up", w_up, Some(d))?;
            let wd_d_by_m = state_mat("FullDownstreamMlp.w_down", w_down, Some(m_f))?;
            if post_attn_norm_weight.len() != d || final_norm_weight.len() != d {
                return Err(GarakSolverError::InvalidVectorLength {
                    name: "downstream_norm_weight".to_string(),
                    expected_dim: d,
                    actual: post_attn_norm_weight.len().min(final_norm_weight.len()),
                });
            }
            let mut wd_cols_flat = vec![0.0; m_f * d];
            for r in 0..d {
                let row = wd_d_by_m.row(r);
                for m in 0..m_f {
                    wd_cols_flat[m * d + r] = row[m];
                }
            }
            let wd_cols = StateMatrix::try_new("garak_wd_cols", m_f, d, wd_cols_flat)
                .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
            let norm_scale: Vec<f64> = post_attn_norm_weight.iter().map(|&w| 1.0 + w).collect();
            let block = DownstreamSwigluBlock::try_new(layer_idx + 1, norm_scale, wg, wu, wd_cols)
                .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;

            let mut eff: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
            for &tid in &GARAK_CONTRAST_TOKEN_IDS {
                let raw = w_u_rows
                    .get(&tid.to_string())
                    .ok_or(GarakSolverError::MissingTokenUnembeddingRow { token_id: tid })?;
                if raw.len() != d {
                    return Err(GarakSolverError::InvalidVectorLength {
                        name: format!("w_u_rows[{tid}]"),
                        expected_dim: d,
                        actual: raw.len(),
                    });
                }
                eff.insert(
                    tid,
                    raw.iter()
                        .zip(final_norm_weight)
                        .map(|(&u, &w)| u * (1.0 + w))
                        .collect(),
                );
            }
            let mut v_no = vec![0.0; d];
            let mut v_act = vec![0.0; d];
            for j in 0..d {
                let no_c = (eff[&1206][j] + eff[&9764][j] + eff[&27775][j] + eff[&3783][j]) / 4.0;
                v_no[j] = eff[&2665][j] - 0.6 * eff[&9175][j] - 0.5 * eff[&332][j] - 0.3 * no_c;
                let act_c =
                    (eff[&760][j] + eff[&8160][j] + eff[&1206][j] + eff[&9175][j] + eff[&3783][j])
                        / 5.0;
                v_act[j] = eff[&50821][j] - 0.5 * eff[&332][j] - 0.4 * act_c;
            }
            let vn =
                normalize_unit_vec(&v_no).ok_or_else(|| GarakSolverError::DegenerateDirection {
                    context: "v_no_raw".to_string(),
                    norm: l2_norm(&v_no),
                })?;
            let va = normalize_unit_vec(&v_act).ok_or_else(|| {
                GarakSolverError::DegenerateDirection {
                    context: "v_act_raw".to_string(),
                    norm: l2_norm(&v_act),
                }
            })?;
            Ok((Some(block), Some((vn, va))))
        }
        _ => Ok((None, None)),
    }
}

pub(crate) fn solver_cfg_for_scale(sx: f64) -> V2SolverConfig {
    V2SolverConfig {
        g_pos_target: 3.50,
        g_pos_band: 1.20,
        g_neg_ceiling: -8.0,
        bf16_neg_pad: 0.35,
        neutral_up_tol: 0.06,
        trust_radius: (56.0 * sx).clamp(36.0, 2500.0),
        lambda_ridge: 0.05,
        gamma_within_cov: 0.0,
        nullspace_policy: NullspacePolicy::StrictZeroCone,
    }
}
