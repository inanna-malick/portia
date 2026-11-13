//! Pure-slice / `portia_compiler::numerics` readout & downstream anti-echo projection (`v_no`, `v_act`, `v_trace`).

use crate::specs::{GarakSolverError, GARAK_CONTRAST_TOKEN_IDS};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{l2_norm, normalize_unit_vec, StateMatrix};
use portia_core::TokenId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const fn default_anti_echo_top_k() -> usize {
    24
}

/// Compact or full readout head + downstream (`L_total - 1`) MLP anti-echo specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GarakReadoutSpec {
    WithTrace {
        base: Box<GarakReadoutSpec>,
        trace: Box<GarakTraceReadoutSpec>,
    },
    Precomputed {
        v_no: Vec<f64>,
        v_act: Vec<f64>,
    },
    CompactHeadAntiEcho {
        w_u_rows: BTreeMap<String, Vec<f64>>,
        final_norm_weight: Vec<f64>,
        post_attn_norm_weight: Vec<f64>,
        anti_no_gate_top: Vec<Vec<f64>>,
        anti_no_up_top: Vec<Vec<f64>>,
        anti_act_gate_top: Vec<Vec<f64>>,
        anti_act_up_top: Vec<Vec<f64>>,
    },
    FullDownstreamMlp {
        w_u_rows: BTreeMap<String, Vec<f64>>,
        final_norm_weight: Vec<f64>,
        post_attn_norm_weight: Vec<f64>,
        w_gate: Vec<Vec<f64>>,
        w_up: Vec<Vec<f64>>,
        w_down: Vec<Vec<f64>>,
        #[serde(default = "default_anti_echo_top_k")]
        top_k: usize,
    },
}

/// Natural-token trace contrast exported from the checkpoint's tokenizer/head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakTraceReadoutSpec {
    pub token_weights: BTreeMap<String, f64>,
    pub w_u_rows: BTreeMap<String, Vec<f64>>,
    pub final_norm_weight: Vec<f64>,
    pub post_attn_norm_weight: Vec<f64>,
    pub anti_trace_gate_top: Vec<Vec<f64>>,
    pub anti_trace_up_top: Vec<Vec<f64>>,
}

/// Resolved unit-norm readout target vectors (`v_no`, `v_act`) in $\mathbb{R}^d$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakResolvedReadouts {
    pub hidden_dim: usize,
    pub v_no: Vec<f64>,
    pub v_act: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_trace: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_trace: Option<Vec<Vec<f64>>>,
}

fn check_vec(name: &str, vals: &[f64], exp: Option<usize>) -> Result<usize, GarakSolverError> {
    let d = exp.unwrap_or(vals.len());
    if d == 0 || vals.len() != d || !vals.iter().all(|v| v.is_finite()) {
        return Err(GarakSolverError::InvalidVectorLength {
            name: name.into(),
            expected_dim: d,
            actual: vals.len(),
        });
    }
    Ok(d)
}

fn norm_checked(v: &[f64], ctx: &str) -> Result<Vec<f64>, GarakSolverError> {
    normalize_unit_vec(v).ok_or_else(|| GarakSolverError::DegenerateDirection {
        context: ctx.into(),
        norm: l2_norm(v),
    })
}

fn anti_echo_subspace(
    name: &str,
    gate_rows: &[Vec<f64>],
    up_rows: &[Vec<f64>],
    post_attn_norm: &[f64],
    d: usize,
) -> Result<OrthonormalSubspace, GarakSolverError> {
    check_vec("post_attn_norm_weight", post_attn_norm, Some(d))?;
    let tot = gate_rows.len() + up_rows.len();
    let mut scaled = Vec::with_capacity(tot);
    for (idx, r) in gate_rows.iter().chain(up_rows).enumerate() {
        if r.len() != d || !r.iter().all(|v| v.is_finite()) {
            return Err(GarakSolverError::InvalidMatrixShape {
                name: format!("{name}[{idx}]"),
                nrows: tot,
                ncols: r.len(),
                expected_dim: d,
            });
        }
        let s: Vec<f64> = r
            .iter()
            .zip(post_attn_norm)
            .map(|(&x, &w)| x * (1.0 + w))
            .collect();
        if l2_norm(&s) > 1e-12 {
            scaled.push(s);
        }
    }
    if scaled.is_empty() {
        return Ok(OrthonormalSubspace::empty(d));
    }
    let mat = StateMatrix::from_rows("garak_anti_echo", &scaled)
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    OrthonormalSubspace::from_row_space_svd("garak_anti_echo", &mat, 1e-10, scaled.len())
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))
}

fn w_u_eff_map(
    w_u_rows: &BTreeMap<String, Vec<f64>>,
    final_norm: &[f64],
    exp: Option<usize>,
) -> Result<(usize, BTreeMap<u32, Vec<f64>>), GarakSolverError> {
    let d = check_vec("final_norm_weight", final_norm, exp)?;
    let mut eff = BTreeMap::new();
    for &tid in &GARAK_CONTRAST_TOKEN_IDS {
        let raw = w_u_rows
            .get(&tid.to_string())
            .ok_or(GarakSolverError::MissingTokenUnembeddingRow { token_id: tid })?;
        check_vec("w_u_rows", raw, Some(d))?;
        eff.insert(
            tid,
            (0..d).map(|j| raw[j] * (1.0 + final_norm[j])).collect(),
        );
    }
    Ok((d, eff))
}

type AntiEchoPairs<'a> = (
    &'a [Vec<f64>],
    &'a [Vec<f64>],
    &'a [Vec<f64>],
    &'a [Vec<f64>],
);

fn project_no_and_act(
    d: usize,
    eff: &BTreeMap<u32, Vec<f64>>,
    post_attn_norm: &[f64],
    (no_g, no_u, act_g, act_u): AntiEchoPairs<'_>,
) -> Result<(Vec<f64>, Vec<f64>), GarakSolverError> {
    let vn: Vec<f64> = (0..d)
        .map(|j| {
            let c = (eff[&1206][j] + eff[&9764][j] + eff[&27775][j] + eff[&3783][j]) * 0.25;
            eff[&2665][j] - 0.6 * eff[&9175][j] - 0.5 * eff[&332][j] - 0.3 * c
        })
        .collect();
    let va: Vec<f64> = (0..d)
        .map(|j| {
            let c = (eff[&760][j] + eff[&8160][j] + eff[&1206][j] + eff[&9175][j] + eff[&3783][j])
                * 0.2;
            eff[&50821][j] - 0.5 * eff[&332][j] - 0.4 * c
        })
        .collect();
    let q_no = anti_echo_subspace("anti_no", no_g, no_u, post_attn_norm, d)?;
    let q_act = anti_echo_subspace("anti_act", act_g, act_u, post_attn_norm, d)?;
    Ok((
        norm_checked(&q_no.project_out(&vn), "v_no_projected")?,
        norm_checked(&q_act.project_out(&va), "v_act_projected")?,
    ))
}

/// Resolve unit-norm `(v_no, v_act)` readout vectors in `f64` from any [`GarakReadoutSpec`] variant.
pub fn resolve_garak_readouts(
    spec: &GarakReadoutSpec,
    expected_dim: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>), GarakSolverError> {
    match spec {
        GarakReadoutSpec::WithTrace { base, .. } => resolve_garak_readouts(base, expected_dim),
        GarakReadoutSpec::Precomputed { v_no, v_act } => {
            let d = check_vec("readout.v_no", v_no, expected_dim)?;
            check_vec("readout.v_act", v_act, Some(d))?;
            Ok((norm_checked(v_no, "v_no")?, norm_checked(v_act, "v_act")?))
        }
        GarakReadoutSpec::CompactHeadAntiEcho {
            w_u_rows,
            final_norm_weight,
            post_attn_norm_weight,
            anti_no_gate_top: ng,
            anti_no_up_top: nu,
            anti_act_gate_top: ag,
            anti_act_up_top: au,
        } => {
            let (d, eff) = w_u_eff_map(w_u_rows, final_norm_weight, expected_dim)?;
            project_no_and_act(d, &eff, post_attn_norm_weight, (ng, nu, ag, au))
        }
        GarakReadoutSpec::FullDownstreamMlp {
            w_u_rows,
            final_norm_weight,
            post_attn_norm_weight,
            w_gate,
            w_up,
            w_down,
            top_k,
        } => {
            let (d, eff) = w_u_eff_map(w_u_rows, final_norm_weight, expected_dim)?;
            let m_f = w_gate.len();
            if m_f == 0
                || w_up.len() != m_f
                || w_down.len() != d
                || w_gate.iter().chain(w_up).any(|r| r.len() != d)
                || w_down.iter().any(|r| r.len() != m_f)
            {
                return Err(GarakSolverError::InvalidMatrixShape {
                    name: "FullDownstreamMlp".into(),
                    nrows: m_f,
                    ncols: w_down.len(),
                    expected_dim: d,
                });
            }
            let k_take = (*top_k).max(1).min(m_f);
            let pick = |dir: Vec<f64>| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
                let mut sc: Vec<(usize, f64)> = (0..m_f)
                    .map(|m| (m, (0..d).map(|r| w_down[r][m] * dir[r]).sum()))
                    .collect();
                sc.sort_by(|a, b| b.1.total_cmp(&a.1));
                sc.iter()
                    .take(k_take)
                    .map(|&(m, _)| (w_gate[m].clone(), w_up[m].clone()))
                    .unzip()
            };
            let (ng, nu) = pick(
                (0..d)
                    .map(|j| eff[&9175][j] + eff[&332][j] - 2.0 * eff[&2665][j])
                    .collect(),
            );
            let (ag, au) = pick(
                (0..d)
                    .map(|j| eff[&760][j] + eff[&8160][j] + eff[&332][j] - 3.0 * eff[&50821][j])
                    .collect(),
            );
            project_no_and_act(d, &eff, post_attn_norm_weight, (&ng, &nu, &ag, &au))
        }
    }
}

fn resolve_trace_readout(
    spec: &GarakReadoutSpec,
    d: usize,
) -> Result<Option<(Vec<f64>, OrthonormalSubspace)>, GarakSolverError> {
    let GarakReadoutSpec::WithTrace { trace, .. } = spec else {
        return Ok(None);
    };
    let inv = |r: &str| GarakSolverError::InvalidSplit {
        name: "trace.token_weights".into(),
        reason: r.into(),
    };
    if trace.token_weights.get("9175").copied().unwrap_or(0.0) >= 0.0
        || trace.token_weights.get("2665").copied().unwrap_or(0.0) >= 0.0
        || !trace.token_weights.values().any(|w| *w > 0.0)
    {
        return Err(inv("invalid trace token_weights contrast"));
    }
    check_vec("trace.final_norm_weight", &trace.final_norm_weight, Some(d))?;
    let mut raw = vec![0.0; d];
    for (token_str, &weight) in &trace.token_weights {
        let tid = TokenId::new(token_str.parse::<u32>().map_err(|_| inv("bad token ID"))?);
        if !weight.is_finite() {
            return Err(inv("nonfinite contrast"));
        }
        let row = trace.w_u_rows.get(&tid.get().to_string()).ok_or(
            GarakSolverError::MissingTokenUnembeddingRow {
                token_id: tid.get(),
            },
        )?;
        check_vec("trace.unembedding", row, Some(d))?;
        for j in 0..d {
            raw[j] += weight * row[j] * (1.0 + trace.final_norm_weight[j]);
        }
    }
    let q = anti_echo_subspace(
        "anti_trace",
        &trace.anti_trace_gate_top,
        &trace.anti_trace_up_top,
        &trace.post_attn_norm_weight,
        d,
    )?;
    Ok(Some((norm_checked(&q.project_out(&raw), "v_trace")?, q)))
}

/// Resolve [`GarakReadoutSpec`] into a serializable [`GarakResolvedReadouts`] payload.
pub fn resolve_garak_readout_payload(
    spec: &GarakReadoutSpec,
) -> Result<GarakResolvedReadouts, GarakSolverError> {
    let (v_no, v_act) = resolve_garak_readouts(spec, None)?;
    let d = v_no.len();
    let trace = resolve_trace_readout(spec, d)?;
    Ok(GarakResolvedReadouts {
        hidden_dim: d,
        v_no,
        v_act,
        v_trace: trace.as_ref().map(|(v, _)| v.clone()),
        q_trace: trace.map(|(_, q)| {
            (0..d)
                .map(|r| (0..q.rank()).map(|c| q.basis_col(c)[r]).collect())
                .collect()
        }),
    })
}
