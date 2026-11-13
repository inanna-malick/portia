//! Exact Orthonormal Row-Space Reduction (`w = U z`) for `conic_qp`.
//!
//! Because `solve_conic_linear_vector`'s objective ($\frac{\lambda}{2}\|\mathbf{w}\|_2^2 + \frac{\gamma}{2 n_{\text{on}}}\|(\mathbf{X}_{\text{on}} - \mathbf{1}\boldsymbol{\mu}^\top)\mathbf{w}\|_2^2$)
//! and trust-region cone ($\|\mathbf{w}\|_2 \le R$) are rotationally isotropic outside the active
//! constraint row-space $V$, any component $\mathbf{w}_\perp \perp V$ has zero effect on every linear
//! constraint while strictly increasing $\|\mathbf{w}\|_2^2$. Thus $\mathbf{w}^* \in V$ holds
//! identically:
//! - Under `StrictZeroCone`, $\mathbf{w} \in \mathcal{K} = \ker(\mathbf{Q}_{\text{preserve}}^\top)$,
//!   so $V = \operatorname{span}(\mathbf{P}_{\mathcal{K}}\mathbf{X}_{\text{on}}^\top \cup \mathbf{P}_{\mathcal{K}}\mathbf{X}_{\text{neg}}^\top)$
//!   has dimension $r \le n_{\text{on}} + n_{\text{neg}}$ (independent of `q_rank`!) and satisfies
//!   $\mathbf{Q}_{\text{preserve}}^\top \mathbf{U} = \mathbf{0}$ by construction.
//! - Under `BoundedSocSlack`, $V = \operatorname{span}(\mathbf{X}_{\text{on}}^\top \cup \mathbf{X}_{\text{neg}}^\top \cup \mathbf{Q}_{\text{preserve}})$
//!   has dimension $r \le n_{\text{on}} + n_{\text{neg}} + q_{\text{rank}}$.

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, l2_norm, StateMatrix};
use crate::solvers::NullspacePolicy;

pub(crate) struct ReducedConicProblem {
    /// Orthonormal columns `U = [u_0, .., u_{r-1}]` in `R^d` (`1 <= r < d`, `U^T U = I_r`).
    pub basis_cols: Vec<Vec<f64>>,
    /// `(n_on, r)` projected on-cursor activations `X_on * U`.
    pub on_red: StateMatrix,
    /// Optional `(n_neg, r)` projected off-cursor activations `(X_neg * U, ceiling)`.
    pub off_red: Option<(StateMatrix, f64)>,
    /// `q_rank` rows of length `r` (`Q_preserve^T * U`) when `BoundedSocSlack` is active.
    pub q_slack_red: Vec<Vec<f64>>,
}

impl ReducedConicProblem {
    #[inline]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis_cols.len()
    }

    /// Lift reduced coordinates $\mathbf{z} \in \mathbb{R}^r$ back to $\mathbf{w} = \mathbf{U}\mathbf{z} \in \mathbb{R}^d$.
    #[must_use]
    pub fn lift_to_full(&self, z: &[f64], d: usize) -> Vec<f64> {
        debug_assert_eq!(z.len(), self.basis_cols.len());
        let mut w = vec![0.0_f64; d];
        for (col, &zc) in self.basis_cols.iter().zip(z.iter()) {
            if zc != 0.0 {
                for (wi, &uc) in w.iter_mut().zip(col.iter()) {
                    *wi += zc * uc;
                }
            }
        }
        w
    }
}

fn orthonormalize_cgs2(
    candidates: Vec<Vec<f64>>,
    strict_qp: Option<&OrthonormalSubspace>,
) -> Vec<Vec<f64>> {
    let max_norm = candidates
        .iter()
        .map(|v| l2_norm(v))
        .fold(0.0_f64, f64::max);
    if max_norm <= 1e-12 {
        return Vec::new();
    }
    let tol = 1e-12 * max_norm;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(candidates.len());
    for mut v in candidates {
        for u in &cols {
            let alpha = dot_f64(u, &v);
            for (vi, &ui) in v.iter_mut().zip(u.iter()) {
                *vi -= alpha * ui;
            }
        }
        if let Some(qp) = strict_qp {
            v = qp.project_out(&v);
        }
        for u in &cols {
            let alpha = dot_f64(u, &v);
            for (vi, &ui) in v.iter_mut().zip(u.iter()) {
                *vi -= alpha * ui;
            }
        }
        let nrm = l2_norm(&v);
        if nrm > tol {
            let inv = 1.0 / nrm;
            for vi in &mut v {
                *vi *= inv;
            }
            cols.push(v);
        }
    }
    cols
}

fn orthonormalize_svd(
    d: usize,
    candidates: Vec<Vec<f64>>,
    strict_qp: Option<&OrthonormalSubspace>,
) -> Result<Vec<Vec<f64>>, V2CompilerError> {
    let m = candidates.len();
    let mut flat = Vec::with_capacity(m * d);
    for row in candidates {
        flat.extend_from_slice(&row);
    }
    let stacked = StateMatrix::try_new("conic_subspace_svd", m, d, flat)?;
    let sub = OrthonormalSubspace::from_row_space_svd("conic_subspace_svd", &stacked, 1e-12, m)?;
    if sub.rank() == 0 {
        return Ok(Vec::new());
    }
    let raw_cols: Vec<Vec<f64>> = (0..sub.rank()).map(|c| sub.basis_col(c).to_vec()).collect();
    if strict_qp.is_some() {
        Ok(orthonormalize_cgs2(raw_cols, strict_qp))
    } else {
        Ok(raw_cols)
    }
}

fn project_matrix_rows(
    ctx: &'static str,
    mat: &StateMatrix,
    cols: &[Vec<f64>],
) -> Result<StateMatrix, V2CompilerError> {
    let r = cols.len();
    let n = mat.nrows();
    let mut data = Vec::with_capacity(n * r);
    for i in 0..n {
        let row = mat.row(i);
        for col in cols {
            data.push(dot_f64(row, col));
        }
    }
    StateMatrix::try_new(ctx, n, r, data)
}

/// Construct an exact orthonormal row-space reduction whenever the active constraint count
/// `m_cand <= (2 * d) / 3` (`m_cand * 3 <= d * 2`), returning `Ok(None)` when full-space solving
/// is smaller or when all projected constraint vectors vanish (`r == 0`).
pub(crate) fn try_reduce_to_orthonormal_row_space(
    on_cursor_bf16: &StateMatrix,
    off_cursor_ceiling: Option<(&StateMatrix, f64)>,
    q_preserve: Option<&OrthonormalSubspace>,
    policy: NullspacePolicy,
) -> Result<Option<ReducedConicProblem>, V2CompilerError> {
    let d = on_cursor_bf16.hidden_dim();
    let n_on = on_cursor_bf16.nrows();
    let n_neg = off_cursor_ceiling.map_or(0, |(m, _)| m.nrows());
    let q_rank = q_preserve.map_or(0, OrthonormalSubspace::rank);

    let is_strict = matches!(policy, NullspacePolicy::StrictZeroCone);
    let m_cand = if is_strict {
        n_on + n_neg
    } else {
        n_on + n_neg + q_rank
    };

    if m_cand == 0 || m_cand * 3 > d * 2 {
        return Ok(None);
    }

    let strict_qp = if is_strict {
        q_preserve.filter(|q| q.rank() > 0)
    } else {
        None
    };

    let mut candidates = Vec::with_capacity(m_cand);
    for i in 0..n_on {
        let row = on_cursor_bf16.row(i);
        candidates.push(match strict_qp {
            Some(qp) => qp.project_out(row),
            None => row.to_vec(),
        });
    }
    if let Some((neg_mat, _)) = off_cursor_ceiling {
        for i in 0..n_neg {
            let row = neg_mat.row(i);
            candidates.push(match strict_qp {
                Some(qp) => qp.project_out(row),
                None => row.to_vec(),
            });
        }
    }
    if !is_strict {
        if let Some(qp) = q_preserve {
            for k in 0..q_rank {
                candidates.push(qp.basis_col(k).to_vec());
            }
        }
    }

    let basis_cols = if m_cand <= 80 {
        orthonormalize_cgs2(candidates, strict_qp)
    } else {
        orthonormalize_svd(d, candidates, strict_qp)?
    };

    let r = basis_cols.len();
    if r == 0 || r >= d {
        return Ok(None);
    }

    let on_red = project_matrix_rows("on_red", on_cursor_bf16, &basis_cols)?;
    let off_red = match off_cursor_ceiling {
        Some((neg_mat, ceiling)) => Some((
            project_matrix_rows("off_red", neg_mat, &basis_cols)?,
            ceiling,
        )),
        None => None,
    };
    let q_slack_red = if !is_strict && q_rank > 0 {
        let qp = q_preserve.expect("q_rank > 0");
        (0..q_rank)
            .map(|k| {
                let qcol = qp.basis_col(k);
                basis_cols.iter().map(|u| dot_f64(qcol, u)).collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(Some(ReducedConicProblem {
        basis_cols,
        on_red,
        off_red,
        q_slack_red,
    }))
}
