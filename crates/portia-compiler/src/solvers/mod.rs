//! Scope-indexed Nullspace Policies (`AlphaEdit` strict `ZeroCone` vs. `AlphaEdit+` ACL 2026
//! bounded `SecondOrderCone` conflict-slack) and sparse `clarabel::algebra::CscMatrix` helpers.

pub mod clarabel_ternary;
pub mod conic_active_set;
pub mod conic_qp;
pub mod conic_subspace;
pub mod deltanet_scrub;

use clarabel::algebra::CscMatrix;
use serde::{Deserialize, Serialize};

/// Nullspace preservation policy applied to a `Q_preserve` (`Q_gift` / `Q_submerged`) basis:
/// - `StrictZeroCone`: exact $\mathbf{Q}^\top \mathbf{w} = \mathbf{0}$ (`AlphaEdit` ICLR 2025),
///   required for permanently welded static `.safetensors` checkpoints (`StaticPermanentScope`).
/// - `BoundedSocSlack`: bounded $\ell_2$ slack $\|\mathbf{Q}^\top \mathbf{w}\|_2 \le \varepsilon$
///   (`AlphaEdit+` ACL 2026), resolving infeasible conflicts against entrenched priors on dynamic
///   `[t_0, t_e)` `SpanCartridges` that detach cleanly at `</close_tag>`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NullspacePolicy {
    StrictZeroCone,
    BoundedSocSlack { max_l2_leak: f64 },
}

/// Configuration for the `v2` Conic Bilinear `SwiGLU` Solver (`clarabel`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SolverConfig {
    /// Positive cursor pre-SiLU gate target (default: `3.20`).
    pub g_pos_target: f64,
    /// Allowed positive gate band half-width (default: `0.20`).
    pub g_pos_band: f64,
    /// Hard off-cursor / background reverse-bias ceiling (default: `-8.0`).
    pub g_neg_ceiling: f64,
    /// Extra QP margin below `g_neg_ceiling` absorbing `bf16` rounding (default: `0.35`).
    pub bf16_neg_pad: f64,
    /// Allowed neutral-branch (`0 / ASK`) band `|w_up · x_0| <= neutral_up_tol` (default: `0.06`).
    pub neutral_up_tol: f64,
    /// Second-order cone trust-region radius on `||w||_2` (default: `28.0`).
    pub trust_radius: f64,
    /// L2 ridge weight (default: `2e-3`).
    pub lambda_ridge: f64,
    /// Within-class covariance / nuisance dampening weight (default: `15.0`).
    pub gamma_within_cov: f64,
    /// Nullspace policy (`StrictZeroCone` vs. `AlphaEdit+` `BoundedSocSlack`).
    pub nullspace_policy: NullspacePolicy,
}

impl Default for V2SolverConfig {
    fn default() -> Self {
        Self {
            g_pos_target: 3.20,
            g_pos_band: 0.20,
            g_neg_ceiling: -8.0,
            bf16_neg_pad: 0.35,
            neutral_up_tol: 0.06,
            trust_radius: 28.0,
            lambda_ridge: 2e-3,
            gamma_within_cov: 15.0,
            nullspace_policy: NullspacePolicy::StrictZeroCone,
        }
    }
}

impl V2SolverConfig {
    /// Dynamic-span preset enabling `AlphaEdit+` (ACL 2026) bounded conic slack (`max_l2_leak`).
    #[must_use]
    pub fn dynamic_span_with_slack(max_l2_leak: f64) -> Self {
        Self {
            nullspace_policy: NullspacePolicy::BoundedSocSlack { max_l2_leak },
            ..Self::default()
        }
    }
}

/// Build a `Clarabel` column-compressed matrix from row-major dense entries `(nrows, ncols)`.
/// When `upper_triangle_only` is `true` (required for `P` in Clarabel), keeps only `r <= c`.
#[must_use]
pub fn dense_row_major_to_csc(
    nrows: usize,
    ncols: usize,
    data: &[f64],
    upper_triangle_only: bool,
    drop_tol: f64,
) -> CscMatrix<f64> {
    debug_assert_eq!(data.len(), nrows * ncols);
    let mut colptr = Vec::with_capacity(ncols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    colptr.push(0);
    for c in 0..ncols {
        let r_end = if upper_triangle_only {
            (c + 1).min(nrows)
        } else {
            nrows
        };
        for r in 0..r_end {
            let v = data[r * ncols + c];
            if v.abs() > drop_tol {
                rowval.push(r);
                nzval.push(v);
            }
        }
        colptr.push(nzval.len());
    }
    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

/// Build an `(n, n)` purely diagonal `Clarabel` `CscMatrix<f64>` directly in $O(n)$ time and memory,
/// eliminating dense `(n * n)` quadratic objective allocations and Cholesky fill-in.
#[must_use]
pub fn build_diagonal_csc(diag: &[f64], drop_tol: f64) -> CscMatrix<f64> {
    let n = diag.len();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::with_capacity(n);
    let mut nzval = Vec::with_capacity(n);
    colptr.push(0);
    for (c, &v) in diag.iter().enumerate() {
        if v.abs() > drop_tol {
            rowval.push(c);
            nzval.push(v);
        }
        colptr.push(nzval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Build an `(nrows, ncols)` `Clarabel` `CscMatrix<f64>` in two streaming passes from a closure
/// `emit_rows` that yields `(row_idx, col_idx, val)` in non-decreasing `row_idx` order (with
/// distinct `col_idx` within any given row), avoiding any dense `(nrows * ncols)` buffer.
#[must_use]
pub fn build_row_sequential_csc<F>(
    nrows: usize,
    ncols: usize,
    drop_tol: f64,
    mut emit_rows: F,
) -> CscMatrix<f64>
where
    F: FnMut(&mut dyn FnMut(usize, usize, f64)),
{
    let mut counts = vec![0usize; ncols];
    emit_rows(&mut |_r, c, v| {
        if v.abs() > drop_tol {
            counts[c] += 1;
        }
    });

    let mut colptr = Vec::with_capacity(ncols + 1);
    colptr.push(0);
    for &cnt in &counts {
        colptr.push(colptr.last().copied().unwrap_or(0) + cnt);
    }
    let nnz = *colptr.last().unwrap_or(&0);
    let mut rowval = vec![0usize; nnz];
    let mut nzval = vec![0.0_f64; nnz];
    let mut write_pos = colptr[..ncols].to_vec();

    emit_rows(&mut |r, c, v| {
        if v.abs() > drop_tol {
            let idx = write_pos[c];
            rowval[idx] = r;
            nzval[idx] = v;
            write_pos[c] = idx + 1;
        }
    });

    debug_assert!((0..ncols).all(|c| {
        let slice = &rowval[colptr[c]..colptr[c + 1]];
        slice.windows(2).all(|w| w[0] < w[1])
    }));
    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}
