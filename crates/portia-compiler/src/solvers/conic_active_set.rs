//! Exact KKT Active-Set Cutting-Plane Reduction for Large Off-Cursor Halfspace Sets (`conic_active_set`).
//!
//! When `X_neg · w <= ceiling` has `n_neg >> 96` rows (e.g. `2,500+` off-cursor captures) in high
//! hidden dimension `d`, forming all `n_neg × d` dense rows inside `Clarabel`'s scalar `qdldl`
//! $LDL^\top$ factorization creates a dense $K_{d, n_{\text{neg}}}$ Schur clique.
//!
//! Because the conic objective is strictly convex (`lambda_ridge > 0`), solving over an active row
//! subset $S \subseteq \{0, \dots, n_{\text{neg}} - 1\}$ yields a lower-bound relaxation $f(\mathbf{w}_S^*) \le f(\mathbf{w}^*)$.
//! Whenever $\max_{r \notin S} (\mathbf{x}_{\text{neg}, r}^\top \mathbf{w}_S^* - c_{\text{neg}}) \le 10^{-9}$
//! (`< tol_feas = 10^{-8}`), setting $\lambda_r^* = 0$ for all $r \notin S$ satisfies **exact KKT
//! primal feasibility, dual feasibility, complementary slackness, and stationarity** on the full
//! `n_neg`-row problem while keeping $|S| \ll d$ inside [`super::conic_subspace`]'s orthonormal row-space!

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, normalize_unit_vec, StateMatrix};
use crate::solvers::conic_qp::solve_single_conic_pass;
use crate::solvers::{NullspacePolicy, V2SolverConfig};

/// Minimum `n_neg` row count before activating outer KKT cutting-plane generation.
pub const ACTIVE_SET_MIN_NEG_ROWS: usize = 96;
const ACTIVE_SET_BATCH_SIZE: usize = 64;
const ACTIVE_SET_MAX_ROUNDS: usize = 8;
const INACTIVE_VIOLATION_TOL: f64 = 1e-9;

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_with_exact_active_set(
    rule_name: &str,
    on_cursor_bf16: &StateMatrix,
    targets: &[f64],
    bands: &[f64],
    neg_mat: &StateMatrix,
    ceiling: f64,
    q_preserve: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Result<Vec<f64>, V2CompilerError> {
    let n_on = on_cursor_bf16.nrows();
    let d = on_cursor_bf16.hidden_dim();
    let n_neg = neg_mat.nrows();

    if n_neg <= ACTIVE_SET_MIN_NEG_ROWS || (n_on + ACTIVE_SET_MIN_NEG_ROWS) * 3 > d * 2 {
        return solve_single_conic_pass(
            rule_name,
            on_cursor_bf16,
            targets,
            bands,
            Some((neg_mat, ceiling)),
            q_preserve,
            config,
        );
    }

    let mut mu_on = on_cursor_bf16.mean_row();
    let mut mu_neg = neg_mat.mean_row();
    if matches!(config.nullspace_policy, NullspacePolicy::StrictZeroCone) {
        if let Some(qp) = q_preserve {
            mu_on = qp.project_out(&mu_on);
            mu_neg = qp.project_out(&mu_neg);
        }
    }
    let u_on = normalize_unit_vec(&mu_on).unwrap_or_else(|| vec![0.0; d]);
    let u_neg = normalize_unit_vec(&mu_neg).unwrap_or_else(|| vec![0.0; d]);
    let probe: Vec<f64> = (0..d).map(|j| 2.0 * u_on[j] - u_neg[j]).collect();

    let mut init_scores: Vec<(f64, usize)> = (0..n_neg)
        .map(|r| (dot_f64(neg_mat.row(r), &probe), r))
        .collect();
    init_scores.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut in_active = vec![false; n_neg];
    let mut active_indices = Vec::with_capacity(ACTIVE_SET_BATCH_SIZE * 4);
    for &(_, r) in init_scores.iter().take(ACTIVE_SET_BATCH_SIZE.min(n_neg)) {
        in_active[r] = true;
        active_indices.push(r);
    }
    active_indices.sort_unstable();

    for _round in 0..ACTIVE_SET_MAX_ROUNDS {
        if (n_on + active_indices.len()) * 3 > d * 2 {
            break;
        }
        let mut sub_flat = Vec::with_capacity(active_indices.len() * d);
        for &r in &active_indices {
            sub_flat.extend_from_slice(neg_mat.row(r));
        }
        let sub_mat = StateMatrix::try_new("neg_active_set", active_indices.len(), d, sub_flat)?;
        let w_cand = solve_single_conic_pass(
            rule_name,
            on_cursor_bf16,
            targets,
            bands,
            Some((&sub_mat, ceiling)),
            q_preserve,
            config,
        )?;

        let mut max_active_viol = 0.0_f64;
        let mut violated: Vec<(f64, usize)> = Vec::new();
        for (r, &is_act) in in_active.iter().enumerate() {
            let viol = dot_f64(neg_mat.row(r), &w_cand) - ceiling;
            if is_act {
                max_active_viol = max_active_viol.max(viol);
            } else if viol > INACTIVE_VIOLATION_TOL {
                violated.push((viol, r));
            }
        }

        if violated.is_empty() && max_active_viol <= 1e-6 {
            return Ok(w_cand);
        }
        if violated.is_empty() {
            break;
        }

        violated.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, r) in violated.iter().take(ACTIVE_SET_BATCH_SIZE) {
            in_active[r] = true;
            active_indices.push(r);
        }
        active_indices.sort_unstable();
        if active_indices.len() == n_neg {
            break;
        }
    }

    solve_single_conic_pass(
        rule_name,
        on_cursor_bf16,
        targets,
        bands,
        Some((neg_mat, ceiling)),
        q_preserve,
        config,
    )
}
