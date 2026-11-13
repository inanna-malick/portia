//! Conic QP/SOCP Linear Vector Solver (`conic_qp`) with `AlphaEdit` (`StrictZeroCone`),
//! `AlphaEdit+` ACL 2026 (`BoundedSocSlack`), Exact Diagonal-`P` KKT Reformulation, and
//! Exact Orthonormal Row-Space Reduction (`w = U z`).

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::StateMatrix;
use crate::solvers::conic_active_set::{solve_with_exact_active_set, ACTIVE_SET_MIN_NEG_ROWS};
use crate::solvers::conic_subspace::try_reduce_to_orthonormal_row_space;
use crate::solvers::{
    build_diagonal_csc, build_row_sequential_csc, NullspacePolicy, V2SolverConfig,
};
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

/// Solve a constrained linear projection vector $\mathbf{w} \in \mathbb{R}^d$ in Clarabel satisfying:
/// - Band constraints `target[i] - band[i] <= X_on[i] · w <= target[i] + band[i]` for each on-cursor row,
/// - Optional upper ceiling `X_neg · w <= neg_ceiling` (used for `w_gate`, omitted on `w_up`),
/// - `NullspacePolicy` (`StrictZeroCone` vs `AlphaEdit+` `BoundedSocSlack`) on `q_preserve`,
/// - Second-order cone trust region `||w||_2 <= trust_radius`.
///
/// Reformulates within-class variance $\frac{\gamma}{2 n_{\text{on}}}\|(\mathbf{X}_{\text{on}} - \mathbf{1}\boldsymbol{\mu}^\top)\mathbf{w}\|_2^2$
/// via auxiliary equality variable $\mathbf{y} = \sqrt{\gamma / n_{\text{on}}}(\mathbf{X}_{\text{on}} - \mathbf{1}\boldsymbol{\mu}^\top)\mathbf{w}$
/// in `ZeroConeT` so `P` is **100% diagonal** (`nnz(P) <= n_vars`, zero Cholesky fill-in in `P`),
/// and applies exact orthonormal row-space reduction $\mathbf{w} = \mathbf{U}\mathbf{z}$ whenever
/// the active constraint count is $\le \frac{2}{3}d$.
#[allow(clippy::too_many_arguments)]
pub fn solve_conic_linear_vector(
    rule_name: &str,
    on_cursor_bf16: &StateMatrix,
    targets: &[f64],
    bands: &[f64],
    off_cursor_ceiling: Option<(&StateMatrix, f64)>,
    q_preserve: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Result<Vec<f64>, V2CompilerError> {
    let n_on = on_cursor_bf16.nrows();
    let d = on_cursor_bf16.hidden_dim();
    if targets.len() != n_on || bands.len() != n_on {
        return Err(V2CompilerError::DimensionMismatch {
            context: "solve_conic_linear_vector(targets/bands)",
            expected: n_on,
            actual: targets.len().min(bands.len()),
        });
    }
    if let Some((neg_mat, _)) = off_cursor_ceiling {
        if neg_mat.hidden_dim() != d {
            return Err(V2CompilerError::DimensionMismatch {
                context: "solve_conic_linear_vector(off_cursor)",
                expected: d,
                actual: neg_mat.hidden_dim(),
            });
        }
    }
    if let Some(qp) = q_preserve {
        if qp.hidden_dim() != d {
            return Err(V2CompilerError::DimensionMismatch {
                context: "solve_conic_linear_vector(q_preserve)",
                expected: d,
                actual: qp.hidden_dim(),
            });
        }
    }

    if let Some((neg_mat, ceiling)) = off_cursor_ceiling {
        if neg_mat.nrows() > ACTIVE_SET_MIN_NEG_ROWS {
            return solve_with_exact_active_set(
                rule_name,
                on_cursor_bf16,
                targets,
                bands,
                neg_mat,
                ceiling,
                q_preserve,
                config,
            );
        }
    }

    solve_single_conic_pass(
        rule_name,
        on_cursor_bf16,
        targets,
        bands,
        off_cursor_ceiling,
        q_preserve,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_single_conic_pass(
    rule_name: &str,
    on_cursor_bf16: &StateMatrix,
    targets: &[f64],
    bands: &[f64],
    off_cursor_ceiling: Option<(&StateMatrix, f64)>,
    q_preserve: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Result<Vec<f64>, V2CompilerError> {
    let n_on = on_cursor_bf16.nrows();
    let d = on_cursor_bf16.hidden_dim();

    let reduced = try_reduce_to_orthonormal_row_space(
        on_cursor_bf16,
        off_cursor_ceiling,
        q_preserve,
        config.nullspace_policy,
    )?;

    let (on_eff, off_eff, p_dim) = match &reduced {
        Some(red) => (
            &red.on_red,
            red.off_red.as_ref().map(|(m, c)| (m, *c)),
            red.rank(),
        ),
        None => (on_cursor_bf16, off_cursor_ceiling, d),
    };

    let all_same_target = targets
        .first()
        .is_some_and(|&t0| targets.iter().all(|&t| (t - t0).abs() < 1e-9));
    let use_cov = all_same_target && n_on > 1 && config.gamma_within_cov > 0.0;
    let n_cov = if use_cov { n_on } else { 0 };
    let cov_scale = if use_cov {
        (config.gamma_within_cov / (n_on as f64)).sqrt()
    } else {
        0.0
    };
    let mu_eff = if use_cov {
        on_eff.mean_row()
    } else {
        Vec::new()
    };

    let q_rank = q_preserve.map_or(0, OrthonormalSubspace::rank);
    let strict_eq_rows = match (&reduced, config.nullspace_policy) {
        (None, NullspacePolicy::StrictZeroCone) => q_rank,
        _ => 0,
    };

    let n_vars = p_dim + n_on + n_cov;
    let mut p_diag = Vec::with_capacity(n_vars);
    p_diag.resize(p_dim, config.lambda_ridge);
    p_diag.resize(p_dim + n_on, 1200.0);
    p_diag.resize(n_vars, 1.0);
    let p_csc = build_diagonal_csc(&p_diag, 1e-15);

    let n_eq = n_on + n_cov + strict_eq_rows;
    let n_neg = off_eff.map_or(0, |(m, _)| m.nrows());
    let n_ineq = 2 * n_on + n_neg;
    let n_trust_soc = p_dim + 1;
    let n_slack_soc = match config.nullspace_policy {
        NullspacePolicy::BoundedSocSlack { .. } if q_rank > 0 => q_rank + 1,
        _ => 0,
    };

    let ineq0 = n_eq;
    let soc0 = n_eq + n_ineq;
    let s0 = soc0 + n_trust_soc;
    let total_rows = s0 + n_slack_soc;

    let mut b_vec = vec![0.0_f64; total_rows];
    b_vec[..n_on].copy_from_slice(targets);
    for i in 0..n_on {
        b_vec[ineq0 + i] = bands[i];
        b_vec[ineq0 + n_on + i] = bands[i];
    }
    if let Some((_, ceiling)) = off_eff {
        for r in 0..n_neg {
            b_vec[ineq0 + 2 * n_on + r] = ceiling;
        }
    }
    b_vec[soc0] = config.trust_radius;
    if let NullspacePolicy::BoundedSocSlack { max_l2_leak } = config.nullspace_policy {
        if q_rank > 0 {
            b_vec[s0] = max_l2_leak;
        }
    }

    let a_csc = build_row_sequential_csc(total_rows, n_vars, 1e-15, |push| {
        for i in 0..n_on {
            let row = on_eff.row(i);
            for (j, &val) in row.iter().enumerate() {
                push(i, j, val);
            }
            push(i, p_dim + i, -1.0);
        }
        if use_cov {
            for i in 0..n_on {
                let row = on_eff.row(i);
                let r_idx = n_on + i;
                for j in 0..p_dim {
                    push(r_idx, j, cov_scale * (row[j] - mu_eff[j]));
                }
                push(r_idx, p_dim + n_on + i, -1.0);
            }
        }
        if strict_eq_rows > 0 {
            let qp = q_preserve.expect("q_rank > 0");
            for k in 0..strict_eq_rows {
                let col = qp.basis_col(k);
                let r_idx = n_on + n_cov + k;
                for (j, &val) in col.iter().enumerate() {
                    push(r_idx, j, val);
                }
            }
        }
        for i in 0..n_on {
            push(ineq0 + i, p_dim + i, 1.0);
        }
        for i in 0..n_on {
            push(ineq0 + n_on + i, p_dim + i, -1.0);
        }
        if let Some((neg_mat, _)) = off_eff {
            for i in 0..n_neg {
                let row = neg_mat.row(i);
                let r_idx = ineq0 + 2 * n_on + i;
                for (j, &val) in row.iter().enumerate() {
                    push(r_idx, j, val);
                }
            }
        }
        for j in 0..p_dim {
            push(soc0 + 1 + j, j, -1.0);
        }
        if n_slack_soc > 0 {
            match &reduced {
                Some(red) => {
                    for (k, qrow) in red.q_slack_red.iter().enumerate() {
                        let r_idx = s0 + 1 + k;
                        for (j, &val) in qrow.iter().enumerate() {
                            push(r_idx, j, -val);
                        }
                    }
                }
                None => {
                    let qp = q_preserve.expect("q_rank > 0");
                    for k in 0..q_rank {
                        let col = qp.basis_col(k);
                        let r_idx = s0 + 1 + k;
                        for (j, &val) in col.iter().enumerate() {
                            push(r_idx, j, -val);
                        }
                    }
                }
            }
        }
    });

    let mut cones = vec![
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(n_ineq),
        SupportedConeT::SecondOrderConeT(n_trust_soc),
    ];
    if n_slack_soc > 0 {
        cones.push(SupportedConeT::SecondOrderConeT(n_slack_soc));
    }

    let q_lin = vec![0.0_f64; n_vars];
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(400)
        .tol_gap_abs(1e-8)
        .tol_gap_rel(1e-8)
        .tol_feas(1e-8)
        .build()
        .expect("valid Clarabel settings");

    let mut solver = DefaultSolver::new(&p_csc, &q_lin, &a_csc, &b_vec, &cones, settings);
    solver.solve();

    let status = solver.solution.status;
    if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        return Err(V2CompilerError::InfeasibleDiode {
            rule: rule_name.to_string(),
            status: format!("{status:?}"),
            min_pos_gate_bf16: f64::NAN,
            req_min_pos: config.g_pos_target - config.g_pos_band,
            max_neg_gate_bf16: f64::NAN,
            req_max_neg: config.g_neg_ceiling,
        });
    }

    let z_sol = &solver.solution.x[0..p_dim];
    let mut w_sol = match &reduced {
        Some(red) => red.lift_to_full(z_sol, d),
        None => z_sol.to_vec(),
    };
    if matches!(config.nullspace_policy, NullspacePolicy::StrictZeroCone) {
        if let Some(qp) = q_preserve {
            w_sol = qp.project_out(&w_sol);
        }
    }
    Ok(w_sol)
}
