//! Differential bisimulation & determinism tests for:
//! 1. Diagonal-`P` + Orthonormal Row-Space (`w = U z`) `solve_conic_linear_vector` vs. reference
//!    dense-`P` full-space `Clarabel` across both reduced (`m_cand <= 2d/3`) and full-space (`m_cand > 2d/3`) regimes,
//! 2. Multi-core `std::thread::scope` deterministic `WireCartridgeManifestJson` emission in `lowering`.

use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::{compile_intent_program, ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{l2_norm, StateMatrix};
use portia_compiler::solvers::conic_qp::solve_conic_linear_vector;
use portia_compiler::solvers::{dense_row_major_to_csc, NullspacePolicy, V2SolverConfig};
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy};
use proptest::prelude::*;
use std::collections::BTreeMap;

/// Reference unreduced dense-`P` solver matching the historical KKT layout bit-for-bit,
/// used strictly as an independent differential oracle in property tests.
fn reference_dense_p_solve(
    on_cursor_bf16: &StateMatrix,
    targets: &[f64],
    bands: &[f64],
    off_cursor_ceiling: Option<(&StateMatrix, f64)>,
    q_preserve: Option<&OrthonormalSubspace>,
    config: &V2SolverConfig,
) -> Option<Vec<f64>> {
    let (n_on, d) = (on_cursor_bf16.nrows(), on_cursor_bf16.hidden_dim());
    let n_vars = d + n_on;
    let mut p_dense = vec![0.0_f64; n_vars * n_vars];
    for j in 0..d {
        p_dense[j * n_vars + j] = config.lambda_ridge;
    }
    let all_same = targets
        .first()
        .is_some_and(|&t0| targets.iter().all(|&t| (t - t0).abs() < 1e-9));
    if all_same && n_on > 1 {
        let mu = on_cursor_bf16.mean_row();
        let scale = config.gamma_within_cov / (n_on as f64);
        for r in 0..n_on {
            let row = on_cursor_bf16.row(r);
            for i in 0..d {
                let di = row[i] - mu[i];
                for j in i..d {
                    p_dense[i * n_vars + j] += scale * di * (row[j] - mu[j]);
                }
            }
        }
    }
    for i in 0..n_on {
        p_dense[(d + i) * n_vars + (d + i)] = 1200.0;
    }

    let q_rank = q_preserve.map_or(0, OrthonormalSubspace::rank);
    let strict_eq = match config.nullspace_policy {
        NullspacePolicy::StrictZeroCone => q_rank,
        NullspacePolicy::BoundedSocSlack { .. } => 0,
    };
    let n_eq = n_on + strict_eq;
    let n_neg = off_cursor_ceiling.map_or(0, |(m, _)| m.nrows());
    let n_ineq = 2 * n_on + n_neg;
    let n_trust = d + 1;
    let n_slack = match config.nullspace_policy {
        NullspacePolicy::BoundedSocSlack { .. } if q_rank > 0 => q_rank + 1,
        _ => 0,
    };
    let total_rows = n_eq + n_ineq + n_trust + n_slack;
    let mut a_dense = vec![0.0_f64; total_rows * n_vars];
    let mut b_vec = vec![0.0_f64; total_rows];

    for i in 0..n_on {
        for (j, &v) in on_cursor_bf16.row(i).iter().enumerate() {
            a_dense[i * n_vars + j] = v;
        }
        a_dense[i * n_vars + (d + i)] = -1.0;
        b_vec[i] = targets[i];
    }
    if strict_eq > 0 {
        let qp = q_preserve.unwrap();
        for k in 0..strict_eq {
            for (j, &v) in qp.basis_col(k).iter().enumerate() {
                a_dense[(n_on + k) * n_vars + j] = v;
            }
        }
    }
    let ineq0 = n_eq;
    for i in 0..n_on {
        a_dense[(ineq0 + i) * n_vars + (d + i)] = 1.0;
        b_vec[ineq0 + i] = bands[i];
        a_dense[(ineq0 + n_on + i) * n_vars + (d + i)] = -1.0;
        b_vec[ineq0 + n_on + i] = bands[i];
    }
    if let Some((neg_mat, ceiling)) = off_cursor_ceiling {
        for i in 0..n_neg {
            for (j, &v) in neg_mat.row(i).iter().enumerate() {
                a_dense[(ineq0 + 2 * n_on + i) * n_vars + j] = v;
            }
            b_vec[ineq0 + 2 * n_on + i] = ceiling;
        }
    }
    let soc0 = n_eq + n_ineq;
    b_vec[soc0] = config.trust_radius;
    for j in 0..d {
        a_dense[(soc0 + 1 + j) * n_vars + j] = -1.0;
    }
    if let NullspacePolicy::BoundedSocSlack { max_l2_leak } = config.nullspace_policy {
        if q_rank > 0 {
            let qp = q_preserve.unwrap();
            let s0 = soc0 + n_trust;
            b_vec[s0] = max_l2_leak;
            for k in 0..q_rank {
                for (j, &v) in qp.basis_col(k).iter().enumerate() {
                    a_dense[(s0 + 1 + k) * n_vars + j] = -v;
                }
            }
        }
    }

    let mut cones = vec![
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(n_ineq),
        SupportedConeT::SecondOrderConeT(n_trust),
    ];
    if n_slack > 0 {
        cones.push(SupportedConeT::SecondOrderConeT(n_slack));
    }
    let p_csc = dense_row_major_to_csc(n_vars, n_vars, &p_dense, true, 1e-15);
    let a_csc = dense_row_major_to_csc(total_rows, n_vars, &a_dense, false, 1e-15);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(400)
        .tol_gap_abs(1e-8)
        .tol_gap_rel(1e-8)
        .tol_feas(1e-8)
        .build()
        .unwrap();
    let mut solver =
        DefaultSolver::new(&p_csc, &vec![0.0; n_vars], &a_csc, &b_vec, &cones, settings);
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return None;
    }
    let mut w = solver.solution.x[0..d].to_vec();
    if matches!(config.nullspace_policy, NullspacePolicy::StrictZeroCone) {
        if let Some(qp) = q_preserve {
            w = qp.project_out(&w);
        }
    }
    Some(w)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_sparse_diagonal_and_subspace_solver_matches_dense_reference(
        d in prop_oneof![Just(32usize), Just(96usize), Just(160usize), Just(220usize)],
        n_on in 2usize..8usize,
        n_neg in prop_oneof![Just(0usize), Just(6usize), Just(120usize)],
        q_rank in 0usize..6usize,
        use_slack in any::<bool>(),
        vary_targets in any::<bool>(),
        seed in 0u64..10_000u64,
    ) {
        let mk = |rows: usize, active: Option<f64>, salt: u64| -> StateMatrix {
            let mut flat = Vec::with_capacity(rows * d);
            for r in 0..rows {
                for c in 0..d {
                    let noise = 0.05 * ((((r as u64) * 31 + (c as u64) * 17 + seed + salt) % 41) as f64 - 20.0);
                    flat.push(match (c, active) {
                        (0, _) => 4.5,
                        (1, Some(amp)) => amp + 0.04 * (r as f64),
                        (2, Some(_)) if vary_targets => if r % 2 == 0 { 2.5 } else { -2.5 },
                        _ => noise,
                    });
                }
            }
            StateMatrix::try_new("prop_mat", rows, d, flat).unwrap().quantize_bf16()
        };

        let on_m = mk(n_on, Some(4.2), 11);
        let off_m = (n_neg > 0).then(|| mk(n_neg, None, 29));
        let qp = (q_rank > 0).then(|| {
            let qrows: Vec<Vec<f64>> = (0..q_rank).map(|k| {
                let mut v = vec![0.0; d];
                v[4 + k] = 1.0;
                v[6 + k] = 0.25;
                v
            }).collect();
            OrthonormalSubspace::from_row_space_svd("qp", &StateMatrix::from_rows("qp", &qrows).unwrap(), 1e-6, q_rank).unwrap()
        });

        let targets: Vec<f64> = (0..n_on).map(|r| if vary_targets { if r % 2 == 0 { 1.0 } else { -1.0 } } else { 3.2 }).collect();
        let bands = vec![0.20; n_on];
        let cfg = if use_slack {
            V2SolverConfig::dynamic_span_with_slack(0.35)
        } else {
            V2SolverConfig::default()
        };
        let off_arg = off_m.as_ref().map(|m| (m, cfg.g_neg_ceiling - cfg.bf16_neg_pad));

        let ref_sol = reference_dense_p_solve(&on_m, &targets, &bands, off_arg, qp.as_ref(), &cfg);
        let fast_sol = solve_conic_linear_vector("prop_rule", &on_m, &targets, &bands, off_arg, qp.as_ref(), &cfg).ok();

        prop_assert_eq!(ref_sol.is_some(), fast_sol.is_some());
        if let (Some(w_ref), Some(w_fast)) = (ref_sol, fast_sol) {
            let diff: Vec<f64> = w_ref.iter().zip(&w_fast).map(|(a, b)| a - b).collect();
            let rel_err = l2_norm(&diff) / l2_norm(&w_ref).max(1e-9);
            prop_assert!(rel_err < 2e-4, "rel_err={rel_err:.3e}");
        }
    }
}

#[test]
fn parallel_thread_scope_lowering_is_byte_exact_deterministic() {
    let d = 64usize;
    let mut caps = BTreeMap::new();
    let mut expr_opt: Option<IntentExpr> = None;
    let off_rows: Vec<Vec<f64>> = (0..24)
        .map(|r| {
            let mut v = vec![0.0; d];
            v[0] = 5.0;
            v[30 + (r % 16)] = 0.2;
            v
        })
        .collect();
    let off_m = StateMatrix::from_rows("off", &off_rows).unwrap();
    let res_m = StateMatrix::from_rows("res", &vec![vec![1.0; d]; 4]).unwrap();

    for idx in 0..10usize {
        let name = format!("par_rule_{idx:02}");
        let mut xp = vec![0.0; d];
        xp[0] = 5.0;
        xp[2 + idx] = 4.5;
        let on_m = StateMatrix::from_rows("on", &[xp.clone(), xp]).unwrap();
        let mut tdir = vec![0.0; d];
        tdir[16 + idx] = 1.0;
        caps.insert(
            name.clone(),
            RuleCaptureData::standard_positive(
                on_m,
                res_m.clone(),
                off_m.clone(),
                V2SolverConfig::default(),
            ),
        );
        let leaf = IntentExpr::rule(IntentRule {
            name,
            guard: RegimeExpr::ConstBool(true),
            mount_policy: if idx < 3 {
                MountPolicy::PermanentStatic
            } else {
                MountPolicy::DynamicBank(CartridgeBank::OrientTrunk)
            },
            grammar: GrammarCartridgeSpec::Unconstrained,
            kind: HardwareIntentKind::StanceRoute {
                layer_idx: 22,
                raw_target_dir: tdir,
                angle: ChordalAngleRad::new(0.30),
                expected_mode: Some(Mode::Read),
            },
        });
        expr_opt = Some(expr_opt.map_or(leaf.clone(), |acc| acc | leaf));
    }

    let expr = expr_opt.unwrap();
    let geom = ModelGeometryTables {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        widened_slot_capacity: 64,
        ..Default::default()
    };
    let baseline = compile_intent_program(&expr, &caps, &geom)
        .unwrap()
        .manifest
        .to_json_pretty()
        .unwrap();
    for _ in 0..6 {
        let again = compile_intent_program(&expr, &caps, &geom)
            .unwrap()
            .manifest
            .to_json_pretty()
            .unwrap();
        assert_eq!(
            baseline, again,
            "parallel lowering must be byte-exact deterministic"
        );
    }
}
