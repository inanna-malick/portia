//! Property-Based & Structural Invariant Suite for `portia_nursery::v2_ternary_hylo`
//! (`tests/v2_ternary_hylo_suite.rs`, strictly `<= 350` lines).

use portia_compiler::lowering::{compile_intent_program, ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::{dot_f64, slice_to_bf16_vec, StateMatrix};
use portia_compiler::solvers::deltanet_scrub::compile_deltanet_head_partition_scrubber;
use portia_compiler::solvers::V2SolverConfig;
use portia_compiler::wire::CompiledSlotBf16;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, MountPolicy};
use portia_nursery::v2_ternary_hylo::{
    build_deltanet_rollback_scrub_intent, build_rust_ast_ternary_world,
    build_signed_ternary_fork_intent, build_ternary_fork_capture_data, classify_rust_todo_seam,
    compose_hylo_scrub_then_ternary_fork, evaluate_compiled_ternary_slots,
    evaluate_rollback_arm_trajectory, partition_deltanet_heads_by_retention,
    synthesize_convex_midpoint_states, DeltaNetRollbackScrubSpec, HyloTernaryBranch,
    Qwen35GatedDeltaNetLayer, SignedTernaryForkSpec,
};
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

fn walk_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if !dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            walk_rs_files(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

#[test]
fn test_v2_ternary_hylo_all_files_strictly_under_350_loc() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut check_files = vec![
        manifest_dir.join("src/v2_ternary_hylo.rs"),
        manifest_dir.join("tests/v2_ternary_hylo_suite.rs"),
    ];
    walk_rs_files(&manifest_dir.join("src/v2_ternary_hylo"), &mut check_files);
    assert!(check_files.len() >= 6);
    for f in check_files {
        let lines = std::fs::read_to_string(&f)
            .expect("read rs")
            .lines()
            .count();
        assert!(
            lines <= 350,
            "350-LOC law violated: {} has {} lines",
            f.display(),
            lines
        );
    }
}

#[test]
fn test_non_causal_deltanet_hop_is_rejected_by_register_allocator() {
    let bad_spec = DeltaNetRollbackScrubSpec {
        rule_name: "bad_non_causal_scrub".to_string(),
        upstream_mlp_layer: 18,
        target_deltanet_layer: 18,
        flush_heads: vec![1, 3],
        lock_heads: vec![0, 2],
        a_drive: 4.5,
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
        guard: RegimeExpr::ConstBool(true),
    };
    let expr = build_deltanet_rollback_scrub_intent(&bad_spec);
    let geom = ModelGeometryTables {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        widened_slot_capacity: 64,
        ..Default::default()
    };
    let res = compile_intent_program(&expr, &BTreeMap::new(), &geom);
    assert!(
        res.is_err(),
        "upstream_mlp_layer >= target_deltanet_layer must fail"
    );
}

fn build_synthetic_deltanet_layer(seed_scale: f64) -> Qwen35GatedDeltaNetLayer {
    let d = 48usize;
    let h = 4usize;
    let dk = 8usize;
    let dv = 8usize;
    let conv_dim = 2 * h * dk + h * dv; // 192
    let z_dim = h * dv; // 32

    let qkv_data: Vec<f64> = (0..conv_dim * d)
        .map(|i| (((i * 19 + 7) % 61) as f64 - 30.0) * 0.015 * seed_scale)
        .collect();
    let mut conv_data = vec![0.0; conv_dim * 4];
    for c in 0..conv_dim {
        conv_data[c * 4] = 0.05;
        conv_data[c * 4 + 1] = 0.12;
        conv_data[c * 4 + 2] = 0.35; // lag-1 bleed!
        conv_data[c * 4 + 3] = 0.78;
    }
    let z_data: Vec<f64> = (0..z_dim * d)
        .map(|i| {
            (((i * 13 + 3) % 43) as f64 - 20.0) * 0.02 + if i % (d + 1) == 0 { 0.4 } else { 0.0 }
        })
        .collect();
    let mut wa_rows = vec![vec![0.0; d]; h];
    let mut wb_rows = vec![vec![0.0; d]; h];
    for hi in 0..h {
        wa_rows[hi][8 + hi] = 1.6;
        wb_rows[hi][16 + hi] = 1.6;
    }
    let out_data: Vec<f64> = (0..d * z_dim)
        .map(|i| (((i * 29 + 11) % 53) as f64 - 26.0) * 0.03)
        .collect();

    Qwen35GatedDeltaNetLayer {
        layer_idx: 18,
        hidden_dim: d,
        num_heads: h,
        head_k_dim: dk,
        head_v_dim: dv,
        conv_kernel_size: 4,
        input_norm_scale: vec![1.0; d],
        in_proj_qkv: StateMatrix::try_new("qkv", conv_dim, d, qkv_data).unwrap(),
        conv1d_weight: StateMatrix::try_new("conv", conv_dim, 4, conv_data).unwrap(),
        in_proj_z: StateMatrix::try_new("z", z_dim, d, z_data).unwrap(),
        in_proj_a: StateMatrix::from_rows("wa", &wa_rows).unwrap(),
        in_proj_b: StateMatrix::from_rows("wb", &wb_rows).unwrap(),
        // Heads 0, 2 are prompt-anchors (dt_bias = -5.5 -> alpha0 ~ 0.997); heads 1, 3 are scratchpad (dt_bias = +0.4 -> alpha0 ~ 0.51):
        a_log: vec![-0.3, -0.3, -0.3, -0.3],
        dt_bias: vec![-5.5, 0.4, -5.5, 0.4],
        gated_norm_weight: vec![1.0; dv],
        out_proj: StateMatrix::try_new("out", d, z_dim, out_data).unwrap(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_ast_seam_classifier_and_world_generator_preserve_polarity(
        branch_idx in 0usize..3,
        variant_idx in 0usize..128,
    ) {
        let branch = HyloTernaryBranch::ALL[branch_idx];
        let world = build_rust_ast_ternary_world(branch, variant_idx).expect("valid AST world");
        let receipt = classify_rust_todo_seam(&world.rust_source).expect("seam classification");
        prop_assert_eq!(receipt.branch, branch);
        prop_assert_eq!(receipt.polarity, branch.to_cursor_polarity());
        match branch {
            HyloTernaryBranch::Fork => {
                prop_assert_eq!(receipt.mut_coupled_hole_count, 0);
                prop_assert!(receipt.decoupled_hole_count >= 2);
                prop_assert!(!receipt.has_underspecified_invariant);
            }
            HyloTernaryBranch::Inline => {
                prop_assert!(receipt.mut_coupled_hole_count >= 1);
                prop_assert!(!receipt.has_underspecified_invariant);
            }
            HyloTernaryBranch::Ask => {
                let midpoint_mix = receipt.mut_coupled_hole_count > 0 && receipt.decoupled_hole_count > 0;
                prop_assert!(receipt.has_underspecified_invariant || midpoint_mix);
            }
        }
    }

    #[test]
    fn prop_convex_midpoint_ternary_fork_solves_in_2_slots_where_linear_diode_fails(
        on_amp in 2.4f64..4.2f64,
        branch_sep in 2.0f64..3.8f64,
        lam in 0.47f64..0.53f64,
        angle in 0.20f64..0.45f64,
    ) {
        let d = 48usize;
        let mut x_fork = vec![0.0; d];
        x_fork[0] = on_amp;
        x_fork[1] = branch_sep;
        let mut x_inline = vec![0.0; d];
        x_inline[0] = on_amp;
        x_inline[1] = -branch_sep;

        let fork_m = StateMatrix::from_rows("fork", &[x_fork.clone()]).unwrap();
        let inline_m = StateMatrix::from_rows("inline", &[x_inline.clone()]).unwrap();
        let ask_m = synthesize_convex_midpoint_states(&fork_m, &inline_m, &[lam]).unwrap();
        let x_ask = ask_m.row(0).to_vec();

        // Mathematical impossibility witness for v1 independent ASK gate clamping FORK & INLINE <= -8:
        let any_linear_gate = vec![-3.0; d];
        if dot_f64(&any_linear_gate, &x_fork) <= -8.0 && dot_f64(&any_linear_gate, &x_inline) <= -8.0 {
            prop_assert!(dot_f64(&any_linear_gate, &x_ask) <= -8.0 + 1e-9);
        }

        let mut off1 = vec![0.0; d];
        off1[0] = -on_amp;
        off1[2] = 1.5;
        let mut off2 = vec![0.0; d];
        off2[0] = -0.8 * on_amp;
        off2[3] = -1.7;
        let off_m = StateMatrix::from_rows("off", &[off1.clone(), off2]).unwrap();

        let mut h_res = vec![0.0; d];
        h_res[4] = 16.0;
        let res_m = StateMatrix::from_rows("res", &[h_res.clone(), h_res.clone(), h_res]).unwrap();

        let mut axis_dir = vec![0.0; d];
        axis_dir[5] = 1.0;

        let spec = SignedTernaryForkSpec::orient_trunk_default("ternary_hylo", 22, axis_dir, angle);
        let expr = build_signed_ternary_fork_intent(&spec);
        let cap = build_ternary_fork_capture_data(
            &fork_m,
            &ask_m,
            &inline_m,
            &res_m,
            &off_m,
            None,
            V2SolverConfig::default(),
        )
        .unwrap();

        let mut caps = BTreeMap::new();
        caps.insert("ternary_hylo".to_string(), cap);
        let geom = ModelGeometryTables {
            model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
            base_intermediate_size: 3584,
            widened_slot_capacity: 64,
            ..Default::default()
        };
        let compiled = compile_intent_program(&expr, &caps, &geom).unwrap();
        prop_assert_eq!(compiled.allocated.len(), 1);
        prop_assert_eq!(compiled.allocated[0].rel_slot_hi - compiled.allocated[0].rel_slot_lo, 2);
        prop_assert!(compiled.manifest.bisimulation_receipt.as_ref().unwrap().bisimulation_holds);

        let wire_entry = &compiled.manifest.banks["OrientTrunk"]["22"][0];
        let to_bf16_slot = |idx: usize| CompiledSlotBf16 {
            slot_offset: idx,
            w_gate: slice_to_bf16_vec(&wire_entry.w_gate[idx].iter().map(|&v| v as f64).collect::<Vec<_>>()),
            w_up: slice_to_bf16_vec(&wire_entry.w_up[idx].iter().map(|&v| v as f64).collect::<Vec<_>>()),
            w_down: slice_to_bf16_vec(&(0..d).map(|r| wire_entry.w_down[r][idx] as f64).collect::<Vec<_>>()),
        };
        let rad = to_bf16_slot(0);
        let tan = to_bf16_slot(1);

        let ev_fork = evaluate_compiled_ternary_slots(&rad, &tan, &x_fork, 0.5, 0.25);
        let ev_ask = evaluate_compiled_ternary_slots(&rad, &tan, &x_ask, 0.5, 0.25);
        let ev_inline = evaluate_compiled_ternary_slots(&rad, &tan, &x_inline, 0.5, 0.25);
        let ev_off = evaluate_compiled_ternary_slots(&rad, &tan, &off1, 0.5, 0.25);

        prop_assert_eq!(ev_fork.predicted_branch, Some(HyloTernaryBranch::Fork));
        prop_assert_eq!(ev_ask.predicted_branch, Some(HyloTernaryBranch::Ask));
        prop_assert_eq!(ev_inline.predicted_branch, Some(HyloTernaryBranch::Inline));
        prop_assert_eq!(ev_off.predicted_branch, None);
        prop_assert!(ev_off.gate_pre_silu <= -7.95);
    }

    #[test]
    fn prop_deltanet_scrubber_flushes_scratchpad_with_zero_beta_crosstalk(
        a_drive in 4.0f64..8.0f64,
        retries in 1usize..=4,
    ) {
        let layer = build_synthetic_deltanet_layer(1.0);
        let (flush_heads, lock_heads) =
            partition_deltanet_heads_by_retention(&layer.a_log, &layer.dt_bias, 0.95);
        prop_assert_eq!(&flush_heads, &vec![1, 3]);
        prop_assert_eq!(&lock_heads, &vec![0, 2]);

        let d = layer.hidden_dim;
        let (mut rb_x17, mut off_x17, mut off2_x17) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
        rb_x17[0] = 3.2; rb_x17[2] = 1.5;
        off_x17[0] = -2.5; off_x17[1] = 3.2;
        off2_x17[0] = -2.0; off2_x17[3] = -3.0;
        let on_m = StateMatrix::from_rows("rb_on", &[rb_x17.clone()]).unwrap();
        let off_m = StateMatrix::from_rows("rb_off", &[off_x17, off2_x17]).unwrap();

        let compiled_scrub = compile_deltanet_head_partition_scrubber(
            "rb_scrub", 0, &layer.gate_layer_view(), &flush_heads, &lock_heads,
            a_drive, &on_m, &off_m, None, &V2SolverConfig::default(),
        ).unwrap();
        prop_assert!(compiled_scrub.max_beta_crosstalk < 1e-2);
        prop_assert!(compiled_scrub.min_flush_head_da > 1.0);
        prop_assert!(compiled_scrub.max_lock_head_da < -0.3);

        // Also verify combined 1+2=3 slot IntentExpr composition (`scrub >> ternary_fork`):
        let scrub_spec = DeltaNetRollbackScrubSpec {
            rule_name: "rb_scrub".to_string(),
            upstream_mlp_layer: 17,
            target_deltanet_layer: 18,
            flush_heads: flush_heads.clone(),
            lock_heads: lock_heads.clone(),
            a_drive,
            mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
            guard: RegimeExpr::ConstBool(true),
        };
        let fork_spec = SignedTernaryForkSpec::orient_trunk_default("fork22", 22, vec![0.2; d], 0.25);
        let combined = compose_hylo_scrub_then_ternary_fork(&scrub_spec, &fork_spec);
        let mut caps = BTreeMap::new();
        let mut h_res = vec![0.0; d];
        h_res[4] = 10.0;
        let res1 = StateMatrix::from_rows("r1", &[h_res]).unwrap();
        caps.insert("rb_scrub".to_string(), RuleCaptureData::standard_positive(on_m.clone(), res1.clone(), off_m.clone(), V2SolverConfig::default()));
        caps.insert("fork22".to_string(), RuleCaptureData::standard_positive(on_m, res1, off_m, V2SolverConfig::default()));
        let mut geom = ModelGeometryTables {
            model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
            base_intermediate_size: 3584,
            widened_slot_capacity: 64,
            ..Default::default()
        };
        geom.deltanet_layers.insert(18, layer.gate_layer_view());
        let prog = compile_intent_program(&combined, &caps, &geom).unwrap();
        prop_assert_eq!(prog.allocated[0].rel_slot_hi - prog.allocated[0].rel_slot_lo, 1);
        prop_assert_eq!(prog.allocated[1].rel_slot_hi - prog.allocated[1].rel_slot_lo, 2);

        // Compare 4 rollback arms over `retries` failed draft chunks:
        let mut prompt_h: Vec<Vec<f64>> = (0..4)
            .map(|t| (0..d).map(|j| if (16..20).contains(&j) { 2.0 } else { ((t + j) as f64 * 0.4).sin() }).collect())
            .collect();
        for _ in 0..6 {
            let mut pad = vec![0.1; d];
            for v in &mut pad[16..20] { *v = -2.5; }
            prompt_h.push(pad);
        }
        let draft_chunks: Vec<Vec<Vec<f64>>> = (0..retries)
            .map(|r| (0..3).map(|t| (0..d).map(|j| {
                if j == 9 || j == 11 { -2.8 } else if (16..20).contains(&j) { 2.6 } else { ((r * 7 + t * 3 + j) as f64).cos() * 1.8 }
            }).collect()).collect())
            .collect();
        let mut base_rb_h18 = vec![0.0; d];
        base_rb_h18[0..6].fill(1.6);
        base_rb_h18[16..20].fill(-2.8);
        let rb_tokens: Vec<(Vec<f64>, Vec<f64>)> = (0..retries).map(|_| (rb_x17.clone(), base_rb_h18.clone())).collect();
        let query_h: Vec<Vec<f64>> = (0..4).map(|q| prompt_h[q].clone()).collect();
        let all_heads: Vec<usize> = (0..layer.num_heads).collect();
        let naive_latch = layer.synthesize_naive_overwrite_latch_delta(&all_heads, 14.0).unwrap();

        let m_unscrub = evaluate_rollback_arm_trajectory(&layer, &prompt_h, &draft_chunks, &rb_tokens, &query_h, None, None, false, &flush_heads, &lock_heads).unwrap();
        let m_latch = evaluate_rollback_arm_trajectory(&layer, &prompt_h, &draft_chunks, &rb_tokens, &query_h, None, Some(&naive_latch), false, &flush_heads, &lock_heads).unwrap();
        let m_v2 = evaluate_rollback_arm_trajectory(&layer, &prompt_h, &draft_chunks, &rb_tokens, &query_h, Some(&compiled_scrub.slot), None, false, &flush_heads, &lock_heads).unwrap();
        let m_snap = evaluate_rollback_arm_trajectory(&layer, &prompt_h, &draft_chunks, &rb_tokens, &query_h, None, None, true, &flush_heads, &lock_heads).unwrap();

        prop_assert!(m_snap.rel_l2_err_vs_clean < 1e-12);
        prop_assert!(m_v2.flush_heads_frobenius_drift < m_unscrub.flush_heads_frobenius_drift);
        prop_assert!(m_v2.flush_heads_frobenius_drift < m_latch.flush_heads_frobenius_drift);
        prop_assert!(m_v2.lock_heads_cosine_retention > 0.85);
        prop_assert!(m_v2.lock_heads_cosine_retention > m_latch.lock_heads_cosine_retention);
    }
}
