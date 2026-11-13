//! `AlphaEdit+` SOC Slack, `DeltaNet` Zero-`\beta` Scrubber, & `WireCartridgeManifestJson` Wire Roundtrip.

use portia_compiler::concolic::ConcolicMapElitesArchive;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::{compile_intent_program, ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::StateMatrix;
use portia_compiler::solvers::clarabel_ternary::{compile_coupled_ternary_pair, CursorPolarity};
use portia_compiler::solvers::deltanet_scrub::{
    compile_deltanet_head_partition_scrubber, DeltaNetGateLayer,
};
use portia_compiler::solvers::V2SolverConfig;
use portia_compiler::wire::WireCartridgeManifestJson;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy};
use std::collections::BTreeMap;

#[test]
fn test_alphaedit_plus_slack_and_deltanet_zero_beta_and_v1_wire_roundtrip() {
    let d = 48;
    let mut q_conflict_row = vec![0.0; d];
    q_conflict_row[0] = 1.0;
    let q_preserve = OrthonormalSubspace::from_row_space_svd(
        "conflict",
        &StateMatrix::from_rows("q", &[q_conflict_row]).unwrap(),
        1e-6,
        1,
    )
    .unwrap();

    let mut xp = vec![0.0; d];
    xp[0] = 3.0;
    xp[4] = 2.0;
    let mut xn = vec![0.0; d];
    xn[0] = -3.0;
    xn[4] = 2.0;
    let on_m = StateMatrix::from_rows("on", &[xp.clone()]).unwrap();
    let off_m = StateMatrix::from_rows("off", &[xn]).unwrap();
    let mut h = vec![0.0; d];
    h[2] = 10.0;
    let res_m = StateMatrix::from_rows("res", &[h]).unwrap();
    let mut tdir = vec![0.0; d];
    tdir[3] = 1.0;

    let strict_res = compile_coupled_ternary_pair(
        "conflict_rule",
        0,
        1,
        &on_m,
        &[CursorPolarity::PlusOne],
        &res_m,
        &off_m,
        &tdir,
        ChordalAngleRad::new(0.25),
        Some(&q_preserve),
        None,
        &V2SolverConfig::default(),
    );
    assert!(
        strict_res.is_err(),
        "StrictZeroCone must reject pure conflict"
    );

    let slack_cfg = V2SolverConfig::dynamic_span_with_slack(4.5);
    let slack_res = compile_coupled_ternary_pair(
        "conflict_rule",
        0,
        1,
        &on_m,
        &[CursorPolarity::PlusOne],
        &res_m,
        &off_m,
        &tdir,
        ChordalAngleRad::new(0.25),
        Some(&q_preserve),
        None,
        &slack_cfg,
    )
    .expect("AlphaEdit+ BoundedSocSlack must solve cleanly");
    assert!(slack_res.preserve_leak_gate_bf16 <= 4.55);

    let mut wa = vec![vec![0.0; d]; 4];
    let mut wb = vec![vec![0.0; d]; 4];
    for h_i in 0..4 {
        wa[h_i][10 + h_i] = 1.5;
        wb[h_i][20 + h_i] = 1.5;
    }
    let dn = DeltaNetGateLayer {
        layer_idx: 18,
        hidden_dim: d,
        num_heads: 4,
        input_norm_scale: vec![1.0; d],
        w_a_proj: StateMatrix::from_rows("wa", &wa).unwrap(),
        w_b_proj: StateMatrix::from_rows("wb", &wb).unwrap(),
    };
    let scrub = compile_deltanet_head_partition_scrubber(
        "scrub_test",
        0,
        &dn,
        &[0, 1],
        &[2],
        4.0,
        &on_m,
        &off_m,
        None,
        &slack_cfg,
    )
    .unwrap();
    assert!(scrub.min_flush_head_da > 1.5);
    assert!(scrub.max_lock_head_da < -0.5);
    assert!(scrub.max_beta_crosstalk < 1e-2);

    let r_high = IntentExpr::rule(IntentRule {
        name: "probe_high".to_string(),
        guard: RegimeExpr::ConstBool(true),
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
        grammar: GrammarCartridgeSpec::LiteralSet(vec![" PROBE".to_string()]),
        kind: HardwareIntentKind::StanceRoute {
            layer_idx: 22,
            raw_target_dir: tdir.clone(),
            angle: ChordalAngleRad::new(0.3),
            expected_mode: Some(Mode::Probe),
        },
    });
    let mut xp2 = vec![0.0; d];
    xp2[1] = 3.0;
    let r_low = IntentExpr::rule(IntentRule {
        name: "read_low".to_string(),
        guard: RegimeExpr::ConstBool(true),
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
        grammar: GrammarCartridgeSpec::Unconstrained,
        kind: HardwareIntentKind::StanceRoute {
            layer_idx: 22,
            raw_target_dir: tdir,
            angle: ChordalAngleRad::new(0.3),
            expected_mode: Some(Mode::Read),
        },
    });

    let program_expr = r_high | r_low;
    let mut caps = BTreeMap::new();
    caps.insert(
        "probe_high".to_string(),
        RuleCaptureData::standard_positive(
            on_m.clone(),
            res_m.clone(),
            off_m.clone(),
            V2SolverConfig::default(),
        ),
    );
    caps.insert(
        "read_low".to_string(),
        RuleCaptureData::standard_positive(
            StateMatrix::from_rows("on2", &[xp2]).unwrap(),
            res_m,
            off_m,
            V2SolverConfig::default(),
        ),
    );

    let geom = ModelGeometryTables {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        widened_slot_capacity: 64,
        ..Default::default()
    };
    let compiled = compile_intent_program(&program_expr, &caps, &geom).unwrap();
    assert_eq!(compiled.allocated[0].rel_slot_lo, 0);
    assert_eq!(compiled.allocated[1].rel_slot_lo, 2);

    let low_entry = &compiled.manifest.banks["OrientTrunk"]["22"][1];
    let g_low_on_high: f64 = low_entry.w_gate[0]
        .iter()
        .zip(&xp)
        .map(|(&w, &x)| (w as f64) * x)
        .sum();
    assert!(g_low_on_high <= -8.0 + 1e-2);

    let json_str = serde_json::to_string(&compiled.manifest).unwrap();
    let parsed = WireCartridgeManifestJson::from_json_str(&json_str).unwrap();
    for entries in parsed.banks["OrientTrunk"].values() {
        for e in entries {
            e.verify_dimensions(d).unwrap();
        }
    }

    let lens = &compiled.concolic_lenses["probe_high"];
    let mut archive = ConcolicMapElitesArchive::new();
    let mut near_break = vec![0.0; d];
    near_break[0] = -0.2;
    near_break[5] = 1.0;
    assert!(archive.observe(lens, 0.5, "adversarial_mutation_1", &near_break));
    assert!(archive.elites_sorted_by_break_proximity()[0].cliff_margin < 0.0);
}
