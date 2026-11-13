//! Shrinkable `proptest` suite verifying Dual-Span (`OrientTrunk` + `ModePayload:Read`) `ASID`
//! grounding and register disjointness across arbitrary `OpaqueGibberishDungeonSpec` worlds.

use portia_compiler::discovery::gqa_bus::{
    synthesize_asid_codebook, GqaPointerLayer, MemorySegmentAsid,
};
use portia_compiler::ir::regalloc::allocate_neural_registers;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, normalize_unit_vec, rms_norm_slice, StateMatrix};
use portia_core::pocket_dsl::RegimeExpr;
use portia_sql::relational_ast::RelationalSchemaGraph;
use portia_sql::roguelite_dungeon::arb_opaque_gibberish_dungeon_spec;
use portia_sql::v2_pointer_bridge::{
    build_dual_span_plan_and_sql_pointer_program, synthesize_fk_passthrough_query_delta,
    SqlPointerBridgeConfig,
};
use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

fn build_realistic_gqa_fixture(seed: u64) -> (GqaPointerLayer, OrthonormalSubspace) {
    let (d, num_q_heads, num_kv_heads, head_dim, rotary_dim) = (128, 4, 1, 32, 8);
    let mut wq_rows = vec![vec![0.0; d]; 2 * num_q_heads * head_dim];
    let mut wk_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];
    let mut wv_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];

    let jitter = 0.05 * ((seed % 7) as f64 - 3.0);
    for i in 0..head_dim {
        wv_rows[i][i] = 1.35 + jitter;
        wv_rows[i][(i + 1) % head_dim] = 0.20;
        wk_rows[i][48 + i] = 1.25;
        for h in 0..num_q_heads {
            wq_rows[h * 2 * head_dim + i][48 + i] = 1.25;
            wq_rows[h * 2 * head_dim + head_dim + i][80 + h * 8 + (i % 8)] = 1.15;
        }
    }

    let mut wu_rows = vec![vec![0.0; d]; 16];
    for i in 0..16 {
        wu_rows[i][32 + i] = 2.0;
    }
    let q_wu = OrthonormalSubspace::from_row_space_svd(
        "wu_top",
        &StateMatrix::from_rows("wu", &wu_rows).unwrap(),
        1e-4,
        16,
    )
    .unwrap();

    let gqa = GqaPointerLayer {
        layer_idx: 15,
        hidden_dim: d,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        input_norm_scale: vec![1.0; d],
        q_norm_scale: vec![1.0; head_dim],
        k_norm_scale: vec![1.0; head_dim],
        w_q_proj: StateMatrix::from_rows("wq", &wq_rows).unwrap(),
        w_k_proj: StateMatrix::from_rows("wk", &wk_rows).unwrap(),
        w_v_proj: StateMatrix::from_rows("wv", &wv_rows).unwrap(),
    };
    (gqa, q_wu)
}

fn eval_head0_qk_logit(gqa: &GqaPointerLayer, h_q: &[f64], h_k: &[f64]) -> f64 {
    let qn_in = rms_norm_slice(h_q, &gqa.input_norm_scale, 1e-6);
    let kn_in = rms_norm_slice(h_k, &gqa.input_norm_scale, 1e-6);
    let q_raw: Vec<f64> = (0..gqa.head_dim)
        .map(|i| dot_f64(gqa.w_q_proj.row(i), &qn_in))
        .collect();
    let k_raw: Vec<f64> = (0..gqa.head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &kn_in))
        .collect();
    let q_vec = rms_norm_slice(&q_raw, &gqa.q_norm_scale, 1e-6);
    let k_vec = rms_norm_slice(&k_raw, &gqa.k_norm_scale, 1e-6);
    dot_f64(&q_vec, &k_vec) / (gqa.head_dim as f64).sqrt()
}

fn eval_head_gate_sigmoid(gqa: &GqaPointerLayer, head: usize, h_q: &[f64]) -> f64 {
    let qn_in = rms_norm_slice(h_q, &gqa.input_norm_scale, 1e-6);
    let r0 = head * 2 * gqa.head_dim + gqa.head_dim;
    let mean_z: f64 = (0..gqa.head_dim)
        .map(|i| dot_f64(gqa.w_q_proj.row(r0 + i), &qn_in))
        .sum::<f64>()
        / (gqa.head_dim as f64);
    1.0 / (1.0 + (-mean_z).exp())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn proptest_dual_span_plan_and_sql_asid_grounds_both_phases_and_respects_register_disjointness(
        spec in arb_opaque_gibberish_dungeon_spec(),
        seed in 0u64..100u64,
    ) {
        let inst = spec.build().unwrap();
        let conn = inst.open_connection().unwrap();
        let schema_graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
        let edge = schema_graph.foreign_keys[0].clone();

        let (gqa, q_wu) = build_realistic_gqa_fixture(seed);
        let d = gqa.hidden_dim;
        let cfg = SqlPointerBridgeConfig::qwen35_08b_l18_to_l19(vec![0, 1], vec![2, 3]);
        let fk_deltas = BTreeMap::from([(edge, vec![0.25; gqa.passthrough_dim()])]);

        let dual_prog = build_dual_span_plan_and_sql_pointer_program(
            &schema_graph,
            &cfg,
            &fk_deltas,
            RegimeExpr::ConstBool(true),
            RegimeExpr::ConstBool(true),
        )
        .unwrap();
        let plan = allocate_neural_registers(&dual_prog, 64).unwrap();
        prop_assert_eq!(plan.allocated_rules.len(), 6);
        let mut static_s = BTreeSet::new();
        let mut dyn_s = BTreeSet::new();
        for r in &plan.allocated_rules {
            if r.wire_bank_key == "UnifiedStatic" {
                static_s.extend(r.rel_slot_lo..r.rel_slot_hi);
            } else {
                dyn_s.extend(r.rel_slot_lo..r.rel_slot_hi);
            }
        }
        prop_assert!(static_s.is_disjoint(&dyn_s));

        let q_forbid = gqa.build_submerged_forbidden_basis(Some(&q_wu), 32).unwrap();
        let (mut fit_fk, mut fit_comp) = (vec![0.0; d], vec![0.0; d]);
        fit_fk[58] = 3.0;
        fit_comp[58] = -2.0;
        fit_comp[60] = 2.0;
        let q_delta = synthesize_fk_passthrough_query_delta(
            &gqa,
            &[0, 1],
            &StateMatrix::from_rows("fit_fk", &[fit_fk]).unwrap(),
            &StateMatrix::from_rows("fit_comp", &[fit_comp]).unwrap(),
            None,
            None,
        )
        .unwrap();
        let mut raw_ptr = vec![0.0; d];
        let mut h_q_base = vec![0.0; d];
        h_q_base[3] = 4.0;
        for &h in &[0usize, 1] {
            let pb_q = gqa.pullback_query_passthrough(h, &q_delta);
            let pb_g = gqa.pullback_head_gate(h, &h_q_base);
            for j in 0..d {
                raw_ptr[j] += 0.65 * pb_q[j] + 0.35 * pb_g[j];
            }
        }
        for &h in &[2usize, 3] {
            let pb_g = gqa.pullback_head_gate(h, &h_q_base);
            for j in 0..d {
                raw_ptr[j] -= 0.35 * pb_g[j];
            }
        }
        let v_ptr = normalize_unit_vec(&q_forbid.project_out(&raw_ptr)).unwrap();
        let codebook = synthesize_asid_codebook(&gqa, &[0, 1], &q_forbid).unwrap();
        prop_assert!(codebook.max_wv_leakage < 1e-12);

        let lex_dim = (seed as usize + inst.true_fk_col.bytes().map(usize::from).sum::<usize>()) % 24;
        let mut h_schema_fk = vec![0.0; d];
        h_schema_fk[lex_dim] = 5.0;
        h_schema_fk[58] = 2.6;
        let mut h_user_comp = vec![0.0; d];
        h_user_comp[lex_dim] = 5.0;
        h_user_comp[58] = -1.8;
        h_user_comp[60] = 2.2;
        let h_think_decoy = h_schema_fk.clone();

        let w_schema = &codebook.write_key_tags[MemorySegmentAsid::AuthoritativeSchema as usize];
        let w_think = &codebook.write_key_tags[MemorySegmentAsid::EphemeralScratchpad as usize];
        let r_schema = &codebook.read_query_selectors[MemorySegmentAsid::AuthoritativeSchema as usize];

        let h_schema_stamped: Vec<f64> = h_schema_fk.iter().zip(w_schema).map(|(a, b)| a + 4.5 * b).collect();
        let h_think_stamped: Vec<f64> = h_think_decoy.iter().zip(w_think).map(|(a, b)| a + 4.5 * b).collect();
        let h_plan_steered: Vec<f64> = (0..d).map(|i| h_q_base[i] + 5.0 * v_ptr[i] + 4.5 * r_schema[i]).collect();

        prop_assert!(eval_head0_qk_logit(&gqa, &h_plan_steered, &h_schema_stamped)
            - eval_head0_qk_logit(&gqa, &h_plan_steered, &h_user_comp) > 0.55);
        prop_assert!(eval_head0_qk_logit(&gqa, &h_plan_steered, &h_schema_stamped)
            - eval_head0_qk_logit(&gqa, &h_plan_steered, &h_think_stamped) > 0.55);
        prop_assert!(eval_head_gate_sigmoid(&gqa, 0, &h_plan_steered) > 0.65);
        prop_assert!(eval_head_gate_sigmoid(&gqa, 2, &h_plan_steered) < 0.35);
    }
}
