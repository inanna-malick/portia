//! Integration invariant suite for `portia_sql::v2_pointer_bridge` (`v2_pointer_asid_suite`).

use portia_compiler::discovery::gqa_bus::GqaPointerLayer;
use portia_compiler::ir::regalloc::allocate_neural_registers;
use portia_compiler::lowering::{compile_intent_program, ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, l2_norm, StateMatrix};
use portia_compiler::solvers::clarabel_ternary::CursorPolarity;
use portia_compiler::solvers::{NullspacePolicy, V2SolverConfig};
use portia_core::pocket_dsl::RegimeExpr;
use portia_sql::relational_ast::{ForeignKeyEdge, RelationalSchemaGraph};
use portia_sql::v2_pointer_bridge::{
    build_dual_span_plan_and_sql_pointer_program, build_per_mode_think_and_sql_pointer_programs,
    build_relational_schema_pointer_program, synthesize_fk_passthrough_query_delta,
    validate_fk_edge_in_schema, SqlPointerBridgeConfig,
};
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

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

#[test]
fn v2_pointer_bridge_suite_and_examples_stay_under_350_loc_limit() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![
        manifest_dir.join("src/v2_pointer_bridge.rs"),
        manifest_dir.join("tests/v2_pointer_asid_suite.rs"),
    ];
    for dir_rel in ["src/v2_pointer_bridge", "examples"] {
        let dir = manifest_dir.join(dir_rel);
        if dir.exists() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let p = entry.unwrap().path();
                if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    files.push(p);
                }
            }
        }
    }
    for path in files {
        let lines = std::fs::read_to_string(&path).unwrap().lines().count();
        assert!(
            lines <= 350,
            "{path:?} has {lines} lines (exceeds 350 LOC hard invariant)"
        );
    }
}

#[test]
fn dual_span_and_per_mode_programs_lower_with_disjoint_static_dynamic_slots_and_zero_wv_leak() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE t_41c (c_009 INTEGER PRIMARY KEY, c_label TEXT NOT NULL);
         CREATE TABLE t_9f2 (id INTEGER PRIMARY KEY, c_77b INTEGER NOT NULL REFERENCES t_41c(c_009));",
    )
    .unwrap();
    let schema = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
    let edge = schema.foreign_keys[0].clone();
    validate_fk_edge_in_schema(&schema, &edge).unwrap();
    assert!(validate_fk_edge_in_schema(
        &schema,
        &ForeignKeyEdge::new("t_9f2", "c_bad", "t_41c", "c_009")
    )
    .is_err());

    let (gqa, q_wu) = build_realistic_gqa_fixture(1);
    let d = gqa.hidden_dim;
    let cfg = SqlPointerBridgeConfig::qwen35_08b_l14_to_l15(vec![0, 1], vec![2, 3]);

    let (mut fk_fit, mut comp_fit) = (vec![0.0; d], vec![0.0; d]);
    fk_fit[58] = 3.2;
    fk_fit[59] = 1.8;
    comp_fit[58] = -1.5;
    comp_fit[60] = 2.4;

    let q_delta = synthesize_fk_passthrough_query_delta(
        &gqa,
        &cfg.copy_heads,
        &StateMatrix::from_rows("fk_fit", &[fk_fit]).unwrap(),
        &StateMatrix::from_rows("comp_fit", &[comp_fit]).unwrap(),
        None,
        None,
    )
    .unwrap();

    let fk_deltas = BTreeMap::from([(edge.clone(), q_delta)]);
    let single_expr = build_relational_schema_pointer_program(
        &schema,
        &cfg,
        &fk_deltas,
        RegimeExpr::ConstBool(true),
    )
    .unwrap();
    assert_eq!(
        allocate_neural_registers(&single_expr, 64)
            .unwrap()
            .allocated_rules
            .len(),
        4
    );

    let dual_expr = build_dual_span_plan_and_sql_pointer_program(
        &schema,
        &cfg,
        &fk_deltas,
        RegimeExpr::ConstBool(true),
        RegimeExpr::ConstBool(true),
    )
    .unwrap();
    let per_mode = build_per_mode_think_and_sql_pointer_programs(
        &schema,
        &cfg,
        &fk_deltas,
        RegimeExpr::ConstBool(true),
        RegimeExpr::ConstBool(true),
    )
    .unwrap();
    let think_plan = allocate_neural_registers(&per_mode.think_program, 64).unwrap();
    let sql_plan = allocate_neural_registers(&per_mode.payload_program, 64).unwrap();
    assert_eq!(think_plan.allocated_rules.len(), 5);
    assert_eq!(sql_plan.allocated_rules.len(), 4);

    let mk_cursor = |idx: usize| {
        let mut x = vec![0.0; d];
        x[112 + idx] = 3.0;
        x[120] = 1.5;
        x
    };
    let (c_schema, c_think, c_plan_q, c_sql_q, c_bg) = (
        mk_cursor(0),
        mk_cursor(1),
        mk_cursor(2),
        mk_cursor(3),
        mk_cursor(5),
    );
    let mut h_res = vec![0.0; d];
    h_res[2] = 16.0;

    let mk_cap = |on: &[f64], offs: Vec<Vec<f64>>, is_static: bool| RuleCaptureData {
        on_cursor_pre_mlp: StateMatrix::from_rows("on", &[on.to_vec()]).unwrap(),
        polarities: vec![CursorPolarity::PlusOne],
        pos_residuals: StateMatrix::from_rows("res", &[h_res.clone()]).unwrap(),
        off_cursor_pre_mlp: StateMatrix::from_rows("off", &offs).unwrap(),
        q_preserve_gate: None,
        solver_config: V2SolverConfig {
            nullspace_policy: if is_static {
                NullspacePolicy::StrictZeroCone
            } else {
                NullspacePolicy::BoundedSocSlack { max_l2_leak: 0.25 }
            },
            ..V2SolverConfig::default()
        },
    };

    let sfx = format!(
        "{}_{}_to_{}_{}",
        edge.child_table, edge.child_col, edge.parent_table, edge.parent_col
    );
    let captures = BTreeMap::from([
        (
            "sql_schema_asid_write_schema".into(),
            mk_cap(
                &c_schema,
                vec![
                    c_think.clone(),
                    c_plan_q.clone(),
                    c_sql_q.clone(),
                    c_bg.clone(),
                ],
                true,
            ),
        ),
        (
            "sql_schema_asid_write_scratchpad".into(),
            mk_cap(
                &c_think,
                vec![
                    c_schema.clone(),
                    c_plan_q.clone(),
                    c_sql_q.clone(),
                    c_bg.clone(),
                ],
                true,
            ),
        ),
        (
            "plan_orient_asid_read_schema".into(),
            mk_cap(
                &c_plan_q,
                vec![c_schema.clone(), c_think.clone(), c_bg.clone()],
                false,
            ),
        ),
        (
            format!("plan_fk_ptr_{sfx}"),
            mk_cap(
                &c_plan_q,
                vec![c_schema.clone(), c_think.clone(), c_bg.clone()],
                false,
            ),
        ),
        (
            "sql_schema_asid_read_schema".into(),
            mk_cap(
                &c_sql_q,
                vec![c_schema.clone(), c_think.clone(), c_bg.clone()],
                false,
            ),
        ),
        (
            format!("fk_ptr_{sfx}"),
            mk_cap(
                &c_sql_q,
                vec![c_schema.clone(), c_think.clone(), c_bg],
                false,
            ),
        ),
    ]);

    let geom = ModelGeometryTables {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        widened_slot_capacity: 64,
        w_u_top_basis: Some(q_wu.clone()),
        downstream_mlps: BTreeMap::new(),
        gqa_layers: BTreeMap::from([(15, gqa.clone())]),
        deltanet_layers: BTreeMap::new(),
    };

    let compiled = compile_intent_program(&dual_expr, &captures, &geom).unwrap();
    let static_slots: BTreeSet<usize> = compiled.manifest.permanent_static["14"]
        .iter()
        .flat_map(|e| e.slot_lo..e.slot_hi)
        .collect();
    let orient_slots: BTreeSet<usize> = compiled.manifest.banks["OrientTrunk"]["14"]
        .iter()
        .flat_map(|e| e.slot_lo..e.slot_hi)
        .collect();
    let read_slots: BTreeSet<usize> = compiled.manifest.banks["ModePayload:Read"]["14"]
        .iter()
        .flat_map(|e| e.slot_lo..e.slot_hi)
        .collect();
    assert_eq!(static_slots.len(), 4);
    assert_eq!(orient_slots.len(), 4);
    assert_eq!(read_slots.len(), 4);
    assert!(static_slots.is_disjoint(&orient_slots));
    assert!(static_slots.is_disjoint(&read_slots));

    for entry in compiled.manifest.permanent_static["14"]
        .iter()
        .chain(compiled.manifest.banks["OrientTrunk"]["14"].iter())
        .chain(compiled.manifest.banks["ModePayload:Read"]["14"].iter())
    {
        let w_down_tan: Vec<f64> = entry.w_down.iter().map(|r| f64::from(r[1])).collect();
        let wv_leak = l2_norm(
            &(0..gqa.w_v_proj.nrows())
                .map(|r| dot_f64(gqa.w_v_proj.row(r), &w_down_tan))
                .collect::<Vec<_>>(),
        );
        assert!(
            wv_leak < 1e-10,
            "tangential W_V leak must be 0.000, got {wv_leak}"
        );
        assert!(q_wu.leakage_norm(&w_down_tan) < 1e-10);
    }
}
