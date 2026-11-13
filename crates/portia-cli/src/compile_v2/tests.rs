//! Integration and governance unit tests for `portia-cli compile`.

use super::*;
use crate::compile::{
    CompileCartridgeInputSpec, CompilerWeightGeometrySpec, DeltaNetLayerWeightSpec,
    DownstreamMlpAntiEchoSpec, PipeSlotObligationSpec, QkLayerVectorsSpec, SchemaGovernanceColumns,
};
use portia_core::pocket_dsl::canonical_compile_fixture_row_states;
use portia_core::Mode;
use std::collections::BTreeMap;

fn unit_vec(d: usize, idx: usize) -> Vec<f64> {
    let mut v = vec![0.0; d];
    v[idx] = 1.0;
    v
}

fn empty_input_spec(
    activations: BTreeMap<String, Vec<Vec<f64>>>,
    allow_synthetic: bool,
) -> CompileCartridgeInputSpec {
    CompileCartridgeInputSpec {
        status: "activations".to_string(),
        activations,
        residuals: None,
        row_states: None,
        obligations: None,
        mode_target_vectors: None,
        payload_target_vectors: None,
        required_payload_modes: Vec::new(),
        capture_row_banks: Vec::new(),
        model_family: Some("Qwen/Qwen3.5-0.8B-Base".to_string()),
        checkpoint_revision: Some("rev-v2-test".to_string()),
        base_intermediate_size: Some(3584),
        widened_slots: Some(64),
        compiler_weight_geometry: None,
        background_activations: None,
        steward_columns: Vec::new(),
        lock_columns: Vec::new(),
        schema_disclosed: false,
        allow_synthetic_fixture_targets: allow_synthetic,
    }
}

#[test]
fn compile_v2_source_files_stay_under_350_loc_limit() {
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = vec![src_dir.join("compile_v2.rs")];
    for entry in std::fs::read_dir(src_dir.join("compile_v2")).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(p);
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
fn compile_v2_enforces_production_geometry_and_governance_gates() {
    let tmp = tempfile::tempdir().unwrap();
    let d = 16usize;
    let row_states = canonical_compile_fixture_row_states();
    let mut rows = vec![vec![0.0; d]; row_states.len()];
    for (i, r) in rows.iter_mut().enumerate() {
        r[0] = 2.0;
        r[1 + (i % 6)] = if i % 2 == 0 { 2.4 } else { -1.2 };
    }
    let activations = BTreeMap::from([("22".to_string(), rows)]);

    let empty_spec = empty_input_spec(activations, true);
    let p_empty = tmp.path().join("empty.json");
    let out = tmp.path().join("out.json");
    std::fs::write(&p_empty, serde_json::to_string(&empty_spec).unwrap()).unwrap();
    assert!(compile_cartridge_v2_with_governance(
        &p_empty,
        None,
        None,
        false,
        &SchemaGovernanceColumns::default(),
        true,
        &out,
    )
    .is_err());

    let mut no_geom_spec = empty_spec;
    no_geom_spec.row_states = Some(row_states);
    no_geom_spec.allow_synthetic_fixture_targets = false;
    let p_nogeom = tmp.path().join("nogeom.json");
    std::fs::write(&p_nogeom, serde_json::to_string(&no_geom_spec).unwrap()).unwrap();
    let err = compile_cartridge_v2_with_governance(
        &p_nogeom,
        None,
        None,
        false,
        &SchemaGovernanceColumns::default(),
        false,
        &out,
    )
    .unwrap_err();
    assert!(err.to_string().contains("compiler_weight_geometry"));
}

#[test]
fn compile_v2_lowers_multilayer_geometry_with_chordal_secant_anti_echo() {
    let tmp = tempfile::tempdir().unwrap();
    let d = 48usize;
    let row_states = canonical_compile_fixture_row_states();
    let mut activations = BTreeMap::new();
    for l_key in ["14", "18", "20", "22"] {
        let mut rows = vec![vec![0.0; d]; row_states.len()];
        for (idx, row) in rows.iter_mut().enumerate() {
            row[0] = 2.0;
            row[1] = if idx < 2 {
                2.5 + 0.04 * (idx as f64)
            } else {
                -1.1
            };
            row[2] = if (2..4).contains(&idx) {
                2.5 + 0.04 * (idx as f64)
            } else {
                -1.0 - 0.02 * (idx as f64)
            };
            row[3 + (idx % 6)] = 0.05 * ((idx + 1) as f64);
        }
        activations.insert(l_key.to_string(), rows);
    }

    let mut vocab_top_k_eigvecs = vec![vec![0.0; 2]; d];
    vocab_top_k_eigvecs[0][0] = 1.0;
    vocab_top_k_eigvecs[1][1] = 1.0;
    let protected_sql_embeddings = vec![unit_vec(d, 2), unit_vec(d, 3)];
    let norm_weight: Vec<f64> = (0..d).map(|idx| 0.05 * ((idx % 5) as f64)).collect();

    let qk_spec = QkLayerVectorsSpec {
        v_set: Vec::new(),
        v_where: Vec::new(),
        v_pos_gate: Some(unit_vec(d, 5)),
        v_neg_gate: Some(unit_vec(d, 6)),
        v_qk_set: Some(unit_vec(d, 7)),
        v_qk_where: Some(unit_vec(d, 8)),
        w_q: None,
        w_k: None,
        ln_gamma: None,
        k_set: None,
        k_where: None,
    };
    let qk_layer_vectors = BTreeMap::from([
        ("14".to_string(), qk_spec.clone()),
        ("18".to_string(), qk_spec),
    ]);

    let in_proj_a: Vec<Vec<f64>> = (0..4).map(|h| unit_vec(d, 9 + h)).collect();
    let in_proj_b: Vec<Vec<f64>> = (0..4).map(|h| unit_vec(d, 13 + h)).collect();
    let deltanet_layer_weights = BTreeMap::from([(
        "20".to_string(),
        DeltaNetLayerWeightSpec {
            in_proj_a,
            in_proj_b,
            input_layernorm_weight: norm_weight.clone(),
            v_mmu: None,
        },
    )]);

    let mut named_token_unembeddings = BTreeMap::new();
    for (idx, tid_key) in [
        "5140", "18104", "5574", "37272", "27604", "13732", "29211", "3563", "2152", "58498",
        "44807", "31942", "4913",
    ]
    .into_iter()
    .enumerate()
    {
        let mut v = vec![0.0; d];
        v[(10 + idx) % d] = 1.0;
        v[(11 + idx) % d] = -0.4;
        named_token_unembeddings.insert(tid_key.to_string(), v);
    }

    let m_ds = 6usize;
    let mut ds_gate = vec![vec![0.0; d]; m_ds];
    let mut ds_up = vec![vec![0.0; d]; m_ds];
    let mut ds_down = vec![vec![0.0; m_ds]; d];
    for j in 0..m_ds {
        ds_gate[j][(18 + j) % d] = 1.4;
        ds_up[j][(24 + j) % d] = 1.1;
        ds_down[(10 + j) % d][j] = -0.9;
    }

    let mk_ob = |id: &str,
                 bank: &str,
                 mode: Option<Mode>,
                 layer: usize,
                 hi: usize,
                 pos: Vec<usize>,
                 neg: Vec<usize>| {
        PipeSlotObligationSpec {
            readout_contrast: None,
            obligation_id: id.to_string(),
            bank_name: bank.to_string(),
            expected_mode: mode,
            layer,
            slot_lo: 0,
            slot_hi: hi,
            positive_prompt_indices: pos,
            negative_prompt_indices: neg,
            background_prompt_indices: vec![4, 5, 6, 7],
            v_target: Vec::new(),
            theta_rad: 0.65,
            gate_target: 3.45,
            gate_neg_target: -8.2,
        }
    };
    let explicit_obs = vec![
        mk_ob(
            "pre_gqa_l14",
            "UnifiedStatic",
            None,
            14,
            2,
            vec![0, 1],
            vec![2, 3],
        ),
        mk_ob(
            "pre_gqa_l18",
            "UnifiedStatic",
            None,
            18,
            2,
            vec![0, 1],
            vec![2, 3],
        ),
        mk_ob(
            "deltanet_mmu_l20",
            "UnifiedStatic",
            None,
            20,
            1,
            vec![0, 1],
            vec![2, 3],
        ),
        mk_ob(
            "orient_read_l22",
            "OrientTrunk",
            Some(Mode::Read),
            22,
            2,
            vec![0, 1],
            vec![2, 3],
        ),
        mk_ob(
            "orient_ask_l22",
            "OrientTrunk",
            Some(Mode::Ask),
            22,
            2,
            vec![2, 3],
            vec![0, 1],
        ),
    ];

    let mut spec = empty_input_spec(activations, false);
    spec.row_states = Some(row_states);
    spec.obligations = Some(explicit_obs);
    spec.compiler_weight_geometry = Some(CompilerWeightGeometrySpec {
        vocab_top_k_eigvecs,
        protected_sql_embeddings,
        norm_weight,
        mean_natural_token_norm: 0.72,
        qk_layer_vectors,
        deltanet_layer_weights,
        named_token_unembeddings,
        downstream_mlp_anti_echo: Some(DownstreamMlpAntiEchoSpec {
            post_attn_norm_weight: vec![0.0; d],
            w_gate: ds_gate,
            w_up: ds_up,
            w_down: ds_down,
            top_k_anti_echo_neurons: 4,
        }),
    });

    let in_path = tmp.path().join("v2_multilayer_in.json");
    let out_path = tmp.path().join("v2_multilayer_out.json");
    std::fs::write(&in_path, serde_json::to_string(&spec).unwrap()).unwrap();

    let (cartridge, prov) = compile_cartridge_v2_with_governance(
        &in_path,
        None,
        None,
        false,
        &SchemaGovernanceColumns::default(),
        false,
        &out_path,
    )
    .expect("multilayer compilation should succeed");

    assert!(prov.bisimulation_verified);
    assert!(cartridge.permanent_static.contains_key("14"));
    assert!(cartridge.permanent_static.contains_key("18"));
    assert_eq!(cartridge.permanent_static["20"][0].slot_span(), 1);
    assert_eq!(cartridge.banks["OrientTrunk"]["22"].len(), 2);
}
