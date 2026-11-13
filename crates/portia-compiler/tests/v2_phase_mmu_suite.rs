//! Integration & invariant suite for `portia_compiler::phase_mmu` (`7` deterministic host-phase
//! banks time-multiplexed into `2` physical SwiGLU slots with 4-color `ASID` isolation).

use portia_compiler::discovery::gqa_bus::GqaPointerLayer;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, rms_norm_slice, StateMatrix};
use portia_compiler::phase_mmu::{
    compile_cursor_gated_seven_bank_phase_mmu, compile_seven_bank_phase_asid_mmu,
    IntraPhaseCursorCapture, PhaseMmuCompileConfig, PhaseMmuSegment, SevenBankCursorCaptures,
};
use portia_compiler::solvers::V2SolverConfig;
use portia_compiler::wire::WireCartridgeManifestJson;
use portia_core::{CartridgeBank, Mode, MountPolicy, PrefillSegmentKind};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn seven_phase_banks_time_multiplex_into_two_slots_and_verify_v1_wire_integrity() {
    let d = 128usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize; // 2 KV groups!
    let head_dim = 32usize;
    let rotary_dim = 8usize;

    let mut wq_rows = vec![vec![0.0; d]; 2 * num_q_heads * head_dim];
    let mut wk_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];
    let mut wv_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];

    // W_V occupies [0..64) (`2 * 32`), W_U occupies [64..80), submerged W_K/W_Q occupies [80..128):
    for g in 0..num_kv_heads {
        for i in 0..head_dim {
            wv_rows[g * head_dim + i][(g * head_dim + i) % 64] = 1.3;
            wk_rows[g * head_dim + i][80 + ((g * 7 + i) % 48)] = 1.25;
        }
    }
    for h in 0..num_q_heads {
        let g = h / 2;
        for i in 0..head_dim {
            wq_rows[h * 2 * head_dim + i][80 + ((g * 7 + i) % 48)] = 1.25;
            wq_rows[h * 2 * head_dim + head_dim + i][80 + ((h * 11 + i) % 48)] = 1.1;
        }
    }

    let mut wu_rows = vec![vec![0.0; d]; 16];
    for i in 0..16 {
        wu_rows[i][64 + i] = 2.0;
    }
    let q_wu = OrthonormalSubspace::from_row_space_svd(
        "wu16",
        &StateMatrix::from_rows("wu", &wu_rows).unwrap(),
        1e-5,
        16,
    )
    .unwrap();

    let gqa = GqaPointerLayer {
        layer_idx: 19,
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

    // Calibration DC carrier (`dim 0 = 4.0` across pre-MLP states):
    let mut cal_rows = Vec::new();
    for r in 0..8 {
        let mut x = vec![0.02 * ((r as f64) - 3.5); d];
        x[0] = 4.2;
        cal_rows.push(x);
    }
    let cal_mat = StateMatrix::from_rows("cal", &cal_rows).unwrap();
    let mut cursor_ref = vec![0.0; d];
    cursor_ref[82] = 2.0;

    let cfg = PhaseMmuCompileConfig {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        upstream_mlp_layer: 18,
        copy_heads: vec![0, 2], // spans BOTH KV groups 0 and 1!
        distractor_heads: vec![1, 3],
        gate_drive: 4.5,
        write_impulse_scale: 1.8,
        read_impulse_scale: 1.8,
    };

    let (manifest, codebook) = compile_seven_bank_phase_asid_mmu(
        &cfg,
        &gqa,
        Some(&q_wu),
        &cal_mat,
        &cursor_ref,
        12.0,
        BTreeMap::new(),
    )
    .expect("7-bank phase MMU must compile cleanly");

    // 1. All 7 phase banks share ONLY 2 physical SwiGLU slots (`3584..3586`)!
    assert_eq!(manifest.widened_slots, 2);
    assert_eq!(manifest.banks.len(), 7);
    for expected_key in [
        "Prefill:Schema",
        "Prefill:User",
        "Prefill:ToolObs",
        "ModePlan:Read",
        "ModePlan:Result",
        "ModePayload:Read",
        "ModePayload:Result",
    ] {
        assert!(
            manifest.banks.contains_key(expected_key),
            "missing bank {expected_key}"
        );
    }

    // 2. Machine-zero submerged leakage (`ker(W_V) ∩ ker(W_U_top)`) & pairwise 4-color orthogonality:
    assert!(
        codebook.max_submerged_leakage < 1e-12,
        "leakage = {:e}",
        codebook.max_submerged_leakage
    );
    for a in 0..4 {
        for b in (a + 1)..4 {
            let dot_ab = dot_f64(&codebook.write_key_tags[a], &codebook.write_key_tags[b]).abs();
            assert!(
                dot_ab < 1e-10,
                "colors {a} and {b} must be orthogonal in residual space, got {dot_ab:e}"
            );
        }
    }

    let json = serde_json::to_string(&manifest).unwrap();
    let verified = WireCartridgeManifestJson::from_json_str(&json)
        .expect("WireCartridgeManifestJson::verify_schema_integrity must accept all 7 phase banks");
    assert_eq!(verified.widened_slots, 2);

    // 4. Verify `ModePayload:Result` (`VerifiedObservation` reader) prefers `<tool_response>` over
    //    `<user>` presupposition AND `<think>` scratchpad simultaneously:
    let base_tok = vec![0.1; d];
    let tag = |seg: PhaseMmuSegment| -> Vec<f64> {
        let w = &codebook.write_key_tags[seg as usize];
        base_tok.iter().zip(w).map(|(&x, &t)| x + 3.0 * t).collect()
    };
    let k_obs = rms_norm_slice(
        &tag(PhaseMmuSegment::VerifiedObservation),
        &gqa.input_norm_scale,
        1e-6,
    );
    let k_usr = rms_norm_slice(
        &tag(PhaseMmuSegment::UserPresupposition),
        &gqa.input_norm_scale,
        1e-6,
    );
    let k_thk = rms_norm_slice(
        &tag(PhaseMmuSegment::EphemeralScratchpad),
        &gqa.input_norm_scale,
        1e-6,
    );

    let r_res = &codebook.read_query_selectors[PhaseMmuSegment::VerifiedObservation as usize];
    let q_cur: Vec<f64> = base_tok
        .iter()
        .zip(r_res)
        .map(|(&x, &q)| x + 3.0 * q)
        .collect();
    let q_in = rms_norm_slice(&q_cur, &gqa.input_norm_scale, 1e-6);

    let q0: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_q_proj.row(i), &q_in))
        .collect();
    let ko0: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &k_obs))
        .collect();
    let ku0: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &k_usr))
        .collect();
    let kt0: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &k_thk))
        .collect();

    assert!(dot_f64(&q0, &ko0) > dot_f64(&q0, &ku0) + 0.4);
    assert!(dot_f64(&q0, &ko0) > dot_f64(&q0, &kt0) + 0.4);

    // 5. Verify seamless fallback on legacy `v1` bank sets vs `v2` phase-specialized bank sets:
    let legacy_banks: BTreeSet<String> = ["Base", "OrientTrunk", "ModePayload:Read"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        MountPolicy::ModePlan(Mode::Read).resolve_fallback_wire_key(&legacy_banks),
        "ModePayload:Read"
    );
    assert_eq!(
        MountPolicy::PrefillSegment(PrefillSegmentKind::Schema)
            .resolve_fallback_wire_key(&legacy_banks),
        "Base"
    );
    let v2_banks: BTreeSet<String> = verified.banks.keys().cloned().collect();
    assert_eq!(
        MountPolicy::ModePlan(Mode::Read).resolve_fallback_wire_key(&v2_banks),
        "ModePlan:Read"
    );
    assert_eq!(
        MountPolicy::PrefillSegment(PrefillSegmentKind::ToolObs)
            .resolve_fallback_wire_key(&v2_banks),
        "Prefill:ToolObs"
    );
    assert!(
        !MountPolicy::ModePlan(Mode::Read).co_mounted_with(MountPolicy::DynamicBank(
            CartridgeBank::ModePayload(Mode::Read)
        ))
    );
}

#[test]
fn cursor_gated_seven_bank_phase_mmu_preserves_non_cursor_syntax_tokens() {
    let d = 128usize;
    let head_dim = 32usize;
    let rotary_dim = 8usize;
    let mut wq_rows = vec![vec![0.0; d]; 2 * 4 * head_dim];
    let mut wk_rows = vec![vec![0.0; d]; 2 * head_dim];
    let mut wv_rows = vec![vec![0.0; d]; 2 * head_dim];
    for g in 0..2 {
        for i in 0..head_dim {
            wv_rows[g * head_dim + i][(g * head_dim + i) % 64] = 1.3;
            wk_rows[g * head_dim + i][80 + ((g * 7 + i) % 48)] = 1.25;
        }
    }
    for h in 0..4 {
        let g = h / 2;
        for i in 0..head_dim {
            wq_rows[h * 2 * head_dim + i][80 + ((g * 7 + i) % 48)] = 1.25;
            wq_rows[h * 2 * head_dim + head_dim + i][80 + ((h * 11 + i) % 48)] = 1.1;
        }
    }
    let gqa = GqaPointerLayer {
        layer_idx: 19,
        hidden_dim: d,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim,
        rotary_dim,
        input_norm_scale: vec![1.0; d],
        q_norm_scale: vec![1.0; head_dim],
        k_norm_scale: vec![1.0; head_dim],
        w_q_proj: StateMatrix::from_rows("wq", &wq_rows).unwrap(),
        w_k_proj: StateMatrix::from_rows("wk", &wk_rows).unwrap(),
        w_v_proj: StateMatrix::from_rows("wv", &wv_rows).unwrap(),
    };

    // Each phase has its own distinct intra-phase cursor axis `dim_on` vs SQL/prose syntax tokens:
    let mk_cap = |dim_on: usize| -> IntraPhaseCursorCapture {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for r in 0..4 {
            let mut xp = vec![0.05 * (r as f64); d];
            xp[0] = 3.0;
            xp[dim_on] = 4.5;
            pos.push(xp);
            for k in 0..5 {
                let mut xn = vec![0.03 * ((r + k) as f64); d];
                xn[0] = 3.0;
                xn[(dim_on + 3 + k) % 40 + 1] = 3.5;
                neg.push(xn);
            }
        }
        IntraPhaseCursorCapture {
            on_cursor_pre_mlp: StateMatrix::from_rows("on", &pos).unwrap(),
            off_cursor_pre_mlp: StateMatrix::from_rows("off", &neg).unwrap(),
        }
    };

    let captures = SevenBankCursorCaptures {
        schema_fk_write: mk_cap(2),
        user_claim_write: mk_cap(3),
        tool_obs_value_write: mk_cap(4),
        plan_read_scratch_write: mk_cap(5),
        plan_read_schema_query: mk_cap(6),
        payload_read_schema_query: mk_cap(7),
        plan_result_scratch_write: mk_cap(8),
        plan_result_obs_query: mk_cap(9),
        payload_result_obs_query: mk_cap(10),
    };
    let mut cursor_ref = vec![0.0; d];
    cursor_ref[82] = 2.0;

    let cfg = PhaseMmuCompileConfig {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        upstream_mlp_layer: 18,
        copy_heads: vec![0, 2],
        distractor_heads: vec![1, 3],
        gate_drive: 3.2,
        write_impulse_scale: 1.8,
        read_impulse_scale: 1.8,
    };

    let (manifest, codebook) = compile_cursor_gated_seven_bank_phase_mmu(
        &cfg,
        &gqa,
        None,
        &captures,
        &cursor_ref,
        12.0,
        &V2SolverConfig::default(),
        BTreeMap::new(),
    )
    .expect("cursor-gated 7-bank phase MMU must solve cleanly in Clarabel");

    assert_eq!(manifest.widened_slots, 2);
    assert_eq!(manifest.banks.len(), 7);
    assert!(codebook.max_submerged_leakage < 1e-12);
    let receipt = manifest.bisimulation_receipt.as_ref().unwrap();
    assert!(receipt.bisimulation_holds);
    assert!(receipt.min_positive_gate_pre_silu >= 2.8);
    assert!(receipt.max_negative_gate_pre_silu <= -7.99);
    assert!(
        receipt.max_negative_swiglu_leak_l2 < 5e-2,
        "non-cursor SQL syntax tokens must remain unperturbed, got {}",
        receipt.max_negative_swiglu_leak_l2
    );

    let json = serde_json::to_string(&manifest).unwrap();
    let verified = WireCartridgeManifestJson::from_json_str(&json).unwrap();
    assert_eq!(verified.widened_slots, 2);
}
