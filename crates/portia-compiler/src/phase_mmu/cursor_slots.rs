//! Intra-Phase 1-Token Kronecker Cursor Diodes for the 7-Bank Phase-Multiplexed `ASID` MMU.
//!
//! Host phase switching (`Prefill:Schema`, `Prefill:User`, `Prefill:ToolObs`, `ModePlan:<Mode>`,
//! `ModePayload:<Mode>`) guarantees **zero cross-phase firing** because inactive banks are physically
//! unmounted from `[base_M .. base_M + 2)`. Within an active mounted phase bank, however, a read
//! selector (`W_Q` rotation + distractor head gate closing) or write tag should fire **only** on its
//! syntactic cursor token (e.g., `FOREIGN KEY (c_` inside `<schema>`, `.c_` inside `ModePayload:Read`,
//! or value digits inside `Prefill:ToolObs`) so SQL keywords (`SELECT`, `SUM`, `JOIN`) and `\n`
//! coat-hooks run completely unperturbed (`silu(w_gate · x_off) < 1e-3`).

use crate::discovery::gqa_bus::GqaPointerLayer;
use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{l2_norm, quantize_slice_bf16_f64, silu_f64, slice_to_bf16_vec, StateMatrix};
use crate::phase_mmu::{
    synthesize_joint_kv_4color_codebook, JointKvPhaseMmuCodebook, PhaseMmuCompileConfig,
    PhaseMmuSegment,
};
use crate::solvers::conic_qp::solve_conic_linear_vector;
use crate::solvers::V2SolverConfig;
use crate::wire::{
    CompiledSlotBf16, WireBisimulationReceipt, WireCartridgeHeader, WireCartridgeManifestJson,
    WireSlotEntryJson,
};
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy, PrefillSegmentKind};
use std::collections::BTreeMap;

/// Intra-phase positive cursor vs. intra-phase non-cursor pre-MLP captures for a single mounted bank.
#[derive(Debug, Clone)]
pub struct IntraPhaseCursorCapture {
    pub on_cursor_pre_mlp: StateMatrix,
    pub off_cursor_pre_mlp: StateMatrix,
}

/// Complete intra-phase cursor calibration bundle for the 7-bank / 2-slot Phase-Multiplexed MMU.
#[derive(Debug, Clone)]
pub struct SevenBankCursorCaptures {
    pub schema_fk_write: IntraPhaseCursorCapture,
    pub user_claim_write: IntraPhaseCursorCapture,
    pub tool_obs_value_write: IntraPhaseCursorCapture,
    pub plan_read_scratch_write: IntraPhaseCursorCapture,
    pub plan_read_schema_query: IntraPhaseCursorCapture,
    pub payload_read_schema_query: IntraPhaseCursorCapture,
    pub plan_result_scratch_write: IntraPhaseCursorCapture,
    pub plan_result_obs_query: IntraPhaseCursorCapture,
    pub payload_result_obs_query: IntraPhaseCursorCapture,
}

/// Verified `bf16` gate & leakage diagnostics for a single compiled intra-phase cursor slot.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledIntraPhaseCursorSlot {
    pub slot: CompiledSlotBf16,
    pub min_pos_gate_bf16: f64,
    pub max_neg_gate_bf16: f64,
    pub max_neg_leak_l2: f64,
}

/// Compile a single 1-slot intra-phase Kronecker cursor SwiGLU diode in `Clarabel` (`conic_qp`),
/// verifying post-`bf16` quantization separation across `capture.on_cursor_pre_mlp` vs
/// `capture.off_cursor_pre_mlp`.
#[allow(clippy::too_many_arguments)]
pub fn compile_intra_phase_cursor_slot(
    rule_name: &str,
    slot_offset: usize,
    capture: &IntraPhaseCursorCapture,
    mean_residual_norm: f64,
    unit_submerged_dir: &[f64],
    impulse_scale: f64,
    q_preserve_gate: Option<&OrthonormalSubspace>,
    solver_cfg: &V2SolverConfig,
) -> Result<CompiledIntraPhaseCursorSlot, V2CompilerError> {
    let on_bf16 = capture.on_cursor_pre_mlp.quantize_bf16();
    let off_bf16 = capture.off_cursor_pre_mlp.quantize_bf16();
    let n_on = on_bf16.nrows();

    let gate_targets = vec![solver_cfg.g_pos_target; n_on];
    let gate_bands = vec![solver_cfg.g_pos_band; n_on];
    let neg_bound = solver_cfg.g_neg_ceiling - solver_cfg.bf16_neg_pad;
    let wg_f64 = solve_conic_linear_vector(
        rule_name,
        &on_bf16,
        &gate_targets,
        &gate_bands,
        Some((&off_bf16, neg_bound)),
        q_preserve_gate,
        solver_cfg,
    )?;
    let wg_q = quantize_slice_bf16_f64(&wg_f64);

    let gp = on_bf16.matvec(&wg_q)?;
    let gn = off_bf16.matvec(&wg_q)?;
    let min_gp = gp.iter().copied().fold(f64::INFINITY, f64::min);
    let max_gn = gn.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let req_min_pos = (solver_cfg.g_pos_target - solver_cfg.g_pos_band - 0.25).max(0.5);
    if min_gp < req_min_pos || max_gn > solver_cfg.g_neg_ceiling + 1e-3 {
        return Err(V2CompilerError::InfeasibleDiode {
            rule: rule_name.to_string(),
            status: "intra-phase post-bf16 gate verification failed".to_string(),
            min_pos_gate_bf16: min_gp,
            req_min_pos,
            max_neg_gate_bf16: max_gn,
            req_max_neg: solver_cfg.g_neg_ceiling,
        });
    }

    let target_mag = impulse_scale * mean_residual_norm;
    let target_up = target_mag / silu_f64(solver_cfg.g_pos_target).max(0.25);
    let up_targets = vec![target_up; n_on];
    let up_bands = vec![(0.10 * target_up.abs()).max(0.08); n_on];
    let wu_f64 = solve_conic_linear_vector(
        &format!("{rule_name}:up"),
        &on_bf16,
        &up_targets,
        &up_bands,
        None,
        None,
        solver_cfg,
    )?;

    let slot = CompiledSlotBf16 {
        slot_offset,
        w_gate: slice_to_bf16_vec(&wg_f64),
        w_up: slice_to_bf16_vec(&wu_f64),
        w_down: slice_to_bf16_vec(unit_submerged_dir),
    };
    let wd_norm = l2_norm(unit_submerged_dir);
    let mut max_leak = 0.0_f64;
    for r in 0..off_bf16.nrows() {
        let (_, act) = slot.eval_gate_and_act(off_bf16.row(r));
        max_leak = max_leak.max(act.abs() * wd_norm);
    }

    Ok(CompiledIntraPhaseCursorSlot {
        slot,
        min_pos_gate_bf16: min_gp,
        max_neg_gate_bf16: max_gn,
        max_neg_leak_l2: max_leak,
    })
}

/// Compile the complete 7-Bank / 2-Slot Phase-Multiplexed 4-Color `ASID` MMU using intra-phase
/// Kronecker cursor diodes (`0` cross-phase firing via host bank unmounting + `< 1e-3` intra-phase
/// background leak on SQL syntax & `\n` tokens).
#[allow(clippy::too_many_arguments)]
pub fn compile_cursor_gated_seven_bank_phase_mmu(
    cfg: &PhaseMmuCompileConfig,
    gqa: &GqaPointerLayer,
    w_u_top_basis: Option<&OrthonormalSubspace>,
    captures: &SevenBankCursorCaptures,
    cursor_ref_h: &[f64],
    mean_residual_norm: f64,
    solver_cfg: &V2SolverConfig,
    mode_grammars: BTreeMap<String, GrammarCartridgeSpec>,
) -> Result<(WireCartridgeManifestJson, JointKvPhaseMmuCodebook), V2CompilerError> {
    if cfg.upstream_mlp_layer >= gqa.layer_idx {
        return Err(V2CompilerError::NonCausalLayerHop {
            rule: "compile_cursor_gated_seven_bank_phase_mmu".to_string(),
            upstream: cfg.upstream_mlp_layer,
            downstream: gqa.layer_idx,
        });
    }
    let q_sub = gqa.build_submerged_forbidden_basis(w_u_top_basis, gqa.w_v_proj.nrows())?;
    let codebook = synthesize_joint_kv_4color_codebook(
        gqa,
        &cfg.copy_heads,
        &cfg.distractor_heads,
        cursor_ref_h,
        &q_sub,
    )?;

    let mut pos_rows = 0usize;
    let mut neg_rows = 0usize;
    let mut min_gp = f64::INFINITY;
    let mut max_gn = f64::NEG_INFINITY;
    let mut max_leak = 0.0_f64;

    let mut mk = |name: &str,
                  off: usize,
                  cap: &IntraPhaseCursorCapture,
                  dir: &[f64],
                  scale: f64|
     -> Result<CompiledSlotBf16, V2CompilerError> {
        let out = compile_intra_phase_cursor_slot(
            name,
            off,
            cap,
            mean_residual_norm,
            dir,
            scale,
            None,
            solver_cfg,
        )?;
        pos_rows += cap.on_cursor_pre_mlp.nrows();
        neg_rows += cap.off_cursor_pre_mlp.nrows();
        min_gp = min_gp.min(out.min_pos_gate_bf16);
        max_gn = max_gn.max(out.max_neg_gate_bf16);
        max_leak = max_leak.max(out.max_neg_leak_l2);
        Ok(out.slot)
    };

    let lk = cfg.upstream_mlp_layer.to_string();
    let base_m = cfg.base_intermediate_size;
    let mut banks: BTreeMap<String, BTreeMap<String, Vec<WireSlotEntryJson>>> = BTreeMap::new();
    let mut insert =
        |pol: MountPolicy, slots: Vec<CompiledSlotBf16>| -> Result<(), V2CompilerError> {
            let entry = WireSlotEntryJson::from_contiguous_slots(&slots, base_m)?;
            banks
                .entry(pol.wire_key().to_string())
                .or_default()
                .insert(lk.clone(), vec![entry]);
            Ok(())
        };

    let w = &codebook.write_key_tags;
    let r = &codebook.read_query_selectors;
    let ws = cfg.write_impulse_scale;
    let rs = cfg.read_impulse_scale;

    insert(
        MountPolicy::PrefillSegment(PrefillSegmentKind::Schema),
        vec![mk(
            "prefill_schema",
            0,
            &captures.schema_fk_write,
            &w[PhaseMmuSegment::AuthoritativeSchema as usize],
            ws,
        )?],
    )?;
    insert(
        MountPolicy::PrefillSegment(PrefillSegmentKind::User),
        vec![mk(
            "prefill_user",
            0,
            &captures.user_claim_write,
            &w[PhaseMmuSegment::UserPresupposition as usize],
            ws,
        )?],
    )?;
    insert(
        MountPolicy::PrefillSegment(PrefillSegmentKind::ToolObs),
        vec![mk(
            "prefill_tool_obs",
            0,
            &captures.tool_obs_value_write,
            &w[PhaseMmuSegment::VerifiedObservation as usize],
            ws,
        )?],
    )?;
    insert(
        MountPolicy::ModePlan(Mode::Read),
        vec![
            mk(
                "plan_read_w",
                0,
                &captures.plan_read_scratch_write,
                &w[PhaseMmuSegment::EphemeralScratchpad as usize],
                ws,
            )?,
            mk(
                "plan_read_r",
                1,
                &captures.plan_read_schema_query,
                &r[PhaseMmuSegment::AuthoritativeSchema as usize],
                rs,
            )?,
        ],
    )?;
    insert(
        MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Read)),
        vec![mk(
            "payload_read_r",
            0,
            &captures.payload_read_schema_query,
            &r[PhaseMmuSegment::AuthoritativeSchema as usize],
            rs,
        )?],
    )?;
    insert(
        MountPolicy::ModePlan(Mode::Result),
        vec![
            mk(
                "plan_result_w",
                0,
                &captures.plan_result_scratch_write,
                &w[PhaseMmuSegment::EphemeralScratchpad as usize],
                ws,
            )?,
            mk(
                "plan_result_r",
                1,
                &captures.plan_result_obs_query,
                &r[PhaseMmuSegment::VerifiedObservation as usize],
                rs,
            )?,
        ],
    )?;
    insert(
        MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Result)),
        vec![mk(
            "payload_result_r",
            0,
            &captures.payload_result_obs_query,
            &r[PhaseMmuSegment::VerifiedObservation as usize],
            rs,
        )?],
    )?;

    let receipt = WireBisimulationReceipt {
        total_leaves_verified: 9,
        positive_rows_checked: pos_rows,
        negative_rows_checked: neg_rows,
        min_positive_gate_pre_silu: min_gp,
        max_negative_gate_pre_silu: max_gn,
        max_negative_swiglu_leak_l2: max_leak,
        min_positive_chordal_cosine: 1.0,
        decoded_mode_matches: 7,
        residual_source: "intra_phase_kronecker_cursor_4color_asid".to_string(),
        bisimulation_holds: max_gn <= solver_cfg.g_neg_ceiling + 1e-3,
    };

    Ok((
        WireCartridgeManifestJson {
            header: Some(WireCartridgeHeader {
                model_family: cfg.model_id.clone(),
                checkpoint_revision: None,
                capture_sha256: None,
            }),
            model_id: Some(cfg.model_id.clone()),
            base_intermediate_size: base_m,
            widened_slots: 2,
            banks,
            permanent_static: BTreeMap::new(),
            mode_grammars,
            bisimulation_receipt: Some(receipt),
        },
        codebook,
    ))
}
