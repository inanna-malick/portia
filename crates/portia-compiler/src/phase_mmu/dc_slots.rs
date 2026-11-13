//! Homogeneous DC-Carrier Constant-On SwiGLU Slots for the 7-Bank Phase-Multiplexed MMU.

use crate::discovery::gqa_bus::GqaPointerLayer;
use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{
    dot_f64, l2_norm, normalize_unit_vec, silu_f64, slice_to_bf16_vec, StateMatrix,
};
use crate::phase_mmu::{
    synthesize_joint_kv_4color_codebook, JointKvPhaseMmuCodebook, PhaseMmuCompileConfig,
    PhaseMmuSegment,
};
use crate::wire::{
    CompiledSlotBf16, WireCartridgeHeader, WireCartridgeManifestJson, WireSlotEntryJson,
};
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy, PrefillSegmentKind};
use std::collections::BTreeMap;

/// Compile a single Homogeneous DC-Carrier Constant-On SwiGLU slot (`100%` active while its
/// phase bank is mounted, `0.000%` active when unmounted).
pub fn compile_dc_carrier_phase_slot(
    slot_offset: usize,
    calibration_pre_mlp: &StateMatrix,
    mean_residual_norm: f64,
    unit_submerged_dir: &[f64],
    gate_drive: f64,
    impulse_scale: f64,
) -> Result<CompiledSlotBf16, V2CompilerError> {
    let mean_x = calibration_pre_mlp.mean_row();
    let u_dc = normalize_unit_vec(&mean_x).ok_or(V2CompilerError::DegenerateProjectedTarget {
        rule: "dc_carrier".to_string(),
        norm: l2_norm(&mean_x),
    })?;
    let c_dc = dot_f64(&mean_x, &u_dc).max(0.25);
    let wg_f64: Vec<f64> = u_dc.iter().map(|&u| (gate_drive / c_dc) * u).collect();
    let denom = (silu_f64(gate_drive) * c_dc).max(0.1);
    let target_mag = impulse_scale * mean_residual_norm;
    let wu_f64: Vec<f64> = u_dc.iter().map(|&u| (target_mag / denom) * u).collect();

    Ok(CompiledSlotBf16 {
        slot_offset,
        w_gate: slice_to_bf16_vec(&wg_f64),
        w_up: slice_to_bf16_vec(&wu_f64),
        w_down: slice_to_bf16_vec(unit_submerged_dir),
    })
}

/// Compile the complete 7-bank Phase-Multiplexed 4-Color `ASID` MMU (`Prefill:Schema`,
/// `Prefill:User`, `Prefill:ToolObs`, `ModePlan:Read`, `ModePayload:Read`, `ModePlan:Result`,
/// `ModePayload:Result`) time-multiplexed into **just 2 physical SwiGLU slots (`widened_slots = 2`)**!
pub fn compile_seven_bank_phase_asid_mmu(
    cfg: &PhaseMmuCompileConfig,
    gqa: &GqaPointerLayer,
    w_u_top_basis: Option<&OrthonormalSubspace>,
    calibration_pre_mlp: &StateMatrix,
    cursor_ref_h: &[f64],
    mean_residual_norm: f64,
    mode_grammars: BTreeMap<String, GrammarCartridgeSpec>,
) -> Result<(WireCartridgeManifestJson, JointKvPhaseMmuCodebook), V2CompilerError> {
    if cfg.upstream_mlp_layer >= gqa.layer_idx {
        return Err(V2CompilerError::NonCausalLayerHop {
            rule: "compile_seven_bank_phase_asid_mmu".to_string(),
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

    let mk_slot = |off: usize, dir: &[f64], scale: f64| {
        compile_dc_carrier_phase_slot(
            off,
            calibration_pre_mlp,
            mean_residual_norm,
            dir,
            cfg.gate_drive,
            scale,
        )
    };
    let lk = cfg.upstream_mlp_layer.to_string();
    let base_m = cfg.base_intermediate_size;
    let mut banks: BTreeMap<String, BTreeMap<String, Vec<WireSlotEntryJson>>> = BTreeMap::new();

    let mut insert_bank =
        |policy: MountPolicy, slots: Vec<CompiledSlotBf16>| -> Result<(), V2CompilerError> {
            let entry = WireSlotEntryJson::from_contiguous_slots(&slots, base_m)?;
            banks
                .entry(policy.wire_key().to_string())
                .or_default()
                .insert(lk.clone(), vec![entry]);
            Ok(())
        };

    for (seg_pol, seg_id) in [
        (
            PrefillSegmentKind::Schema,
            PhaseMmuSegment::AuthoritativeSchema,
        ),
        (
            PrefillSegmentKind::User,
            PhaseMmuSegment::UserPresupposition,
        ),
        (
            PrefillSegmentKind::ToolObs,
            PhaseMmuSegment::VerifiedObservation,
        ),
    ] {
        insert_bank(
            MountPolicy::PrefillSegment(seg_pol),
            vec![mk_slot(
                0,
                &codebook.write_key_tags[seg_id as usize],
                cfg.write_impulse_scale,
            )?],
        )?;
    }

    for (mode, target_seg) in [
        (Mode::Read, PhaseMmuSegment::AuthoritativeSchema),
        (Mode::Result, PhaseMmuSegment::VerifiedObservation),
    ] {
        insert_bank(
            MountPolicy::ModePlan(mode),
            vec![
                mk_slot(
                    0,
                    &codebook.write_key_tags[PhaseMmuSegment::EphemeralScratchpad as usize],
                    cfg.write_impulse_scale,
                )?,
                mk_slot(
                    1,
                    &codebook.read_query_selectors[target_seg as usize],
                    cfg.read_impulse_scale,
                )?,
            ],
        )?;
        insert_bank(
            MountPolicy::DynamicBank(CartridgeBank::ModePayload(mode)),
            vec![mk_slot(
                0,
                &codebook.read_query_selectors[target_seg as usize],
                cfg.read_impulse_scale,
            )?],
        )?;
    }

    let manifest = WireCartridgeManifestJson {
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
        bisimulation_receipt: None,
    };
    Ok((manifest, codebook))
}
