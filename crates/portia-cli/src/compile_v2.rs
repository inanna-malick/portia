//! `portia-cli` Pointer-ISA + `faer` + Clarabel Lowering & Provenance Bridge (`compile_v2`).
//!
//! Lowers `CompileCartridgeInputSpec` + `SchemaGovernanceColumns` through
//! [`portia_compiler::lowering::compile_intent_program`], stamps the cryptographic capture
//! provenance header on [`WireCartridgeManifestJson`], and runs
//! [`verify_wire_cartridge_bisimulation`] to populate [`CompileProvenanceReceipt`].

#[cfg(test)]
mod capture_fixture_tests;
pub(crate) mod derive_bindings;
mod geometry;
mod model_tables;
mod obligations;
#[cfg(test)]
mod tests;

use crate::compile::{
    verify_wire_cartridge_bisimulation, CompileProvenanceReceipt, PipeSlotObligationSpec,
    SchemaGovernanceColumns,
};
use anyhow::{bail, Context, Result};
use portia_compiler::lowering::compile_intent_program;
use portia_compiler::numerics::normalize_unit_vec;
use portia_compiler::wire::{WireCartridgeHeader, WireCartridgeManifestJson};
use portia_core::{CartridgeBank, MountPolicy};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Compile a [`WireCartridgeManifestJson`] and [`CompileProvenanceReceipt`] via `portia-compiler`.
pub fn compile_cartridge_v2_with_governance(
    input_json: &Path,
    db_path: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
    governance: &SchemaGovernanceColumns,
    allow_synthetic_fixture_targets: bool,
    out_json: &Path,
) -> Result<(WireCartridgeManifestJson, CompileProvenanceReceipt)> {
    let prep = obligations::prepare_v2_compile_inputs(
        input_json,
        db_path,
        regime_json,
        read_only,
        governance,
        allow_synthetic_fixture_targets,
    )?;

    let mut lowered = geometry::build_v2_intent_and_geometry(&prep)?;
    let compiled = compile_intent_program(&lowered.expr, &lowered.captures, &lowered.geom)
        .context("portia-compiler compile_intent_program failed")?;

    let capture_sha256 = format!("{:x}", Sha256::digest(prep.raw_json.as_bytes()));
    let mut cartridge = compiled.manifest;
    cartridge.header = Some(WireCartridgeHeader {
        model_family: prep.model_family.clone(),
        checkpoint_revision: prep.checkpoint_revision.clone(),
        capture_sha256: Some(capture_sha256),
    });
    cartridge.model_id = Some(prep.model_family.clone());
    cartridge
        .verify_schema_integrity()
        .context("WireCartridgeManifestJson failed verify_schema_integrity()")?;

    // Align bisimulation obligation slot bounds to `regalloc`'s assigned physical slots
    // and populate 1-slot `DeltaNetRollbackScrub` `v_target` from its compiled `w_down` column:
    let mut bisim_obligations: Vec<PipeSlotObligationSpec> =
        Vec::with_capacity(compiled.allocated.len());
    for alloc in &compiled.allocated {
        let mut ob = lowered
            .resolved_obligations_by_rule
            .remove(&alloc.rule.name)
            .with_context(|| format!("missing obligation record for rule {}", alloc.rule.name))?;
        let abs_lo = prep.base_inter + alloc.rel_slot_lo;
        let abs_hi = prep.base_inter + alloc.rel_slot_hi;
        ob.slot_lo = abs_lo;
        ob.slot_hi = abs_hi;

        if ob.v_target.is_empty() {
            let layer_key = alloc.host_mlp_layer.to_string();
            let entries = match alloc.rule.mount_policy {
                MountPolicy::PermanentStatic => cartridge.permanent_static.get(&layer_key),
                MountPolicy::DynamicBank(_)
                | MountPolicy::ModePlan(_)
                | MountPolicy::PrefillSegment(_) => cartridge
                    .banks
                    .get(&alloc.wire_bank_key)
                    .and_then(|m| m.get(&layer_key)),
            };
            if let Some(entry) = entries.and_then(|es| {
                es.iter()
                    .find(|e| e.slot_lo == abs_lo && e.slot_hi == abs_hi)
            }) {
                let raw_down: Vec<f64> = entry
                    .w_down
                    .iter()
                    .map(|row| f64::from(row.first().copied().unwrap_or(0.0)))
                    .collect();
                ob.v_target = normalize_unit_vec(&raw_down).ok_or_else(|| {
                    anyhow::anyhow!("degenerate w_down on 1-slot rule {}", alloc.rule.name)
                })?;
            }
        }
        bisim_obligations.push(ob);
    }

    let bisim_receipt =
        verify_wire_cartridge_bisimulation(&cartridge, &prep.capture, &bisim_obligations)
            .context("WireCartridgeManifestJson failed verify_wire_cartridge_bisimulation()")?;
    if !bisim_receipt.bisimulation_holds {
        bail!("bisimulation_receipt reported bisimulation_holds=false");
    }
    cartridge.bisimulation_receipt = Some(bisim_receipt.clone());

    for &mode in &prep.required_payload_modes {
        let bank = CartridgeBank::ModePayload(mode).wire_key();
        let has_exec = cartridge.banks.get(bank).is_some_and(|layers| {
            layers.values().flatten().any(|e| {
                [&e.w_gate, &e.w_up, &e.w_down]
                    .iter()
                    .all(|m| m.iter().flatten().any(|v| v.is_finite() && *v != 0.0))
            })
        });
        if !has_exec {
            bail!("required payload bank {bank} has no nonzero verified slots");
        }
    }

    let compact_json = cartridge
        .to_json_compact()
        .context("failed to serialize WireCartridgeManifestJson")?;
    if let Some(parent) = out_json.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_json, compact_json)
        .with_context(|| format!("failed to write compiled cartridge to {out_json:?}"))?;

    let provenance = CompileProvenanceReceipt {
        regime_source: prep.regime_source,
        capture_source: prep.capture_source,
        obligation_source: prep.obligation_source,
        hidden_dim: prep.hidden_dim,
        matched_steward_columns: prep.matched_steward_columns,
        matched_lock_columns: prep.matched_lock_columns,
        read_only,
        active_modes: prep.active_modes,
        guarded_rules: prep.guarded_rules,
        rule_state_bindings: prep.rule_state_bindings,
        bisimulation_verified: bisim_receipt.bisimulation_holds,
        min_positive_gate_pre_silu: bisim_receipt.min_positive_gate_pre_silu,
        max_negative_gate_pre_silu: bisim_receipt.max_negative_gate_pre_silu,
        min_positive_chordal_cosine: bisim_receipt.min_positive_chordal_cosine,
        max_negative_swiglu_leak_l2: bisim_receipt.max_negative_swiglu_leak_l2,
        decoded_mode_matches: bisim_receipt.decoded_mode_matches,
        residual_source: bisim_receipt.residual_source,
    };

    Ok((cartridge, provenance))
}
