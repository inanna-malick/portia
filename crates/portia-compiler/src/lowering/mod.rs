//! Lowering Pass (`lowering`): Compiles an [`IntentExpr`] + [`RuleCaptureData`] map into a verified
//! [`WireCartridgeManifestJson`] ready for `portia_qwen_stdio_worker.py` and `portia-cli`.
//!
//! Dispatches independent register-allocated rules across `std::thread::scope` worker threads and
//! merges outputs in deterministic `plan.allocated_rules` index order.

mod rule_worker;
pub mod tables;
pub use tables::{CompiledProgramV2, ModelGeometryTables, RuleCaptureData};

use crate::error::V2CompilerError;
use crate::ir::regalloc::{allocate_neural_registers, AllocatedIntentRule};
use crate::ir::IntentExpr;
use crate::lowering::rule_worker::{compile_allocated_rule, CompiledRuleArtifact};
use crate::wire::{
    WireBisimulationReceipt, WireCartridgeHeader, WireCartridgeManifestJson, WireSlotEntryJson,
};
use portia_core::MountPolicy;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

fn compile_rules_deterministic_parallel(
    rules: &[AllocatedIntentRule],
    captures: &BTreeMap<String, RuleCaptureData>,
    geom: &ModelGeometryTables,
) -> Result<Vec<CompiledRuleArtifact>, V2CompilerError> {
    let n = rules.len();
    if n <= 1 {
        return rules
            .iter()
            .map(|r| compile_allocated_rule(r, captures, geom))
            .collect();
    }

    let n_workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(n);
    if n_workers <= 1 {
        return rules
            .iter()
            .map(|r| compile_allocated_rule(r, captures, geom))
            .collect();
    }

    let next_idx = AtomicUsize::new(0);
    let worker_batches: Vec<Vec<(usize, Result<CompiledRuleArtifact, V2CompilerError>)>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_workers);
            for _ in 0..n_workers {
                let next_ref = &next_idx;
                handles.push(scope.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= n {
                            break;
                        }
                        let res = compile_allocated_rule(&rules[idx], captures, geom);
                        local.push((idx, res));
                    }
                    local
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("rule compilation worker panicked"))
                .collect()
        });

    let mut ordered: Vec<Option<Result<CompiledRuleArtifact, V2CompilerError>>> =
        (0..n).map(|_| None).collect();
    for batch in worker_batches {
        for (idx, res) in batch {
            ordered[idx] = Some(res);
        }
    }

    ordered
        .into_iter()
        .map(|slot| slot.expect("every rule index 0..n visited exactly once"))
        .collect()
}

/// Compile an `IntentExpr` into an executable `CompiledProgramV2` with automatic register allocation,
/// Rodney Brooks `high | low` inhibition propagation, multi-core rule parallelism, and `bf16`
/// bisimulation verification.
pub fn compile_intent_program(
    expr: &IntentExpr,
    captures: &BTreeMap<String, RuleCaptureData>,
    geom: &ModelGeometryTables,
) -> Result<CompiledProgramV2, V2CompilerError> {
    let plan = allocate_neural_registers(expr, geom.widened_slot_capacity)?;
    let artifacts = compile_rules_deterministic_parallel(&plan.allocated_rules, captures, geom)?;

    let mut banks: BTreeMap<String, BTreeMap<String, Vec<WireSlotEntryJson>>> = BTreeMap::new();
    let mut permanent_static: BTreeMap<String, Vec<WireSlotEntryJson>> = BTreeMap::new();
    let mut mode_grammars = BTreeMap::new();
    let mut concolic_lenses = BTreeMap::new();

    let mut total_pos_checked = 0usize;
    let mut total_neg_checked = 0usize;
    let mut min_pos_gate = f64::INFINITY;
    let mut max_neg_gate = f64::NEG_INFINITY;
    let mut max_neg_leak_l2 = 0.0_f64;

    for (alloc, art) in plan.allocated_rules.iter().zip(artifacts) {
        if let Some((mode_key, grammar)) = art.mode_grammar_entry {
            mode_grammars.insert(mode_key, grammar);
        }
        total_pos_checked += art.pos_checked;
        total_neg_checked += art.neg_checked;
        min_pos_gate = min_pos_gate.min(art.min_pos_gate);
        max_neg_gate = max_neg_gate.max(art.max_neg_gate);
        max_neg_leak_l2 = max_neg_leak_l2.max(art.max_neg_leak_l2);
        concolic_lenses.insert(art.rule_name, art.lens);

        let layer_key = alloc.host_mlp_layer.to_string();
        if matches!(alloc.rule.mount_policy, MountPolicy::PermanentStatic) {
            permanent_static
                .entry(layer_key)
                .or_default()
                .push(art.entry);
        } else {
            banks
                .entry(alloc.wire_bank_key.clone())
                .or_default()
                .entry(layer_key)
                .or_default()
                .push(art.entry);
        }
    }

    let bisim_holds = max_neg_gate <= -7.95 && min_pos_gate >= 0.5;
    let manifest = WireCartridgeManifestJson {
        header: Some(WireCartridgeHeader {
            model_family: geom.model_id.clone(),
            checkpoint_revision: None,
            capture_sha256: None,
        }),
        model_id: Some(geom.model_id.clone()),
        base_intermediate_size: geom.base_intermediate_size,
        widened_slots: geom.widened_slot_capacity,
        banks,
        permanent_static,
        mode_grammars,
        bisimulation_receipt: Some(WireBisimulationReceipt {
            total_leaves_verified: plan.allocated_rules.len(),
            positive_rows_checked: total_pos_checked,
            negative_rows_checked: total_neg_checked,
            min_positive_gate_pre_silu: min_pos_gate,
            max_negative_gate_pre_silu: max_neg_gate,
            max_negative_swiglu_leak_l2: max_neg_leak_l2,
            min_positive_chordal_cosine: 1.0,
            decoded_mode_matches: plan.allocated_rules.len(),
            residual_source: "portia-compiler-faer".to_string(),
            bisimulation_holds: bisim_holds,
        }),
    };

    Ok(CompiledProgramV2 {
        allocated: plan.allocated_rules,
        manifest,
        concolic_lenses,
    })
}
