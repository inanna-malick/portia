//! Independent per-rule compilation worker (`lowering::rule_worker`) executed across
//! `std::thread::scope` worker threads and merged deterministically in `lowering::compile_intent_program`.

use crate::concolic::ConcolicDiodeLens;
use crate::discovery::discover_and_project_anti_echo;
use crate::discovery::gqa_bus::synthesize_asid_codebook;
use crate::error::V2CompilerError;
use crate::ir::regalloc::AllocatedIntentRule;
use crate::ir::HardwareIntentKind;
use crate::lowering::tables::{ModelGeometryTables, RuleCaptureData};
use crate::numerics::{l2_norm, normalize_unit_vec, StateMatrix};
use crate::solvers::clarabel_ternary::compile_coupled_ternary_pair;
use crate::solvers::deltanet_scrub::compile_deltanet_head_partition_scrubber;
use crate::wire::{CompiledSlotBf16, WireSlotEntryJson};
use portia_core::GrammarCartridgeSpec;
use std::collections::BTreeMap;

pub(crate) struct CompiledRuleArtifact {
    pub rule_name: String,
    pub mode_grammar_entry: Option<(String, GrammarCartridgeSpec)>,
    pub lens: ConcolicDiodeLens,
    pub entry: WireSlotEntryJson,
    pub pos_checked: usize,
    pub neg_checked: usize,
    pub min_pos_gate: f64,
    pub max_neg_gate: f64,
    pub max_neg_leak_l2: f64,
}

/// Compile a single register-allocated [`AllocatedIntentRule`] from shared immutable capture &
/// geometry tables (`Send + Sync`, zero shared mutable state).
pub(crate) fn compile_allocated_rule(
    alloc: &AllocatedIntentRule,
    captures: &BTreeMap<String, RuleCaptureData>,
    geom: &ModelGeometryTables,
) -> Result<CompiledRuleArtifact, V2CompilerError> {
    let name = &alloc.rule.name;
    let cap = captures
        .get(name)
        .ok_or_else(|| V2CompilerError::InvalidConfig {
            context: "compile_intent_program",
            reason: format!("missing RuleCaptureData for rule `{name}`"),
        })?;

    let mut neg_blocks: Vec<&StateMatrix> = vec![&cap.off_cursor_pre_mlp];
    for high_name in &alloc.inhibited_by_rules {
        if let Some(high_cap) = captures.get(high_name) {
            neg_blocks.push(&high_cap.on_cursor_pre_mlp);
        }
    }
    let merged_off = StateMatrix::vstack("inhibited_off_cursor", &neg_blocks)?;

    let mode_grammar_entry = if alloc.rule.grammar.is_constrained() {
        if let HardwareIntentKind::StanceRoute {
            expected_mode: Some(m),
            ..
        } = &alloc.rule.kind
        {
            Some((format!("{m:?}"), alloc.rule.grammar.clone()))
        } else {
            None
        }
    } else {
        None
    };

    let mut min_pos_gate = f64::INFINITY;
    let mut max_neg_gate = f64::NEG_INFINITY;

    let compiled_slots: Vec<CompiledSlotBf16> = match &alloc.rule.kind {
        HardwareIntentKind::StanceRoute {
            layer_idx,
            raw_target_dir,
            angle,
            ..
        } => {
            let (safe_dir, q_anti_opt) =
                if let Some(ds_block) = geom.downstream_mlps.get(&(layer_idx + 1)) {
                    let (u_safe, q_anti, _) = discover_and_project_anti_echo(
                        name,
                        ds_block,
                        &cap.pos_residuals,
                        raw_target_dir,
                        0.20,
                        32,
                    )?;
                    (u_safe, Some(q_anti))
                } else {
                    (raw_target_dir.clone(), None)
                };
            let pair = compile_coupled_ternary_pair(
                name,
                alloc.rel_slot_lo,
                alloc.rel_slot_lo + 1,
                &cap.on_cursor_pre_mlp,
                &cap.polarities,
                &cap.pos_residuals,
                &merged_off,
                &safe_dir,
                *angle,
                cap.q_preserve_gate.as_ref(),
                q_anti_opt.as_ref(),
                &cap.solver_config,
            )?;
            min_pos_gate = min_pos_gate.min(pair.min_pos_gate_bf16);
            max_neg_gate = max_neg_gate.max(pair.max_neg_gate_bf16);
            vec![pair.radial_slot, pair.tangential_slot]
        }

        HardwareIntentKind::SignedTernaryFork {
            layer_idx,
            branch_axis_dir,
            angle,
        } => {
            let q_anti_opt = if let Some(ds_block) = geom.downstream_mlps.get(&(layer_idx + 1)) {
                let (_, q_anti, _) = discover_and_project_anti_echo(
                    name,
                    ds_block,
                    &cap.pos_residuals,
                    branch_axis_dir,
                    0.20,
                    32,
                )?;
                Some(q_anti)
            } else {
                None
            };
            let pair = compile_coupled_ternary_pair(
                name,
                alloc.rel_slot_lo,
                alloc.rel_slot_lo + 1,
                &cap.on_cursor_pre_mlp,
                &cap.polarities,
                &cap.pos_residuals,
                &merged_off,
                branch_axis_dir,
                *angle,
                cap.q_preserve_gate.as_ref(),
                q_anti_opt.as_ref(),
                &cap.solver_config,
            )?;
            min_pos_gate = min_pos_gate.min(pair.min_pos_gate_bf16);
            max_neg_gate = max_neg_gate.max(pair.max_neg_gate_bf16);
            vec![pair.radial_slot, pair.tangential_slot]
        }

        HardwareIntentKind::SubmergedPointerRoute {
            target_gqa_layer,
            copy_heads,
            distractor_heads,
            query_passthrough_delta,
            angle,
            ..
        } => {
            let gqa = geom.gqa_layers.get(target_gqa_layer).ok_or_else(|| {
                V2CompilerError::InvalidConfig {
                    context: "SubmergedPointerRoute",
                    reason: format!("missing GqaPointerLayer for layer {target_gqa_layer}"),
                }
            })?;
            let q_submerged =
                gqa.build_submerged_forbidden_basis(geom.w_u_top_basis.as_ref(), 512)?;
            let mu_h = cap.pos_residuals.mean_row();
            let mut raw = vec![0.0; gqa.hidden_dim];
            for &h in copy_heads {
                let pb_q = gqa.pullback_query_passthrough(h, query_passthrough_delta);
                let pb_g = gqa.pullback_head_gate(h, &mu_h);
                for j in 0..gqa.hidden_dim {
                    raw[j] += 0.65 * pb_q[j] + 0.35 * pb_g[j];
                }
            }
            for &h in distractor_heads {
                let pb_g = gqa.pullback_head_gate(h, &mu_h);
                for j in 0..gqa.hidden_dim {
                    raw[j] -= 0.35 * pb_g[j];
                }
            }
            let submerged = q_submerged.project_out(&raw);
            let unit_dir = normalize_unit_vec(&submerged).ok_or_else(|| {
                V2CompilerError::DegenerateProjectedTarget {
                    rule: name.clone(),
                    norm: l2_norm(&submerged),
                }
            })?;
            let pair = compile_coupled_ternary_pair(
                name,
                alloc.rel_slot_lo,
                alloc.rel_slot_lo + 1,
                &cap.on_cursor_pre_mlp,
                &cap.polarities,
                &cap.pos_residuals,
                &merged_off,
                &unit_dir,
                *angle,
                cap.q_preserve_gate.as_ref(),
                Some(&q_submerged),
                &cap.solver_config,
            )?;
            min_pos_gate = min_pos_gate.min(pair.min_pos_gate_bf16);
            max_neg_gate = max_neg_gate.max(pair.max_neg_gate_bf16);
            vec![pair.radial_slot, pair.tangential_slot]
        }

        HardwareIntentKind::AsidSegmentColor {
            target_gqa_layer,
            copy_heads,
            segment,
            is_write_side_key_stamp,
            angle,
            ..
        } => {
            let gqa = geom.gqa_layers.get(target_gqa_layer).ok_or_else(|| {
                V2CompilerError::InvalidConfig {
                    context: "AsidSegmentColor",
                    reason: format!("missing GqaPointerLayer for layer {target_gqa_layer}"),
                }
            })?;
            let q_submerged =
                gqa.build_submerged_forbidden_basis(geom.w_u_top_basis.as_ref(), 512)?;
            let codebook = synthesize_asid_codebook(gqa, copy_heads, &q_submerged)?;
            let dir = if *is_write_side_key_stamp {
                &codebook.write_key_tags[*segment as usize]
            } else {
                &codebook.read_query_selectors[*segment as usize]
            };
            let pair = compile_coupled_ternary_pair(
                name,
                alloc.rel_slot_lo,
                alloc.rel_slot_lo + 1,
                &cap.on_cursor_pre_mlp,
                &cap.polarities,
                &cap.pos_residuals,
                &merged_off,
                dir,
                *angle,
                cap.q_preserve_gate.as_ref(),
                Some(&q_submerged),
                &cap.solver_config,
            )?;
            min_pos_gate = min_pos_gate.min(pair.min_pos_gate_bf16);
            max_neg_gate = max_neg_gate.max(pair.max_neg_gate_bf16);
            vec![pair.radial_slot, pair.tangential_slot]
        }

        HardwareIntentKind::DeltaNetRollbackScrub {
            target_deltanet_layer,
            flush_heads,
            lock_heads,
            a_drive,
            ..
        } => {
            let dn = geom
                .deltanet_layers
                .get(target_deltanet_layer)
                .ok_or_else(|| V2CompilerError::InvalidConfig {
                    context: "DeltaNetRollbackScrub",
                    reason: format!("missing DeltaNetGateLayer for layer {target_deltanet_layer}"),
                })?;
            let scrubbed = compile_deltanet_head_partition_scrubber(
                name,
                alloc.rel_slot_lo,
                dn,
                flush_heads,
                lock_heads,
                *a_drive,
                &cap.on_cursor_pre_mlp,
                &merged_off,
                geom.w_u_top_basis.as_ref(),
                &cap.solver_config,
            )?;
            for r in 0..cap.on_cursor_pre_mlp.nrows() {
                let (g, _) = scrubbed
                    .slot
                    .eval_gate_and_act(cap.on_cursor_pre_mlp.row(r));
                min_pos_gate = min_pos_gate.min(g);
            }
            vec![scrubbed.slot]
        }
    };

    let mut max_neg_leak_l2 = 0.0_f64;
    for r in 0..merged_off.nrows() {
        let x = merged_off.row(r);
        let mut delta = vec![0.0_f64; x.len()];
        for s in &compiled_slots {
            let (g, act) = s.eval_gate_and_act(x);
            max_neg_gate = max_neg_gate.max(g);
            for (di, &wd) in delta.iter_mut().zip(&s.w_down) {
                *di += act * wd.to_f64();
            }
        }
        max_neg_leak_l2 = max_neg_leak_l2.max(l2_norm(&delta));
    }

    let lens = ConcolicDiodeLens::build(
        name.clone(),
        alloc.host_mlp_layer,
        &compiled_slots[0].w_gate,
        geom.w_u_top_basis.as_ref(),
        &cap.on_cursor_pre_mlp,
        4,
    )?;
    let entry =
        WireSlotEntryJson::from_contiguous_slots(&compiled_slots, geom.base_intermediate_size)?;

    Ok(CompiledRuleArtifact {
        rule_name: name.clone(),
        mode_grammar_entry,
        lens,
        entry,
        pos_checked: cap.on_cursor_pre_mlp.nrows(),
        neg_checked: merged_off.nrows(),
        min_pos_gate,
        max_neg_gate,
        max_neg_leak_l2,
    })
}
