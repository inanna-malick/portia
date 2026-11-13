//! `ModelGeometryTables` (`DownstreamSwigluBlock`, `DeltaNetGateLayer`, `W_U_top`) and
//! `IntentExpr` + `RuleCaptureData` lowering preparation (`compile_v2::geometry`).

use crate::compile::PipeSlotObligationSpec;
use crate::compile_v2::obligations::PreparedV2CompileSpec;
use anyhow::{Context, Result};
use portia_compiler::discovery::discover_and_project_anti_echo;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::{ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::{normalize_unit_vec, StateMatrix};
use portia_compiler::solvers::clarabel_ternary::CursorPolarity;
use portia_compiler::solvers::{NullspacePolicy, V2SolverConfig};
use portia_compiler::wire::parse_mount_policy_and_pen;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{GrammarCartridgeSpec, Mode, MountPolicy};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct LoweredV2ProgramInputs {
    pub expr: IntentExpr,
    pub captures: BTreeMap<String, RuleCaptureData>,
    pub geom: ModelGeometryTables,
    pub resolved_obligations_by_rule: BTreeMap<String, PipeSlotObligationSpec>,
}

pub(crate) fn build_v2_intent_and_geometry(
    prep: &PreparedV2CompileSpec,
) -> Result<LoweredV2ProgramInputs> {
    let d = prep.hidden_dim;
    let geom = crate::compile_v2::model_tables::populate_model_geometry_tables(prep)?;

    // Collect union of positive prompt indices per `(MountPolicy, layer)` for clean sibling inhibition:
    let mut sibling_pos_by_mount_layer: BTreeMap<(MountPolicy, usize), BTreeSet<usize>> =
        BTreeMap::new();
    for ob in &prep.obligations {
        let (_, mp, _) = parse_mount_policy_and_pen(&ob.bank_name)?;
        sibling_pos_by_mount_layer
            .entry((mp, ob.layer))
            .or_default()
            .extend(ob.positive_prompt_indices.iter().copied());
    }

    let mut used_names = BTreeSet::new();
    let mut expr_opt: Option<IntentExpr> = None;
    let mut captures = BTreeMap::new();
    let mut resolved_obligations_by_rule = BTreeMap::new();

    for (idx, ob) in prep.obligations.iter().enumerate() {
        let mut eff_ob = ob.clone();
        let (_, mount_policy, _) = parse_mount_policy_and_pen(&eff_ob.bank_name)?;
        let rule_name = if used_names.insert(eff_ob.obligation_id.clone()) {
            eff_ob.obligation_id.clone()
        } else {
            let disamb = format!("{}#{idx}", eff_ob.obligation_id);
            used_names.insert(disamb.clone());
            disamb
        };

        let layer_key = eff_ob.layer.to_string();
        let layer_acts = prep
            .capture
            .activations
            .get(&layer_key)
            .with_context(|| format!("missing layer {layer_key} in activations"))?;
        let pos_rows: Vec<Vec<f64>> = eff_ob
            .positive_prompt_indices
            .iter()
            .map(|&r| {
                layer_acts
                    .get(r)
                    .cloned()
                    .with_context(|| format!("pos row {r} out of bounds"))
            })
            .collect::<Result<_>>()?;
        let on_mat = StateMatrix::from_rows("on_cursor", &pos_rows)?;

        let pos_res_rows: Vec<Vec<f64>> = if let Some(res_layer) = prep
            .capture
            .residuals
            .as_ref()
            .and_then(|m| m.get(&layer_key))
        {
            eff_ob
                .positive_prompt_indices
                .iter()
                .map(|&r| {
                    res_layer
                        .get(r)
                        .cloned()
                        .with_context(|| format!("res row {r} out of bounds"))
                })
                .collect::<Result<_>>()?
        } else {
            pos_rows
                .iter()
                .map(|row| row.iter().map(|&v| v * 7.25).collect())
                .collect()
        };
        let pos_res_mat = StateMatrix::from_rows("pos_residuals", &pos_res_rows)?;

        let pos_set: BTreeSet<usize> = eff_ob.positive_prompt_indices.iter().copied().collect();
        let mut neg_idx_set: BTreeSet<usize> = eff_ob
            .negative_prompt_indices
            .iter()
            .chain(&eff_ob.background_prompt_indices)
            .copied()
            .filter(|i| !pos_set.contains(i))
            .collect();
        if let Some(sibs) = sibling_pos_by_mount_layer.get(&(mount_policy, eff_ob.layer)) {
            for &s_idx in sibs {
                if !pos_set.contains(&s_idx) {
                    neg_idx_set.insert(s_idx);
                }
            }
        }
        let mut neg_rows: Vec<Vec<f64>> = neg_idx_set
            .iter()
            .map(|&r| {
                layer_acts
                    .get(r)
                    .cloned()
                    .with_context(|| format!("neg row {r} out of bounds"))
            })
            .collect::<Result<_>>()?;
        if let Some(ref bg_spec) = prep.background_activations {
            if let Some(extra) = bg_spec.rows_for_layer(eff_ob.layer) {
                neg_rows.extend(extra.iter().cloned());
            }
        }
        let off_mat = StateMatrix::from_rows("off_cursor", &neg_rows)?;

        let is_deltanet_1slot =
            eff_ob.slot_span() == 1 && geom.deltanet_layers.contains_key(&(eff_ob.layer + 1));
        let kind = if is_deltanet_1slot {
            let dn = &geom.deltanet_layers[&(eff_ob.layer + 1)];
            let half_h = (dn.num_heads / 2).max(1);
            let flush_heads: Vec<usize> = (0..half_h).collect();
            let lock_heads: Vec<usize> = (half_h..dn.num_heads).collect();
            HardwareIntentKind::DeltaNetRollbackScrub {
                upstream_mlp_layer: eff_ob.layer,
                target_deltanet_layer: eff_ob.layer + 1,
                flush_heads,
                lock_heads,
                a_drive: eff_ob.gate_target.max(3.2),
            }
        } else {
            if eff_ob.v_target.is_empty() {
                if let Some(qk_spec) = prep
                    .geometry
                    .as_ref()
                    .and_then(|g| g.qk_layer_vectors.get(&layer_key))
                {
                    eff_ob.v_target = qk_spec.resolve_combined_direction(eff_ob.layer, d)?;
                }
            }
            let unit_v = normalize_unit_vec(&eff_ob.v_target).ok_or_else(|| {
                anyhow::anyhow!(
                    "obligation {} has empty or zero v_target",
                    eff_ob.obligation_id
                )
            })?;
            // Record post-secant-anti-echo target on `eff_ob.v_target` for exact bisimulation audit:
            if let Some(ds_block) = geom.downstream_mlps.get(&(eff_ob.layer + 1)) {
                let (u_safe, _, _) = discover_and_project_anti_echo(
                    &rule_name,
                    ds_block,
                    &pos_res_mat,
                    &unit_v,
                    0.20,
                    32,
                )?;
                eff_ob.v_target = u_safe;
            } else {
                eff_ob.v_target = unit_v.clone();
            }
            let expected_mode: Option<Mode> = eff_ob.expected_mode;
            HardwareIntentKind::StanceRoute {
                layer_idx: eff_ob.layer,
                raw_target_dir: unit_v,
                angle: ChordalAngleRad::new(eff_ob.theta_rad.max(0.25)),
                expected_mode,
            }
        };

        let nullspace_policy = match mount_policy {
            MountPolicy::PermanentStatic => NullspacePolicy::StrictZeroCone,
            MountPolicy::DynamicBank(_)
            | MountPolicy::ModePlan(_)
            | MountPolicy::PrefillSegment(_) => {
                NullspacePolicy::BoundedSocSlack { max_l2_leak: 0.25 }
            }
        };
        let solver_config = V2SolverConfig {
            g_pos_target: eff_ob.gate_target,
            g_pos_band: 0.20,
            g_neg_ceiling: eff_ob.gate_neg_target,
            bf16_neg_pad: 0.35,
            trust_radius: 36.0,
            nullspace_policy,
            ..V2SolverConfig::default()
        };

        let n_pos = on_mat.nrows();
        captures.insert(
            rule_name.clone(),
            RuleCaptureData {
                on_cursor_pre_mlp: on_mat,
                polarities: vec![CursorPolarity::PlusOne; n_pos],
                pos_residuals: pos_res_mat,
                off_cursor_pre_mlp: off_mat,
                q_preserve_gate: None,
                solver_config,
            },
        );
        resolved_obligations_by_rule.insert(rule_name.clone(), eff_ob);

        let rule_leaf = IntentExpr::rule(IntentRule {
            name: rule_name,
            guard: RegimeExpr::ConstBool(true),
            mount_policy,
            grammar: GrammarCartridgeSpec::Unconstrained,
            kind,
        });
        expr_opt = Some(match expr_opt {
            None => rule_leaf,
            Some(acc) => acc >> rule_leaf,
        });
    }

    let expr = expr_opt.ok_or_else(|| anyhow::anyhow!("no obligations to compile"))?;
    Ok(LoweredV2ProgramInputs {
        expr,
        captures,
        geom,
        resolved_obligations_by_rule,
    })
}
