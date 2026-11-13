//! Input validation and `RegimeExpr` schema governance specialization for
//! `portia-cli compile` (`compile_v2::obligations`).

use crate::compile::{
    specialize_regime_and_columns_with_governance, BackgroundActivationsSpec,
    CompileCartridgeInputSpec, CompilerWeightGeometrySpec, PipeSlotObligationSpec,
    SchemaGovernanceColumns, SpecializedRegimeSummary, TensorPipeCaptureResponse,
    DEFAULT_QWEN35_08B_BASE_INTERMEDIATE_SIZE,
};
use crate::compile_v2::derive_bindings::{
    derive_obligations_from_bindings, lookup_mode_target_override,
};
use anyhow::{bail, Context, Result};
use portia_compiler::wire::{parse_mount_policy_and_pen, TOTAL_RESERVED_SLOTS};
use portia_core::pocket_dsl::{GuardedModeRule, GuardedRuleStateBinding, RegimeExpr};
use portia_core::{CartridgeBank, Mode, MountPolicy, WireBankKey};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) struct PreparedV2CompileSpec {
    pub raw_json: String,
    pub capture: TensorPipeCaptureResponse,
    pub hidden_dim: usize,
    pub model_family: String,
    pub checkpoint_revision: Option<String>,
    pub base_inter: usize,
    pub widened_slots: usize,
    pub geometry: Option<CompilerWeightGeometrySpec>,
    pub background_activations: Option<BackgroundActivationsSpec>,
    pub required_payload_modes: Vec<Mode>,
    pub obligations: Vec<PipeSlotObligationSpec>,
    pub regime_source: String,
    pub capture_source: String,
    pub obligation_source: String,
    pub matched_steward_columns: Vec<String>,
    pub matched_lock_columns: Vec<String>,
    pub active_modes: Vec<Mode>,
    pub guarded_rules: Vec<GuardedModeRule>,
    pub rule_state_bindings: Vec<GuardedRuleStateBinding>,
    #[allow(dead_code)]
    pub specialized_regime: RegimeExpr,
}

pub(crate) fn prepare_v2_compile_inputs(
    input_json: &Path,
    db: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
    governance: &SchemaGovernanceColumns,
    allow_synthetic_cli: bool,
) -> Result<PreparedV2CompileSpec> {
    let raw_json = std::fs::read_to_string(input_json)
        .with_context(|| format!("failed to read {input_json:?}"))?;
    let spec: CompileCartridgeInputSpec = serde_json::from_str(&raw_json)
        .with_context(|| format!("failed to parse {input_json:?}"))?;

    let mut eff_gov = governance.clone();
    for c in &spec.steward_columns {
        if !eff_gov.steward_columns.contains(c) {
            eff_gov.steward_columns.push(c.clone());
        }
    }
    for c in &spec.lock_columns {
        if !eff_gov.lock_columns.contains(c) {
            eff_gov.lock_columns.push(c.clone());
        }
    }
    eff_gov.schema_disclosed |= spec.schema_disclosed;

    let SpecializedRegimeSummary {
        specialized_regime,
        active_modes,
        guarded_rules,
        matched_steward_columns,
        matched_lock_columns,
        regime_source,
    } = specialize_regime_and_columns_with_governance(db, regime_json, read_only, &eff_gov)?;

    let capture = TensorPipeCaptureResponse {
        status: spec.status,
        activations: spec.activations,
        residuals: spec.residuals,
    };

    let has_explicit = spec.obligations.is_some();
    let has_states = spec.row_states.is_some();
    if !has_explicit && !has_states {
        bail!("External capture {input_json:?} omits both `obligations` and `row_states`");
    }
    if regime_json.is_some() && has_explicit && !has_states {
        bail!("--regime-json cannot rebind pre-baked `obligations` in {input_json:?} without `row_states`");
    }

    let d = capture.infer_hidden_dim()?;
    let (first_layer_key, first_layer) = capture
        .activations
        .keys()
        .filter_map(|k| k.parse::<usize>().ok().map(|l| (k.clone(), l)))
        .max_by_key(|&(_, l)| l)
        .unwrap_or_else(|| ("22".to_string(), 22));
    let num_rows = capture
        .activations
        .get(&first_layer_key)
        .map_or(0, Vec::len);

    let allow_synthetic = allow_synthetic_cli || spec.allow_synthetic_fixture_targets;
    let has_geom = spec.compiler_weight_geometry.is_some()
        || spec
            .mode_target_vectors
            .as_ref()
            .is_some_and(|m| !m.is_empty());
    if !allow_synthetic && !has_geom {
        bail!(
            "Production compile requires `compiler_weight_geometry` or explicit `mode_target_vectors` in {input_json:?}"
        );
    }

    let rule_state_bindings = match spec.row_states {
        Some(states) => {
            if states.len() != num_rows {
                bail!(
                    "`row_states` length ({}) != captured row count ({num_rows}) at layer {first_layer}",
                    states.len()
                );
            }
            specialized_regime.bind_capture_row_states(&states)
        }
        None => Vec::new(),
    };

    // Strip static anti-echo heuristic during raw unembedding extraction so contextual
    // chordal-secant `DownstreamSwigluBlock` discovery operates directly on `pos_residuals`:
    let raw_geom_opt = spec.compiler_weight_geometry.as_ref().map(|g| {
        let mut raw_g = g.clone();
        raw_g.downstream_mlp_anti_echo = None;
        raw_g
    });
    let eff_mode_targets: Option<BTreeMap<String, Vec<f64>>> = if let Some(ref g) = raw_geom_opt {
        let derived = g
            .derive_mode_readout_targets()
            .context("failed to derive natural BPE mode readout targets")?;
        let mut merged: BTreeMap<String, Vec<f64>> = derived
            .into_iter()
            .map(|(m, v)| (m.as_str().to_string(), v))
            .collect();
        if let Some(ref ov) = spec.mode_target_vectors {
            for (k, v) in ov {
                merged.insert(k.clone(), v.clone());
            }
        }
        Some(merged)
    } else {
        spec.mode_target_vectors.clone()
    };

    let (raw_obs, obligation_source, explicit_flag) =
        match (spec.obligations, has_states, regime_json.is_some()) {
            (Some(exp_obs), true, true) => {
                let mut reb = derive_obligations_from_bindings(
                    d,
                    first_layer,
                    &active_modes,
                    &rule_state_bindings,
                    eff_mode_targets.as_ref(),
                    spec.payload_target_vectors.as_ref(),
                    allow_synthetic,
                )?;
                reb.retain(|ob| {
                    parse_mount_policy_and_pen(&ob.bank_name).is_ok_and(|(_, mp, _)| {
                        mp == MountPolicy::DynamicBank(CartridgeBank::OrientTrunk)
                    })
                });
                for ob in exp_obs {
                    let (_, mp, _) = parse_mount_policy_and_pen(&ob.bank_name)?;
                    if mp != MountPolicy::DynamicBank(CartridgeBank::OrientTrunk) {
                        reb.push(ob);
                    }
                }
                (
                    reb,
                    "rebound_from_regime_over_input_row_states".to_string(),
                    true,
                )
            }
            (Some(exp_obs), _, _) => (exp_obs, "input_json".to_string(), true),
            (None, true, _) => (
                derive_obligations_from_bindings(
                    d,
                    first_layer,
                    &active_modes,
                    &rule_state_bindings,
                    eff_mode_targets.as_ref(),
                    spec.payload_target_vectors.as_ref(),
                    allow_synthetic,
                )?,
                "derived_from_regime_over_input_row_states".to_string(),
                false,
            ),
            (None, false, _) => unreachable!(),
        };

    let mut obligations = Vec::with_capacity(raw_obs.len());
    for mut ob in raw_obs {
        let (norm_bank, mp, _) = parse_mount_policy_and_pen(&ob.bank_name)?;
        if let MountPolicy::DynamicBank(CartridgeBank::ModePayload(m)) = mp {
            if !active_modes.contains(&m) {
                if explicit_flag {
                    bail!(
                        "Obligation {:?} targets {norm_bank} which is excluded by specialized active_modes {active_modes:?}",
                        ob.obligation_id
                    );
                }
                continue;
            }
        }
        if let Some(ref contrast) = ob.readout_contrast {
            let g = raw_geom_opt.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "obligation {} readout_contrast requires compiler_weight_geometry",
                    ob.obligation_id
                )
            })?;
            ob.v_target = g.derive_safe_token_readout_vector(contrast)?;
        } else if mp == MountPolicy::DynamicBank(CartridgeBank::OrientTrunk) {
            if let (Some(mode), Some(ref map)) = (ob.expected_mode, &eff_mode_targets) {
                if ob.v_target.is_empty() || spec.mode_target_vectors.is_some() {
                    if let Some(v_ov) =
                        lookup_mode_target_override(map, mode, d, "mode_target_vectors")?
                    {
                        ob.v_target = v_ov.clone();
                    }
                }
            }
        }
        obligations.push(ob);
    }

    validate_mount_scopes_and_payload_coverage(
        &capture,
        &spec.capture_row_banks,
        &spec.required_payload_modes,
        &obligations,
    )?;
    if !spec.required_payload_modes.is_empty() && spec.compiler_weight_geometry.is_none() {
        bail!("required payload banks need compiler_weight_geometry");
    }

    Ok(PreparedV2CompileSpec {
        raw_json,
        capture,
        hidden_dim: d,
        model_family: spec
            .model_family
            .unwrap_or_else(|| "Qwen/Qwen3.5-0.8B-Base".to_string()),
        checkpoint_revision: spec.checkpoint_revision,
        base_inter: spec
            .base_intermediate_size
            .unwrap_or(DEFAULT_QWEN35_08B_BASE_INTERMEDIATE_SIZE),
        widened_slots: spec.widened_slots.unwrap_or(TOTAL_RESERVED_SLOTS),
        geometry: spec.compiler_weight_geometry,
        background_activations: spec.background_activations,
        required_payload_modes: spec.required_payload_modes,
        obligations,
        regime_source,
        capture_source: "input_json".to_string(),
        obligation_source,
        matched_steward_columns,
        matched_lock_columns,
        active_modes,
        guarded_rules,
        rule_state_bindings,
        specialized_regime,
    })
}

fn validate_mount_scopes_and_payload_coverage(
    capture: &TensorPipeCaptureResponse,
    capture_row_banks: &[WireBankKey],
    required_payload_modes: &[Mode],
    obligations: &[PipeSlotObligationSpec],
) -> Result<()> {
    if !capture_row_banks.is_empty() {
        if capture
            .activations
            .values()
            .any(|rows| rows.len() != capture_row_banks.len())
        {
            bail!("capture_row_banks length must match captured rows at every layer");
        }
        for ob in obligations {
            let (_, mp, _) = parse_mount_policy_and_pen(&ob.bank_name)?;
            let expected_bank = match mp {
                MountPolicy::DynamicBank(bank) => Some(WireBankKey(bank)),
                MountPolicy::ModePlan(mode) => Some(WireBankKey(CartridgeBank::ModePayload(mode))),
                MountPolicy::PrefillSegment(_) => Some(WireBankKey(CartridgeBank::Base)),
                MountPolicy::PermanentStatic => None,
            };
            if let Some(exp) = expected_bank {
                for idx in ob
                    .positive_prompt_indices
                    .iter()
                    .chain(&ob.negative_prompt_indices)
                    .chain(&ob.background_prompt_indices)
                {
                    if capture_row_banks.get(*idx) != Some(&exp) {
                        bail!(
                            "obligation {} row {idx} lies outside its declared mount scope {}",
                            ob.obligation_id,
                            ob.bank_name
                        );
                    }
                }
            }
        }
    }
    for &req_mode in required_payload_modes {
        let bank = CartridgeBank::ModePayload(req_mode).wire_key();
        if !obligations.iter().any(|ob| {
            ob.bank_name == bank
                && ob.readout_contrast.is_some()
                && !ob.positive_prompt_indices.is_empty()
                && (!ob.negative_prompt_indices.is_empty()
                    || !ob.background_prompt_indices.is_empty())
        }) {
            bail!(
                "required payload bank {bank} is missing a natural-token obligation with positive and negative coverage"
            );
        }
    }
    Ok(())
}
