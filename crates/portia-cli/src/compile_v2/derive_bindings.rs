//! Guarded `RegimeExpr` rule-state binding -> `PipeSlotObligationSpec` derivation (`compile_v2::derive_bindings`).

use crate::compile::PipeSlotObligationSpec;
use anyhow::{bail, Result};
use portia_compiler::wire::canonical_mode_readout_vector;
use portia_core::pocket_dsl::GuardedRuleStateBinding;
use portia_core::{CartridgeBank, Mode};
use std::collections::BTreeMap;

fn mode_target_vector(mode: Mode, d: usize) -> Result<(Vec<f64>, f64)> {
    let v = canonical_mode_readout_vector(mode, d)?;
    let theta_rad = match mode {
        Mode::Probe => 0.58,
        Mode::Read => 0.65,
        Mode::Ask => 0.70,
        Mode::Write => 0.74,
        Mode::Abstain => 0.62,
        Mode::Result => 0.60,
    };
    Ok((v, theta_rad))
}

fn split_non_empty_indices(
    pos: &[usize],
    neg: &[usize],
    bg: &[usize],
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if pos.is_empty() {
        return None;
    }
    if !neg.is_empty() {
        return Some((pos.to_vec(), neg.to_vec(), bg.to_vec()));
    }
    if bg.len() >= 2 {
        return Some((pos.to_vec(), vec![bg[0]], bg[1..].to_vec()));
    }
    None
}

pub(crate) fn lookup_mode_target_override<'a>(
    map: &'a BTreeMap<String, Vec<f64>>,
    mode: Mode,
    d: usize,
    field_name: &str,
) -> Result<Option<&'a Vec<f64>>> {
    for (k, v) in map {
        let matched = k.eq_ignore_ascii_case(mode.as_str())
            || Mode::parse_flexible(&k.to_ascii_uppercase()) == Some(mode);
        if matched {
            if v.len() != d {
                bail!(
                    "{field_name}[{k:?}] length ({}) does not match hidden_dim ({d})",
                    v.len()
                );
            }
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn partition_payload_indices_for_mode(
    target_mode: Mode,
    bindings: &[GuardedRuleStateBinding],
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    let bg = bindings
        .first()
        .map(|b| b.background_prompt_indices.clone())
        .unwrap_or_default();
    for b in bindings {
        let dst = if b.mode == target_mode {
            &mut pos
        } else {
            &mut neg
        };
        for &idx in &b.positive_prompt_indices {
            if !dst.contains(&idx) {
                dst.push(idx);
            }
        }
    }
    if neg.is_empty() {
        for b in bindings.iter().filter(|b| b.mode == target_mode) {
            for &idx in &b.negative_prompt_indices {
                if !pos.contains(&idx) && !neg.contains(&idx) {
                    neg.push(idx);
                }
            }
        }
    }
    pos.sort_unstable();
    neg.sort_unstable();
    split_non_empty_indices(&pos, &neg, &bg).ok_or_else(|| {
        anyhow::anyhow!(
            "ModePayload({target_mode:?}) obligation has insufficient coverage on `row_states`"
        )
    })
}

pub(crate) fn derive_obligations_from_bindings(
    d: usize,
    layer: usize,
    active_modes: &[Mode],
    bindings: &[GuardedRuleStateBinding],
    mode_targets: Option<&BTreeMap<String, Vec<f64>>>,
    payload_targets: Option<&BTreeMap<String, Vec<f64>>>,
    allow_synthetic: bool,
) -> Result<Vec<PipeSlotObligationSpec>> {
    let mut obs = Vec::new();
    for (slot_idx, b) in bindings.iter().enumerate() {
        let Some((pos_idx, neg_idx, bg_idx)) = split_non_empty_indices(
            &b.positive_prompt_indices,
            &b.negative_prompt_indices,
            &b.background_prompt_indices,
        ) else {
            continue;
        };
        let explicit_v = match mode_targets {
            Some(m) => lookup_mode_target_override(m, b.mode, d, "mode_target_vectors")?.cloned(),
            None => None,
        };
        let theta_rad = match b.mode {
            Mode::Probe => 0.58,
            Mode::Read => 0.65,
            Mode::Ask => 0.70,
            Mode::Write => 0.74,
            Mode::Abstain => 0.62,
            Mode::Result => 0.60,
        };
        let v_target = match explicit_v {
            Some(v) => v,
            None if allow_synthetic => mode_target_vector(b.mode, d)?.0,
            None => bail!(
                "Production compile missing readout target vector for mode {:?} in `compiler_weight_geometry` / `mode_target_vectors`",
                b.mode
            ),
        };
        let slot_lo = 2 * (slot_idx % 16);
        obs.push(PipeSlotObligationSpec {
            readout_contrast: None,
            obligation_id: format!("{}_l{}", b.rule_id, layer),
            bank_name: "OrientTrunk".to_string(),
            expected_mode: Some(b.mode),
            layer,
            slot_lo,
            slot_hi: slot_lo + 2,
            positive_prompt_indices: pos_idx,
            negative_prompt_indices: neg_idx,
            background_prompt_indices: bg_idx,
            v_target,
            theta_rad,
            gate_target: 3.4,
            gate_neg_target: -8.0,
        });
    }
    if obs.is_empty() {
        bail!(
            "No guarded regime rule had non-empty positive and negative/background coverage over the supplied `row_states` (active_modes={active_modes:?})"
        );
    }
    if let Some(pmap) = payload_targets {
        for (p_idx, &mode) in active_modes.iter().enumerate() {
            if let Some(vp) = lookup_mode_target_override(pmap, mode, d, "payload_target_vectors")?
            {
                let (pos, neg, bg) = partition_payload_indices_for_mode(mode, bindings)?;
                let slot_lo = 32 + 2 * (p_idx % 16);
                obs.push(PipeSlotObligationSpec {
                    readout_contrast: None,
                    obligation_id: format!("payload_{}_col_l{layer}", mode.as_str().to_lowercase()),
                    bank_name: CartridgeBank::ModePayload(mode).wire_key().to_string(),
                    expected_mode: Some(mode),
                    layer,
                    slot_lo,
                    slot_hi: slot_lo + 2,
                    positive_prompt_indices: pos,
                    negative_prompt_indices: neg,
                    background_prompt_indices: bg,
                    v_target: vp.clone(),
                    theta_rad: 0.72,
                    gate_target: 3.5,
                    gate_neg_target: -8.0,
                });
            }
        }
    }
    Ok(obs)
}
