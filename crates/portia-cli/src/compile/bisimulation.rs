//! Authoritative `half::bf16` SwiGLU forward-pass bisimulation verifier over
//! [`WireCartridgeManifestJson`] (`compile::bisimulation`).

use crate::compile::input_specs::{PipeSlotObligationSpec, TensorPipeCaptureResponse};
use anyhow::{bail, Result};
use portia_compiler::numerics::{
    dot_f64, l2_norm, normalize_unit_vec, quantize_slice_bf16_f64, silu_f64, StateMatrix,
};
use portia_compiler::wire::{
    canonical_mode_readout_vector, parse_mount_policy_and_pen, WireBisimulationReceipt,
    WireCartridgeManifestJson, WireSlotEntryJson, TOTAL_RESERVED_SLOTS,
};
use portia_core::{CartridgeBank, LayerIndex, Mode, MountPolicy};
use std::collections::{BTreeMap, BTreeSet};

fn evaluate_wire_entry_bf16(
    entry: &WireSlotEntryJson,
    x_bf16: &[f64],
) -> Result<(f64, f64, Vec<f64>)> {
    let d = x_bf16.len();
    entry.verify_dimensions(d)?;
    let span = entry.slot_span();
    let mut g = Vec::with_capacity(span);
    let mut act = Vec::with_capacity(span);
    for (wg_row, wu_row) in entry.w_gate.iter().zip(entry.w_up.iter()).take(span) {
        let wg_q: Vec<f64> = wg_row.iter().map(|&v| f64::from(v)).collect();
        let wu_q: Vec<f64> = wu_row.iter().map(|&v| f64::from(v)).collect();
        let wg_bf16 = quantize_slice_bf16_f64(&wg_q);
        let wu_bf16 = quantize_slice_bf16_f64(&wu_q);
        let gk = dot_f64(&wg_bf16, x_bf16);
        let uk = dot_f64(&wu_bf16, x_bf16);
        g.push(gk);
        act.push(silu_f64(gk) * uk);
    }
    let mut dh = vec![0.0_f64; d];
    for (r, wd_row) in entry.w_down.iter().enumerate().take(d) {
        let mut acc = 0.0_f64;
        for (&act_k, &wd_rk) in act.iter().zip(wd_row.iter()).take(span) {
            let wd_q = quantize_slice_bf16_f64(&[f64::from(wd_rk)])[0];
            acc += act_k * wd_q;
        }
        dh[r] = acc;
    }
    let g0 = g[0];
    let g1_or_g0 = if span >= 2 { g[1] } else { g[0] };
    Ok((g0, g1_or_g0, dh))
}

fn project_canonical_mode_tangent(mode: Mode, u_rad: &[f64]) -> Option<Vec<f64>> {
    let d = u_rad.len();
    let base = canonical_mode_readout_vector(mode, d).ok()?;
    let rad_dot = dot_f64(&base, u_rad);
    let proj: Vec<f64> = base
        .iter()
        .zip(u_rad)
        .map(|(&b, &r)| b - rad_dot * r)
        .collect();
    normalize_unit_vec(&proj)
}

fn match_synthetic_canonical_mode(target: &[f64], u_rad: &[f64]) -> Option<Mode> {
    let d = target.len();
    if d < 8 {
        return None;
    }
    let t_rad = dot_f64(target, u_rad);
    let t_tan_raw: Vec<f64> = target
        .iter()
        .zip(u_rad)
        .map(|(&t, &r)| t - t_rad * r)
        .collect();
    let t_tan = normalize_unit_vec(&t_tan_raw)?;
    for mode in Mode::ALL {
        if let Some(m_tan) = project_canonical_mode_tangent(mode, u_rad) {
            if dot_f64(&t_tan, &m_tan) > 0.999_999 {
                return Some(mode);
            }
        }
    }
    None
}

fn decode_synthetic_mode_from_tangential_delta(delta_tan: &[f64], u_rad: &[f64]) -> Option<Mode> {
    let unit_delta = normalize_unit_vec(delta_tan)?;
    let mut best: Option<(Mode, f64)> = None;
    for mode in Mode::ALL {
        if let Some(m_tan) = project_canonical_mode_tangent(mode, u_rad) {
            let score = dot_f64(&unit_delta, &m_tan);
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((mode, score));
            }
        }
    }
    best.filter(|&(_, s)| s > 0.25).map(|(m, _)| m)
}

/// Verify pure-Rust `bf16` SwiGLU forward-pass bisimulation on a [`WireCartridgeManifestJson`].
pub fn verify_wire_cartridge_bisimulation(
    cartridge: &WireCartridgeManifestJson,
    capture: &TensorPipeCaptureResponse,
    obligations: &[PipeSlotObligationSpec],
) -> Result<WireBisimulationReceipt> {
    if obligations.is_empty() {
        bail!("verify_wire_cartridge_bisimulation requires at least 1 obligation");
    }
    cartridge.verify_schema_integrity()?;
    let d = capture.infer_hidden_dim()?;
    let base_inter = cartridge.base_intermediate_size;
    let cap = if cartridge.widened_slots == 0 {
        TOTAL_RESERVED_SLOTS
    } else {
        cartridge.widened_slots
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum CoveredEntrySource {
        PermanentStaticMap,
        BankMap(MountPolicy),
    }

    let mut union_pos_by_mount_layer: BTreeMap<(MountPolicy, LayerIndex), BTreeSet<usize>> =
        BTreeMap::new();
    for spec in obligations {
        let (_, mp, _) = parse_mount_policy_and_pen(&spec.bank_name)?;
        union_pos_by_mount_layer
            .entry((mp, LayerIndex::new(spec.layer)))
            .or_default()
            .extend(spec.positive_prompt_indices.iter().copied());
    }

    let mut covered: BTreeSet<(CoveredEntrySource, LayerIndex, usize, usize)> = BTreeSet::new();
    let mut positive_rows_checked = 0usize;
    let mut negative_rows_checked = 0usize;
    let mut min_positive_gate_pre_silu = f64::INFINITY;
    let mut max_negative_gate_pre_silu = f64::NEG_INFINITY;
    let mut max_negative_swiglu_leak_l2 = 0.0_f64;
    let mut min_positive_chordal_cosine = f64::INFINITY;
    let mut decoded_mode_matches = 0usize;
    let mut all_layers_have_observed_residuals = true;

    for spec in obligations {
        if !capture.has_observed_residuals(spec.layer) {
            all_layers_have_observed_residuals = false;
        }
        if spec.v_target.len() != d {
            bail!(
                "obligation {} v_target len ({}) != hidden_dim ({d})",
                spec.obligation_id,
                spec.v_target.len()
            );
        }
        let (norm_bank_key, mount_policy, _) = parse_mount_policy_and_pen(&spec.bank_name)?;
        let (abs_lo, abs_hi) = spec.resolve_physical_slot_bounds(base_inter, cap)?;
        let layer_idx = LayerIndex::new(spec.layer);
        let layer_key = layer_idx.to_string();

        let (source_key, target_entries) = match mount_policy {
            MountPolicy::PermanentStatic => {
                if let Some(entries) = cartridge.permanent_static.get(&layer_key) {
                    (CoveredEntrySource::PermanentStaticMap, entries)
                } else {
                    let legacy = cartridge
                        .banks
                        .get(&norm_bank_key)
                        .and_then(|m| m.get(&layer_key))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Missing permanent_static layer {layer_key:?} in WireCartridgeManifestJson"
                            )
                        })?;
                    (CoveredEntrySource::BankMap(mount_policy), legacy)
                }
            }
            MountPolicy::DynamicBank(_)
            | MountPolicy::ModePlan(_)
            | MountPolicy::PrefillSegment(_) => {
                let entries = cartridge
                    .banks
                    .get(&norm_bank_key)
                    .and_then(|m| m.get(&layer_key))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Missing bank {norm_bank_key:?} layer {layer_key:?} in WireCartridgeManifestJson"
                        )
                    })?;
                (CoveredEntrySource::BankMap(mount_policy), entries)
            }
        };

        let entry = target_entries
            .iter()
            .find(|e| e.slot_lo == abs_lo && e.slot_hi == abs_hi)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Obligation {} slots [{abs_lo}..{abs_hi}) not found in {:?} layer {layer_key:?}",
                    spec.obligation_id,
                    mount_policy.wire_key()
                )
            })?;
        if !covered.insert((source_key, layer_idx, entry.slot_lo, entry.slot_hi)) {
            bail!(
                "Duplicate obligation coverage for slot entry [{}..{}) in {:?} layer {layer_key:?}",
                entry.slot_lo,
                entry.slot_hi,
                mount_policy.wire_key()
            );
        }

        let co_mounted_static: &[WireSlotEntryJson] =
            if matches!(mount_policy, MountPolicy::DynamicBank(_)) {
                cartridge
                    .permanent_static
                    .get(&layer_key)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            } else {
                &[]
            };

        let pos_pre = StateMatrix::from_rows(
            "pos_pre",
            &capture.extract_pre_mlp_rows(spec.layer, &spec.positive_prompt_indices)?,
        )?
        .quantize_bf16();
        let pos_res = StateMatrix::from_rows(
            "pos_res",
            &capture.extract_residual_rows(spec.layer, &spec.positive_prompt_indices)?,
        )?;

        let mut all_neg_indices = spec.negative_prompt_indices.clone();
        for &bg_idx in &spec.background_prompt_indices {
            if !all_neg_indices.contains(&bg_idx) {
                all_neg_indices.push(bg_idx);
            }
        }
        let neg_pre = StateMatrix::from_rows(
            "neg_pre",
            &capture.extract_pre_mlp_rows(spec.layer, &all_neg_indices)?,
        )?
        .quantize_bf16();

        let h0_bar = pos_res.mean_row();
        let u_rad = normalize_unit_vec(&h0_bar).unwrap_or_else(|| vec![0.0; d]);
        let rad_dot_t = dot_f64(&u_rad, &spec.v_target);
        let v_tan_raw: Vec<f64> = (0..d)
            .map(|c| spec.v_target[c] - rad_dot_t * u_rad[c])
            .collect();
        let u_tan = normalize_unit_vec(&v_tan_raw)
            .or_else(|| normalize_unit_vec(&spec.v_target))
            .ok_or_else(|| {
                anyhow::anyhow!("degenerate v_target on obligation {}", spec.obligation_id)
            })?;

        let expected_canonical_mode =
            if mount_policy == MountPolicy::DynamicBank(CartridgeBank::OrientTrunk) {
                spec.expected_mode
                    .and_then(|m| {
                        match_synthetic_canonical_mode(&spec.v_target, &u_rad)
                            .filter(|&matched| matched == m)
                    })
                    .or_else(|| match_synthetic_canonical_mode(&spec.v_target, &u_rad))
            } else {
                None
            };

        for r in 0..pos_pre.nrows() {
            let x_row = pos_pre.row(r);
            let (g1, g2, dh_pair) = evaluate_wire_entry_bf16(entry, x_row)?;
            min_positive_gate_pre_silu = min_positive_gate_pre_silu.min(g1.min(g2));

            let dh_pair_norm = l2_norm(&dh_pair);
            let pair_cos = if dh_pair_norm > 1e-12 {
                dot_f64(&dh_pair, &u_tan) / dh_pair_norm
            } else {
                0.0
            };

            let mut dh_comp = vec![0.0_f64; d];
            for e in target_entries.iter().chain(co_mounted_static.iter()) {
                let (_, _, dh_e) = evaluate_wire_entry_bf16(e, x_row)?;
                for c in 0..d {
                    dh_comp[c] += dh_e[c];
                }
            }
            let comp_rad = dot_f64(&u_rad, &dh_comp);
            let delta_comp_tan: Vec<f64> =
                (0..d).map(|c| dh_comp[c] - comp_rad * u_rad[c]).collect();
            let comp_tan_norm = l2_norm(&delta_comp_tan);
            let composed_cos = if comp_tan_norm > 1e-12 {
                dot_f64(&delta_comp_tan, &u_tan) / comp_tan_norm
            } else {
                0.0
            };

            min_positive_chordal_cosine =
                min_positive_chordal_cosine.min(pair_cos.min(composed_cos));

            if g1 > 0.0 && g2 > 0.0 && pair_cos > 0.0 && composed_cos > 0.0 {
                if let Some(exp_mode) = expected_canonical_mode {
                    let decoded =
                        decode_synthetic_mode_from_tangential_delta(&delta_comp_tan, &u_rad);
                    if decoded != Some(exp_mode) {
                        bail!(
                            "Obligation {} composed bank-layer output decoded mode {decoded:?} != expected {exp_mode:?}",
                            spec.obligation_id
                        );
                    }
                }
                decoded_mode_matches += 1;
            }
            positive_rows_checked += 1;
        }

        let mut co_mounted_union_pos = BTreeSet::new();
        for (&(mp, lk), pset) in &union_pos_by_mount_layer {
            if lk == layer_idx && mp.co_mounted_with(mount_policy) {
                co_mounted_union_pos.extend(pset.iter().copied());
            }
        }
        for (r, &prompt_idx) in all_neg_indices.iter().enumerate().take(neg_pre.nrows()) {
            let x_row = neg_pre.row(r);
            let (g1, g2, dh_neg) = evaluate_wire_entry_bf16(entry, x_row)?;
            max_negative_gate_pre_silu = max_negative_gate_pre_silu.max(g1.max(g2));
            max_negative_swiglu_leak_l2 = max_negative_swiglu_leak_l2.max(l2_norm(&dh_neg));

            if !co_mounted_union_pos.contains(&prompt_idx) {
                let mut dh_comp_neg = vec![0.0_f64; d];
                for e in target_entries.iter().chain(co_mounted_static.iter()) {
                    let (_, _, dh_e) = evaluate_wire_entry_bf16(e, x_row)?;
                    for c in 0..d {
                        dh_comp_neg[c] += dh_e[c];
                    }
                }
                max_negative_swiglu_leak_l2 =
                    max_negative_swiglu_leak_l2.max(l2_norm(&dh_comp_neg));
            }
            negative_rows_checked += 1;
        }
    }

    for (layer_key, entries) in &cartridge.permanent_static {
        let layer_idx: LayerIndex = layer_key
            .parse()
            .map_err(|_| anyhow::anyhow!("non-numeric layer key {layer_key:?}"))?;
        for entry in entries {
            if !covered.contains(&(
                CoveredEntrySource::PermanentStaticMap,
                layer_idx,
                entry.slot_lo,
                entry.slot_hi,
            )) {
                bail!(
                    "Unaccounted executable slot entry [{}..{}) in permanent_static layer {layer_key:?}",
                    entry.slot_lo,
                    entry.slot_hi
                );
            }
        }
    }
    for (bank_name, layer_map) in &cartridge.banks {
        let (_, mp, _) = parse_mount_policy_and_pen(bank_name)?;
        for (layer_key, entries) in layer_map {
            let layer_idx: LayerIndex = layer_key
                .parse()
                .map_err(|_| anyhow::anyhow!("non-numeric layer key {layer_key:?}"))?;
            for entry in entries {
                if !covered.contains(&(
                    CoveredEntrySource::BankMap(mp),
                    layer_idx,
                    entry.slot_lo,
                    entry.slot_hi,
                )) {
                    bail!(
                        "Unaccounted executable slot entry [{}..{}) in bank {bank_name:?} layer {layer_key:?}",
                        entry.slot_lo,
                        entry.slot_hi
                    );
                }
            }
        }
    }

    let bisimulation_holds = !obligations.is_empty()
        && positive_rows_checked > 0
        && negative_rows_checked > 0
        && decoded_mode_matches == positive_rows_checked
        && min_positive_gate_pre_silu > 0.0
        && min_positive_chordal_cosine > 0.0
        && max_negative_gate_pre_silu < 0.0
        && max_negative_swiglu_leak_l2 < 0.25;

    let residual_source = if all_layers_have_observed_residuals {
        "observed_residuals".to_string()
    } else {
        "scaled_pre_mlp_7.25x_fallback".to_string()
    };

    let receipt = WireBisimulationReceipt {
        total_leaves_verified: obligations.len(),
        positive_rows_checked,
        negative_rows_checked,
        min_positive_gate_pre_silu,
        max_negative_gate_pre_silu,
        max_negative_swiglu_leak_l2,
        min_positive_chordal_cosine,
        decoded_mode_matches,
        residual_source,
        bisimulation_holds,
    };

    if !bisimulation_holds {
        bail!("WireCartridgeManifestJson forward bf16 SwiGLU bisimulation failed: {receipt:?}");
    }

    Ok(receipt)
}
