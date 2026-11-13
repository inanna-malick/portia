//! Real captured `Qwen3.5-0.8B` fixture test for `compile_v2` (`compile_v2::capture_fixture_tests`).

use super::*;
use crate::compile::{CompileCartridgeInputSpec, SchemaGovernanceColumns};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn compile_v2_compiles_real_captured_qwen35_08b_astra_verification_payload() {
    let Some(real_path_str) = std::env::var("PORTIA_CAPTURE_FIXTURE_JSON").ok() else {
        return;
    };
    let real_path = std::path::Path::new(&real_path_str);
    if !real_path.exists() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let in_path = if cfg!(debug_assertions) {
        let raw = std::fs::read_to_string(real_path).unwrap();
        let mut spec: CompileCartridgeInputSpec = serde_json::from_str(&raw).unwrap();
        spec.row_states = None;
        spec.capture_row_banks.clear();
        spec.required_payload_modes.clear();
        if let Some(ref mut obs) = spec.obligations {
            obs.truncate(3);
            let mut keep_rows = BTreeSet::new();
            for ob in obs.iter() {
                keep_rows.extend(&ob.positive_prompt_indices);
                keep_rows.extend(ob.negative_prompt_indices.iter().take(12));
                keep_rows.extend(ob.background_prompt_indices.iter().take(12));
            }
            let remap: BTreeMap<usize, usize> = keep_rows
                .iter()
                .copied()
                .enumerate()
                .map(|(ni, oi)| (oi, ni))
                .collect();
            for rows in spec.activations.values_mut() {
                *rows = keep_rows.iter().map(|&oi| rows[oi].clone()).collect();
            }
            if let Some(ref mut res_map) = spec.residuals {
                for rows in res_map.values_mut() {
                    *rows = keep_rows.iter().map(|&oi| rows[oi].clone()).collect();
                }
            }
            for ob in obs.iter_mut() {
                ob.positive_prompt_indices.retain_mut(|i| {
                    if let Some(&n) = remap.get(i) {
                        *i = n;
                        true
                    } else {
                        false
                    }
                });
                ob.negative_prompt_indices.retain(|i| remap.contains_key(i));
                for i in &mut ob.negative_prompt_indices {
                    *i = remap[i];
                }
                ob.background_prompt_indices
                    .retain(|i| remap.contains_key(i));
                for i in &mut ob.background_prompt_indices {
                    *i = remap[i];
                }
            }
        }
        if let Some(ref mut g) = spec.compiler_weight_geometry {
            if g.vocab_top_k_eigvecs.len() > 16 && g.vocab_top_k_eigvecs[0].len() == 1024 {
                g.vocab_top_k_eigvecs.truncate(16);
            }
        }
        let sliced = tmp.path().join("sliced_real_08b.json");
        std::fs::write(&sliced, serde_json::to_string(&spec).unwrap()).unwrap();
        sliced
    } else {
        real_path.to_path_buf()
    };
    let out_path = tmp.path().join("v2_real_08b_cartridge.json");
    let (cartridge, prov) = compile_cartridge_v2_with_governance(
        &in_path,
        None,
        None,
        false,
        &SchemaGovernanceColumns::default(),
        false,
        &out_path,
    )
    .expect("v2 compile on real Qwen3.5-0.8B captured activations must succeed");

    assert!(prov.bisimulation_verified);
    assert_eq!(prov.hidden_dim, 1024);
    cartridge.verify_schema_integrity().unwrap();

    for (layer_k, static_entries) in &cartridge.permanent_static {
        let static_slots: BTreeSet<usize> = static_entries
            .iter()
            .flat_map(|e| e.slot_lo..e.slot_hi)
            .collect();
        for bank_layers in cartridge.banks.values() {
            if let Some(dyn_entries) = bank_layers.get(layer_k) {
                for de in dyn_entries {
                    for s in de.slot_lo..de.slot_hi {
                        assert!(
                            !static_slots.contains(&s),
                            "layer {layer_k} slot {s} collides between permanent_static and dynamic bank"
                        );
                    }
                }
            }
        }
    }
}
