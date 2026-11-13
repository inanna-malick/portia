//! Bounded-memory (`< 12 MiB` RSS, zero Python/PyTorch) streaming `.safetensors` checkpoint
//! exporter (`portia-cli export-safetensors`) powered by [`safetensors_surgeon`].
//!
//! Materializes a compiled [`WireCartridgeManifestJson`]'s `permanent_static` (and any opted-in
//! `--include-bank`) SwiGLU slots directly into a single-file or multi-shard HuggingFace
//! checkpoint directory, widening `intermediate_size: M -> M + K` uniformly across all layers.

use anyhow::{bail, Context, Result};
use portia_compiler::wire::{WireCartridgeManifestJson, WireSlotEntryJson};
use portia_core::{BaseIntermediateSize, PhysicalSlotIndex};
use safetensors_surgeon::{
    apply_checkpoint_patch, LayerMlpAppendSpec, MlpWidenPlan, OutputStrategy, PatchReceipt,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
struct HfMlpConfigSummary {
    layer_prefix: String,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

fn inspect_hf_config(checkpoint_dir: &Path) -> Result<HfMlpConfigSummary> {
    let cfg_path = checkpoint_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("failed to read {}", cfg_path.display()))?;
    let val: Value = serde_json::from_str(&raw)
        .with_context(|| format!("invalid JSON in {}", cfg_path.display()))?;

    let (section, has_text_cfg) = match val.get("text_config") {
        Some(tc) if tc.is_object() => (tc, true),
        _ => (&val, false),
    };
    let num_layers = section
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing num_hidden_layers in {}", cfg_path.display()))?
        as usize;
    let hidden_size = section
        .get("hidden_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing hidden_size in {}", cfg_path.display()))?
        as usize;
    let intermediate_size = section
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing intermediate_size in {}", cfg_path.display()))?
        as usize;

    // Detect `model.language_model.layers` vs `model.layers` from index or fallback to `text_config`:
    let idx_path = checkpoint_dir.join("model.safetensors.index.json");
    let layer_prefix = if idx_path.exists() {
        let idx_raw = std::fs::read_to_string(&idx_path)?;
        if idx_raw.contains("\"model.language_model.layers.") {
            "model.language_model.layers".to_string()
        } else {
            "model.layers".to_string()
        }
    } else if has_text_cfg {
        "model.language_model.layers".to_string()
    } else {
        "model.layers".to_string()
    };

    Ok(HfMlpConfigSummary {
        layer_prefix,
        num_layers,
        hidden_size,
        intermediate_size,
    })
}

struct LayerAppendBuffers {
    occupied: Vec<bool>,
    gate_rows: Vec<f32>,
    up_rows: Vec<f32>,
    down_cols: Vec<f32>,
}

fn stamp_entry_into_layer_buffers(
    entry: &WireSlotEntryJson,
    hf: &HfMlpConfigSummary,
    added_slots: usize,
    bufs: &mut LayerAppendBuffers,
) -> Result<()> {
    let base_m = hf.intermediate_size;
    let d = hf.hidden_size;
    let k = entry.slot_hi.saturating_sub(entry.slot_lo);
    let (rel_lo_typed, _) =
        PhysicalSlotIndex::resolve_flexible_boundary(entry.slot_lo, BaseIntermediateSize(base_m));
    let rel_lo = rel_lo_typed.get();
    let rel_hi = rel_lo + k;
    if rel_hi > added_slots {
        bail!(
            "slot entry [{}..{}) exceeds widened capacity {}",
            rel_lo,
            rel_hi,
            added_slots
        );
    }
    entry
        .verify_dimensions(d)
        .map_err(|e| anyhow::anyhow!("invalid WireSlotEntryJson dimensions: {e}"))?;

    for s in 0..k {
        let rel_s = rel_lo + s;
        if bufs.occupied[rel_s] {
            bail!("overlapping static slot export at relative slot {rel_s}");
        }
        bufs.occupied[rel_s] = true;
        bufs.gate_rows[rel_s * d..(rel_s + 1) * d].copy_from_slice(&entry.w_gate[s]);
        bufs.up_rows[rel_s * d..(rel_s + 1) * d].copy_from_slice(&entry.w_up[s]);
        for r in 0..d {
            bufs.down_cols[r * added_slots + rel_s] = entry.w_down[r][s];
        }
    }
    Ok(())
}

/// Stream-patch `checkpoint_dir` into `out_dir` with `< 12 MiB` bounded working memory, widening
/// all layers `M -> M + added_slots` and stamping `cartridge`'s `permanent_static` (+ `include_banks`).
pub fn export_cartridge_to_safetensors_dir(
    checkpoint_dir: &Path,
    cartridge_json: &Path,
    out_dir: &Path,
    include_banks: &[String],
    copy_unmodified_shards: bool,
) -> Result<PatchReceipt> {
    let hf = inspect_hf_config(checkpoint_dir)?;
    let raw_cart = std::fs::read_to_string(cartridge_json)
        .with_context(|| format!("failed to read {}", cartridge_json.display()))?;
    let cart = WireCartridgeManifestJson::from_json_str(&raw_cart).with_context(|| {
        format!(
            "invalid WireCartridgeManifestJson in {}",
            cartridge_json.display()
        )
    })?;
    cart.verify_schema_integrity()
        .context("cartridge failed verify_schema_integrity() before safetensors export")?;

    if cart.base_intermediate_size != hf.intermediate_size {
        bail!(
            "cartridge base_intermediate_size ({}) != checkpoint intermediate_size ({})",
            cart.base_intermediate_size,
            hf.intermediate_size
        );
    }
    let added_slots = cart.widened_slots;
    if added_slots == 0 {
        bail!("cartridge widened_slots must be > 0 for safetensors export");
    }

    let mut by_layer: BTreeMap<usize, Vec<&WireSlotEntryJson>> = BTreeMap::new();
    for (lk, entries) in &cart.permanent_static {
        let l_idx: usize = lk
            .parse()
            .with_context(|| format!("invalid layer key {lk:?}"))?;
        by_layer.entry(l_idx).or_default().extend(entries.iter());
    }
    for bank in include_banks {
        let bank_map = cart.banks.get(bank).ok_or_else(|| {
            anyhow::anyhow!("requested --include-bank {bank:?} not found in cartridge")
        })?;
        for (lk, entries) in bank_map {
            let l_idx: usize = lk
                .parse()
                .with_context(|| format!("invalid layer key {lk:?}"))?;
            by_layer.entry(l_idx).or_default().extend(entries.iter());
        }
    }
    if by_layer.is_empty() {
        bail!("no slot entries found in permanent_static or requested --include-bank set");
    }

    let mut target_layers = Vec::with_capacity(by_layer.len());
    for (layer_index, entries) in by_layer {
        let len = added_slots * hf.hidden_size;
        let mut bufs = LayerAppendBuffers {
            occupied: vec![false; added_slots],
            gate_rows: vec![0.0_f32; len],
            up_rows: vec![0.0_f32; len],
            down_cols: vec![0.0_f32; len],
        };
        for e in entries {
            stamp_entry_into_layer_buffers(e, &hf, added_slots, &mut bufs)?;
        }
        target_layers.push(LayerMlpAppendSpec {
            layer_index,
            gate_rows: bufs.gate_rows,
            up_rows: bufs.up_rows,
            down_cols: bufs.down_cols,
        });
    }

    let plan = MlpWidenPlan {
        layer_prefix: hf.layer_prefix,
        num_layers: hf.num_layers,
        hidden_size: hf.hidden_size,
        base_intermediate_size: hf.intermediate_size,
        added_slots,
        target_layers,
        uniform_across_all_layers: true,
        output_strategy: if copy_unmodified_shards {
            OutputStrategy::CopyUnmodifiedShards
        } else {
            OutputStrategy::HardlinkUnmodifiedShards
        },
    }
    .into_patch_plan()
    .context("safetensors-surgeon MlpWidenPlan validation failed")?;

    apply_checkpoint_patch(checkpoint_dir, out_dir, &plan)
        .context("safetensors-surgeon apply_checkpoint_patch failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use tempfile::TempDir;

    #[test]
    fn exports_qwen35_nested_text_config_safetensors_in_pure_rust() {
        let tmp = TempDir::new().unwrap();
        let in_dir = tmp.path().join("hf_in");
        let out_dir = tmp.path().join("hf_out");
        std::fs::create_dir_all(&in_dir).unwrap();

        std::fs::write(
            in_dir.join("config.json"),
            r#"{"model_type":"qwen3_5","text_config":{"num_hidden_layers":2,"hidden_size":4,"intermediate_size":6}}"#,
        )
        .unwrap();

        let bf_zeros = |n: usize| -> Vec<u8> {
            vec![bf16::from_f32(0.5); n]
                .into_iter()
                .flat_map(|x| x.to_le_bytes())
                .collect()
        };
        let g_buf = bf_zeros(6 * 4);
        let d_buf = bf_zeros(4 * 6);
        let mut tensors = Vec::new();
        for l in 0..2 {
            tensors.push((
                format!("model.language_model.layers.{l}.mlp.gate_proj.weight"),
                vec![6, 4],
                g_buf.clone(),
            ));
            tensors.push((
                format!("model.language_model.layers.{l}.mlp.up_proj.weight"),
                vec![6, 4],
                g_buf.clone(),
            ));
            tensors.push((
                format!("model.language_model.layers.{l}.mlp.down_proj.weight"),
                vec![4, 6],
                d_buf.clone(),
            ));
        }
        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .map(|(k, sh, b)| {
                (
                    k.as_str(),
                    TensorView::new(Dtype::BF16, sh.clone(), b).unwrap(),
                )
            })
            .collect();
        serialize_to_file(views, None, &in_dir.join("model.safetensors")).unwrap();

        let mut permanent_static = BTreeMap::new();
        permanent_static.insert(
            "1".to_string(),
            vec![WireSlotEntryJson {
                slot_lo: 6,
                slot_hi: 8,
                w_gate: vec![vec![1.0; 4], vec![2.0; 4]],
                w_up: vec![vec![0.5; 4], vec![-0.5; 4]],
                w_down: vec![vec![0.25, -0.25]; 4],
            }],
        );
        let cart = WireCartridgeManifestJson {
            header: None,
            model_id: Some("Qwen/Qwen3.5-0.8B-Base".into()),
            base_intermediate_size: 6,
            widened_slots: 2,
            banks: BTreeMap::new(),
            permanent_static,
            mode_grammars: BTreeMap::new(),
            bisimulation_receipt: None,
        };
        let cart_path = tmp.path().join("cart.json");
        std::fs::write(&cart_path, cart.to_json_compact().unwrap()).unwrap();

        let receipt =
            export_cartridge_to_safetensors_dir(&in_dir, &cart_path, &out_dir, &[], false).unwrap();
        assert_eq!(receipt.tensors_modified, 6);
        let patched_cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(out_dir.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(patched_cfg["text_config"]["intermediate_size"], 8);
    }
}
