//! Pure-Rust zero-Python `.safetensors` tensor loader (`safetensors_io`).
//!
//! Reads `BF16`, `F16`, and `F32` tensors directly from single-file or sharded HuggingFace
//! `.safetensors` checkpoints (`[u64 LE header_len][JSON header][raw tensor bytes]`) straight into
//! `portia-compiler`'s [`StateMatrix`], eliminating Python `safetensors` bridge overhead when
//! extracting `W_U`, `GqaPointerLayer`, `DownstreamSwigluBlock`, and `DeltaNetGateLayer` geometry.

use crate::error::V2CompilerError;
use crate::numerics::StateMatrix;
use half::{bf16, f16};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct SafetensorEntryMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
struct ShardedIndexJson {
    weight_map: BTreeMap<String, String>,
}

fn candidate_tensor_names(tensor_name: &str) -> Vec<String> {
    let mut out = vec![tensor_name.to_string()];
    if let Some(rest) = tensor_name.strip_prefix("model.language_model.") {
        out.push(format!("model.{rest}"));
    } else if let Some(rest) = tensor_name.strip_prefix("model.") {
        out.push(format!("model.language_model.{rest}"));
    }
    if tensor_name == "lm_head.weight" {
        out.push("model.embed_tokens.weight".to_string());
        out.push("model.language_model.embed_tokens.weight".to_string());
    }
    out
}

/// Resolve which `.safetensors` file inside `snapshot_dir` holds `tensor_name` (supporting both
/// `Qwen3_5ForCausalLM` `model.layers.*` and multimodal `model.language_model.layers.*` keys).
fn resolve_shard_path(
    snapshot_dir: &Path,
    tensor_name: &str,
) -> Result<(PathBuf, Vec<String>), V2CompilerError> {
    let candidates = candidate_tensor_names(tensor_name);
    let index_path = snapshot_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| V2CompilerError::WireIntegrity(format!("read index.json: {e}")))?;
        let idx: ShardedIndexJson = serde_json::from_str(&raw)
            .map_err(|e| V2CompilerError::WireIntegrity(format!("parse index.json: {e}")))?;
        for cand in &candidates {
            if let Some(shard) = idx.weight_map.get(cand) {
                return Ok((snapshot_dir.join(shard), vec![cand.clone()]));
            }
        }
        Err(V2CompilerError::WireIntegrity(format!(
            "tensor `{tensor_name}` not in weight_map"
        )))
    } else {
        Ok((snapshot_dir.join("model.safetensors"), candidates))
    }
}

/// Load any 1D or 2D (`BF16`, `F16`, or `F32`) tensor by name from a HuggingFace snapshot directory
/// directly into `(shape, Vec<f64>)` in pure Rust.
pub fn load_safetensor_f64(
    snapshot_dir: &Path,
    tensor_name: &str,
) -> Result<(Vec<usize>, Vec<f64>), V2CompilerError> {
    let (file_path, candidates) = resolve_shard_path(snapshot_dir, tensor_name)?;
    let mut f = File::open(&file_path).map_err(|e| {
        V2CompilerError::WireIntegrity(format!("open {}: {e}", file_path.display()))
    })?;

    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("read header len: {e}")))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    let mut hdr_bytes = vec![0u8; header_len];
    f.read_exact(&mut hdr_bytes)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("read header json: {e}")))?;

    let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hdr_bytes)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("parse safetensors header: {e}")))?;
    let val = candidates
        .iter()
        .find_map(|cand| raw_map.get(cand))
        .ok_or_else(|| {
            V2CompilerError::WireIntegrity(format!(
                "tensor `{tensor_name}` missing in shard header"
            ))
        })?;
    let meta: SafetensorEntryMeta = serde_json::from_value(val.clone())
        .map_err(|e| V2CompilerError::WireIntegrity(format!("parse tensor meta: {e}")))?;

    let byte_start = 8u64 + (header_len as u64) + meta.data_offsets[0];
    let byte_len = (meta.data_offsets[1] - meta.data_offsets[0]) as usize;
    f.seek(SeekFrom::Start(byte_start))
        .map_err(|e| V2CompilerError::WireIntegrity(format!("seek tensor: {e}")))?;
    let mut raw = vec![0u8; byte_len];
    f.read_exact(&mut raw)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("read tensor bytes: {e}")))?;

    let elems: Vec<f64> = match meta.dtype.as_str() {
        "BF16" => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16::from_bits(u16::from_le_bytes(c)).to_f64())
            .collect(),
        "F16" => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| f16::from_bits(u16::from_le_bytes(c)).to_f64())
            .collect(),
        "F32" => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c) as f64)
            .collect(),
        other => {
            return Err(V2CompilerError::WireIntegrity(format!(
                "unsupported safetensors dtype `{other}` on `{tensor_name}`"
            )));
        }
    };
    Ok((meta.shape, elems))
}

/// Load a 2D weight matrix from `.safetensors` straight into a validated [`StateMatrix`].
pub fn load_safetensor_matrix_2d(
    snapshot_dir: &Path,
    tensor_name: &str,
) -> Result<StateMatrix, V2CompilerError> {
    let (shape, data) = load_safetensor_f64(snapshot_dir, tensor_name)?;
    if shape.len() != 2 {
        return Err(V2CompilerError::InvalidConfig {
            context: "load_safetensor_matrix_2d",
            reason: format!("expected 2D tensor for `{tensor_name}`, got shape {shape:?}"),
        });
    }
    StateMatrix::try_new("safetensors_2d", shape[0], shape[1], data)
}

/// Load a 1D `RMSNorm` weight vector `w` from `.safetensors` and convert to Qwen3.5's `(1.0 + w)` scale!
pub fn load_qwen35_rmsnorm_scale_1p(
    snapshot_dir: &Path,
    tensor_name: &str,
) -> Result<Vec<f64>, V2CompilerError> {
    let (shape, data) = load_safetensor_f64(snapshot_dir, tensor_name)?;
    if shape.len() != 1 {
        return Err(V2CompilerError::InvalidConfig {
            context: "load_qwen35_rmsnorm_scale_1p",
            reason: format!("expected 1D norm tensor `{tensor_name}`, got shape {shape:?}"),
        });
    }
    Ok(data.into_iter().map(|w| 1.0 + w).collect())
}
