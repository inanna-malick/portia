//! Pure-Rust Qwen3.5 `.safetensors` GQA loader, selective `BF16` row reader, and exact
//! submerged `SO(d)` Gram-Schmidt chordal rotator (`v2_pointer_bridge::qwen_loader`).

use crate::v2_pointer_bridge::Qwen35GqaEvalBlock;
use portia_compiler::discovery::gqa_bus::GqaPointerLayer;
use portia_compiler::numerics::safetensors_io::{
    load_qwen35_rmsnorm_scale_1p, load_safetensor_matrix_2d,
};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, l2_norm, normalize_unit_vec};
use portia_compiler::V2CompilerError;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Locate the latest cached HuggingFace snapshot directory under `~/.cache/huggingface/hub/<repo_dir_name>/snapshots`.
#[must_use]
pub fn find_latest_hf_snapshot(repo_dir_name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir_name)
        .join("snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .collect();
    dirs.sort();
    dirs.pop()
}

/// Load a complete [`Qwen35GqaEvalBlock`] (`q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, input_layernorm, norm`)
/// from a real Qwen3.5 HuggingFace `.safetensors` snapshot directory.
pub fn load_qwen35_gqa_eval_block(
    snap: &Path,
    layer: usize,
    hidden_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rope_theta: f64,
) -> Result<Qwen35GqaEvalBlock, V2CompilerError> {
    let pfx = "model.language_model.";
    let gqa = GqaPointerLayer {
        layer_idx: layer,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        head_dim: 256,
        rotary_dim: 64,
        input_norm_scale: load_qwen35_rmsnorm_scale_1p(
            snap,
            &format!("{pfx}layers.{layer}.input_layernorm.weight"),
        )?,
        q_norm_scale: load_qwen35_rmsnorm_scale_1p(
            snap,
            &format!("{pfx}layers.{layer}.self_attn.q_norm.weight"),
        )?,
        k_norm_scale: load_qwen35_rmsnorm_scale_1p(
            snap,
            &format!("{pfx}layers.{layer}.self_attn.k_norm.weight"),
        )?,
        w_q_proj: load_safetensor_matrix_2d(
            snap,
            &format!("{pfx}layers.{layer}.self_attn.q_proj.weight"),
        )?,
        w_k_proj: load_safetensor_matrix_2d(
            snap,
            &format!("{pfx}layers.{layer}.self_attn.k_proj.weight"),
        )?,
        w_v_proj: load_safetensor_matrix_2d(
            snap,
            &format!("{pfx}layers.{layer}.self_attn.v_proj.weight"),
        )?,
    };
    let w_o_proj = load_safetensor_matrix_2d(
        snap,
        &format!("{pfx}layers.{layer}.self_attn.o_proj.weight"),
    )?;
    let final_norm = load_qwen35_rmsnorm_scale_1p(snap, &format!("{pfx}norm.weight"))?;
    Qwen35GqaEvalBlock::try_new(gqa, w_o_proj, final_norm, rope_theta)
}

/// Selective `BF16` row reader for large embedding tables (`embed_tokens.weight`), avoiding a
/// multi-GB full-vocabulary `f64` allocation on `4B`+ models.
pub fn load_selective_bf16_rows(
    snap: &Path,
    tensor_name: &str,
    row_indices: &[usize],
) -> Result<BTreeMap<usize, Vec<f64>>, V2CompilerError> {
    let idx_path = snap.join("model.safetensors.index.json");
    let shard_path = if idx_path.exists() {
        let raw = std::fs::read_to_string(&idx_path)
            .map_err(|e| V2CompilerError::WireIntegrity(format!("read index.json: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| V2CompilerError::WireIntegrity(format!("parse index.json: {e}")))?;
        let shard = v["weight_map"][tensor_name]
            .as_str()
            .ok_or_else(|| V2CompilerError::WireIntegrity(format!("missing `{tensor_name}`")))?;
        snap.join(shard)
    } else {
        snap.join("model.safetensors")
    };
    let mut f = File::open(&shard_path)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("open shard: {e}")))?;
    let mut len_b = [0u8; 8];
    f.read_exact(&mut len_b)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("read hdr len: {e}")))?;
    let hlen = u64::from_le_bytes(len_b) as usize;
    let mut hdr_b = vec![0u8; hlen];
    f.read_exact(&mut hdr_b)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("read hdr: {e}")))?;
    let hdr: serde_json::Value = serde_json::from_slice(&hdr_b)
        .map_err(|e| V2CompilerError::WireIntegrity(format!("parse hdr: {e}")))?;
    let meta = &hdr[tensor_name];
    let cols = meta["shape"][1].as_u64().unwrap_or(0) as usize;
    let base_off = 8u64 + (hlen as u64) + meta["data_offsets"][0].as_u64().unwrap_or(0);

    let mut out = BTreeMap::new();
    let mut row_bytes = vec![0u8; cols * 2];
    for &r in row_indices {
        if out.contains_key(&r) {
            continue;
        }
        f.seek(SeekFrom::Start(base_off + (r * cols * 2) as u64))
            .map_err(|e| V2CompilerError::WireIntegrity(format!("seek row {r}: {e}")))?;
        f.read_exact(&mut row_bytes)
            .map_err(|e| V2CompilerError::WireIntegrity(format!("read row {r}: {e}")))?;
        let vec_f64: Vec<f64> = row_bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16) as f64)
            .collect();
        out.insert(r, vec_f64);
    }
    Ok(out)
}

/// Exact norm-preserving `SO(d)` chordal rotation `h' = cos(theta)*h + ||h||*sin(theta)*v_perp`
/// where `v_perp` is orthogonal to BOTH `h` AND `q_forbidden` via 1-step joint Gram-Schmidt (`O(k*d)`).
#[must_use]
pub fn rotate_chordal_submerged(
    h: &[f64],
    raw_dir: &[f64],
    theta_rad: f64,
    q_forbidden: Option<&OrthonormalSubspace>,
) -> (Vec<f64>, Vec<f64>) {
    let r = l2_norm(h);
    let Some(h_hat) = normalize_unit_vec(h) else {
        return (h.to_vec(), vec![0.0; h.len()]);
    };
    let mut v = match q_forbidden {
        Some(q) => q.project_out(raw_dir),
        None => raw_dir.to_vec(),
    };
    let h_residual_dir = match q_forbidden {
        Some(q) => normalize_unit_vec(&q.project_out(&h_hat)),
        None => Some(h_hat),
    };
    if let Some(u_h) = h_residual_dir {
        let alpha = dot_f64(&u_h, &v);
        for (vi, &ui) in v.iter_mut().zip(&u_h) {
            *vi -= alpha * ui;
        }
    }
    if let Some(q) = q_forbidden {
        v = q.project_out(&v);
    }
    let v_perp = normalize_unit_vec(&v).unwrap_or_else(|| vec![0.0; h.len()]);
    let (s, c) = theta_rad.sin_cos();
    let mut h_new = vec![0.0; h.len()];
    let mut delta_tan = vec![0.0; h.len()];
    for i in 0..h.len() {
        delta_tan[i] = r * s * v_perp[i];
        h_new[i] = c * h[i] + delta_tan[i];
    }
    (h_new, delta_tan)
}
