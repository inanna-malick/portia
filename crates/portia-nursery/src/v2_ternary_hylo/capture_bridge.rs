//! Real `Qwen3.5-0.8B-Base` CUDA/PyTorch Activation Capture Bridge (`v2_ternary_hylo::capture_bridge`).
//!
//! Invokes local PyTorch (`transformers` + `safetensors` in `bfloat16`) in a single batched pass
//! to capture authentic Layer-17, Layer-18, and Layer-22 hidden states on Rust-AST `todo!()` prompts,
//! co-mounted `SQL` + `Garak` crowding prompts, background prompts, and `<rollback/>` draft episodes.

use portia_compiler::numerics::StateMatrix;
use portia_compiler::V2CompilerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Captured Layer-22 cursor matrices (`x22_pre_mlp`, `h22_mid`, `h22_post_base`) for a prompt group.
#[derive(Debug, Clone)]
pub struct Layer22GroupCapture {
    pub x22_pre_mlp: StateMatrix,
    pub h22_mid: StateMatrix,
    pub h22_post_base: StateMatrix,
}

/// Input specification for a multi-draft `<rollback/>` episode on Layer 17 -> Layer 18.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackEpisodeSpec {
    pub episode_id: String,
    pub prompt_text: String,
    pub draft_texts: Vec<String>,
    pub query_text: String,
}

/// Real captured Layer-17/18 token trajectories for a single [`RollbackEpisodeSpec`].
#[derive(Debug, Clone)]
pub struct CapturedRollbackEpisode {
    pub episode_id: String,
    pub prompt_h18: Vec<Vec<f64>>,
    pub draft_chunks_h18: Vec<Vec<Vec<f64>>>,
    pub rollback_tokens: Vec<(Vec<f64>, Vec<f64>)>,
    pub query_h18: Vec<Vec<f64>>,
    pub off_cursor_x17_samples: Vec<Vec<f64>>,
}

/// Combined single-pass CUDA capture bundle for Scoreboards A and B.
#[derive(Debug, Clone)]
pub struct RealModelCaptureBundle {
    pub snapshot_dir: PathBuf,
    pub device_used: String,
    pub l22_groups: BTreeMap<String, Layer22GroupCapture>,
    pub rollback_episodes: Vec<CapturedRollbackEpisode>,
}

/// Locate the latest cached `Qwen/Qwen3.5-0.8B-Base` HuggingFace snapshot directory.
#[must_use]
pub fn latest_qwen35_08b_snapshot() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .collect();
    dirs.sort();
    dirs.pop()
}

fn resolve_torch_python() -> String {
    if let Ok(p) = std::env::var("PORTIA_BENCH_PYTHON") {
        if !p.trim().is_empty() {
            return p;
        }
    }
    let ws_venv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    if ws_venv.exists() {
        return ws_venv.display().to_string();
    }
    std::env::var("PORTIA_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

#[derive(Serialize)]
struct PyCaptureRequest<'a> {
    snapshot_dir: &'a str,
    l22_groups: &'a BTreeMap<String, Vec<String>>,
    rollback_episodes: &'a [RollbackEpisodeSpec],
    bin_out_path: &'a str,
}

#[derive(Deserialize)]
struct PyGroupMeta {
    nrows: usize,
    x22_offset: usize,
    h22_mid_offset: usize,
    h22_post_offset: usize,
}

#[derive(Deserialize)]
struct PyEpisodeMeta {
    episode_id: String,
    prompt_len: usize,
    prompt_h18_off: usize,
    draft_lens: Vec<usize>,
    draft_h18_offs: Vec<usize>,
    rb_x17_offs: Vec<usize>,
    rb_h18_offs: Vec<usize>,
    query_len: usize,
    query_h18_off: usize,
    off_x17_len: usize,
    off_x17_off: usize,
}

#[derive(Deserialize)]
struct PyCaptureIndex {
    device: String,
    hidden_dim: usize,
    groups: BTreeMap<String, PyGroupMeta>,
    episodes: Vec<PyEpisodeMeta>,
}

const PY_CAPTURE_SCRIPT: &str = r#"
import json, sys, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

req = json.load(open(sys.argv[1], "r"))
snap = req["snapshot_dir"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16).to(dev).eval()
d = model.config.hidden_size

c = {}
h17 = model.model.layers[17].post_attention_layernorm.register_forward_hook(
    lambda m, i, o: c.update(x17=o.detach().float().cpu().numpy())
)
h18 = model.model.layers[18].register_forward_pre_hook(
    lambda m, i: c.update(h18=i[0].detach().float().cpu().numpy())
)
h22_pre = model.model.layers[22].post_attention_layernorm.register_forward_hook(
    lambda m, i, o: c.update(x22=o.detach().float().cpu().numpy(), h22_mid=i[0].detach().float().cpu().numpy())
)
h22_post = model.model.layers[22].register_forward_hook(
    lambda m, i, o: c.update(h22_post=(o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy())
)

chunks = []
cur_floats = 0
def push_arr(arr):
    global cur_floats
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    off = cur_floats
    chunks.append(a)
    cur_floats += a.size
    return off

groups_meta = {}
with torch.no_grad():
    for gname, prompts in req["l22_groups"].items():
        xs, mids, posts = [], [], []
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(dev)
            model(**inp)
            xs.append(c["x22"][0, -1])
            mids.append(c["h22_mid"][0, -1])
            posts.append(c["h22_post"][0, -1])
        groups_meta[gname] = {
            "nrows": len(prompts),
            "x22_offset": push_arr(np.stack(xs)),
            "h22_mid_offset": push_arr(np.stack(mids)),
            "h22_post_offset": push_arr(np.stack(posts)),
        }

    eps_meta = []
    for ep in req["rollback_episodes"]:
        p_ids = tok.encode(ep["prompt_text"], add_special_tokens=False)
        q_ids = tok.encode(ep["query_text"], add_special_tokens=False)
        clean_ids = torch.tensor([p_ids + q_ids], dtype=torch.long, device=dev)
        model(input_ids=clean_ids)
        p_len, q_len = len(p_ids), len(q_ids)
        p_h18_off = push_arr(c["h18"][0, :p_len])
        q_h18_off = push_arr(c["h18"][0, p_len:p_len + q_len])
        off_samples = [c["x17"][0, max(0, p_len // 2)], c["x17"][0, p_len - 1], c["x17"][0, p_len + q_len - 1]]

        full_ids = list(p_ids)
        spans = []
        for dt in ep["draft_texts"]:
            d_ids = tok.encode(dt, add_special_tokens=False)
            s0 = len(full_ids)
            full_ids.extend(d_ids)
            spans.append((s0, len(full_ids)))
        model(input_ids=torch.tensor([full_ids], dtype=torch.long, device=dev))
        d_lens, d_offs, rx_offs, rh_offs = [], [], [], []
        for (s0, s1) in spans:
            body_len = max(1, (s1 - 1) - s0)
            d_lens.append(body_len)
            d_offs.append(push_arr(c["h18"][0, s0:s0 + body_len]))
            rx_offs.append(push_arr(c["x17"][0, s1 - 1]))
            rh_offs.append(push_arr(c["h18"][0, s1 - 1]))
            off_samples.append(c["x17"][0, s0 + body_len // 2])
        eps_meta.append({
            "episode_id": ep["episode_id"],
            "prompt_len": p_len,
            "prompt_h18_off": p_h18_off,
            "draft_lens": d_lens,
            "draft_h18_offs": d_offs,
            "rb_x17_offs": rx_offs,
            "rb_h18_offs": rh_offs,
            "query_len": q_len,
            "query_h18_off": q_h18_off,
            "off_x17_len": len(off_samples),
            "off_x17_off": push_arr(np.stack(off_samples)),
        })

with open(req["bin_out_path"], "wb") as bf:
    for ch in chunks:
        bf.write(ch.tobytes())

print(json.dumps({"device": dev, "hidden_dim": d, "groups": groups_meta, "episodes": eps_meta}))
"#;

fn slice_rows(buf: &[f32], off: usize, nrows: usize, d: usize) -> Vec<Vec<f64>> {
    (0..nrows)
        .map(|r| {
            let base = off + r * d;
            buf[base..base + d].iter().map(|&v| v as f64).collect()
        })
        .collect()
}

/// Execute a single batched `Qwen3.5-0.8B-Base` forward pass capturing all Layer-22 & Layer-17/18 states.
pub fn capture_real_qwen35_bundle(
    snapshot_dir: &Path,
    l22_groups: &BTreeMap<String, Vec<String>>,
    rollback_episodes: &[RollbackEpisodeSpec],
) -> Result<RealModelCaptureBundle, V2CompilerError> {
    let pid = std::process::id();
    let req_path = std::env::temp_dir().join(format!("portia_v2_cap_req_{pid}.json"));
    let bin_path = std::env::temp_dir().join(format!("portia_v2_cap_bin_{pid}.f32"));
    let req = PyCaptureRequest {
        snapshot_dir: snapshot_dir.to_str().unwrap_or_default(),
        l22_groups,
        rollback_episodes,
        bin_out_path: bin_path.to_str().unwrap_or_default(),
    };
    let req_json = serde_json::to_vec(&req).map_err(|e| V2CompilerError::InvalidConfig {
        context: "capture_real_qwen35_bundle(serialize)",
        reason: e.to_string(),
    })?;
    let io_err = |e: std::io::Error| V2CompilerError::InvalidConfig {
        context: "capture_real_qwen35_bundle(io)",
        reason: e.to_string(),
    };
    std::fs::write(&req_path, req_json).map_err(io_err)?;

    let py = resolve_torch_python();
    let out = Command::new(&py)
        .args([
            "-c",
            PY_CAPTURE_SCRIPT,
            req_path.to_str().unwrap_or_default(),
        ])
        .output()
        .map_err(io_err)?;
    let _ = std::fs::remove_file(&req_path);
    if !out.status.success() {
        let _ = std::fs::remove_file(&bin_path);
        return Err(V2CompilerError::InvalidConfig {
            context: "capture_real_qwen35_bundle(python)",
            reason: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }

    let stdout_str = String::from_utf8_lossy(&out.stdout);
    let last_line = stdout_str
        .lines()
        .rev()
        .find(|l| l.starts_with('{'))
        .unwrap_or("{}");
    let idx: PyCaptureIndex = serde_json::from_str(last_line).map_err(|e| {
        let _ = std::fs::remove_file(&bin_path);
        V2CompilerError::InvalidConfig {
            context: "capture_real_qwen35_bundle(index)",
            reason: format!("{e}: {last_line}"),
        }
    })?;
    let raw_bytes = std::fs::read(&bin_path).map_err(io_err)?;
    let _ = std::fs::remove_file(&bin_path);
    let floats: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let d = idx.hidden_dim;

    let mut groups_out = BTreeMap::new();
    for (name, gm) in idx.groups {
        let x22 =
            StateMatrix::from_rows("l22_x22", &slice_rows(&floats, gm.x22_offset, gm.nrows, d))?;
        let mid = StateMatrix::from_rows(
            "l22_mid",
            &slice_rows(&floats, gm.h22_mid_offset, gm.nrows, d),
        )?;
        let post = StateMatrix::from_rows(
            "l22_post",
            &slice_rows(&floats, gm.h22_post_offset, gm.nrows, d),
        )?;
        groups_out.insert(
            name,
            Layer22GroupCapture {
                x22_pre_mlp: x22,
                h22_mid: mid,
                h22_post_base: post,
            },
        );
    }

    let mut eps_out = Vec::with_capacity(idx.episodes.len());
    for em in idx.episodes {
        let draft_chunks_h18 = em
            .draft_lens
            .iter()
            .zip(&em.draft_h18_offs)
            .map(|(&len, &off)| slice_rows(&floats, off, len, d))
            .collect();
        let rollback_tokens = em
            .rb_x17_offs
            .iter()
            .zip(&em.rb_h18_offs)
            .map(|(&xo, &ho)| {
                (
                    slice_rows(&floats, xo, 1, d).remove(0),
                    slice_rows(&floats, ho, 1, d).remove(0),
                )
            })
            .collect();
        eps_out.push(CapturedRollbackEpisode {
            episode_id: em.episode_id,
            prompt_h18: slice_rows(&floats, em.prompt_h18_off, em.prompt_len, d),
            draft_chunks_h18,
            rollback_tokens,
            query_h18: slice_rows(&floats, em.query_h18_off, em.query_len, d),
            off_cursor_x17_samples: slice_rows(&floats, em.off_x17_off, em.off_x17_len, d),
        });
    }

    Ok(RealModelCaptureBundle {
        snapshot_dir: snapshot_dir.to_path_buf(),
        device_used: idx.device,
        l22_groups: groups_out,
        rollback_episodes: eps_out,
    })
}
