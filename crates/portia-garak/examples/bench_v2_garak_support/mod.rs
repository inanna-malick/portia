//! Support routines (`Qwen/Qwen3.5-0.8B` geometry loader, forward-capture cache, and `v2` spec
//! builder) for `examples/bench_v2_garak_map_elites_ab.rs`.

use anyhow::{bail, Context, Result};
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::numerics::safetensors_io::{
    load_qwen35_rmsnorm_scale_1p, load_safetensor_f64, load_safetensor_matrix_2d,
};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::StateMatrix;
use portia_garak::v2_concolic::{DiscreteGarakProbe, GarakV2GeometryExtras};
use portia_garak::{
    GarakCalibratedAngles, GarakCalibrationFamily, GarakCompileInputSpec, GarakPartition,
    GarakPreMlpSplits, GarakProbeFamily, GarakReadoutSpec, GarakResidualSplits, GarakRoleSplit,
    GarakV2Spec, GARAK_CONTRAST_TOKEN_IDS, GARAK_SLOT_NAMES,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize, Deserialize)]
pub struct CapturedPair {
    pub x: Vec<f64>,
    pub h: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct RealQwen08bCaptureCache {
    pub hidden_dim: usize,
    pub prompt_count: usize,
    pub by_id: BTreeMap<String, CapturedPair>,
    pub continuations: Vec<CapturedPair>,
}

pub fn resolve_qwen08b_snapshot() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    let preferred = snaps.join("2fc06364715b967f1860aea9cf38778875588b17");
    if preferred.exists() {
        return Ok(preferred);
    }
    for entry in std::fs::read_dir(&snaps).context("read Qwen3.5-0.8B snapshots dir")? {
        let p = entry?.path();
        if p.is_dir() {
            return Ok(p);
        }
    }
    bail!(
        "no cached Qwen3.5-0.8B snapshot found under {}",
        snaps.display()
    )
}

pub fn load_real_qwen08b_geometry(
    snap: &Path,
) -> Result<(GarakReadoutSpec, GarakV2GeometryExtras)> {
    let (_, final_norm_raw) = load_safetensor_f64(snap, "model.language_model.norm.weight")?;
    let (_, post_norm_raw) = load_safetensor_f64(
        snap,
        "model.language_model.layers.23.post_attention_layernorm.weight",
    )?;
    let post_scale = load_qwen35_rmsnorm_scale_1p(
        snap,
        "model.language_model.layers.23.post_attention_layernorm.weight",
    )?;
    let (emb_shape, emb_flat) =
        load_safetensor_f64(snap, "model.language_model.embed_tokens.weight")?;
    let d = emb_shape[1];
    let mut w_u_rows = BTreeMap::new();
    for &tid in &GARAK_CONTRAST_TOKEN_IDS {
        let off = (tid as usize) * d;
        w_u_rows.insert(tid.to_string(), emb_flat[off..off + d].to_vec());
    }

    let w_gate =
        load_safetensor_matrix_2d(snap, "model.language_model.layers.23.mlp.gate_proj.weight")?;
    let w_up =
        load_safetensor_matrix_2d(snap, "model.language_model.layers.23.mlp.up_proj.weight")?;
    let w_down_raw =
        load_safetensor_matrix_2d(snap, "model.language_model.layers.23.mlp.down_proj.weight")?;
    let m_f = w_gate.nrows();
    let mut wd_t = vec![0.0; m_f * d];
    for r in 0..d {
        let row = w_down_raw.row(r);
        for c in 0..m_f {
            wd_t[c * d + r] = row[c];
        }
    }
    let w_down_cols = StateMatrix::try_new("l23_w_down_cols", m_f, d, wd_t)?;
    let ds_block = DownstreamSwigluBlock::try_new(23, post_scale, w_gate, w_up, w_down_cols)?;

    let vocab_sub = 2048usize.min(emb_shape[0]);
    let wu_mat = StateMatrix::try_new("wu_top", vocab_sub, d, emb_flat[..vocab_sub * d].to_vec())?;
    let w_u_top_basis = OrthonormalSubspace::from_row_space_svd("w_u_top48", &wu_mat, 1e-3, 48)?;

    let z2 = vec![vec![0.0; d]; 2];
    let readout = GarakReadoutSpec::CompactHeadAntiEcho {
        w_u_rows,
        final_norm_weight: final_norm_raw,
        post_attn_norm_weight: post_norm_raw,
        anti_no_gate_top: z2.clone(),
        anti_no_up_top: z2.clone(),
        anti_act_gate_top: z2.clone(),
        anti_act_up_top: z2,
    };
    Ok((
        readout,
        GarakV2GeometryExtras {
            downstream_mlp: Some(ds_block),
            w_u_top_basis: Some(w_u_top_basis),
        },
    ))
}

pub fn ensure_real_qwen08b_captures(
    snap: &Path,
    prompts: &[(String, String)],
    cont_prompts: &[(String, String)],
) -> Result<RealQwen08bCaptureCache> {
    let cache_path = Path::new("/tmp/portia_v2_garak_qwen08b_lattice_states_v2.json");
    if let Ok(raw) = std::fs::read_to_string(cache_path) {
        if let Ok(parsed) = serde_json::from_str::<RealQwen08bCaptureCache>(&raw) {
            if parsed.hidden_dim == 1024 && parsed.prompt_count == prompts.len() {
                return Ok(parsed);
            }
        }
    }
    let req_path = Path::new("/tmp/portia_v2_garak_capture_req.json");
    let py_path = Path::new("/tmp/portia_v2_garak_capture_driver.py");
    std::fs::write(
        req_path,
        serde_json::to_string(&serde_json::json!({
            "snapshot": snap.display().to_string(),
            "prompts": prompts,
            "continuations": cont_prompts,
            "out_path": cache_path.display().to_string(),
        }))?,
    )?;
    let py_code = r#"import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained(cfg["snapshot"], local_files_only=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(cfg["snapshot"], local_files_only=True, dtype=dt).to(dev).eval()
layers = getattr(getattr(model, "model", model), "layers", None) or model.model.language_model.layers
box = {}
def hook(_m, inp, out):
    box["h"], box["x"] = inp[0].detach().float().cpu(), out.detach().float().cpu()
handle = layers[22].post_attention_layernorm.register_forward_hook(hook)
fmt_u = lambda u: f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
by_id, conts = {}, []
with torch.no_grad():
    for pid, text in cfg["prompts"]:
        model(**tok(fmt_u(text), return_tensors="pt").to(dev), use_cache=False)
        by_id[pid] = {"x": box["x"][0, -1].tolist(), "h": box["h"][0, -1].tolist()}
    for u, a in cfg["continuations"]:
        model(**tok(fmt_u(u) + a, return_tensors="pt").to(dev), use_cache=False)
        for k in [-4, -3, -2, -1]:
            conts.append({"x": box["x"][0, k].tolist(), "h": box["h"][0, k].tolist()})
handle.remove()
json.dump({"hidden_dim": len(conts[0]["x"]), "prompt_count": len(cfg["prompts"]), "by_id": by_id, "continuations": conts}, open(cfg["out_path"], "w"))
"#;
    std::fs::write(py_path, py_code)?;
    let ws_venv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    let py_bin = std::env::var("PORTIA_BENCH_PYTHON")
        .or_else(|_| std::env::var("PORTIA_PYTHON"))
        .unwrap_or_else(|_| {
            if ws_venv.exists() {
                ws_venv.display().to_string()
            } else {
                "python3".into()
            }
        });
    let status = Command::new(&py_bin).arg(py_path).arg(req_path).status()?;
    if !status.success() {
        bail!("real Qwen3.5-0.8B forward capture failed with {status}");
    }
    Ok(serde_json::from_str(&std::fs::read_to_string(cache_path)?)?)
}

pub fn build_spec(
    readout: &GarakReadoutSpec,
    cache: &RealQwen08bCaptureCache,
    seeds: &[DiscreteGarakProbe],
    extra: &[DiscreteGarakProbe],
    guard_ids: &[String],
    wk_true_ids: &[String],
) -> GarakCompileInputSpec {
    let get_xh = |ids: &[String]| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        (
            ids.iter().map(|id| cache.by_id[id].x.clone()).collect(),
            ids.iter().map(|id| cache.by_id[id].h.clone()).collect(),
        )
    };
    let mut by_fam: BTreeMap<GarakProbeFamily, Vec<String>> = BTreeMap::new();
    for p in seeds.iter().chain(extra.iter()) {
        let v = by_fam.entry(p.family).or_default();
        if !v.contains(&p.id) {
            v.push(p.id.clone());
        }
    }
    let mk_split = |part: GarakPartition, fam: GarakCalibrationFamily, ids: &[String]| {
        let pos = fam.expected_positive_slot();
        let (x, h) = get_xh(ids);
        GarakRoleSplit {
            name: format!("{part}_{fam}"),
            partition: part,
            family: fam,
            row_ids: ids.iter().map(|s| format!("{part}:{s}")).collect(),
            x,
            h,
            positive_slots: pos.into_iter().map(str::to_string).collect(),
            cold_slots: GARAK_SLOT_NAMES
                .into_iter()
                .filter(|&s| Some(s) != pos)
                .map(str::to_string)
                .collect(),
            globally_benign: pos.is_none(),
        }
    };
    let gc_ids = &by_fam[&GarakProbeFamily::SnowballGraphConnectivity];
    let fc_ids = &by_fam[&GarakProbeFamily::MisleadingFalseAssertion];
    let wk_ids = &by_fam[&GarakProbeFamily::SnowballSenators];
    let pk_ids = &by_fam[&GarakProbeFamily::PackageHallucination];
    let (real_pkg, true_clm) = (&guard_ids[0..4], &guard_ids[4..12]);
    let cont_x: Vec<Vec<f64>> = cache.continuations.iter().map(|c| c.x.clone()).collect();
    let cont_h: Vec<Vec<f64>> = cache.continuations.iter().map(|c| c.h.clone()).collect();

    let mut splits = Vec::new();
    for part in [GarakPartition::Fit, GarakPartition::Dev] {
        splits.extend([
            mk_split(part, GarakCalibrationFamily::GcConnected, &gc_ids[0..2]),
            mk_split(part, GarakCalibrationFamily::GcDisconnected, &gc_ids[2..]),
            mk_split(part, GarakCalibrationFamily::FalseClaim, fc_ids),
            mk_split(part, GarakCalibrationFamily::WkFalse, wk_ids),
            mk_split(part, GarakCalibrationFamily::WkTrue, wk_true_ids),
            mk_split(part, GarakCalibrationFamily::UnrealPackage, pk_ids),
            mk_split(part, GarakCalibrationFamily::RealPackage, real_pkg),
            mk_split(part, GarakCalibrationFamily::TrueClaim, true_clm),
            GarakRoleSplit {
                name: format!("{part}_continuation"),
                partition: part,
                family: GarakCalibrationFamily::Continuation,
                row_ids: (0..cont_x.len())
                    .map(|i| format!("{part}:cont:{i}"))
                    .collect(),
                x: cont_x.clone(),
                h: cont_h.clone(),
                positive_slots: Vec::new(),
                cold_slots: GARAK_SLOT_NAMES.into_iter().map(str::to_string).collect(),
                globally_benign: true,
            },
        ]);
    }
    let ((xgc, hgc), (xfc, hfc), (xtr, htr), (xup, hup), (xrp, _)) = (
        get_xh(gc_ids),
        get_xh(fc_ids),
        get_xh(true_clm),
        get_xh(&pk_ids[0..4]),
        get_xh(real_pkg),
    );
    GarakCompileInputSpec {
        v2: Some(GarakV2Spec {
            splits,
            ridge: 1e-3,
            cold_weight: 2.0,
            candidate_index: 0,
        }),
        model_id: "Qwen/Qwen3.5-0.8B".into(),
        target_layer_idx: 22,
        base_intermediate_size: 3584,
        widened_slots: Some(8),
        slot_offset: None,
        pkg_template_count: 1,
        x_splits: GarakPreMlpSplits {
            train_gc: xgc,
            train_fc: xfc,
            train_true_claims: xtr,
            train_unreal_py: xup,
            train_real_py: xrp,
            test_gc: vec![],
            test_fc: vec![],
            test_unreal_py: vec![],
            transfer_senators: vec![],
            transfer_primes: vec![],
            transfer_unreal_rs: vec![],
            guard_real_code: vec![],
            guard_true_claims: vec![],
        },
        h_splits: GarakResidualSplits {
            train_gc: hgc,
            train_fc: hfc,
            train_true_claims: htr,
            train_unreal_py: hup,
        },
        x_cont_all: cont_x,
        calibrated_angles: GarakCalibratedAngles::default(),
        readout: readout.clone(),
        max_bg_rss_leakage: 0.25,
        strict_guard_check: true,
    }
}
