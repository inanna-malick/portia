//! `portia-compiler` Pit-Wall Dyno (`compiler_dyno`):
//! 1. Benchmarks `faer` thin-SVD subspace extraction across `0.8B`..`27B` widths (`d = 1024..5120`),
//! 2. Benchmarks `Clarabel` sparse diagonal-`P` & orthonormal row-space (`w = U z`) conic solves,
//! 3. Compiles and verifies a `12`-rule declarative `IntentExpr` program (`StanceRoute`,
//!    `SignedTernaryFork`, `SubmergedPointerRoute`, `AsidSegmentColor`, `DeltaNetRollbackScrub`,
//!    `regalloc`, multi-core `std::thread::scope` lowering, and `WireCartridgeManifestJson` bisimulation),
//! 4. Inspects cached `Qwen3.5-0.8B-Base` `.safetensors` `ASID` & Layer-23 secant geometry when present.

mod compiler_dyno_support;

use compiler_dyno_support::{axis_subspace, build_demo_program, pre_mlp_rows};
use portia_compiler::discovery::gqa_bus::{
    synthesize_asid_codebook, GqaPointerLayer, MemorySegmentAsid,
};
use portia_compiler::discovery::{discover_and_project_anti_echo, DownstreamSwigluBlock};
use portia_compiler::lowering::compile_intent_program;
use portia_compiler::numerics::safetensors_io::{
    load_qwen35_rmsnorm_scale_1p, load_safetensor_matrix_2d,
};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, rms_norm_slice, StateMatrix};
use portia_compiler::solvers::conic_qp::solve_conic_linear_vector;
use portia_compiler::solvers::V2SolverConfig;
use portia_compiler::wire::WireCartridgeManifestJson;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn latest_qwen08b_snapshot() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .collect();
    entries.sort();
    entries.pop()
}

fn bench_conic_and_full_lowering() {
    println!("\n=== [2] Clarabel Sparse Conic SOCP & Parallel Pointer-ISA Lowering Dyno ===");
    let cfg = V2SolverConfig::default();

    let d_4b = 2560usize;
    let on_4b = pre_mlp_rows(24, d_4b, Some((8, 4.5)), 101).quantize_bf16();
    let q_4b = axis_subspace(d_4b, 100, 48);
    let t0 = Instant::now();
    let w_up_4b = solve_conic_linear_vector(
        "dyno_4b_up",
        &on_4b,
        &[1.0; 24],
        &[0.18; 24],
        None,
        Some(&q_4b),
        &cfg,
    )
    .unwrap();
    println!(
        "  4B w_up solve (d={d_4b}, n_on=24, q_rank={}): {:.2} ms (leak={:.2e})",
        q_4b.rank(),
        t0.elapsed().as_secs_f64() * 1000.0,
        q_4b.leakage_norm(&w_up_4b)
    );

    let d_08b = 1024usize;
    let on_08b = pre_mlp_rows(16, d_08b, Some((4, 5.2)), 202).quantize_bf16();
    let off_08b = pre_mlp_rows(2500, d_08b, None, 203).quantize_bf16();
    let q_08b = axis_subspace(d_08b, 100, 64);
    let t1 = Instant::now();
    let w_gate_08b = solve_conic_linear_vector(
        "dyno_08b_gate",
        &on_08b,
        &[cfg.g_pos_target; 16],
        &[cfg.g_pos_band; 16],
        Some((&off_08b, cfg.g_neg_ceiling - cfg.bf16_neg_pad)),
        Some(&q_08b),
        &cfg,
    )
    .unwrap();
    println!(
        "  0.8B w_gate solve (d={d_08b}, n_on=16, n_neg=2500, q_rank={}): {:.2} ms (leak={:.2e})",
        q_08b.rank(),
        t1.elapsed().as_secs_f64() * 1000.0,
        q_08b.leakage_norm(&w_gate_08b)
    );

    let (expr, caps, geom) = build_demo_program(d_08b, &cfg);
    let t2 = Instant::now();
    let out = compile_intent_program(&expr, &caps, &geom).unwrap();
    let prog_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let rt =
        WireCartridgeManifestJson::from_json_str(&out.manifest.to_json_pretty().unwrap()).unwrap();
    rt.verify_schema_integrity().unwrap();
    let rcpt = rt.bisimulation_receipt.as_ref().unwrap();
    assert!(rcpt.bisimulation_holds);
    println!(
        "  12-rule IntentExpr lowering + regalloc + bf16 bisimulation: {prog_ms:.2} ms (min_pos={:.2}, max_neg={:.2})",
        rcpt.min_positive_gate_pre_silu, rcpt.max_negative_gate_pre_silu
    );
}

fn inspect_cached_qwen08b(snap: &Path) {
    println!("\n=== [3] Pure-Rust Qwen3.5-0.8B-Base .safetensors ASID & Secant Verification ===");
    let gqa = GqaPointerLayer {
        layer_idx: 15,
        hidden_dim: 1024,
        num_q_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        input_norm_scale: load_qwen35_rmsnorm_scale_1p(
            snap,
            "model.layers.15.input_layernorm.weight",
        )
        .unwrap(),
        q_norm_scale: load_qwen35_rmsnorm_scale_1p(snap, "model.layers.15.self_attn.q_norm.weight")
            .unwrap(),
        k_norm_scale: load_qwen35_rmsnorm_scale_1p(snap, "model.layers.15.self_attn.k_norm.weight")
            .unwrap(),
        w_q_proj: load_safetensor_matrix_2d(snap, "model.layers.15.self_attn.q_proj.weight")
            .unwrap(),
        w_k_proj: load_safetensor_matrix_2d(snap, "model.layers.15.self_attn.k_proj.weight")
            .unwrap(),
        w_v_proj: load_safetensor_matrix_2d(snap, "model.layers.15.self_attn.v_proj.weight")
            .unwrap(),
    };
    let q_forbid = gqa.build_submerged_forbidden_basis(None, 32).unwrap();
    let cb = synthesize_asid_codebook(&gqa, &[0, 1], &q_forbid).unwrap();
    let qn = rms_norm_slice(
        &cb.read_query_selectors[MemorySegmentAsid::AuthoritativeSchema as usize],
        &gqa.input_norm_scale,
        1e-6,
    );
    let kn = rms_norm_slice(
        &cb.write_key_tags[MemorySegmentAsid::AuthoritativeSchema as usize],
        &gqa.input_norm_scale,
        1e-6,
    );
    let qk_dot: f64 = (0..gqa.head_dim)
        .map(|i| dot_f64(gqa.w_q_proj.row(i), &qn) * dot_f64(gqa.w_k_proj.row(i), &kn))
        .sum();
    println!(
        "  L15 ASID max_wv_leakage={:.2e}, same-color QK dot={qk_dot:.3}",
        cb.max_wv_leakage
    );

    let wg23 = load_safetensor_matrix_2d(snap, "model.layers.23.mlp.gate_proj.weight").unwrap();
    let wu23 = load_safetensor_matrix_2d(snap, "model.layers.23.mlp.up_proj.weight").unwrap();
    let wd23 = load_safetensor_matrix_2d(snap, "model.layers.23.mlp.down_proj.weight").unwrap();
    let post23 =
        load_qwen35_rmsnorm_scale_1p(snap, "model.layers.23.post_attention_layernorm.weight")
            .unwrap();
    let (d, m_f) = (wd23.nrows(), wg23.nrows());
    let wd_cols: Vec<f64> = (0..m_f * d).map(|k| wd23.row(k % d)[k / d]).collect();
    let ds = DownstreamSwigluBlock::try_new(
        23,
        post23,
        wg23,
        wu23,
        StateMatrix::try_new("wd23_cols", m_f, d, wd_cols).unwrap(),
    )
    .unwrap();
    let (mut hb, mut vr) = (vec![0.0; d], vec![0.0; d]);
    (hb[0], vr[1]) = (24.0, 1.0);
    let (_, _, rcpt) = discover_and_project_anti_echo(
        "dyno_secant",
        &ds,
        &StateMatrix::from_rows("hb", &[hb]).unwrap(),
        &vr,
        0.15,
        16,
    )
    .unwrap();
    println!(
        "  L23 Secant Anti-Echo: nullspace_rank={}, retained_energy={:.3}",
        rcpt.nullspace_rank, rcpt.target_retained_energy
    );
}

fn main() {
    println!("=== [1] Pit-Wall Dyno: faer Thin-SVD Subspace Extraction (m = 512 rows) ===");
    for &(label, d_model) in &[
        ("0.8B (d=1024)", 1024usize),
        ("4B   (d=2560)", 2560),
        ("9B   (d=4096)", 4096),
        ("27B  (d=5120)", 5120),
    ] {
        let m = 512usize;
        let flat: Vec<f64> = (0..m * d_model)
            .map(|k| ((((k / d_model) * 31 + (k % d_model) * 17) % 97) as f64 - 48.0) * 0.02)
            .collect();
        let mat = StateMatrix::try_new("bench", m, d_model, flat).unwrap();
        let t0 = Instant::now();
        let sub = OrthonormalSubspace::from_row_space_svd("bench_svd", &mat, 1e-4, 64).unwrap();
        println!(
            "  {label} -> rank={} extracted in {:.2} ms",
            sub.rank(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    bench_conic_and_full_lowering();
    if let Some(snap) = latest_qwen08b_snapshot() {
        inspect_cached_qwen08b(&snap);
    }
}
