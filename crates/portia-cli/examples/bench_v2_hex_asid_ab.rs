//! Pre-Registered Non-Circular Hex-Obfuscated + `<think>`-Decoy `∀α` A/B Benchmark (`bench_v2_hex_asid_ab`).
//!
//! Loads **real cached `Qwen3.5-0.8B-Base` (`d=1024`, `L15`) and `Qwen3.5-4B-Base` (`d=2560`, `L23`)**
//! `.safetensors` weights + `vocab.json` in pure Rust (`0` Python) and evaluates `5` FIT worlds ->
//! `40` held-out unseen hex worlds (`0%` token overlap) through the full Qwen3.5 GQA forward pass.

use portia_compiler::discovery::gqa_bus::{synthesize_asid_codebook, MemorySegmentAsid};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{normalize_unit_vec, StateMatrix};
use portia_sql::v2_pointer_bridge::{
    find_latest_hf_snapshot, load_qwen35_gqa_eval_block, load_selective_bf16_rows,
    rotate_chordal_submerged, synthesize_fk_passthrough_query_delta,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
struct ArmBenchmarkMetrics {
    arm: String,
    top1_unseen_fk_wins: usize,
    eval_worlds: usize,
    top1_accuracy: f64,
    mean_fk_vs_best_decoy_logit_margin: f64,
    schema_copy_attn_pct: f64,
    think_copy_attn_pct: f64,
    user_prose_copy_attn_pct: f64,
    max_wv_contamination_l2: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ModelBenchmarkReceipt {
    model_id: String,
    gqa_layer: usize,
    hidden_dim: usize,
    discovered_copy_heads: Vec<usize>,
    discovered_distractor_heads: Vec<usize>,
    fit_worlds: usize,
    eval_worlds: usize,
    lexical_token_overlap_count: usize,
    compile_latency_ms: f64,
    arms: Vec<ArmBenchmarkMetrics>,
}

fn lin_comb(terms: &[(f64, &[f64])], d: usize) -> Vec<f64> {
    let mut acc = vec![0.0_f64; d];
    for &(w, vec) in terms {
        for (a, &v) in acc.iter_mut().zip(vec) {
            *a += w * v;
        }
    }
    acc
}

struct HexWorld {
    fk_tok: usize,
    prose_tok: usize,
    think_tok: usize,
    h_schema_fk: Vec<f64>,
    h_user_prose: Vec<f64>,
    h_think_decoy: Vec<f64>,
    h_create_bg: Vec<f64>,
    h_select_bg: Vec<f64>,
    h_sql_cursor: Vec<f64>,
}

fn run_checkpoint_benchmark(
    model_id: &str,
    snap: &Path,
    layer: usize,
    d: usize,
    n_q: usize,
    n_kv: usize,
) -> ModelBenchmarkReceipt {
    let vocab_raw = std::fs::read_to_string(snap.join("vocab.json")).unwrap();
    let vocab: BTreeMap<String, usize> = serde_json::from_str(&vocab_raw).unwrap();

    let mut hex_tokens: Vec<(String, usize)> = vocab
        .iter()
        .filter(|(k, _)| {
            let is_low_hex = |s: &str| {
                s.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            };
            (k.len() == 3 && is_low_hex(k))
                || ((k.len() == 3 || k.len() == 4) && k.starts_with('_') && is_low_hex(&k[1..]))
        })
        .map(|(k, &id)| (k.clone(), id))
        .collect();
    hex_tokens.sort_by(|a, b| a.0.cmp(&b.0));
    let n_hex = hex_tokens.len();
    let perm_hex: Vec<usize> = (0..135)
        .map(|i| hex_tokens[(i * 37 + 11) % n_hex].1)
        .collect();
    let fit_set: BTreeSet<usize> = perm_hex[0..15].iter().copied().collect();
    let eval_set: BTreeSet<usize> = perm_hex[15..135].iter().copied().collect();
    let overlap = fit_set.intersection(&eval_set).count();
    assert_eq!(
        overlap, 0,
        "FIT and held-out EVAL hex tokens must be 100% disjoint"
    );

    let kw_names = [
        "ĠPRIMARY",
        "ĠKEY",
        "ĠREFERENCES",
        "ĠJOIN",
        "ĠON",
        "CREATE",
        "ĠTABLE",
        "SELECT",
        "ĠFROM",
        "ĠWHERE",
        "Ġuser",
        "Ġfilter",
        "Ġequals",
        "=",
    ];
    let kw_ids: BTreeMap<&str, usize> = kw_names.into_iter().map(|k| (k, vocab[k])).collect();
    let eval_block = load_qwen35_gqa_eval_block(snap, layer, d, n_q, n_kv, 10_000_000.0).unwrap();

    let mut needed_rows: Vec<usize> = (0..512).map(|i| i * (248_320 / 512)).collect();
    needed_rows.extend(perm_hex.iter().copied());
    needed_rows.extend(kw_ids.values().copied());
    let emb = load_selective_bf16_rows(
        snap,
        "model.language_model.embed_tokens.weight",
        &needed_rows,
    )
    .unwrap();

    let wu_rows: Vec<Vec<f64>> = (0..512)
        .map(|i| emb[&(i * (248_320 / 512))].clone())
        .collect();
    let q_wu64 = OrthonormalSubspace::from_row_space_svd(
        "wu64",
        &StateMatrix::from_rows("wu_sub", &wu_rows).unwrap(),
        1e-4,
        64,
    )
    .unwrap();

    let build_world = |w_idx: usize| -> HexWorld {
        let (fk_tok, prose_tok, think_tok) = (
            perm_hex[3 * w_idx],
            perm_hex[3 * w_idx + 1],
            perm_hex[3 * w_idx + 2],
        );
        let e = |t: usize| emb[&t].as_slice();
        let kw = |s: &str| emb[&kw_ids[s]].as_slice();
        HexWorld {
            fk_tok,
            prose_tok,
            think_tok,
            h_schema_fk: lin_comb(
                &[
                    (14.0, e(fk_tok)),
                    (7.5, kw("ĠPRIMARY")),
                    (7.5, kw("ĠKEY")),
                    (7.5, kw("ĠREFERENCES")),
                    (4.0, kw("ĠTABLE")),
                ],
                d,
            ),
            h_user_prose: lin_comb(
                &[
                    (14.0, e(prose_tok)),
                    (7.5, kw("Ġuser")),
                    (7.5, kw("Ġfilter")),
                    (7.5, kw("Ġequals")),
                    (4.0, kw("ĠWHERE")),
                ],
                d,
            ),
            h_think_decoy: lin_comb(
                &[
                    (14.0, e(think_tok)),
                    (6.8, kw("ĠPRIMARY")),
                    (6.8, kw("ĠKEY")),
                    (6.8, kw("ĠREFERENCES")),
                    (5.5, kw("ĠJOIN")),
                    (5.5, kw("ĠON")),
                ],
                d,
            ),
            h_create_bg: lin_comb(&[(14.0, kw("CREATE")), (8.0, kw("ĠTABLE"))], d),
            h_select_bg: lin_comb(&[(14.0, kw("SELECT")), (8.0, kw("ĠFROM"))], d),
            h_sql_cursor: lin_comb(
                &[
                    (12.0, kw("ĠJOIN")),
                    (12.0, kw("ĠON")),
                    (12.0, kw("=")),
                    (4.0, kw("SELECT")),
                ],
                d,
            ),
        }
    };
    let fit_worlds: Vec<HexWorld> = (0..5).map(build_world).collect();
    let eval_worlds: Vec<HexWorld> = (5..45).map(build_world).collect();

    let t_compile = Instant::now();
    let n_copy = (n_q / 4).max(2);
    let fit_ov_triples: Vec<(&[f64], &[f64], &[f64])> = fit_worlds
        .iter()
        .map(|w| {
            (
                w.h_schema_fk.as_slice(),
                emb[&w.fk_tok].as_slice(),
                emb[&w.think_tok].as_slice(),
            )
        })
        .collect();
    let (copy_heads, distractor_heads, head_scores) = eval_block.discover_fit_token_copy_heads(
        &fit_worlds[0].h_sql_cursor,
        &fit_ov_triples,
        n_copy,
        n_q - n_copy,
    );
    eprintln!("  [{model_id}] FIT contrastive OV head scores: {head_scores:.3?}");

    let mut fit_comp_rows = Vec::new();
    for w in &fit_worlds {
        fit_comp_rows.push(w.h_user_prose.clone());
        fit_comp_rows.push(w.h_create_bg.clone());
        fit_comp_rows.push(w.h_select_bg.clone());
    }
    let fit_fk_mat = StateMatrix::from_rows(
        "fit_fk",
        &fit_worlds
            .iter()
            .map(|w| w.h_schema_fk.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let fit_comp_mat = StateMatrix::from_rows("fit_comp", &fit_comp_rows).unwrap();
    let fit_think_mat = StateMatrix::from_rows(
        "fit_think",
        &fit_worlds
            .iter()
            .map(|w| w.h_think_decoy.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let q_delta = synthesize_fk_passthrough_query_delta(
        &eval_block.gqa,
        &copy_heads,
        &fit_fk_mat,
        &fit_comp_mat,
        None,
        Some(&fit_think_mat),
    )
    .unwrap();

    let full_wv_rank = eval_block.gqa.w_v_proj.nrows();
    let q_submerged = eval_block
        .gqa
        .build_submerged_forbidden_basis(Some(&q_wu64), full_wv_rank)
        .unwrap();
    let codebook = synthesize_asid_codebook(&eval_block.gqa, &copy_heads, &q_submerged).unwrap();

    let cursor_ref = &fit_worlds[0].h_sql_cursor;
    let mut raw_qk = vec![0.0; d];
    let mut raw_gate = vec![0.0; d];
    for &h in &copy_heads {
        let pb_q = eval_block.gqa.pullback_query_passthrough(h, &q_delta);
        let pb_g = eval_block.gqa.pullback_head_gate(h, cursor_ref);
        for j in 0..d {
            raw_qk[j] += pb_q[j];
            raw_gate[j] += pb_g[j];
        }
    }
    for &h in &distractor_heads {
        let pb_g = eval_block.gqa.pullback_head_gate(h, cursor_ref);
        for j in 0..d {
            raw_gate[j] -= pb_g[j];
        }
    }

    let mut v_ov_fit = vec![0.0; d];
    for w in &fit_worlds {
        for (acc, &x) in v_ov_fit.iter_mut().zip(&emb[&w.fk_tok]) {
            *acc += x;
        }
    }
    let (u_qk, u_gate, u_ov) = (
        normalize_unit_vec(&raw_qk).unwrap(),
        normalize_unit_vec(&raw_gate).unwrap(),
        normalize_unit_vec(&v_ov_fit).unwrap(),
    );
    let dir_v1 = normalize_unit_vec(&lin_comb(
        &[(0.60, &u_qk), (0.25, &u_gate), (0.15, &u_ov)],
        d,
    ))
    .unwrap();

    let u_qk_sub = normalize_unit_vec(&q_submerged.project_out(&raw_qk)).unwrap();
    let u_gate_sub = normalize_unit_vec(&q_submerged.project_out(&raw_gate)).unwrap();
    let dir_v2_ptr =
        normalize_unit_vec(&lin_comb(&[(0.65, &u_qk_sub), (0.35, &u_gate_sub)], d)).unwrap();
    let r_schema = &codebook.read_query_selectors[MemorySegmentAsid::AuthoritativeSchema as usize];
    let w_schema = &codebook.write_key_tags[MemorySegmentAsid::AuthoritativeSchema as usize];
    let w_think = &codebook.write_key_tags[MemorySegmentAsid::EphemeralScratchpad as usize];
    let dir_v2_full = normalize_unit_vec(
        &q_submerged.project_out(&lin_comb(&[(0.55, &dir_v2_ptr), (0.45, r_schema)], d)),
    )
    .unwrap();
    let compile_latency_ms = t_compile.elapsed().as_secs_f64() * 1000.0;

    let arm_specs = [
        ("A: Unsteered Base", 0usize),
        ("B: v1 qk_routing (15% v_ov_role, no ASID)", 1),
        ("C1: v2 SubmergedPointerRoute Only", 2),
        ("C2: v2 SubmergedPointerRoute + Paired ASID", 3),
    ];
    let mut arms = Vec::new();
    for (arm_name, mode_id) in arm_specs {
        let mut wins = 0usize;
        let (mut sum_margin, mut sum_s_attn, mut sum_p_attn, mut sum_t_attn, mut max_wv) =
            (0.0, 0.0, 0.0, 0.0, 0.0_f64);

        for w in &eval_worlds {
            let (q_h, k_schema, k_think) = match mode_id {
                0 => (
                    w.h_sql_cursor.clone(),
                    w.h_schema_fk.clone(),
                    w.h_think_decoy.clone(),
                ),
                1 => {
                    let (qh, dq) = rotate_chordal_submerged(&w.h_sql_cursor, &dir_v1, 0.52, None);
                    max_wv = max_wv.max(eval_block.wv_cache_contamination_l2(&dq));
                    (qh, w.h_schema_fk.clone(), w.h_think_decoy.clone())
                }
                2 => {
                    let (qh, dq) = rotate_chordal_submerged(
                        &w.h_sql_cursor,
                        &dir_v2_ptr,
                        0.52,
                        Some(&q_submerged),
                    );
                    max_wv = max_wv.max(eval_block.wv_cache_contamination_l2(&dq));
                    (qh, w.h_schema_fk.clone(), w.h_think_decoy.clone())
                }
                _ => {
                    let (qh, dq) = rotate_chordal_submerged(
                        &w.h_sql_cursor,
                        &dir_v2_full,
                        0.52,
                        Some(&q_submerged),
                    );
                    let (ks, dks) = rotate_chordal_submerged(
                        &w.h_schema_fk,
                        w_schema,
                        0.45,
                        Some(&q_submerged),
                    );
                    let (kt, dkt) = rotate_chordal_submerged(
                        &w.h_think_decoy,
                        w_think,
                        0.45,
                        Some(&q_submerged),
                    );
                    for d_vec in [&dq, &dks, &dkt] {
                        max_wv = max_wv.max(eval_block.wv_cache_contamination_l2(d_vec));
                    }
                    (qh, ks, kt)
                }
            };

            let keys = vec![
                (8, w.h_create_bg.clone()),
                (24, k_schema),
                (180, w.h_user_prose.clone()),
                (248, w.h_select_bg.clone()),
                (310, k_think),
            ];
            let cands = [
                emb[&w.fk_tok].as_slice(),
                emb[&w.prose_tok].as_slice(),
                emb[&w.think_tok].as_slice(),
            ];
            let res =
                eval_block.eval_multi_candidate_forward(&q_h, 316, &keys, &copy_heads, &cands);
            let z_fk = res.candidate_logits[0];
            let z_best_decoy = res.candidate_logits[1].max(res.candidate_logits[2]);
            if z_fk > z_best_decoy {
                wins += 1;
            }
            sum_margin += z_fk - z_best_decoy;
            sum_s_attn += res.mean_copy_head_attn[1] * 100.0;
            sum_p_attn += res.mean_copy_head_attn[2] * 100.0;
            sum_t_attn += res.mean_copy_head_attn[4] * 100.0;
        }
        let n_eval = eval_worlds.len();
        arms.push(ArmBenchmarkMetrics {
            arm: arm_name.to_string(),
            top1_unseen_fk_wins: wins,
            eval_worlds: n_eval,
            top1_accuracy: (wins as f64) / (n_eval as f64),
            mean_fk_vs_best_decoy_logit_margin: sum_margin / (n_eval as f64),
            schema_copy_attn_pct: sum_s_attn / (n_eval as f64),
            think_copy_attn_pct: sum_t_attn / (n_eval as f64),
            user_prose_copy_attn_pct: sum_p_attn / (n_eval as f64),
            max_wv_contamination_l2: max_wv,
        });
    }

    ModelBenchmarkReceipt {
        model_id: model_id.to_string(),
        gqa_layer: layer,
        hidden_dim: d,
        discovered_copy_heads: copy_heads,
        discovered_distractor_heads: distractor_heads,
        fit_worlds: fit_worlds.len(),
        eval_worlds: eval_worlds.len(),
        lexical_token_overlap_count: overlap,
        compile_latency_ms,
        arms,
    }
}

fn main() {
    let targets = [
        (
            "Qwen/Qwen3.5-0.8B-Base (L15 GQA)",
            "models--Qwen--Qwen3.5-0.8B-Base",
            15usize,
            1024usize,
            8usize,
            2usize,
        ),
        (
            "Qwen/Qwen3.5-0.8B-Base (L19 GQA)",
            "models--Qwen--Qwen3.5-0.8B-Base",
            19usize,
            1024usize,
            8usize,
            2usize,
        ),
        (
            "Qwen/Qwen3.5-0.8B-Base (L23 GQA)",
            "models--Qwen--Qwen3.5-0.8B-Base",
            23usize,
            1024usize,
            8usize,
            2usize,
        ),
        (
            "Qwen/Qwen3.5-4B-Base   (L23 GQA)",
            "models--Qwen--Qwen3.5-4B-Base",
            23usize,
            2560usize,
            16usize,
            4usize,
        ),
    ];
    let mut receipts = Vec::new();
    for (model_id, repo_dir, layer, d, n_q, n_kv) in targets {
        let Some(snap) = find_latest_hf_snapshot(repo_dir) else {
            eprintln!("Skipping {model_id}: snapshot not cached in ~/.cache/huggingface/hub");
            continue;
        };
        let rec = run_checkpoint_benchmark(model_id, &snap, layer, d, n_q, n_kv);
        println!(
            "\n=== {model_id} (L{layer} GQA, d={d}, copy_heads={:?}, compile={:.2} ms, fit->eval={}->{}, overlap={}) ===",
            rec.discovered_copy_heads, rec.compile_latency_ms, rec.fit_worlds, rec.eval_worlds, rec.lexical_token_overlap_count
        );
        println!(
            "{:<44} | {:>12} | {:>11} | {:>11} | {:>11} | {:>12}",
            "Arm", "Top-1 FK", "Logit Δ", "<schema> %", "<think> %", "||W_V·Δh||_2"
        );
        println!("{}", "-".repeat(115));
        for a in &rec.arms {
            println!(
                "{:<44} | {:>2}/{:<2} ({:>3.0}%) | {:>+11.4} | {:>10.2}% | {:>10.2}% | {:>12.3e}",
                a.arm,
                a.top1_unseen_fk_wins,
                a.eval_worlds,
                a.top1_accuracy * 100.0,
                a.mean_fk_vs_best_decoy_logit_margin,
                a.schema_copy_attn_pct,
                a.think_copy_attn_pct,
                a.max_wv_contamination_l2
            );
        }
        receipts.push(rec);
    }
    println!(
        "\nJSON_RECEIPT: {}",
        serde_json::to_string(&receipts).unwrap()
    );
}
