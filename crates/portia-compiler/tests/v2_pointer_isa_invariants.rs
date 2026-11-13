//! Pointer-ISA, Submerged `ASID`, Convex-Midpoint Ternary, & Anti-Megafile (`<= 350` LOC) Invariants.

use portia_compiler::discovery::gqa_bus::{
    synthesize_asid_codebook, GqaPointerLayer, MemorySegmentAsid,
};
use portia_compiler::discovery::{discover_and_project_anti_echo, DownstreamSwigluBlock};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, rms_norm_slice, StateMatrix};
use portia_compiler::solvers::clarabel_ternary::{compile_coupled_ternary_pair, CursorPolarity};
use portia_compiler::solvers::V2SolverConfig;
use std::fs;
use std::path::Path;

fn walk_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if !dir.exists() {
        return;
    }
    for entry in fs::read_dir(dir).expect("read_dir") {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            walk_rs_files(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

#[test]
fn test_module_tree_file_size_hygiene_under_350_loc() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for sub in ["src", "tests", "examples"] {
        walk_rs_files(&manifest_dir.join(sub), &mut files);
    }
    assert!(files.len() >= 14, "expected modular tree, got {:?}", files);
    for f in files {
        let content = fs::read_to_string(&f).expect("read source");
        let lines = content.lines().count();
        assert!(
            lines <= 350,
            "Anti-megafile law violated: {} has {} lines (limit 350)",
            f.display(),
            lines
        );
    }
}

#[test]
fn test_convex_midpoint_ternary_swiglu_fork_solves_and_preserves_zero_at_ask() {
    let d = 64;
    let mut x_fork = vec![0.0; d];
    x_fork[0] = 3.0;
    x_fork[1] = 2.5;

    let mut x_inline = vec![0.0; d];
    x_inline[0] = 3.0;
    x_inline[1] = -2.5;

    let x_ask_midpoint: Vec<f64> = x_fork
        .iter()
        .zip(&x_inline)
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();

    let mut off1 = vec![0.0; d];
    off1[0] = -2.5;
    off1[2] = 1.5;
    let mut off2 = vec![0.0; d];
    off2[0] = -1.8;
    off2[3] = -2.0;

    let on_mat = StateMatrix::from_rows(
        "on_ternary",
        &[x_fork.clone(), x_ask_midpoint.clone(), x_inline.clone()],
    )
    .unwrap();
    let off_mat = StateMatrix::from_rows("off_ternary", &[off1.clone(), off2]).unwrap();

    let mut h_res = vec![0.0; d];
    h_res[4] = 18.0;
    let res_mat =
        StateMatrix::from_rows("pos_res", &[h_res.clone(), h_res.clone(), h_res]).unwrap();

    let mut target_dir = vec![0.0; d];
    target_dir[5] = 1.0;

    let pair = compile_coupled_ternary_pair(
        "hylo_fork_ask_inline",
        0,
        1,
        &on_mat,
        &[
            CursorPolarity::PlusOne,
            CursorPolarity::ZeroNeutral,
            CursorPolarity::MinusOne,
        ],
        &res_mat,
        &off_mat,
        &target_dir,
        ChordalAngleRad::new(0.28),
        None,
        None,
        &V2SolverConfig::default(),
    )
    .expect("convex-midpoint ternary pair must solve cleanly");

    let (_, act_fork) = pair.tangential_slot.eval_gate_and_act(&x_fork);
    let (_, act_ask) = pair.tangential_slot.eval_gate_and_act(&x_ask_midpoint);
    let (_, act_inline) = pair.tangential_slot.eval_gate_and_act(&x_inline);
    let (g_off, _) = pair.tangential_slot.eval_gate_and_act(&off1);

    assert!(
        act_fork > 2.0,
        "expected strong +tan on FORK, got {act_fork}"
    );
    assert!(
        act_inline < -2.0,
        "expected strong -tan on INLINE, got {act_inline}"
    );
    assert!(
        act_ask.abs() < 0.25,
        "expected near-zero tangential kick at convex midpoint ASK, got {act_ask}"
    );
    assert!(
        g_off <= -8.0 + 1e-3,
        "expected off-cursor <= -8.0, got {g_off}"
    );
}

#[test]
fn test_submerged_asid_coloring_suppresses_think_hallucination_with_zero_wv_leak() {
    let d = 128;
    let num_q_heads = 4;
    let num_kv_heads = 1;
    let head_dim = 32;
    let rotary_dim = 8;

    let mut wq_rows = vec![vec![0.0; d]; 2 * num_q_heads * head_dim];
    let mut wk_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];
    let mut wv_rows = vec![vec![0.0; d]; num_kv_heads * head_dim];

    for i in 0..head_dim {
        wv_rows[i][i] = 1.4;
        wk_rows[i][48 + i] = 1.2;
        for h in 0..num_q_heads {
            wq_rows[h * 2 * head_dim + i][48 + i] = 1.2;
            wq_rows[h * 2 * head_dim + head_dim + i][80 + (i % 16)] = 1.0;
        }
    }

    let mut wu_rows = vec![vec![0.0; d]; 16];
    for i in 0..16 {
        wu_rows[i][32 + i] = 2.0;
    }
    let q_wu = OrthonormalSubspace::from_row_space_svd(
        "wu_top",
        &StateMatrix::from_rows("wu", &wu_rows).unwrap(),
        1e-4,
        16,
    )
    .unwrap();

    let gqa = GqaPointerLayer {
        layer_idx: 15,
        hidden_dim: d,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        input_norm_scale: vec![1.0; d],
        q_norm_scale: vec![1.0; head_dim],
        k_norm_scale: vec![1.0; head_dim],
        w_q_proj: StateMatrix::from_rows("wq", &wq_rows).unwrap(),
        w_k_proj: StateMatrix::from_rows("wk", &wk_rows).unwrap(),
        w_v_proj: StateMatrix::from_rows("wv", &wv_rows).unwrap(),
    };

    let q_submerged = gqa
        .build_submerged_forbidden_basis(Some(&q_wu), 64)
        .unwrap();
    let codebook = synthesize_asid_codebook(&gqa, &[0, 1], &q_submerged).unwrap();
    assert!(
        codebook.max_wv_leakage < 1e-12,
        "expected machine-zero W_V + W_U leakage, got {:e}",
        codebook.max_wv_leakage
    );

    let mut base_col = vec![0.0; d];
    base_col[10] = 3.5;
    base_col[52] = 1.0;

    let k_schema_tagged: Vec<f64> = base_col
        .iter()
        .zip(&codebook.write_key_tags[MemorySegmentAsid::AuthoritativeSchema as usize])
        .map(|(&b, &t)| b + 3.0 * t)
        .collect();
    let k_think_tagged: Vec<f64> = base_col
        .iter()
        .zip(&codebook.write_key_tags[MemorySegmentAsid::EphemeralScratchpad as usize])
        .map(|(&b, &t)| b + 3.0 * t)
        .collect();

    let v_orig = gqa.w_v_proj.matvec(&base_col).unwrap();
    let v_tagged = gqa.w_v_proj.matvec(&k_schema_tagged).unwrap();
    for (a, b) in v_orig.iter().zip(&v_tagged) {
        assert!((a - b).abs() < 1e-11, "W_V payload corrupted: {a} vs {b}");
    }

    let mut cursor = vec![0.0; d];
    cursor[52] = 1.0;
    let cursor_steered: Vec<f64> = cursor
        .iter()
        .zip(&codebook.read_query_selectors[MemorySegmentAsid::AuthoritativeSchema as usize])
        .map(|(&c, &q)| c + 3.0 * q)
        .collect();

    let c_norm = rms_norm_slice(&cursor_steered, &gqa.input_norm_scale, 1e-6);
    let ks_norm = rms_norm_slice(&k_schema_tagged, &gqa.input_norm_scale, 1e-6);
    let kt_norm = rms_norm_slice(&k_think_tagged, &gqa.input_norm_scale, 1e-6);

    let q0: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_q_proj.row(i), &c_norm))
        .collect();
    let k_schema: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &ks_norm))
        .collect();
    let k_think: Vec<f64> = (0..head_dim)
        .map(|i| dot_f64(gqa.w_k_proj.row(i), &kt_norm))
        .collect();

    assert!(
        dot_f64(&q0, &k_schema) > dot_f64(&q0, &k_think) + 0.5,
        "expected ASID selector to strongly prefer schema over <think> scratchpad"
    );
}

#[test]
fn test_contextual_secant_anti_echo_isolates_true_suppressors_only() {
    let d = 32;
    let m = 4;
    let mut wg = vec![vec![0.0; d]; m];
    let mut wu = vec![vec![0.0; d]; m];
    let mut wd = vec![vec![0.0; d]; m];

    wg[0][0] = 2.0;
    wu[0][0] = 2.0;
    wd[0][0] = 10.0;

    wg[1][1] = 2.5;
    wu[1][1] = 2.5;
    wd[1][0] = -6.0;

    let block = DownstreamSwigluBlock::try_new(
        23,
        vec![1.0; d],
        StateMatrix::from_rows("wg", &wg).unwrap(),
        StateMatrix::from_rows("wu", &wu).unwrap(),
        StateMatrix::from_rows("wd", &wd).unwrap(),
    )
    .unwrap();

    let mut h0 = vec![0.05; d];
    h0[2] = 4.0;
    let pos_res = StateMatrix::from_rows("res", &[h0]).unwrap();

    let mut raw_u = vec![0.0; d];
    raw_u[0] = 0.8;
    raw_u[1] = 0.6;

    let (u_safe, q_anti, receipt) =
        discover_and_project_anti_echo("test_secant", &block, &pos_res, &raw_u, 0.2, 4).unwrap();
    assert_eq!(receipt.suppressor_neuron_indices, vec![1]);
    assert!(q_anti.leakage_norm(&u_safe) < 1e-12);
    assert!(u_safe[0] > 0.95);
}
