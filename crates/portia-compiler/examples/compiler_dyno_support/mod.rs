//! Synthetic activation & geometry builders for `examples/compiler_dyno.rs`.

use portia_compiler::discovery::gqa_bus::{GqaPointerLayer, MemorySegmentAsid};
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::{ModelGeometryTables, RuleCaptureData};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::StateMatrix;
use portia_compiler::solvers::clarabel_ternary::CursorPolarity;
use portia_compiler::solvers::deltanet_scrub::DeltaNetGateLayer;
use portia_compiler::solvers::V2SolverConfig;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy};
use std::collections::BTreeMap;

fn prng_f64(mut z: u64) -> f64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (((z ^ (z >> 31)) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

pub fn pre_mlp_rows(
    n: usize,
    d: usize,
    active_axis: Option<(usize, f64)>,
    seed: u64,
) -> StateMatrix {
    let flat: Vec<f64> = (0..n * d)
        .map(|k| {
            let (r, c) = (k / d, k % d);
            if Some(c) == active_axis.map(|(ax, _)| ax) {
                active_axis.unwrap().1 + 0.03 * prng_f64(seed.wrapping_add(999) ^ (r as u64))
            } else if c == 0 {
                6.0
            } else if c >= 64 {
                0.04 * prng_f64(seed ^ ((r as u64) << 20) ^ (c as u64))
            } else {
                0.0
            }
        })
        .collect();
    StateMatrix::try_new("dyno_pre_mlp", n, d, flat).unwrap()
}

pub fn axis_subspace(d: usize, start: usize, rank: usize) -> OrthonormalSubspace {
    let rows: Vec<Vec<f64>> = (0..rank)
        .map(|k| {
            let mut v = vec![0.0; d];
            v[start + k] = 1.0;
            v
        })
        .collect();
    OrthonormalSubspace::from_row_space_svd(
        "axis_q",
        &StateMatrix::from_rows("axis_q", &rows).unwrap(),
        1e-4,
        rank,
    )
    .unwrap()
}

pub fn build_demo_program(
    d: usize,
    cfg: &V2SolverConfig,
) -> (
    IntentExpr,
    BTreeMap<String, RuleCaptureData>,
    ModelGeometryTables,
) {
    let (num_q, num_kv, head_dim, rotary_dim) = (4usize, 1usize, 64usize, 16usize);
    let mut wq = vec![vec![0.0; d]; 2 * num_q * head_dim];
    let mut wk = vec![vec![0.0; d]; num_kv * head_dim];
    let mut wv = vec![vec![0.0; d]; num_kv * head_dim];
    for i in 0..head_dim {
        wv[i][200 + i] = 1.3;
        wk[i][300 + i] = 1.2;
        for h in 0..num_q {
            wq[h * 2 * head_dim + i][300 + i] = 1.2;
            wq[h * 2 * head_dim + head_dim + i][400 + (i % 32)] = 1.0;
        }
    }
    let mut wa = vec![vec![0.0; d]; 8];
    let mut wb = vec![vec![0.0; d]; 8];
    for h in 0..8 {
        wa[h][500 + h] = 1.4;
        wb[h][520 + h] = 1.4;
    }
    let mut geom = ModelGeometryTables {
        model_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
        base_intermediate_size: 3584,
        widened_slot_capacity: 128,
        ..Default::default()
    };
    geom.gqa_layers.insert(
        15,
        GqaPointerLayer {
            layer_idx: 15,
            hidden_dim: d,
            num_q_heads: num_q,
            num_kv_heads: num_kv,
            head_dim,
            rotary_dim,
            input_norm_scale: vec![1.0; d],
            q_norm_scale: vec![1.0; head_dim],
            k_norm_scale: vec![1.0; head_dim],
            w_q_proj: StateMatrix::from_rows("wq", &wq).unwrap(),
            w_k_proj: StateMatrix::from_rows("wk", &wk).unwrap(),
            w_v_proj: StateMatrix::from_rows("wv", &wv).unwrap(),
        },
    );
    geom.deltanet_layers.insert(
        21,
        DeltaNetGateLayer {
            layer_idx: 21,
            hidden_dim: d,
            num_heads: 8,
            input_norm_scale: vec![1.0; d],
            w_a_proj: StateMatrix::from_rows("wa", &wa).unwrap(),
            w_b_proj: StateMatrix::from_rows("wb", &wb).unwrap(),
        },
    );

    let shared_off = pre_mlp_rows(800, d, None, 303);
    let res_mat = pre_mlp_rows(12, d, Some((700, 18.0)), 404);
    let mut caps = BTreeMap::new();
    let mut expr_opt: Option<IntentExpr> = None;

    for idx in 0..12usize {
        let name = format!("dyno_rule_{idx:02}");
        let ax = 6 + idx * 2;
        let mut tdir = vec![0.0; d];
        tdir[600 + idx] = 1.0;
        let on_m = pre_mlp_rows(12, d, Some((ax, 5.2)), 500 + idx as u64);
        let std_cap = || {
            RuleCaptureData::standard_positive(
                on_m.clone(),
                res_mat.clone(),
                shared_off.clone(),
                cfg.clone(),
            )
        };

        let (kind, mount_policy, cap) = match idx {
            0..=7 => (
                HardwareIntentKind::StanceRoute {
                    layer_idx: 22,
                    raw_target_dir: tdir,
                    angle: ChordalAngleRad::new(0.32),
                    expected_mode: Some(Mode::Read),
                },
                if idx < 4 {
                    MountPolicy::PermanentStatic
                } else {
                    MountPolicy::DynamicBank(CartridgeBank::OrientTrunk)
                },
                std_cap(),
            ),
            8 => {
                let (mut xp, mut xm) = (vec![0.0; d], vec![0.0; d]);
                (xp[0], xp[ax], xp[ax + 1]) = (6.0, 5.2, 2.8);
                (xm[0], xm[ax], xm[ax + 1]) = (6.0, 5.2, -2.8);
                let x0: Vec<f64> = xp.iter().zip(&xm).map(|(a, b)| 0.5 * (a + b)).collect();
                (
                    HardwareIntentKind::SignedTernaryFork {
                        layer_idx: 22,
                        branch_axis_dir: tdir,
                        angle: ChordalAngleRad::new(0.30),
                    },
                    MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Ask)),
                    RuleCaptureData {
                        on_cursor_pre_mlp: StateMatrix::from_rows("ton", &[xp, x0, xm]).unwrap(),
                        polarities: vec![
                            CursorPolarity::PlusOne,
                            CursorPolarity::ZeroNeutral,
                            CursorPolarity::MinusOne,
                        ],
                        pos_residuals: pre_mlp_rows(3, d, Some((700, 18.0)), 601),
                        off_cursor_pre_mlp: shared_off.clone(),
                        q_preserve_gate: None,
                        solver_config: cfg.clone(),
                    },
                )
            }
            9 => {
                let mut q_delta = vec![0.0; 48];
                q_delta[0] = 1.0;
                (
                    HardwareIntentKind::SubmergedPointerRoute {
                        upstream_mlp_layer: 14,
                        target_gqa_layer: 15,
                        copy_heads: vec![0, 1],
                        distractor_heads: vec![2],
                        query_passthrough_delta: q_delta,
                        angle: ChordalAngleRad::new(0.28),
                    },
                    MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Read)),
                    std_cap(),
                )
            }
            10 => (
                HardwareIntentKind::AsidSegmentColor {
                    upstream_mlp_layer: 14,
                    target_gqa_layer: 15,
                    copy_heads: vec![0, 1],
                    segment: MemorySegmentAsid::AuthoritativeSchema,
                    is_write_side_key_stamp: true,
                    angle: ChordalAngleRad::new(0.28),
                },
                MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Probe)),
                std_cap(),
            ),
            _ => (
                HardwareIntentKind::DeltaNetRollbackScrub {
                    upstream_mlp_layer: 20,
                    target_deltanet_layer: 21,
                    flush_heads: vec![0, 1],
                    lock_heads: vec![2, 3],
                    a_drive: 4.0,
                },
                MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Write)),
                std_cap(),
            ),
        };

        caps.insert(name.clone(), cap);
        let leaf = IntentExpr::rule(IntentRule {
            name,
            guard: RegimeExpr::ConstBool(true),
            mount_policy,
            grammar: GrammarCartridgeSpec::Unconstrained,
            kind,
        });
        expr_opt = Some(expr_opt.map_or(leaf.clone(), |acc| acc & leaf));
    }

    (expr_opt.unwrap(), caps, geom)
}
