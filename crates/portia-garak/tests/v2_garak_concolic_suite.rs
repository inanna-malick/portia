//! Integration & property tests for `portia_garak::v2_concolic` (unified `portia-compiler` lowering +
//! white-box concolic `MAP-Elites` discrete probe boundary explorer).

use half::bf16;
use portia_compiler::concolic::ConcolicDiodeLens;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{l2_norm, StateMatrix};
use portia_compiler::wire::WireCartridgeManifestJson;
use portia_garak::v2_concolic::splits::prepare_garak_v2_inputs;
use portia_garak::v2_concolic::{
    compile_garak_v2_cartridge, compile_garak_v2_with_geometry, run_blind_blackbox_probe_search,
    run_whitebox_concolic_map_elites_search, DiscreteGarakProbe, GarakV2GeometryExtras,
};
use portia_garak::{
    GarakCalibratedAngles, GarakCalibrationFamily, GarakCompileInputSpec, GarakPartition,
    GarakPreMlpSplits, GarakProbeFamily, GarakReadoutSpec, GarakResidualSplits, GarakRoleSplit,
    GarakSolverError, GarakV2Spec, GARAK_CONTRAST_TOKEN_IDS, GARAK_SLOT_NAMES,
};
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

fn sample_full_mlp_garak_spec(d: usize) -> GarakCompileInputSpec {
    let mk_rows = |n: usize, axis: usize, amp: f64, seed: usize| -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                let mut v: Vec<f64> = (0..d)
                    .map(|c| 0.025 * (((r * 13 + c * 7 + seed * 17) % 19) as f64 - 9.0))
                    .collect();
                v[0] = 9.5;
                v[axis] = amp + 0.05 * ((r % 5) as f64 - 2.0);
                v
            })
            .collect()
    };
    let mut train_true_claims = mk_rows(6, 4, 7.2, 5);
    train_true_claims.extend(mk_rows(6, 5, 7.2, 6));

    let mk_pkg = |n_ent: usize, sign: f64, seed: usize| -> Vec<Vec<f64>> {
        (0..4 * n_ent)
            .map(|k| {
                let (t, e) = (k / n_ent, k % n_ent);
                let mut v: Vec<f64> = (0..d)
                    .map(|c| 0.025 * (((t * 11 + e * 5 + c * 3 + seed) % 17) as f64 - 8.0))
                    .collect();
                v[0] = 9.5;
                v[6] = sign * (6.6 + 0.04 * (e as f64));
                v[7] = 0.3 * ((t as f64) - 1.5);
                v[8] = 7.6;
                v
            })
            .collect()
    };

    let m_f = 16;
    let (mut w_gate, mut w_up, mut w_down) = (
        vec![vec![0.0; d]; m_f],
        vec![vec![0.0; d]; m_f],
        vec![vec![0.0; m_f]; d],
    );
    for j in 0..m_f {
        let ax = (j + 14) % d;
        (w_gate[j][ax], w_up[j][ax], w_down[ax][j]) = (0.35, 0.30, 0.25);
    }
    let w_u_rows: BTreeMap<String, Vec<f64>> = GARAK_CONTRAST_TOKEN_IDS
        .iter()
        .enumerate()
        .map(|(idx, &tid)| {
            let mut row = vec![0.0; d];
            row[16 + (idx % (d - 18))] = 1.8;
            (tid.to_string(), row)
        })
        .collect();

    GarakCompileInputSpec {
        v2: None,
        model_id: "Qwen/Qwen3.5-0.8B".to_string(),
        target_layer_idx: 22,
        base_intermediate_size: 3584,
        widened_slots: Some(128),
        slot_offset: Some(48),
        pkg_template_count: 4,
        x_splits: GarakPreMlpSplits {
            train_gc: mk_rows(10, 2, 7.6, 1),
            train_fc: mk_rows(10, 3, 7.8, 2),
            train_true_claims,
            train_unreal_py: mk_pkg(5, 1.0, 3),
            train_real_py: mk_pkg(5, -1.0, 4),
            test_gc: mk_rows(6, 2, 7.5, 7),
            test_fc: mk_rows(6, 3, 7.6, 8),
            test_unreal_py: mk_pkg(2, 1.0, 9),
            transfer_senators: mk_rows(6, 10, 7.9, 14),
            transfer_primes: mk_rows(6, 4, 7.0, 10),
            transfer_unreal_rs: mk_pkg(2, 1.0, 11),
            guard_real_code: mk_pkg(2, -1.0, 12),
            guard_true_claims: mk_rows(6, 5, 7.0, 13),
        },
        h_splits: GarakResidualSplits {
            train_gc: mk_rows(10, 2, 25.0, 21),
            train_fc: mk_rows(10, 3, 25.0, 22),
            train_true_claims: mk_rows(12, 4, 25.0, 23),
            train_unreal_py: mk_rows(20, 6, 25.0, 24),
        },
        x_cont_all: mk_rows(20, 1, 10.5, 25),
        calibrated_angles: GarakCalibratedAngles::default(),
        readout: GarakReadoutSpec::FullDownstreamMlp {
            w_u_rows,
            final_norm_weight: vec![0.1; d],
            post_attn_norm_weight: vec![0.1; d],
            w_gate,
            w_up,
            w_down,
            top_k: 8,
        },
        max_bg_rss_leakage: 0.25,
        strict_guard_check: true,
    }
}

#[test]
fn v2_concolic_source_files_strictly_within_350_loc_invariant() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<_> = "src/lib.rs|src/specs.rs|src/readout.rs|src/v2_concolic.rs|tests/v2_garak_concolic_suite.rs|tests/detectors_and_readouts_suite.rs|examples/bench_v2_garak_map_elites_ab.rs|examples/bench_v2_garak_support/mod.rs"
        .split('|')
        .map(|p| dir.join(p))
        .collect();
    for entry in std::fs::read_dir(dir.join("src/v2_concolic")).expect("read src/v2_concolic") {
        let p = entry.expect("dir entry").path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(p);
        }
    }
    for path in files.into_iter().filter(|p| p.exists()) {
        let n = std::fs::read_to_string(&path).unwrap().lines().count();
        assert!(n <= 350, "{} has {} lines (> 350)", path.display(), n);
    }
    for sub in ["src/specs.rs", "src/readout.rs"] {
        let n = std::fs::read_to_string(dir.join(sub))
            .unwrap()
            .lines()
            .count();
        assert!(n <= 320, "{sub} has {n} lines (> 320)");
    }
}

#[test]
fn v2_lowering_full_downstream_mlp_secant_nullspace_and_slot_relocation() {
    let spec = sample_full_mlp_garak_spec(48);
    let prepared = prepare_garak_v2_inputs(&spec).expect("prepare inputs");
    let out = compile_garak_v2_with_geometry(&spec, GarakV2GeometryExtras::default()).unwrap();
    assert_eq!(
        (out.receipt.target_layer, out.receipt.widened_slots),
        (22, 128)
    );
    assert_eq!(
        (out.receipt.slot_lo, out.receipt.slot_hi),
        (3584 + 48, 3584 + 56)
    );
    assert!(out.receipt.guard_bg_max <= 0.25 && out.concolic_lenses.len() == 4);

    let roundtrip =
        WireCartridgeManifestJson::from_json_str(&out.manifest.to_json_pretty().unwrap()).unwrap();
    roundtrip.verify_schema_integrity().unwrap();
    let entries = &roundtrip.permanent_static["22"];
    assert_eq!(
        (entries.len(), entries[0].slot_lo, entries[3].slot_hi),
        (4, 3584 + 48, 3584 + 56)
    );

    for (slot_idx, entry) in entries.iter().enumerate() {
        let w_gate_f64: Vec<f64> = entry.w_gate[0].iter().map(|&v| v as f64).collect();
        let w_norm = l2_norm(&w_gate_f64).max(1e-12);
        if let Some(q_pres) = &prepared.captures_by_slot[slot_idx].q_preserve_gate {
            let resid = q_pres.project_out(&w_gate_f64);
            let leaked_sq: f64 = w_gate_f64
                .iter()
                .zip(&resid)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            assert!(leaked_sq.sqrt() / w_norm < 1e-2);
        }
    }

    let mut bad_widened = spec.clone();
    bad_widened.widened_slots = Some(52);
    assert!(matches!(
        compile_garak_v2_cartridge(&bad_widened),
        Err(GarakSolverError::SlotOffsetOutOfBounds { .. })
    ));

    if out.receipt.guard_bg_max > 0.0 {
        let mut tight = spec;
        tight.max_bg_rss_leakage = out.receipt.guard_bg_max * 0.1;
        assert!(matches!(
            compile_garak_v2_cartridge(&tight),
            Err(GarakSolverError::GuardLeakageV2Exceeded { .. })
        ));
    }
}

#[test]
fn v2_role_tagged_spec_compiles_and_enforces_partition_invariants() {
    let d = 48;
    let mut spec = sample_full_mlp_garak_spec(d);
    let mut splits = Vec::new();
    for (p_idx, part) in [GarakPartition::Fit, GarakPartition::Dev]
        .into_iter()
        .enumerate()
    {
        for (f_idx, fam) in GarakCalibrationFamily::REQUIRED_V2_FAMILIES
            .into_iter()
            .enumerate()
        {
            let pos = fam.expected_positive_slot();
            let x: Vec<Vec<f64>> = (0..6)
                .map(|r| {
                    let mut v: Vec<f64> = (0..d)
                        .map(|c| {
                            0.02 * (((r * 11 + c * 5 + f_idx * 19 + p_idx * 7) % 17) as f64 - 8.0)
                        })
                        .collect();
                    v[0] = 9.5;
                    v[1 + f_idx] =
                        if pos.is_some() { 8.4 } else { 7.0 } + 0.04 * ((r % 3) as f64 - 1.0);
                    v
                })
                .collect();
            let h = x
                .iter()
                .map(|v| v.iter().map(|u| 2.5 * u).collect())
                .collect();
            splits.push(GarakRoleSplit {
                name: format!("{part}_{fam}"),
                partition: part,
                family: fam,
                row_ids: (0..6).map(|r| format!("{part}_{fam}_{r}")).collect(),
                x,
                h,
                positive_slots: pos.into_iter().map(str::to_string).collect(),
                cold_slots: GARAK_SLOT_NAMES
                    .into_iter()
                    .filter(|&s| Some(s) != pos)
                    .map(str::to_string)
                    .collect(),
                globally_benign: pos.is_none(),
            });
        }
    }
    spec.v2 = Some(GarakV2Spec {
        splits,
        ridge: 1e-3,
        cold_weight: 2.0,
        candidate_index: 0,
    });
    let (manifest, receipt, lenses) = compile_garak_v2_cartridge(&spec).unwrap();
    manifest.verify_schema_integrity().unwrap();
    assert!(receipt.guard_bg_max <= 0.25 && lenses.len() == 4);

    let mut dup = spec.clone();
    dup.v2.as_mut().unwrap().splits[1].row_ids[0] =
        dup.v2.as_ref().unwrap().splits[0].row_ids[0].clone();
    assert!(matches!(
        compile_garak_v2_cartridge(&dup),
        Err(GarakSolverError::InvalidSplit { .. })
    ));

    let mut bad = spec;
    bad.v2.as_mut().unwrap().splits[0].cold_slots.clear();
    assert!(matches!(
        compile_garak_v2_cartridge(&bad),
        Err(GarakSolverError::InvalidSplit { .. })
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn whitebox_concolic_map_elites_outperforms_blind_search_across_seeds(
        rng_seed in 0u64..10_000u64,
        slope in 0.85f64..1.35f64,
    ) {
        let d = 24usize;
        let mut lenses = BTreeMap::new();
        let mut w_u_row = vec![0.0; d];
        w_u_row[d - 1] = 1.0;
        let w_u_mat = StateMatrix::from_rows("w_u", &[w_u_row]).unwrap();
        let w_u_sub = OrthonormalSubspace::from_row_space_svd("w_u", &w_u_mat, 1e-6, 2).unwrap();

        for (f_idx, fam) in GarakProbeFamily::ALL.into_iter().enumerate() {
            let mut w_gate = vec![bf16::from_f64(0.0); d];
            w_gate[f_idx] = bf16::from_f64(1.0);
            let on_rows: Vec<Vec<f64>> = (0..8).map(|r| {
                let mut row = vec![0.0; d];
                row[f_idx] = 3.0;
                row[4 + (r % 4)] = if r % 2 == 0 { 1.5 } else { -1.5 };
                row
            }).collect();
            let on_mat = StateMatrix::from_rows("on", &on_rows).unwrap();
            let lens = ConcolicDiodeLens::build(fam.rule_name(), 22, &w_gate, Some(&w_u_sub), &on_mat, 4).unwrap();
            lenses.insert(fam.rule_name().to_string(), lens);
        }

        let mut seeds = Vec::new();
        let mut candidates = Vec::new();
        for (f_idx, fam) in GarakProbeFamily::ALL.into_iter().enumerate() {
            for s in 0..2usize {
                let mut x = vec![0.0; d];
                x[f_idx] = 2.6 * slope;
                x[4 + s] = 1.0;
                seeds.push((DiscreteGarakProbe { id: format!("s_{f_idx}_{s}"), family: fam, coords: [s, 0, 0], prompt: String::new() }, x));
            }
            for c0 in 0..2usize {
                for c1 in 0..6usize {
                    for c2 in 0..3usize {
                        let mut x = vec![0.0; d];
                        x[f_idx] = if c1 <= 3 { (2.6 - 0.65 * (c1 as f64) - 0.45 * (c2 as f64)) * slope } else { (2.4 + 0.3 * (c1 as f64)) * slope };
                        x[4 + (c0 % 4)] = (c1 as f64) - 2.0;
                        x[8 + (c2 % 4)] = (c2 as f64) - 1.0;
                        candidates.push((DiscreteGarakProbe { id: format!("c_{f_idx}_{c0}_{c1}_{c2}"), family: fam, coords: [c0, c1, c2], prompt: String::new() }, x));
                    }
                }
            }
        }

        let (mut blind_byp, mut conc_byp, mut blind_min, mut conc_min) = (0usize, 0usize, 0.0f64, 0.0f64);
        for off in 0..8u64 {
            let s = rng_seed.wrapping_add(off * 1009);
            let b = run_blind_blackbox_probe_search(&lenses, &seeds, &candidates, 36, 1.5, s);
            let c = run_whitebox_concolic_map_elites_search(&lenses, &seeds, &candidates, 36, 1.5, s);
            blind_byp += b.bypassed_probes.len();
            conc_byp += c.bypassed_probes.len();
            blind_min += b.min_cliff_margin;
            conc_min += c.min_cliff_margin;
        }
        prop_assert!(conc_min <= blind_min + 1e-9);
        prop_assert!(conc_byp > blind_byp, "conc_byp={conc_byp} <= blind_byp={blind_byp}");
    }
}
