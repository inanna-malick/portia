use super::*;
use clap::Parser;
use portia_garak::{
    GarakCalibratedAngles, GarakPreMlpSplits, GarakResidualSplits, GARAK_CONTRAST_TOKEN_IDS,
};
use std::collections::BTreeMap;
use tempfile::TempDir;

fn sample_cli_garak_spec(d: usize) -> GarakCompileInputSpec {
    let mk_rows = |n: usize, axis: usize, amp: f64, seed: usize| -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                (0..d)
                    .map(|c| match c {
                        0 => 9.5,
                        x if x == axis => amp + 0.06 * ((r % 5) as f64 - 2.0),
                        _ => 0.03 * (((r * 13 + c * 7 + seed * 17) % 19) as f64 - 9.0),
                    })
                    .collect()
            })
            .collect()
    };

    let mut train_true_claims = mk_rows(6, 4, 7.0, 5);
    train_true_claims.extend(mk_rows(6, 5, 7.0, 6));

    let mk_pkg = |n_ent: usize, sign: f64, seed: usize| -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(4 * n_ent);
        for t in 0..4 {
            for e in 0..n_ent {
                out.push(
                    (0..d)
                        .map(|c| match c {
                            0 => 9.5,
                            6 => sign * (6.4 + 0.04 * (e as f64)),
                            7 => 0.3 * ((t as f64) - 1.5),
                            8 => 7.8,
                            _ => 0.03 * (((t * 11 + e * 5 + c * 3 + seed) % 17) as f64 - 8.0),
                        })
                        .collect(),
                );
            }
        }
        out
    };

    let x_cont_all: Vec<Vec<f64>> = (0..24)
        .map(|r| {
            (0..d)
                .map(|c| match c {
                    0 => 8.0,
                    1 => 11.0 + 0.05 * ((r % 5) as f64),
                    _ => 0.02 * (((r * 7 + c * 9) % 15) as f64 - 7.0),
                })
                .collect()
        })
        .collect();

    let mk_res = |n: usize, axis: usize| -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                (0..d)
                    .map(|c| {
                        if c == axis {
                            60.0 + 0.2 * (r as f64)
                        } else {
                            0.1
                        }
                    })
                    .collect()
            })
            .collect()
    };

    let mut w_u_rows = BTreeMap::new();
    for (idx, &tid) in GARAK_CONTRAST_TOKEN_IDS.iter().enumerate() {
        let mut v = vec![0.0; d];
        v[12 + idx] = 1.8;
        w_u_rows.insert(tid.to_string(), v);
    }

    GarakCompileInputSpec {
        v2: None,
        model_id: "Qwen/Qwen3.5-0.8B".to_string(),
        target_layer_idx: 22,
        base_intermediate_size: 3584,
        widened_slots: None,
        slot_offset: None,
        pkg_template_count: 4,
        x_splits: GarakPreMlpSplits {
            train_gc: mk_rows(10, 2, 7.5, 1),
            train_fc: mk_rows(10, 3, 7.7, 2),
            train_true_claims,
            train_unreal_py: mk_pkg(6, 1.0, 3),
            train_real_py: mk_pkg(6, -1.0, 4),
            test_gc: mk_rows(6, 2, 7.4, 7),
            test_fc: mk_rows(6, 3, 7.5, 8),
            test_unreal_py: mk_pkg(2, 1.0, 9),
            transfer_senators: Vec::new(),
            transfer_primes: mk_rows(6, 4, 6.9, 10),
            transfer_unreal_rs: mk_pkg(2, 1.0, 11),
            guard_real_code: mk_pkg(2, -1.0, 12),
            guard_true_claims: mk_rows(6, 5, 6.9, 13),
        },
        h_splits: GarakResidualSplits {
            train_gc: mk_res(10, 2),
            train_fc: mk_res(10, 3),
            train_true_claims: mk_res(10, 4),
            train_unreal_py: mk_res(10, 6),
        },
        x_cont_all,
        calibrated_angles: GarakCalibratedAngles::default(),
        readout: GarakReadoutSpec::CompactHeadAntiEcho {
            w_u_rows,
            final_norm_weight: vec![0.2; d],
            post_attn_norm_weight: vec![0.1; d],
            anti_no_gate_top: vec![vec![0.0; d]; 2],
            anti_no_up_top: vec![vec![0.0; d]; 2],
            anti_act_gate_top: vec![vec![0.0; d]; 2],
            anti_act_up_top: vec![vec![0.0; d]; 2],
        },
        max_bg_rss_leakage: 0.25,
        strict_guard_check: false,
    }
}

#[test]
fn cli_compile_garak_parses_flags_and_roundtrips_manifest_on_disk() {
    let tmp = TempDir::new().unwrap();
    let in_path = tmp.path().join("garak_input.json");
    let out_path = tmp.path().join("garak_cartridge.json");

    let spec = sample_cli_garak_spec(40);
    std::fs::write(&in_path, serde_json::to_string(&spec).unwrap()).unwrap();

    let cli = crate::Cli::try_parse_from([
        "portia-cli",
        "compile-garak",
        "--input-json",
        in_path.to_str().unwrap(),
        "--out-json",
        out_path.to_str().unwrap(),
        "--slot-offset",
        "48",
        "--widened-slots",
        "128",
        "--max-bg-leakage",
        "0.25",
    ])
    .unwrap();

    match cli.command {
        crate::Commands::CompileGarak {
            input_json,
            out_json,
            slot_offset,
            widened_slots,
            max_bg_leakage,
            readout_only,
            ..
        } => {
            assert!(!readout_only);
            let out_p = out_json.expect("out_json present");
            let (manifest, receipt) = compile_garak_command(
                &input_json,
                &out_p,
                &CompileGarakCliOverrides {
                    slot_offset,
                    widened_slots,
                    max_bg_leakage,
                },
            )
            .unwrap();

            assert_eq!(manifest.widened_slots, 128);
            assert_eq!(receipt.slot_lo, 3584 + 48);
            assert_eq!(receipt.slot_hi, 3584 + 56);
            assert!(receipt.guard_bg_max <= 0.25);
            assert!(receipt.elapsed_ms.is_some());

            let on_disk = std::fs::read_to_string(&out_p).unwrap();
            let parsed = WireCartridgeManifestJson::from_json_str(&on_disk).unwrap();
            assert_eq!(parsed.permanent_static["22"].len(), 4);
        }
        other => panic!("unexpected command variant: {other:?}"),
    }

    let readouts = resolve_garak_readout_command(&in_path).unwrap();
    assert_eq!(readouts.hidden_dim, 40);
    assert_eq!(readouts.v_no.len(), 40);
    assert_eq!(readouts.v_act.len(), 40);
}

#[test]
fn cli_compile_garak_subcommand_stdout_receipt_readout_and_strict_leakage_gate() {
    let tmp = TempDir::new().unwrap();
    let in_path = tmp.path().join("garak_subproc_in.json");
    let out_path = tmp.path().join("garak_subproc_out.json");

    let spec = sample_cli_garak_spec(40);
    std::fs::write(&in_path, serde_json::to_string(&spec).unwrap()).unwrap();

    let cli = crate::Cli::try_parse_from([
        "portia-cli",
        "compile-garak",
        "--input-json",
        in_path.to_str().unwrap(),
        "--out-json",
        out_path.to_str().unwrap(),
        "--slot-offset",
        "120",
        "--widened-slots",
        "128",
    ])
    .unwrap();

    let mut stdout_buf = Vec::new();
    match cli.command {
        crate::Commands::CompileGarak {
            input_json,
            out_json,
            slot_offset,
            widened_slots,
            max_bg_leakage,
            readout_only,
            ..
        } => {
            run_compile_garak_subcommand(
                &input_json,
                out_json.as_deref(),
                &CompileGarakCliOverrides {
                    slot_offset,
                    widened_slots,
                    max_bg_leakage,
                },
                readout_only,
                &mut stdout_buf,
            )
            .unwrap();
        }
        other => panic!("unexpected command variant: {other:?}"),
    }

    let receipt: GarakCompileReceipt = serde_json::from_slice(&stdout_buf).unwrap();
    assert_eq!(receipt.target_layer, 22);
    assert_eq!(receipt.slot_lo, 3584 + 120);
    assert_eq!(receipt.slot_hi, 3584 + 128);
    assert!(receipt.guard_bg_max <= 0.25);

    let manifest_raw = std::fs::read_to_string(&out_path).unwrap();
    let manifest = WireCartridgeManifestJson::from_json_str(&manifest_raw).unwrap();
    assert_eq!(manifest.permanent_static["22"].len(), 4);

    let mut readout_buf = Vec::new();
    run_compile_garak_subcommand(
        &in_path,
        None,
        &CompileGarakCliOverrides::default(),
        true,
        &mut readout_buf,
    )
    .unwrap();
    let resolved: GarakResolvedReadouts = serde_json::from_slice(&readout_buf).unwrap();
    assert_eq!(resolved.hidden_dim, 40);

    let err = run_compile_garak_subcommand(
        &in_path,
        None,
        &CompileGarakCliOverrides::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("--out-json is required"));
}

#[test]
fn cli_compile_garak_v2_engine_roundtrips_and_enforces_leakage_and_bounds() {
    let tmp = TempDir::new().unwrap();
    let in_path = tmp.path().join("garak_v2_in.json");
    let out_path = tmp.path().join("garak_v2_out.json");

    let spec = sample_cli_garak_spec(40);
    std::fs::write(&in_path, serde_json::to_string(&spec).unwrap()).unwrap();

    let cli = crate::Cli::try_parse_from([
        "portia-cli",
        "compile-garak",
        "--input-json",
        in_path.to_str().unwrap(),
        "--out-json",
        out_path.to_str().unwrap(),
        "--slot-offset",
        "48",
        "--widened-slots",
        "128",
        "--max-bg-leakage",
        "0.25",
        "--compiler-engine",
        "v2",
    ])
    .unwrap();

    let mut stdout_buf = Vec::new();
    match cli.command {
        crate::Commands::CompileGarak {
            input_json,
            out_json,
            slot_offset,
            widened_slots,
            max_bg_leakage,
            readout_only,
            compiler_engine,
        } => {
            ensure_v2_compiler_engine_flag(&compiler_engine).unwrap();
            run_compile_garak_subcommand(
                &input_json,
                out_json.as_deref(),
                &CompileGarakCliOverrides {
                    slot_offset,
                    widened_slots,
                    max_bg_leakage,
                },
                readout_only,
                &mut stdout_buf,
            )
            .unwrap();
        }
        other => panic!("unexpected command variant: {other:?}"),
    }

    assert!(ensure_v2_compiler_engine_flag("v1").is_err());

    let receipt: GarakCompileReceipt = serde_json::from_slice(&stdout_buf).unwrap();
    assert_eq!(receipt.target_layer, 22);
    assert_eq!(receipt.slot_lo, 3584 + 48);
    assert_eq!(receipt.slot_hi, 3584 + 56);
    assert_eq!(receipt.widened_slots, 128);
    assert!(
        receipt.guard_bg_max <= 0.25,
        "guard_bg_max={}",
        receipt.guard_bg_max
    );

    let manifest_raw = std::fs::read_to_string(&out_path).unwrap();
    let manifest = WireCartridgeManifestJson::from_json_str(&manifest_raw).unwrap();
    manifest.verify_schema_integrity().unwrap();
    assert_eq!(manifest.permanent_static["22"].len(), 4);

    let narrow_out = tmp.path().join("garak_v2_narrow.json");
    let narrow_err = compile_garak_v2_command(
        &in_path,
        &narrow_out,
        &CompileGarakCliOverrides {
            slot_offset: None,
            widened_slots: Some(4),
            max_bg_leakage: Some(0.25),
        },
    )
    .unwrap_err();
    assert!(
        narrow_err.to_string().contains("rejection.json"),
        "unexpected error: {narrow_err}"
    );
    assert!(narrow_out.with_extension("rejection.json").exists());
}
