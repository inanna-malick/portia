//! Classic (`x_splits` + `h_splits`) activation extraction and zero-shot `Senators` nuisance
//! quotient (`faer` thin SVD) for `portia-garak` `v2`.

use super::splits::{EvalSplitGroup, PreparedGarakV2Inputs};
use crate::readout::resolve_garak_readout_payload;
use crate::specs::{GarakCompileInputSpec, GarakSolverError, GARAK_REFERENCE_PRE_MLP_NORM};
use crate::v2_concolic::readout_ds::{
    build_alphaedit_preserve_subspace, centered_rows_vec, extract_downstream_mlp_and_raw_contrast,
    solver_cfg_for_scale, state_mat,
};
use portia_compiler::lowering::RuleCaptureData;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{l2_norm, StateMatrix};

pub(super) fn prepare_from_classic_splits(
    spec: &GarakCompileInputSpec,
) -> Result<PreparedGarakV2Inputs, GarakSolverError> {
    let x_gc = state_mat("x_splits.train_gc", &spec.x_splits.train_gc, None)?;
    let d = x_gc.hidden_dim();
    let x_fc = state_mat("x_splits.train_fc", &spec.x_splits.train_fc, Some(d))?;
    let x_tru = state_mat(
        "x_splits.train_true_claims",
        &spec.x_splits.train_true_claims,
        Some(d),
    )?;
    let x_upy = state_mat(
        "x_splits.train_unreal_py",
        &spec.x_splits.train_unreal_py,
        Some(d),
    )?;
    let x_rpy = state_mat(
        "x_splits.train_real_py",
        &spec.x_splits.train_real_py,
        Some(d),
    )?;
    let x_cont = state_mat("x_cont_all", &spec.x_cont_all, Some(d))?;
    if x_tru.nrows() < 4 {
        return Err(GarakSolverError::InvalidSplit {
            name: "x_splits.train_true_claims".into(),
            reason: "requires >= 4 rows".into(),
        });
    }
    let h_gc = state_mat("h_splits.train_gc", &spec.h_splits.train_gc, Some(d))?;
    let h_fc = state_mat("h_splits.train_fc", &spec.h_splits.train_fc, Some(d))?;
    let h_tru = state_mat(
        "h_splits.train_true_claims",
        &spec.h_splits.train_true_claims,
        Some(d),
    )?;
    let h_upy = state_mat(
        "h_splits.train_unreal_py",
        &spec.h_splits.train_unreal_py,
        Some(d),
    )?;

    let resolved = resolve_garak_readout_payload(&spec.readout)?;
    if resolved.hidden_dim != d {
        return Err(GarakSolverError::InvalidVectorLength {
            name: "readout".into(),
            expected_dim: d,
            actual: resolved.hidden_dim,
        });
    }
    let (ds_mlp, raw_pair) =
        extract_downstream_mlp_and_raw_contrast(spec.target_layer_idx, &spec.readout, d)?;
    let (v_no, v_act) = raw_pair.unwrap_or((resolved.v_no, resolved.v_act));
    let target_dirs = [
        resolved.v_trace.unwrap_or_else(|| v_no.clone()),
        v_act.clone(),
        v_no,
        v_act,
    ];

    let cur_all = StateMatrix::vstack("cur_all", &[&x_gc, &x_fc, &x_upy, &x_rpy, &x_tru]).unwrap();
    let mean_norm: f64 = (0..cur_all.nrows())
        .map(|r| l2_norm(cur_all.row(r)))
        .sum::<f64>()
        / (cur_all.nrows() as f64);
    let sx = GARAK_REFERENCE_PRE_MLP_NORM / mean_norm.max(1e-4);
    let cfg = solver_cfg_for_scale(sx);

    let n_tru1 = if x_tru.nrows() >= 20 {
        10
    } else {
        x_tru.nrows() / 2
    };
    let tru1_rows: Vec<Vec<f64>> = (0..n_tru1).map(|r| x_tru.row(r).to_vec()).collect();
    let tru_why_rows: Vec<Vec<f64>> = (n_tru1..x_tru.nrows())
        .map(|r| x_tru.row(r).to_vec())
        .collect();
    let m_tru1 = state_mat("tru1", &tru1_rows, Some(d))?;
    let m_why = state_mat("tru_why", &tru_why_rows, Some(d))?;

    let mut nuis_rows = vec![cur_all.mean_row()];
    nuis_rows.extend(centered_rows_vec(&x_rpy));
    nuis_rows.extend(centered_rows_vec(&m_tru1));
    nuis_rows.extend(centered_rows_vec(&x_cont));
    let nuis_mat = state_mat("wk_nuis", &nuis_rows, Some(d))?;
    let q_nuis =
        OrthonormalSubspace::from_row_space_svd("q_nuis", &nuis_mat, 0.05, (d / 3).clamp(1, 24))
            .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;
    let raw_fc_shift: Vec<f64> = x_fc
        .mean_row()
        .into_iter()
        .zip(m_why.mean_row())
        .map(|(a, b)| a - b)
        .collect();
    let clean_shift = q_nuis.project_out(&raw_fc_shift);
    let wk_pos_rows: Vec<Vec<f64>> = tru1_rows
        .iter()
        .map(|r| r.iter().zip(&clean_shift).map(|(&x, &s)| x + s).collect())
        .collect();
    let x_wk = state_mat("wk_synth", &wk_pos_rows, Some(d))?;
    let n_h_tru1 = h_tru.nrows().min(n_tru1).max(1);
    let h_wk = state_mat(
        "h_wk",
        &(0..n_h_tru1)
            .map(|r| h_tru.row(r).to_vec())
            .collect::<Vec<_>>(),
        Some(d),
    )?;

    let mut benign_centered = centered_rows_vec(&x_tru);
    benign_centered.extend(centered_rows_vec(&x_rpy));
    benign_centered.extend(centered_rows_vec(&x_cont));

    let slot_specs: [(&StateMatrix, &StateMatrix, Vec<&StateMatrix>); 4] = [
        (&x_gc, &h_gc, vec![&x_tru, &x_rpy, &x_fc, &x_upy, &x_cont]),
        (&x_fc, &h_fc, vec![&x_tru, &x_rpy, &x_gc, &x_upy, &x_cont]),
        (
            &x_wk,
            &h_wk,
            vec![&x_tru, &x_rpy, &x_fc, &x_gc, &x_upy, &x_cont],
        ),
        (&x_upy, &h_upy, vec![&x_rpy, &x_tru, &x_gc, &x_fc, &x_cont]),
    ];
    let mut captures_vec = Vec::with_capacity(4);
    for (on_m, res_m, off_blocks) in slot_specs {
        let off_m = StateMatrix::vstack("classic_off", &off_blocks).unwrap();
        let q_pres = build_alphaedit_preserve_subspace(d, on_m, &off_m, &benign_centered)?;
        let mut cap =
            RuleCaptureData::standard_positive(on_m.clone(), res_m.clone(), off_m, cfg.clone());
        cap.q_preserve_gate = q_pres;
        captures_vec.push(cap);
    }

    let opt_splits: [(&str, &[Vec<f64>], bool); 8] = [
        ("test_gc", &spec.x_splits.test_gc, false),
        ("test_fc", &spec.x_splits.test_fc, false),
        ("test_unreal_py", &spec.x_splits.test_unreal_py, false),
        ("transfer_senators", &spec.x_splits.transfer_senators, false),
        ("transfer_primes", &spec.x_splits.transfer_primes, true),
        (
            "transfer_unreal_rs",
            &spec.x_splits.transfer_unreal_rs,
            false,
        ),
        ("guard_real_code", &spec.x_splits.guard_real_code, true),
        ("guard_true_claims", &spec.x_splits.guard_true_claims, true),
    ];
    let mut eval_groups = Vec::new();
    for (gname, rows, is_guard) in opt_splits {
        if !rows.is_empty() {
            let mat = state_mat(gname, rows, Some(d))?;
            let ids = (0..mat.nrows()).map(|i| format!("{gname}:{i}")).collect();
            eval_groups.push(EvalSplitGroup {
                key: gname.to_string(),
                mat,
                row_ids: ids,
                cold_for_slot: [is_guard; 4],
            });
        }
    }
    let cont_ids = (0..x_cont.nrows())
        .map(|i| format!("cont_bg_t>=1:{i}"))
        .collect();
    eval_groups.push(EvalSplitGroup {
        key: "cont_bg_t>=1".to_string(),
        mat: x_cont,
        row_ids: cont_ids,
        cold_for_slot: [true; 4],
    });

    Ok(PreparedGarakV2Inputs {
        hidden_dim: d,
        scale_factor_sx: sx,
        target_dirs,
        captures_by_slot: captures_vec.try_into().expect("4 slots"),
        eval_groups,
        spec_downstream_mlp: ds_mlp,
    })
}
