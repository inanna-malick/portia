//! Activation split extraction and zero-shot `Senators` nuisance quotient (`faer` thin SVD)
//! across both role-tagged (`spec.v2`) and classic (`x_splits` + `h_splits`) Garak inputs.

use crate::readout::resolve_garak_readout_payload;
use crate::specs::{
    GarakCalibrationFamily, GarakCompileInputSpec, GarakPartition, GarakSolverError, GarakV2Spec,
    GARAK_REFERENCE_PRE_MLP_NORM, GARAK_SLOT_NAMES,
};
use crate::v2_concolic::readout_ds::{
    build_alphaedit_preserve_subspace, centered_rows_vec, extract_downstream_mlp_and_raw_contrast,
    solver_cfg_for_scale, state_mat,
};
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::lowering::RuleCaptureData;
use portia_compiler::numerics::{l2_norm, StateMatrix};
use std::collections::BTreeSet;

pub struct EvalSplitGroup {
    pub key: String,
    pub mat: StateMatrix,
    pub row_ids: Vec<String>,
    pub cold_for_slot: [bool; 4],
}

pub struct PreparedGarakV2Inputs {
    pub hidden_dim: usize,
    pub scale_factor_sx: f64,
    pub target_dirs: [Vec<f64>; 4],
    pub captures_by_slot: [RuleCaptureData; 4],
    pub eval_groups: Vec<EvalSplitGroup>,
    pub spec_downstream_mlp: Option<DownstreamSwigluBlock>,
}

pub fn prepare_garak_v2_inputs(
    spec: &GarakCompileInputSpec,
) -> Result<PreparedGarakV2Inputs, GarakSolverError> {
    if spec.base_intermediate_size == 0 {
        return Err(GarakSolverError::InvalidSplit {
            name: "base_intermediate_size".to_string(),
            reason: "must be > 0".to_string(),
        });
    }
    if let Some(v2) = &spec.v2 {
        prepare_from_role_splits(spec, v2)
    } else {
        super::splits_classic::prepare_from_classic_splits(spec)
    }
}

fn prepare_from_role_splits(
    spec: &GarakCompileInputSpec,
    v2: &GarakV2Spec,
) -> Result<PreparedGarakV2Inputs, GarakSolverError> {
    let invalid = |r: &str| GarakSolverError::InvalidSplit {
        name: "v2".into(),
        reason: r.into(),
    };
    let first = v2
        .splits
        .first()
        .ok_or_else(|| invalid("no role-tagged splits"))?;
    for part in [GarakPartition::Fit, GarakPartition::Dev] {
        for fam in GarakCalibrationFamily::REQUIRED_V2_FAMILIES {
            if !v2
                .splits
                .iter()
                .any(|s| s.partition == part && s.family == fam && !s.x.is_empty())
            {
                return Err(invalid("missing required fit/dev calibration family"));
            }
        }
    }
    let d = state_mat("v2.first", &first.x, None)?.hidden_dim();
    let mut seen_ids = BTreeSet::new();
    for s in &v2.splits {
        let exp_pos = s.family.expected_positive_slot();
        let pos_ok = match exp_pos {
            Some(slot) => s.positive_slots.len() == 1 && s.positive_slots[0] == slot,
            None => s.positive_slots.is_empty(),
        };
        let slots_partition_all = GARAK_SLOT_NAMES.iter().all(|&name| {
            s.positive_slots.iter().any(|p| p == name) ^ s.cold_slots.iter().any(|c| c == name)
        }) && s.positive_slots.len() + s.cold_slots.len() == 4;
        if s.row_ids.len() != s.x.len()
            || s.h.len() != s.x.len()
            || !pos_ok
            || !slots_partition_all
            || s.globally_benign != exp_pos.is_none()
        {
            return Err(invalid(
                "invalid row_ids, residual length, or role-split slot partition",
            ));
        }
        for id in &s.row_ids {
            if !seen_ids.insert(id.as_str()) {
                return Err(invalid("duplicate capture row_id"));
            }
        }
        state_mat(&s.name, &s.x, Some(d))?;
        state_mat(&s.name, &s.h, Some(d))?;
    }

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

    let fit_cur: Vec<Vec<f64>> = v2
        .splits
        .iter()
        .filter(|s| {
            s.partition == GarakPartition::Fit && s.family != GarakCalibrationFamily::Continuation
        })
        .flat_map(|s| s.x.clone())
        .collect();
    let cur_mat = state_mat("v2.fit.cursor", &fit_cur, Some(d))?;
    let mean_norm: f64 = (0..cur_mat.nrows())
        .map(|r| l2_norm(cur_mat.row(r)))
        .sum::<f64>()
        / (cur_mat.nrows() as f64);
    let sx = GARAK_REFERENCE_PRE_MLP_NORM / mean_norm.max(1e-4);
    let cfg = solver_cfg_for_scale(sx);

    let mut benign_centered = Vec::new();
    for s in v2
        .splits
        .iter()
        .filter(|s| s.partition == GarakPartition::Fit && s.globally_benign)
    {
        let m = state_mat(&s.name, &s.x, Some(d))?;
        benign_centered.extend(centered_rows_vec(&m));
    }

    let mut captures_vec = Vec::with_capacity(4);
    for slot_name in GARAK_SLOT_NAMES {
        let mut pos_x = Vec::new();
        let mut pos_h = Vec::new();
        let mut off_x = Vec::new();
        for s in v2
            .splits
            .iter()
            .filter(|s| s.partition == GarakPartition::Fit)
        {
            if s.positive_slots.iter().any(|p| p == slot_name) {
                pos_x.extend(s.x.iter().cloned());
                pos_h.extend(s.h.iter().cloned());
            } else if s.cold_slots.iter().any(|c| c == slot_name) {
                off_x.extend(s.x.iter().cloned());
            }
        }
        let on_m = state_mat(slot_name, &pos_x, Some(d))?;
        let res_m = state_mat(slot_name, &pos_h, Some(d))?;
        let off_m = state_mat(slot_name, &off_x, Some(d))?;
        let q_pres = build_alphaedit_preserve_subspace(d, &on_m, &off_m, &benign_centered)?;
        let mut cap = RuleCaptureData::standard_positive(on_m, res_m, off_m, cfg.clone());
        cap.q_preserve_gate = q_pres;
        captures_vec.push(cap);
    }

    let eval_groups = v2
        .splits
        .iter()
        .map(|s| {
            let mat = state_mat(&s.name, &s.x, Some(d))?;
            let cold_for_slot =
                std::array::from_fn(|idx| s.cold_slots.iter().any(|c| c == GARAK_SLOT_NAMES[idx]));
            Ok(EvalSplitGroup {
                key: format!("{}:{}:{}", s.partition, s.name, s.family),
                mat,
                row_ids: s.row_ids.clone(),
                cold_for_slot,
            })
        })
        .collect::<Result<Vec<_>, GarakSolverError>>()?;

    Ok(PreparedGarakV2Inputs {
        hidden_dim: d,
        scale_factor_sx: sx,
        target_dirs,
        captures_by_slot: captures_vec.try_into().expect("4 slots"),
        eval_groups,
        spec_downstream_mlp: ds_mlp,
    })
}
