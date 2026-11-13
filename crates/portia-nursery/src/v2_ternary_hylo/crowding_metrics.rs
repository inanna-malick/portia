//! Real `.safetensors` Layer-23 MLP + Vocabulary Readout Loader & Co-Mounted Bank Collateral
//! Evaluator (`v2_ternary_hylo::crowding_metrics`).

use crate::v2_ternary_hylo::builders::evaluate_bank_collateral_and_l23_drift;
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::numerics::safetensors_io::{
    load_qwen35_rmsnorm_scale_1p, load_safetensor_matrix_2d,
};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{
    dot_f64, rms_norm_slice, silu_f64, slice_to_bf16_vec, StateMatrix,
};
use portia_compiler::wire::{CompiledSlotBf16, WireSlotEntryJson};
use portia_compiler::V2CompilerError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Real `Qwen3.5-0.8B-Base` `.safetensors` readout & Layer-23 geometry bundle.
#[derive(Debug, Clone)]
pub struct Qwen35ReadoutAndL23Bundle {
    pub hidden_dim: usize,
    pub embed_tokens: StateMatrix,
    pub final_norm_scale: Vec<f64>,
    pub wu_probe: StateMatrix,
    pub q_wu64: OrthonormalSubspace,
    pub mlp23: DownstreamSwigluBlock,
}

/// Collateral drift summary of a co-mounted bank across unrelated background prompts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BankCollateralSummary {
    pub guard_bg_max: f64,
    pub mean_l22_drift: f64,
    pub max_l22_drift: f64,
    pub max_l23_drift: f64,
    pub max_logit_drift: f64,
}

/// Load `embed_tokens`, `norm.weight`, a 515-row vocabulary probe (anchoring `" FOR"`, `" AS"`, `" INLINE"`),
/// `q_wu64`, and the full 3,584-neuron Layer-23 `DownstreamSwigluBlock` from `.safetensors`.
pub fn load_qwen35_readout_and_l23(
    snapshot_dir: &Path,
) -> Result<Qwen35ReadoutAndL23Bundle, V2CompilerError> {
    let pfx = "model.language_model.";
    let emb = load_safetensor_matrix_2d(snapshot_dir, &format!("{pfx}embed_tokens.weight"))?;
    let final_norm = load_qwen35_rmsnorm_scale_1p(snapshot_dir, &format!("{pfx}norm.weight"))?;
    let d = emb.hidden_dim();

    let mut wu_rows = vec![
        emb.row(4463).to_vec(),
        emb.row(5574).to_vec(),
        emb.row(63373).to_vec(),
    ];
    let step = emb.nrows() / 512;
    for i in 0..512 {
        wu_rows.push(emb.row(i * step).to_vec());
    }
    let wu_probe = StateMatrix::from_rows("wu_probe", &wu_rows)?;
    let q_wu64 = OrthonormalSubspace::from_row_space_svd("wu64", &wu_probe, 1e-4, 64)?;

    let wd23_raw = load_safetensor_matrix_2d(
        snapshot_dir,
        &format!("{pfx}layers.23.mlp.down_proj.weight"),
    )?;
    let mut wd23_cols = vec![0.0; 3584 * d];
    for r in 0..d {
        let row = wd23_raw.row(r);
        for m in 0..3584 {
            wd23_cols[m * d + r] = row[m];
        }
    }
    let mlp23 = DownstreamSwigluBlock::try_new(
        23,
        load_qwen35_rmsnorm_scale_1p(
            snapshot_dir,
            &format!("{pfx}layers.23.post_attention_layernorm.weight"),
        )?,
        load_safetensor_matrix_2d(
            snapshot_dir,
            &format!("{pfx}layers.23.mlp.gate_proj.weight"),
        )?,
        load_safetensor_matrix_2d(snapshot_dir, &format!("{pfx}layers.23.mlp.up_proj.weight"))?,
        StateMatrix::try_new("wd23_cols", 3584, d, wd23_cols)?,
    )?;

    Ok(Qwen35ReadoutAndL23Bundle {
        hidden_dim: d,
        embed_tokens: emb,
        final_norm_scale: final_norm,
        wu_probe,
        q_wu64,
        mlp23,
    })
}

/// Convert a 2-slot [`WireSlotEntryJson`] into `(radial, tangential)` [`CompiledSlotBf16`] structs.
#[must_use]
pub fn wire_pair_to_bf16(
    entry: &WireSlotEntryJson,
    d: usize,
) -> (CompiledSlotBf16, CompiledSlotBf16) {
    let slot = |idx: usize| CompiledSlotBf16 {
        slot_offset: idx,
        w_gate: slice_to_bf16_vec(
            &entry.w_gate[idx]
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
        ),
        w_up: slice_to_bf16_vec(
            &entry.w_up[idx]
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
        ),
        w_down: slice_to_bf16_vec(
            &(0..d)
                .map(|r| entry.w_down[r][idx] as f64)
                .collect::<Vec<_>>(),
        ),
    };
    (slot(0), slot(1))
}

/// Measure `guard_bg_max`, Layer-22 L2 drift, Layer-23 output L2 drift, and `max_logit_drift`
/// across a batch of unrelated background prompts.
#[must_use]
pub fn evaluate_bank_background_collateral(
    entries: &[WireSlotEntryJson],
    bg_x22: &StateMatrix,
    bg_post22: &StateMatrix,
    bundle: &Qwen35ReadoutAndL23Bundle,
) -> BankCollateralSummary {
    let d = bg_x22.hidden_dim();
    let (mut guard_max, mut sum_l22, mut max_l22, mut max_l23, mut max_logit) =
        (f64::NEG_INFINITY, 0.0, 0.0_f64, 0.0_f64, 0.0_f64);
    for r in 0..bg_x22.nrows() {
        let x22 = bg_x22.row(r);
        let h_post22 = bg_post22.row(r);
        let mut delta22 = vec![0.0_f64; d];
        for e in entries {
            for s in 0..e.w_gate.len() {
                let g: f64 = e.w_gate[s]
                    .iter()
                    .zip(x22)
                    .map(|(&w, &x)| (w as f64) * x)
                    .sum();
                let u: f64 = e.w_up[s]
                    .iter()
                    .zip(x22)
                    .map(|(&w, &x)| (w as f64) * x)
                    .sum();
                guard_max = guard_max.max(g);
                let act = silu_f64(g) * u;
                for (i, dh) in delta22.iter_mut().enumerate() {
                    *dh += act * (e.w_down[i][s] as f64);
                }
            }
        }
        let (l22, l23) =
            evaluate_bank_collateral_and_l23_drift(entries, x22, h_post22, Some(&bundle.mlp23));
        sum_l22 += l22;
        max_l22 = max_l22.max(l22);
        max_l23 = max_l23.max(l23);
        let h_pert: Vec<f64> = h_post22
            .iter()
            .zip(&delta22)
            .map(|(&h, &dh)| h + dh)
            .collect();
        let fn0 = rms_norm_slice(h_post22, &bundle.final_norm_scale, 1e-6);
        let fn1 = rms_norm_slice(&h_pert, &bundle.final_norm_scale, 1e-6);
        let dfn: Vec<f64> = fn1.iter().zip(&fn0).map(|(&a, &b)| a - b).collect();
        for vr in 0..bundle.wu_probe.nrows() {
            max_logit = max_logit.max(dot_f64(bundle.wu_probe.row(vr), &dfn).abs());
        }
    }
    BankCollateralSummary {
        guard_bg_max: guard_max,
        mean_l22_drift: sum_l22 / (bg_x22.nrows() as f64),
        max_l22_drift: max_l22,
        max_l23_drift: max_l23,
        max_logit_drift: max_logit,
    }
}
