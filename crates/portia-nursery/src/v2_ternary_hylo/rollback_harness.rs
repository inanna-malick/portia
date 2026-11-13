//! 4-Arm `DeltaNet-1` Static Rollback Scrubber vs. `Host-RAM` Snapshot Trajectory Evaluator
//! (`v2_ternary_hylo::rollback_harness`).

use crate::v2_ternary_hylo::deltanet_eval::{DeltaNetRecurrentSnapshot, Qwen35GatedDeltaNetLayer};
use portia_compiler::numerics::{dot_f64, l2_norm};
use portia_compiler::wire::CompiledSlotBf16;
use portia_compiler::V2CompilerError;
use serde::{Deserialize, Serialize};

/// Measured fidelity metrics of a single rollback arm at a given retry count `k`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RollbackArmMetrics {
    pub output_cosine_vs_clean: f64,
    pub rel_l2_err_vs_clean: f64,
    pub flush_heads_frobenius_drift: f64,
    pub lock_heads_cosine_retention: f64,
}

fn head_subset_frob_diff(
    a: &DeltaNetRecurrentSnapshot,
    b: &DeltaNetRecurrentSnapshot,
    heads: &[usize],
) -> f64 {
    let mut sq = 0.0;
    for &h in heads {
        for (&x, &y) in a.s_heads[h].iter().zip(&b.s_heads[h]) {
            let d = x - y;
            sq += d * d;
        }
    }
    sq.sqrt()
}

fn head_subset_cosine(
    a: &DeltaNetRecurrentSnapshot,
    b: &DeltaNetRecurrentSnapshot,
    heads: &[usize],
) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for &h in heads {
        dot += dot_f64(&a.s_heads[h], &b.s_heads[h]);
        na += dot_f64(&a.s_heads[h], &a.s_heads[h]);
        nb += dot_f64(&b.s_heads[h], &b.s_heads[h]);
    }
    let denom = (na * nb).sqrt();
    if denom <= 1e-12 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Evaluate one rollback arm after `prompt_h`, `k` injected failed draft chunks (each ending with
/// `(rollback_l17_pre_mlp, rollback_l18_h_in)`), and final target `query_h` tokens.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rollback_arm_trajectory(
    layer: &Qwen35GatedDeltaNetLayer,
    prompt_h: &[Vec<f64>],
    draft_chunks: &[Vec<Vec<f64>>],
    rollback_tokens: &[(Vec<f64>, Vec<f64>)],
    query_h: &[Vec<f64>],
    v2_scrub_slot: Option<&CompiledSlotBf16>,
    naive_latch_delta: Option<&[f64]>,
    use_host_ram_snapshot: bool,
    flush_heads: &[usize],
    lock_heads: &[usize],
) -> Result<RollbackArmMetrics, V2CompilerError> {
    let mut clean_state = layer.zero_state();
    for h in prompt_h {
        let _ = layer.step_token(&mut clean_state, h)?;
    }
    let post_prompt_snapshot = clean_state.clone();
    let mut clean_out = vec![0.0; layer.hidden_dim];
    for q in query_h {
        clean_out = layer.step_token(&mut clean_state, q)?;
    }

    let mut state = post_prompt_snapshot.clone();
    for (chunk, (rb_x17, rb_h18)) in draft_chunks.iter().zip(rollback_tokens.iter()) {
        for h in chunk {
            let _ = layer.step_token(&mut state, h)?;
        }
        if use_host_ram_snapshot {
            state = post_prompt_snapshot.clone();
        } else {
            let mut h_rb = rb_h18.clone();
            if let Some(slot) = v2_scrub_slot {
                let (_, act) = slot.eval_gate_and_act(rb_x17);
                for (hi, &wd) in h_rb.iter_mut().zip(&slot.w_down) {
                    *hi += act * wd.to_f64();
                }
            } else if let Some(latch) = naive_latch_delta {
                for (hi, &dl) in h_rb.iter_mut().zip(latch.iter()) {
                    *hi += dl;
                }
            }
            let _ = layer.step_token(&mut state, &h_rb)?;
        }
    }

    let mut arm_out = vec![0.0; layer.hidden_dim];
    for q in query_h {
        arm_out = layer.step_token(&mut state, q)?;
    }

    let clean_norm = l2_norm(&clean_out).max(1e-12);
    let arm_norm = l2_norm(&arm_out).max(1e-12);
    let cos = (dot_f64(&clean_out, &arm_out) / (clean_norm * arm_norm)).clamp(-1.0, 1.0);
    let diff: Vec<f64> = arm_out
        .iter()
        .zip(&clean_out)
        .map(|(&a, &b)| a - b)
        .collect();

    Ok(RollbackArmMetrics {
        output_cosine_vs_clean: cos,
        rel_l2_err_vs_clean: l2_norm(&diff) / clean_norm,
        flush_heads_frobenius_drift: head_subset_frob_diff(&state, &clean_state, flush_heads),
        lock_heads_cosine_retention: head_subset_cosine(&state, &clean_state, lock_heads),
    })
}
