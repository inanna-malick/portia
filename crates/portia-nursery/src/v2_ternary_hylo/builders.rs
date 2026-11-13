//! Declarative `IntentExpr` Builders & Evaluation Helpers for `SignedTernaryFork` (`2` slots)
//! and `DeltaNetRollbackScrub` (`1` slot on `in_proj_a`) (`v2_ternary_hylo::builders`).

use crate::v2_ternary_hylo::ast_seam::HyloTernaryBranch;
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::RuleCaptureData;
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{dot_f64, l2_norm, rms_norm_slice, silu_f64, StateMatrix};
use portia_compiler::solvers::clarabel_ternary::CursorPolarity;
use portia_compiler::solvers::V2SolverConfig;
use portia_compiler::wire::{CompiledSlotBf16, WireSlotEntryJson};
use portia_compiler::V2CompilerError;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, MountPolicy};
use serde::{Deserialize, Serialize};

/// Specification for a 2-slot `SignedTernaryFork` (`FORK` `+1` / `ASK` `0` / `INLINE` `-1`) intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedTernaryForkSpec {
    pub rule_name: String,
    pub layer_idx: usize,
    pub branch_axis_dir: Vec<f64>,
    pub angle: ChordalAngleRad,
    pub mount_policy: MountPolicy,
    pub guard: RegimeExpr,
    pub grammar: GrammarCartridgeSpec,
}

impl SignedTernaryForkSpec {
    /// Canonical Layer-22 `OrientTrunk` constructor for Rust-AST `todo!()` ternary routing.
    #[must_use]
    pub fn orient_trunk_default(
        rule_name: impl Into<String>,
        layer_idx: usize,
        branch_axis_dir: Vec<f64>,
        angle_rad: f64,
    ) -> Self {
        Self {
            rule_name: rule_name.into(),
            layer_idx,
            branch_axis_dir,
            angle: ChordalAngleRad::new(angle_rad),
            mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
            guard: RegimeExpr::ConstBool(true),
            grammar: GrammarCartridgeSpec::Unconstrained,
        }
    }
}

/// Build a 2-slot `HardwareIntentKind::SignedTernaryFork` [`IntentExpr`] leaf.
#[must_use]
pub fn build_signed_ternary_fork_intent(spec: &SignedTernaryForkSpec) -> IntentExpr {
    IntentExpr::rule(IntentRule {
        name: spec.rule_name.clone(),
        guard: spec.guard.clone(),
        mount_policy: spec.mount_policy,
        grammar: spec.grammar.clone(),
        kind: HardwareIntentKind::SignedTernaryFork {
            layer_idx: spec.layer_idx,
            branch_axis_dir: spec.branch_axis_dir.clone(),
            angle: spec.angle,
        },
    })
}

/// Build validated [`RuleCaptureData`] for a 3-way `FORK / ASK / INLINE` ternary rule from
/// separate per-branch pre-MLP matrices and corresponding un-normalized positive residuals.
#[allow(clippy::too_many_arguments)]
pub fn build_ternary_fork_capture_data(
    fork_pre_mlp: &StateMatrix,
    ask_pre_mlp: &StateMatrix,
    inline_pre_mlp: &StateMatrix,
    pos_residuals: &StateMatrix,
    off_cursor_pre_mlp: &StateMatrix,
    q_preserve_gate: Option<OrthonormalSubspace>,
    solver_config: V2SolverConfig,
) -> Result<RuleCaptureData, V2CompilerError> {
    let on_cursor_pre_mlp = StateMatrix::vstack(
        "ternary_on_cursor",
        &[fork_pre_mlp, ask_pre_mlp, inline_pre_mlp],
    )?;
    if pos_residuals.nrows() != on_cursor_pre_mlp.nrows() {
        return Err(V2CompilerError::DimensionMismatch {
            context: "build_ternary_fork_capture_data(pos_residuals)",
            expected: on_cursor_pre_mlp.nrows(),
            actual: pos_residuals.nrows(),
        });
    }
    let mut polarities = Vec::with_capacity(on_cursor_pre_mlp.nrows());
    polarities.extend(std::iter::repeat_n(
        CursorPolarity::PlusOne,
        fork_pre_mlp.nrows(),
    ));
    polarities.extend(std::iter::repeat_n(
        CursorPolarity::ZeroNeutral,
        ask_pre_mlp.nrows(),
    ));
    polarities.extend(std::iter::repeat_n(
        CursorPolarity::MinusOne,
        inline_pre_mlp.nrows(),
    ));
    Ok(RuleCaptureData {
        on_cursor_pre_mlp,
        polarities,
        pos_residuals: pos_residuals.clone(),
        off_cursor_pre_mlp: off_cursor_pre_mlp.clone(),
        q_preserve_gate,
        solver_config,
    })
}

/// Synthesize convex-combination states $\mathbf{x}(\lambda) = (1 - \lambda)\mathbf{x}_+ + \lambda\mathbf{x}_-$
/// ($\lambda \in [0, 1]$, with $\lambda = 0.5$ yielding exact midpoints where linear `w_gate <= -8.0`
/// clamping on both endpoints forces $w_{\text{gate}} \cdot \mathbf{x}_0 \le -8.0$!).
pub fn synthesize_convex_midpoint_states(
    fork_states: &StateMatrix,
    inline_states: &StateMatrix,
    lambdas: &[f64],
) -> Result<StateMatrix, V2CompilerError> {
    let n_pairs = fork_states.nrows().min(inline_states.nrows());
    let d = fork_states.hidden_dim();
    if inline_states.hidden_dim() != d {
        return Err(V2CompilerError::DimensionMismatch {
            context: "synthesize_convex_midpoint_states",
            expected: d,
            actual: inline_states.hidden_dim(),
        });
    }
    if n_pairs == 0 || lambdas.is_empty() {
        return Err(V2CompilerError::EmptyMatrix {
            context: "synthesize_convex_midpoint_states",
            nrows: 0,
            ncols: d,
        });
    }
    let mut rows = Vec::with_capacity(n_pairs * lambdas.len());
    for i in 0..n_pairs {
        let xp = fork_states.row(i);
        let xm = inline_states.row(i);
        for &lam in lambdas {
            let w_p = 1.0 - lam;
            rows.push(
                xp.iter()
                    .zip(xm.iter())
                    .map(|(&a, &b)| w_p * a + lam * b)
                    .collect(),
            );
        }
    }
    StateMatrix::from_rows("convex_midpoint_ask", &rows)
}

/// Specification for a 1-slot `DeltaNetRollbackScrub` intent targeting `in_proj_a`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaNetRollbackScrubSpec {
    pub rule_name: String,
    pub upstream_mlp_layer: usize,
    pub target_deltanet_layer: usize,
    pub flush_heads: Vec<usize>,
    pub lock_heads: Vec<usize>,
    pub a_drive: f64,
    pub mount_policy: MountPolicy,
    pub guard: RegimeExpr,
}

/// Partition `Qwen3_5GatedDeltaNet` heads by baseline zero-input retention
/// $\alpha_{0,h} = \exp(-\exp(A_{\text{log},h})\operatorname{softplus}(\text{dt\_bias}_h))$:
/// - `flush_heads`: scratchpad-tracking heads with $\alpha_{0,h} < \text{anchor\_alpha\_floor}$,
/// - `lock_heads`: long-horizon prompt-anchor heads with $\alpha_{0,h} \ge \text{anchor\_alpha\_floor}$.
#[must_use]
pub fn partition_deltanet_heads_by_retention(
    a_log: &[f64],
    dt_bias: &[f64],
    anchor_alpha_floor: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut flush_heads = Vec::new();
    let mut lock_heads = Vec::new();
    let n = a_log.len().min(dt_bias.len());
    for h in 0..n {
        let sp = if dt_bias[h] > 20.0 {
            dt_bias[h]
        } else {
            (1.0 + dt_bias[h].exp()).ln()
        };
        let alpha0 = (-a_log[h].exp() * sp).exp();
        if alpha0 < anchor_alpha_floor {
            flush_heads.push(h);
        } else {
            lock_heads.push(h);
        }
    }
    (flush_heads, lock_heads)
}

/// Build a 1-slot `HardwareIntentKind::DeltaNetRollbackScrub` [`IntentExpr`] leaf.
#[must_use]
pub fn build_deltanet_rollback_scrub_intent(spec: &DeltaNetRollbackScrubSpec) -> IntentExpr {
    IntentExpr::rule(IntentRule {
        name: spec.rule_name.clone(),
        guard: spec.guard.clone(),
        mount_policy: spec.mount_policy,
        grammar: GrammarCartridgeSpec::Unconstrained,
        kind: HardwareIntentKind::DeltaNetRollbackScrub {
            upstream_mlp_layer: spec.upstream_mlp_layer,
            target_deltanet_layer: spec.target_deltanet_layer,
            flush_heads: spec.flush_heads.clone(),
            lock_heads: spec.lock_heads.clone(),
            a_drive: spec.a_drive,
        },
    })
}

/// Compose `DeltaNetRollbackScrub >> SignedTernaryFork` into a single multi-layer [`IntentExpr`].
#[must_use]
pub fn compose_hylo_scrub_then_ternary_fork(
    scrub_spec: &DeltaNetRollbackScrubSpec,
    fork_spec: &SignedTernaryForkSpec,
) -> IntentExpr {
    build_deltanet_rollback_scrub_intent(scrub_spec) >> build_signed_ternary_fork_intent(fork_spec)
}

/// Forward evaluation telemetry of a compiled 2-slot `SignedTernaryFork` on a pre-MLP vector `x`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TernarySlotEval {
    pub gate_pre_silu: f64,
    pub radial_act: f64,
    pub tangential_act: f64,
    pub tangential_up_linear: f64,
    pub predicted_branch: Option<HyloTernaryBranch>,
    pub delta_h: Vec<f64>,
}

/// Evaluate a 2-slot `(radial_slot, tangential_slot)` pair on pre-MLP state `x`:
/// - Returns `predicted_branch == None` when off-cursor (`gate_pre_silu <= active_gate_floor`);
/// - Otherwise classifies `tangential_up_linear` (`w_up_tan · x`) into `Fork` (`> +neutral_up_band`),
///   `Ask` (`|u_tan| <= neutral_up_band`), or `Inline` (`< -neutral_up_band`).
#[must_use]
pub fn evaluate_compiled_ternary_slots(
    radial_slot: &CompiledSlotBf16,
    tangential_slot: &CompiledSlotBf16,
    x: &[f64],
    active_gate_floor: f64,
    neutral_up_band: f64,
) -> TernarySlotEval {
    let (g_rad, act_rad) = radial_slot.eval_gate_and_act(x);
    let (_g_tan, act_tan) = tangential_slot.eval_gate_and_act(x);
    let wu_tan: Vec<f64> = tangential_slot.w_up.iter().map(|v| v.to_f64()).collect();
    let u_tan = dot_f64(&wu_tan, x);

    let mut delta_h = vec![0.0; x.len()];
    for ((dh, &wd_r), &wd_t) in delta_h
        .iter_mut()
        .zip(&radial_slot.w_down)
        .zip(&tangential_slot.w_down)
    {
        *dh = act_rad * wd_r.to_f64() + act_tan * wd_t.to_f64();
    }

    let predicted_branch = if g_rad <= active_gate_floor {
        None
    } else if u_tan > neutral_up_band {
        Some(HyloTernaryBranch::Fork)
    } else if u_tan < -neutral_up_band {
        Some(HyloTernaryBranch::Inline)
    } else {
        Some(HyloTernaryBranch::Ask)
    };

    TernarySlotEval {
        gate_pre_silu: g_rad,
        radial_act: act_rad,
        tangential_act: act_tan,
        tangential_up_linear: u_tan,
        predicted_branch,
        delta_h,
    }
}

/// Evaluate the summed residual perturbation $\Delta\mathbf{h}_{\text{bank}}(\mathbf{x})$ across all
/// co-mounted [`WireSlotEntryJson`] blocks on a layer, plus downstream Layer-23 MLP output L2 drift.
#[must_use]
pub fn evaluate_bank_collateral_and_l23_drift(
    entries: &[WireSlotEntryJson],
    x_l22_pre_mlp: &[f64],
    h_l22_post_base: &[f64],
    mlp23: Option<&DownstreamSwigluBlock>,
) -> (f64, f64) {
    let d = x_l22_pre_mlp.len();
    let mut delta_l22 = vec![0.0_f64; d];
    for entry in entries {
        let n_slots = entry.w_gate.len();
        for s in 0..n_slots {
            let g: f64 = entry.w_gate[s]
                .iter()
                .zip(x_l22_pre_mlp)
                .map(|(&w, &x)| (w as f64) * x)
                .sum();
            let u: f64 = entry.w_up[s]
                .iter()
                .zip(x_l22_pre_mlp)
                .map(|(&w, &x)| (w as f64) * x)
                .sum();
            let act = silu_f64(g) * u;
            for (row_i, dh) in delta_l22.iter_mut().enumerate() {
                *dh += act * (entry.w_down[row_i][s] as f64);
            }
        }
    }
    let l2_l22 = l2_norm(&delta_l22);
    let Some(ds) = mlp23 else {
        return (l2_l22, l2_l22);
    };
    let h_pert: Vec<f64> = h_l22_post_base
        .iter()
        .zip(&delta_l22)
        .map(|(&h, &dh)| h + dh)
        .collect();
    let x0 = rms_norm_slice(h_l22_post_base, &ds.post_attn_norm_scale, 1e-6);
    let x1 = rms_norm_slice(&h_pert, &ds.post_attn_norm_scale, 1e-6);
    let mut diff_out = vec![0.0_f64; d];
    for m in 0..ds.intermediate_size {
        let a0 = silu_f64(dot_f64(ds.w_gate.row(m), &x0)) * dot_f64(ds.w_up.row(m), &x0);
        let a1 = silu_f64(dot_f64(ds.w_gate.row(m), &x1)) * dot_f64(ds.w_up.row(m), &x1);
        let da = a1 - a0;
        if da.abs() > 1e-15 {
            for (j, &wd) in ds.w_down_cols.row(m).iter().enumerate() {
                diff_out[j] += da * wd;
            }
        }
    }
    let total_post_l23: Vec<f64> = delta_l22
        .iter()
        .zip(&diff_out)
        .map(|(&a, &b)| a + b)
        .collect();
    (l2_l22, l2_norm(&total_post_l23))
}
