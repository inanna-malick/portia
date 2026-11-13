//! Unified error taxonomy for `portia-compiler` (zero `nalgebra` leakage, strict `#[must_use]`).

use thiserror::Error;

/// Errors emitted across subspace geometry, auto-discovery, conic solving, register allocation,
/// and runtime wire verification in `portia-compiler`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum V2CompilerError {
    #[error("empty state matrix ({context}): nrows={nrows}, ncols={ncols}")]
    EmptyMatrix {
        context: &'static str,
        nrows: usize,
        ncols: usize,
    },

    #[error("dimension mismatch in {context}: expected {expected}, got {actual}")]
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("non-finite value detected in {context}")]
    NonFiniteValue { context: &'static str },

    #[error("degenerate target vector in `{rule}`: residual norm after nullspace projection is {norm:.3e}")]
    DegenerateProjectedTarget { rule: String, norm: f64 },

    #[error("SVD decomposition failed in `{context}`")]
    SvdConvergence { context: &'static str },

    #[error(
        "infeasible diode `{rule}` (status={status}): min_pos_gate_bf16={min_pos_gate_bf16:.4} (req >= {req_min_pos:.4}), max_neg_gate_bf16={max_neg_gate_bf16:.4} (req <= {req_max_neg:.4})"
    )]
    InfeasibleDiode {
        rule: String,
        status: String,
        min_pos_gate_bf16: f64,
        req_min_pos: f64,
        max_neg_gate_bf16: f64,
        req_max_neg: f64,
    },

    #[error(
        "infeasible ternary `w_up` on `{rule}` (status={status}): min_signed_active={min_signed_active:.4}, max_neutral_abs={max_neutral_abs:.4} (limit <= {neutral_tol:.4})"
    )]
    InfeasibleTernaryUp {
        rule: String,
        status: String,
        min_signed_active: f64,
        max_neutral_abs: f64,
        neutral_tol: f64,
    },

    #[error("non-causal layer hop in `{rule}`: upstream MLP layer {upstream} must be < downstream layer {downstream}")]
    NonCausalLayerHop {
        rule: String,
        upstream: usize,
        downstream: usize,
    },

    #[error("slot register capacity exhausted on bank `{bank}` layer {layer}: requested {requested} slots, limit {capacity}")]
    SlotCapacityExhausted {
        bank: String,
        layer: usize,
        requested: usize,
        capacity: usize,
    },

    #[error("invalid configuration in `{context}`: {reason}")]
    InvalidConfig {
        context: &'static str,
        reason: String,
    },

    #[error("wire serialization / bisimulation error: {0}")]
    WireIntegrity(String),
}
