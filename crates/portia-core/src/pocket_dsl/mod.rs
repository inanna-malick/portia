//! Compositional Epistemic Regime & Trailhead Combinator eDSL (`RegimeExpr`),
//! Futamura-1 / Futamura-2 partial evaluation, and differential obligation types.
//!
//! Provides stack-safe [`crate::hylo::MappableFrame`] catamorphisms (`RegimeExprFrame<Child>` +
//! `collapse_frames`) over observable episode states (`PocketObsState`), priority-ordered rule
//! extraction (`GuardedModeRule`), activation row binding (`GuardedRuleStateBinding`), and
//! trajectory-preserving monotonic invariant specialization.

mod atoms;
mod combinators;
mod expr;
mod simplify_rules;
mod specialize;
mod trace_diff;

#[cfg(test)]
mod tests;

pub use atoms::{EpistemicAtom, PocketObsState};
pub use combinators::{
    candidate_count_eq, candidate_count_gt, candidates_observed,
    canonical_compile_fixture_row_states, canonical_pocket_regime_dsl, emit_mode, has_schema_probe,
    is_locked_or_denied, operation_eq, policy_requires_ask, post_write_verified,
    steward_decision_is, writes_applied_eq, writes_applied_gt, zero_row_toctou_race,
};
pub use expr::{RegimeEvaluation, RegimeExpr, RegimeExprFrame};
pub use simplify_rules::{partition_states_by_guard, GuardedModeRule, GuardedRuleStateBinding};
pub use trace_diff::{DivergenceClassification, ExtractedDslObligation};
