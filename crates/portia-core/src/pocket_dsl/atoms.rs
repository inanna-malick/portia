//! Observable episode predicates (`EpistemicAtom`) and episode snapshot state (`PocketObsState`).

use super::{canonical_pocket_regime_dsl, RegimeEvaluation};
use serde::{Deserialize, Serialize};

/// Atomic epistemic predicate inspected on [`PocketObsState`] (zero oracle spoilers!).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicAtom {
    /// `obs.has_schema_probe == true`
    HasSchemaProbe,
    /// `obs.candidate_count.is_some()`
    CandidatesObserved,
    /// `obs.candidate_count == Some(n)`
    CandidateCountEq(usize),
    /// `obs.candidate_count.is_some_and(|c| c > n)` (e.g. `CandidateCountGt(1)` for multi-candidate ambiguity)
    CandidateCountGt(usize),
    /// `obs.operation.eq_ignore_ascii_case(op)`
    OperationEq(String),
    /// `obs.policy_requires_ask == true`
    PolicyRequiresAsk,
    /// `obs.is_locked_or_denied == true`
    IsLockedOrDenied,
    /// `obs.steward_grant_approved == expected`
    StewardDecisionIs(Option<bool>),
    /// `obs.applied_write_count == n`
    WritesAppliedEq(usize),
    /// `obs.applied_write_count > n`
    WritesAppliedGt(usize),
    /// `obs.has_post_write_verification == true`
    PostWriteVerified,
    /// `obs.zero_row_toctou_race == true`
    ZeroRowToctouRace,
}

impl EpistemicAtom {
    /// Evaluate this atomic observable predicate against `state`.
    #[inline]
    pub fn evaluate(&self, state: &PocketObsState) -> bool {
        match self {
            Self::HasSchemaProbe => state.has_schema_probe,
            Self::CandidatesObserved => state.candidate_count.is_some(),
            Self::CandidateCountEq(n) => state.candidate_count == Some(*n),
            Self::CandidateCountGt(n) => state.candidate_count.is_some_and(|c| c > *n),
            Self::OperationEq(op) => state.operation.eq_ignore_ascii_case(op),
            Self::PolicyRequiresAsk => state.policy_requires_ask,
            Self::IsLockedOrDenied => state.is_locked_or_denied,
            Self::StewardDecisionIs(expected) => state.steward_grant_approved == *expected,
            Self::WritesAppliedEq(n) => state.applied_write_count == *n,
            Self::WritesAppliedGt(n) => state.applied_write_count > *n,
            Self::PostWriteVerified => state.has_post_write_verification,
            Self::ZeroRowToctouRace => state.zero_row_toctou_race,
        }
    }

    /// Return `Some(b)` **if and only if** this atom's truth value `b` at `state` is provably
    /// invariant across **all reachable successor states** $s' \succeq \text{state}$ induced by
    /// [`crate::steward_coalgebra::apply_env_observation_to_pocket_state`] (where `operation` is
    /// immutable, `has_schema_probe` is irreversible once `true`, `candidate_count.is_some()` is
    /// irreversible once `applied_write_count > 0` because pre-write `RequestAdditionalRead`
    /// resets only occur when `applied_write_count == 0`, `steward_grant_approved` is latched
    /// final once `Some(_)`, and `applied_write_count` is monotonically non-decreasing).
    #[inline]
    pub fn monotonic_invariant_value(&self, state: &PocketObsState) -> Option<bool> {
        match self {
            Self::OperationEq(op) => Some(op.eq_ignore_ascii_case(&state.operation)),
            Self::HasSchemaProbe => {
                if state.has_schema_probe {
                    Some(true)
                } else {
                    None
                }
            }
            Self::CandidatesObserved => {
                if state.candidate_count.is_some() && state.applied_write_count > 0 {
                    Some(true)
                } else {
                    None
                }
            }
            Self::StewardDecisionIs(expected) => {
                if state.steward_grant_approved.is_some() {
                    Some(state.steward_grant_approved == *expected)
                } else {
                    None
                }
            }
            Self::WritesAppliedGt(n) => {
                if state.applied_write_count > *n {
                    Some(true)
                } else {
                    None
                }
            }
            Self::WritesAppliedEq(n) => {
                if state.applied_write_count > *n {
                    Some(false)
                } else {
                    None
                }
            }
            Self::CandidateCountEq(_)
            | Self::CandidateCountGt(_)
            | Self::PolicyRequiresAsk
            | Self::IsLockedOrDenied
            | Self::PostWriteVerified
            | Self::ZeroRowToctouRace => None,
        }
    }
}

/// Strongly-typed observable state of an episode at turn `turn_index` (zero oracle spoilers!).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PocketObsState {
    pub world_id: String,
    pub turn_index: usize,
    pub table_name: String,
    pub entity_name: String,
    pub operation: String,
    /// True if the table schema was disclosed in the prompt or already inspected via `PROBE`.
    pub has_schema_probe: bool,
    /// Number of candidate rows observed in a pre-write `READ` (`None` before first `READ`).
    pub candidate_count: Option<usize>,
    /// True if the observed row policy or ambiguity (`candidate_count > 1`) requires `ASK`.
    pub policy_requires_ask: bool,
    /// True if the target row is locked or steward denied the operation.
    pub is_locked_or_denied: bool,
    /// `None` before `ASK`, `Some(true)` if steward granted, `Some(false)` if steward denied.
    pub steward_grant_approved: Option<bool>,
    /// Number of successful `WRITE` mutations applied so far in the episode.
    pub applied_write_count: usize,
    /// True once a post-write verification `READ` has observed the updated state.
    pub has_post_write_verification: bool,
    /// True if a concurrent TOCTOU deletion allows a single zero-row `WRITE` attempt before `ABSTAIN`.
    pub zero_row_toctou_race: bool,
}

impl PocketObsState {
    /// Evaluate [`canonical_pocket_regime_dsl`] against `self` via
    /// [`recursion::CollapsibleExt::collapse_frames`], returning the full [`RegimeEvaluation`]
    /// (selected [`crate::Mode`] + active prime-implicant witness predicates).
    pub fn evaluate_regime(&self) -> RegimeEvaluation {
        canonical_pocket_regime_dsl().evaluate(self)
    }
}
