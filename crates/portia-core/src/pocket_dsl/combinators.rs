//! Ergonomic combinator constructors, [`canonical_pocket_regime_dsl`], and the 14-row
//! [`canonical_compile_fixture_row_states`] state bank.

use super::{EpistemicAtom, PocketObsState, RegimeExpr};
use crate::Mode;

/// Combinator: atomic predicate `obs.has_schema_probe`.
#[inline]
pub fn has_schema_probe() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::HasSchemaProbe)
}

/// Combinator: atomic predicate `obs.candidate_count.is_some()`.
#[inline]
pub fn candidates_observed() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::CandidatesObserved)
}

/// Combinator: atomic predicate `obs.candidate_count == Some(n)`.
#[inline]
pub fn candidate_count_eq(n: usize) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::CandidateCountEq(n))
}

/// Combinator: atomic predicate `obs.candidate_count.is_some_and(|c| c > n)`.
#[inline]
pub fn candidate_count_gt(n: usize) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::CandidateCountGt(n))
}

/// Combinator: atomic predicate `obs.operation.eq_ignore_ascii_case(op)`.
#[inline]
pub fn operation_eq(op: impl Into<String>) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::OperationEq(op.into()))
}

/// Combinator: atomic predicate `obs.policy_requires_ask`.
#[inline]
pub fn policy_requires_ask() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::PolicyRequiresAsk)
}

/// Combinator: atomic predicate `obs.is_locked_or_denied`.
#[inline]
pub fn is_locked_or_denied() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::IsLockedOrDenied)
}

/// Combinator: atomic predicate `obs.steward_grant_approved == decision`.
#[inline]
pub fn steward_decision_is(decision: Option<bool>) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::StewardDecisionIs(decision))
}

/// Combinator: atomic predicate `obs.applied_write_count == n`.
#[inline]
pub fn writes_applied_eq(n: usize) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::WritesAppliedEq(n))
}

/// Combinator: atomic predicate `obs.applied_write_count > n`.
#[inline]
pub fn writes_applied_gt(n: usize) -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::WritesAppliedGt(n))
}

/// Combinator: atomic predicate `obs.has_post_write_verification`.
#[inline]
pub fn post_write_verified() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::PostWriteVerified)
}

/// Combinator: atomic predicate `obs.zero_row_toctou_race`.
#[inline]
pub fn zero_row_toctou_race() -> RegimeExpr {
    RegimeExpr::Atom(EpistemicAtom::ZeroRowToctouRace)
}

/// Combinator: terminal epistemic mode emitter `RegimeExpr::EmitMode(mode)`.
#[inline]
pub const fn emit_mode(mode: Mode) -> RegimeExpr {
    RegimeExpr::EmitMode(mode)
}

/// Canonical compositional Pocket Regime & Trailhead Combinator program built from
/// `!`, `&`, `>>`, and `|` over [`RegimeExpr`].
///
/// Rule 3 separates multi-candidate ambiguity (`candidate_count_gt(1)`) from steward policy
/// authorization (`policy_requires_ask()`): when [`RegimeExpr::specialize_for_schema_invariants`]
/// folds `PolicyRequiresAsk -> ConstBool(false)` on tables without steward columns or in `read_only`
/// mode, `(candidate_count_gt(1) & steward_decision_is(None)) >> Mode::Ask` remains active so
/// multi-row candidate ambiguity still triggers `Mode::Ask` clarification.
pub fn canonical_pocket_regime_dsl() -> RegimeExpr {
    (!has_schema_probe() >> Mode::Probe)
        | (!candidates_observed() >> Mode::Read)
        | ((candidate_count_eq(0) & !operation_eq("INSERT")) >> Mode::Abstain)
        | (((policy_requires_ask() | candidate_count_gt(1)) & steward_decision_is(None))
            >> Mode::Ask)
        | (steward_decision_is(Some(false)) >> Mode::Abstain)
        | ((zero_row_toctou_race() & writes_applied_eq(0)) >> Mode::Write)
        | ((zero_row_toctou_race() & writes_applied_gt(0)) >> Mode::Abstain)
        | ((is_locked_or_denied() & !steward_decision_is(Some(true))) >> Mode::Ask)
        | (writes_applied_eq(0) >> Mode::Write)
        | (!post_write_verified() >> Mode::Read)
        | emit_mode(Mode::Result)
}

/// Canonical 14-row episode-state bank (`11` `[MODE:` decision-fork [`PocketObsState`]s covering
/// every priority rule of [`canonical_pocket_regime_dsl`] and both truth polarities of every
/// [`EpistemicAtom`], plus `3` `None` non-fork continuation/background rows) used to bind captured
/// rows to observable states during compilation.
pub fn canonical_compile_fixture_row_states() -> Vec<Option<PocketObsState>> {
    let mk = |world_id: &str,
              turn_index: usize,
              operation: &str,
              has_schema_probe: bool,
              candidate_count: Option<usize>,
              policy_requires_ask: bool,
              is_locked_or_denied: bool,
              steward_grant_approved: Option<bool>,
              applied_write_count: usize,
              has_post_write_verification: bool,
              zero_row_toctou_race: bool| {
        Some(PocketObsState {
            world_id: world_id.to_string(),
            turn_index,
            table_name: "incidents".to_string(),
            entity_name: "INC-101".to_string(),
            operation: operation.to_string(),
            has_schema_probe,
            candidate_count,
            policy_requires_ask,
            is_locked_or_denied,
            steward_grant_approved,
            applied_write_count,
            has_post_write_verification,
            zero_row_toctou_race,
        })
    };
    vec![
        mk(
            "W0_PROBE", 0, "UPDATE", false, None, false, false, None, 0, false, false,
        ),
        mk(
            "W1_PRE_READ",
            1,
            "UPDATE",
            true,
            None,
            false,
            false,
            None,
            0,
            false,
            false,
        ),
        mk(
            "W2_ZERO_CAND",
            2,
            "UPDATE",
            true,
            Some(0),
            false,
            false,
            None,
            0,
            false,
            false,
        ),
        mk(
            "W3_POLICY_ASK",
            2,
            "UPDATE",
            true,
            Some(2),
            true,
            false,
            None,
            0,
            false,
            false,
        ),
        mk(
            "W4_STEWARD_DENY",
            3,
            "DELETE",
            true,
            Some(1),
            true,
            true,
            Some(false),
            0,
            false,
            false,
        ),
        mk(
            "W5_TOCTOU_WRITE",
            2,
            "UPDATE",
            true,
            Some(1),
            false,
            false,
            None,
            0,
            false,
            true,
        ),
        mk(
            "W6_TOCTOU_ABSTAIN",
            3,
            "UPDATE",
            true,
            Some(1),
            false,
            false,
            None,
            1,
            false,
            true,
        ),
        mk(
            "W7_LOCKED_ASK",
            2,
            "UPDATE",
            true,
            Some(1),
            false,
            true,
            None,
            0,
            false,
            false,
        ),
        mk(
            "W8_READY_WRITE",
            2,
            "INSERT",
            true,
            Some(0),
            false,
            false,
            Some(true),
            0,
            false,
            false,
        ),
        mk(
            "W9_VERIFY_READ",
            3,
            "UPDATE",
            true,
            Some(1),
            false,
            false,
            Some(true),
            1,
            false,
            false,
        ),
        mk(
            "W10_RESULT",
            4,
            "UPDATE",
            true,
            Some(1),
            false,
            false,
            Some(true),
            1,
            true,
            false,
        ),
        None,
        None,
        None,
    ]
}
