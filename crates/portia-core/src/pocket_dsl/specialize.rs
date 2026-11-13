//! Futamura-1 schema/operation partial evaluation and Futamura-2 monotonic/snapshot
//! episode residual specialization on [`RegimeExpr`].

use super::{EpistemicAtom, PocketObsState, RegimeExpr, RegimeExprFrame};
use crate::hylo::{CollapsibleExt, Expandable};
use crate::Mode;

impl RegimeExpr {
    /// Futamura-1 partial evaluation: specialize this [`RegimeExpr`] for a known SQL operation
    /// (`"INSERT"`, `"UPDATE"`, `"DELETE"`) and fold away all unreachable branches via
    /// [`recursion::CollapsibleExt::collapse_frames`].
    pub fn specialize_for_operation(&self, operation: &str) -> Self {
        let specialized = self.collapse_frames(|frame| match frame {
            RegimeExprFrame::Atom(EpistemicAtom::OperationEq(op)) => {
                RegimeExpr::ConstBool(op.eq_ignore_ascii_case(operation))
            }
            other => RegimeExpr::from_frame(other),
        });
        specialized.simplify()
    }

    /// Futamura-1 partial evaluation over schema and session invariants via
    /// [`recursion::CollapsibleExt::collapse_frames`] + [`Self::simplify`]:
    /// - `is_schema_disclosed == true` folds `HasSchemaProbe -> ConstBool(true)` (pruning `Mode::Probe`),
    /// - `!has_steward_col` folds `PolicyRequiresAsk -> ConstBool(false)`,
    /// - `!has_lock_col` folds `IsLockedOrDenied -> ConstBool(false)`,
    /// - `read_only == true` prunes mutation/permission branches (`PolicyRequiresAsk -> ConstBool(false)`,
    ///   `IsLockedOrDenied -> ConstBool(false)`, `Mode::Write -> Mode::Result`, and
    ///   `PostWriteVerified -> ConstBool(true)`) while preserving any clarification `Mode::Ask` rules.
    pub fn specialize_for_schema_invariants(
        &self,
        read_only: bool,
        has_steward_col: bool,
        has_lock_col: bool,
        is_schema_disclosed: bool,
    ) -> Self {
        let specialized = self.collapse_frames(|frame| match frame {
            RegimeExprFrame::Atom(EpistemicAtom::HasSchemaProbe) if is_schema_disclosed => {
                RegimeExpr::ConstBool(true)
            }
            RegimeExprFrame::Atom(EpistemicAtom::PolicyRequiresAsk)
                if !has_steward_col || read_only =>
            {
                RegimeExpr::ConstBool(false)
            }
            RegimeExprFrame::Atom(EpistemicAtom::IsLockedOrDenied)
                if !has_lock_col || read_only =>
            {
                RegimeExpr::ConstBool(false)
            }
            RegimeExprFrame::Atom(EpistemicAtom::PostWriteVerified) if read_only => {
                RegimeExpr::ConstBool(true)
            }
            RegimeExprFrame::EmitMode(Mode::Write) if read_only => {
                RegimeExpr::EmitMode(Mode::Result)
            }
            other => RegimeExpr::from_frame(other),
        });
        specialized.simplify()
    }

    /// Reusable Futamura-2 episode residual specialization: fold **only** [`EpistemicAtom`]s
    /// whose truth value is provably invariant across **all reachable successor states**
    /// $s' \succeq \text{state}$ ([`EpistemicAtom::monotonic_invariant_value`] `== Some(b)`)
    /// into `ConstBool(b)` and algebraically simplify the remaining AST.
    ///
    /// ### Multi-Step Trajectory Homomorphism Law
    /// For every reachable successor state $s' \succeq \text{state}$ (under arbitrary future
    /// probes, secondary reads, steward decisions, and additional writes where
    /// `applied_write_count` is non-decreasing):
    /// ```text
    /// self.specialize_monotonic_invariants(state).evaluate(&s').truth == self.evaluate(&s').truth
    /// self.specialize_monotonic_invariants(state).evaluate(&s').selected_mode == self.evaluate(&s').selected_mode
    /// ```
    pub fn specialize_monotonic_invariants(&self, state: &PocketObsState) -> Self {
        let specialized = self.collapse_frames(|frame| match frame {
            RegimeExprFrame::Atom(atom) => match atom.monotonic_invariant_value(state) {
                Some(b) => RegimeExpr::ConstBool(b),
                None => RegimeExpr::Atom(atom),
            },
            other => RegimeExpr::from_frame(other),
        });
        specialized.simplify()
    }

    /// Single-turn snapshot specialization: fold all facts observed at `state` (including
    /// non-monotonic turn-local facts such as `CandidateCountEq`, `PolicyRequiresAsk`,
    /// `IsLockedOrDenied`, `WritesAppliedEq`, and `PostWriteVerified`) for **current-turn**
    /// slot pruning at `state` only.
    ///
    /// Unlike [`Self::specialize_monotonic_invariants`], the AST returned by
    /// `specialize_for_current_turn` is specialized to the exact snapshot `state` and must not
    /// be reused after a subsequent read or write mutates non-monotonic state fields.
    pub fn specialize_for_current_turn(&self, state: &PocketObsState) -> Self {
        let specialized = self.collapse_frames(|frame| match frame {
            RegimeExprFrame::Atom(EpistemicAtom::OperationEq(op)) => {
                RegimeExpr::ConstBool(op.eq_ignore_ascii_case(&state.operation))
            }
            RegimeExprFrame::Atom(EpistemicAtom::HasSchemaProbe) if state.has_schema_probe => {
                RegimeExpr::ConstBool(true)
            }
            RegimeExprFrame::Atom(EpistemicAtom::CandidatesObserved)
                if state.candidate_count.is_some() =>
            {
                RegimeExpr::ConstBool(true)
            }
            RegimeExprFrame::Atom(EpistemicAtom::CandidateCountEq(n))
                if state.candidate_count.is_some() =>
            {
                RegimeExpr::ConstBool(state.candidate_count == Some(n))
            }
            RegimeExprFrame::Atom(EpistemicAtom::CandidateCountGt(n))
                if state.candidate_count.is_some() =>
            {
                RegimeExpr::ConstBool(state.candidate_count.is_some_and(|c| c > n))
            }
            RegimeExprFrame::Atom(EpistemicAtom::PolicyRequiresAsk)
                if state.candidate_count.is_some() =>
            {
                RegimeExpr::ConstBool(state.policy_requires_ask)
            }
            RegimeExprFrame::Atom(EpistemicAtom::IsLockedOrDenied)
                if state.candidate_count.is_some() =>
            {
                RegimeExpr::ConstBool(state.is_locked_or_denied)
            }
            RegimeExprFrame::Atom(EpistemicAtom::StewardDecisionIs(expected))
                if state.steward_grant_approved.is_some()
                    || state.applied_write_count > 0
                    || (state.candidate_count.is_some()
                        && !state.policy_requires_ask
                        && !state.is_locked_or_denied) =>
            {
                RegimeExpr::ConstBool(state.steward_grant_approved == expected)
            }
            RegimeExprFrame::Atom(EpistemicAtom::WritesAppliedEq(n))
                if state.applied_write_count > 0 =>
            {
                RegimeExpr::ConstBool(state.applied_write_count == n)
            }
            RegimeExprFrame::Atom(EpistemicAtom::WritesAppliedGt(n))
                if state.applied_write_count > 0 =>
            {
                RegimeExpr::ConstBool(state.applied_write_count > n)
            }
            RegimeExprFrame::Atom(EpistemicAtom::ZeroRowToctouRace)
                if state.applied_write_count > 0 && !state.zero_row_toctou_race =>
            {
                RegimeExpr::ConstBool(false)
            }
            RegimeExprFrame::Atom(EpistemicAtom::PostWriteVerified)
                if state.has_post_write_verification =>
            {
                RegimeExpr::ConstBool(true)
            }
            other => RegimeExpr::from_frame(other),
        });
        specialized.simplify()
    }

    /// Reusable Futamura-2 mid-episode residual specialization, delegating to
    /// [`Self::specialize_monotonic_invariants`] so the resulting residual is sound across
    /// all reachable future states $s' \succeq \text{state}$.
    #[inline]
    pub fn specialize_for_turn_state(&self, state: &PocketObsState) -> Self {
        self.specialize_monotonic_invariants(state)
    }

    /// Return the deduplicated, priority-ordered list of reachable [`Mode`]s in this
    /// (potentially specialized) [`RegimeExpr`] via [`recursion::CollapsibleExt::collapse_frames`].
    pub fn active_modes(&self) -> Vec<Mode> {
        let simplified = self.simplify();
        simplified.collapse_frames(|frame: RegimeExprFrame<Vec<Mode>>| match frame {
            RegimeExprFrame::ConstBool(_) | RegimeExprFrame::Atom(_) | RegimeExprFrame::Not(_) => {
                Vec::new()
            }
            RegimeExprFrame::EmitMode(m) => vec![m],
            RegimeExprFrame::And(mut l, r)
            | RegimeExprFrame::Or(mut l, r)
            | RegimeExprFrame::Stage(mut l, r) => {
                for m in r {
                    if !l.contains(&m) {
                        l.push(m);
                    }
                }
                l
            }
        })
    }

    /// Count total AST nodes in this [`RegimeExpr`] via [`recursion::CollapsibleExt::collapse_frames`].
    pub fn ast_node_count(&self) -> usize {
        self.collapse_frames(|frame| match frame {
            RegimeExprFrame::ConstBool(_)
            | RegimeExprFrame::Atom(_)
            | RegimeExprFrame::EmitMode(_) => 1,
            RegimeExprFrame::Not(c) => 1 + c,
            RegimeExprFrame::And(l, r)
            | RegimeExprFrame::Or(l, r)
            | RegimeExprFrame::Stage(l, r) => 1 + l + r,
        })
    }
}
