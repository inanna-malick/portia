//! Algebraic AST simplification, priority-ordered rule extraction (`GuardedModeRule`),
//! and activation capture row binding (`GuardedRuleStateBinding`).

use super::{PocketObsState, RegimeExpr, RegimeExprFrame};
use crate::hylo::{Collapsible, CollapsibleExt, Expandable};
use crate::Mode;
use serde::{Deserialize, Serialize};

/// A compiled guarded rule `(guard >> mode)` extracted from a [`RegimeExpr`] via
/// [`RegimeExpr::extract_guarded_rules`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardedModeRule {
    pub rule_index: usize,
    pub rule_id: String,
    pub guard: RegimeExpr,
    pub mode: Mode,
}

/// Semantic binding of a [`GuardedModeRule`] against a sequence of captured activation rows
/// associated with optional [`PocketObsState`] episode states (`Some(obs)` for `[MODE:`
/// decision-fork rows, `None` for non-fork continuation/background rows).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardedRuleStateBinding {
    pub rule_index: usize,
    pub rule_id: String,
    pub guard: RegimeExpr,
    pub mode: Mode,
    /// Captured decision-fork row indices where `guard.evaluate(obs).truth == true`
    /// (independent of priority shadowing by earlier rules).
    pub guard_satisfied_indices: Vec<usize>,
    /// Captured decision-fork row indices where this rule is the **winning** priority rule
    /// (`guard.evaluate(obs).truth == true` and no earlier rule `j < rule_index` fired),
    /// matching `regime.evaluate(obs).selected_mode == Some(mode)`.
    pub positive_prompt_indices: Vec<usize>,
    /// Captured decision-fork row indices where this rule does **not** win
    /// (`guard.evaluate(obs).truth == false` or shadowed by an earlier rule `j < rule_index`).
    pub negative_prompt_indices: Vec<usize>,
    /// Non-fork continuation/background row indices (`row_states[idx].is_none()`).
    pub background_prompt_indices: Vec<usize>,
}

/// Partition a slice of captured row states (`Some(obs)` for decision-fork rows, `None` for
/// continuation/background rows) by evaluating `guard.evaluate(obs).truth`, returning
/// `(positive_indices, negative_indices, background_indices)`.
pub fn partition_states_by_guard(
    guard: &RegimeExpr,
    row_states: &[Option<PocketObsState>],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    let mut bg = Vec::new();
    for (idx, state_opt) in row_states.iter().enumerate() {
        match state_opt {
            Some(obs) => {
                if guard.evaluate(obs).truth {
                    pos.push(idx);
                } else {
                    neg.push(idx);
                }
            }
            None => bg.push(idx),
        }
    }
    (pos, neg, bg)
}

impl RegimeExpr {
    /// Algebraically simplify this [`RegimeExpr`] via a bottom-up
    /// [`recursion::CollapsibleExt::collapse_frames`] catamorphism while strictly preserving
    /// `(truth, selected_mode)` evaluation semantics across all [`PocketObsState`]s:
    /// - Negation: `!true -> false`, `!false -> true`, `!EmitMode(_) -> false`, and `!!x -> x`
    ///   when `!x.can_emit_mode()`
    /// - Conjunction identities & annihilation: `true & x -> x`, `false & x -> false`
    /// - Guarded stage folding: `true >> rhs -> rhs`, `false >> rhs -> ConstBool(false)`,
    ///   `lhs >> false -> ConstBool(false)`
    /// - Priority-fallback & disjunction identities:
    ///   - `EmitMode(m) | _ -> EmitMode(m)`
    ///   - `ConstBool(false) | rhs -> rhs` and `lhs | ConstBool(false) -> lhs`
    ///   - `ConstBool(true) | EmitMode(m) -> EmitMode(m)` (since `ConstBool(true)` has `selected_mode == None`)
    ///   - `ConstBool(true) | rhs -> ConstBool(true)` and `lhs | ConstBool(true) -> ConstBool(true)`
    ///     when the other branch is a pure Boolean guard (`!can_emit_mode()`).
    pub fn simplify(&self) -> Self {
        struct SimplifiedNode {
            expr: RegimeExpr,
            can_emit: bool,
            not_child_can_emit: bool,
        }

        self.collapse_frames(|frame: RegimeExprFrame<SimplifiedNode>| match frame {
            RegimeExprFrame::ConstBool(b) => SimplifiedNode {
                expr: RegimeExpr::ConstBool(b),
                can_emit: false,
                not_child_can_emit: false,
            },
            RegimeExprFrame::Atom(a) => SimplifiedNode {
                expr: RegimeExpr::Atom(a),
                can_emit: false,
                not_child_can_emit: false,
            },
            RegimeExprFrame::Not(inner) => match inner.expr.into_frame() {
                RegimeExprFrame::ConstBool(b) => SimplifiedNode {
                    expr: RegimeExpr::ConstBool(!b),
                    can_emit: false,
                    not_child_can_emit: false,
                },
                RegimeExprFrame::EmitMode(_) => SimplifiedNode {
                    expr: RegimeExpr::ConstBool(false),
                    can_emit: false,
                    not_child_can_emit: false,
                },
                RegimeExprFrame::Not(double_neg) if !inner.not_child_can_emit => {
                    let not_child_can_emit = matches!(&double_neg, RegimeExpr::Not(_));
                    SimplifiedNode {
                        expr: double_neg,
                        can_emit: false,
                        not_child_can_emit,
                    }
                }
                other_frame => SimplifiedNode {
                    expr: !RegimeExpr::from_frame(other_frame),
                    can_emit: false,
                    not_child_can_emit: inner.can_emit,
                },
            },
            RegimeExprFrame::And(lhs, rhs) => match (&lhs.expr, &rhs.expr) {
                (RegimeExpr::ConstBool(false), _) | (_, RegimeExpr::ConstBool(false)) => {
                    SimplifiedNode {
                        expr: RegimeExpr::ConstBool(false),
                        can_emit: false,
                        not_child_can_emit: false,
                    }
                }
                (RegimeExpr::ConstBool(true), _) => rhs,
                (_, RegimeExpr::ConstBool(true)) => lhs,
                _ => SimplifiedNode {
                    can_emit: lhs.can_emit || rhs.can_emit,
                    expr: lhs.expr & rhs.expr,
                    not_child_can_emit: false,
                },
            },
            RegimeExprFrame::Or(lhs, rhs) => match (&lhs.expr, &rhs.expr) {
                (RegimeExpr::EmitMode(m), _) => SimplifiedNode {
                    expr: RegimeExpr::EmitMode(*m),
                    can_emit: true,
                    not_child_can_emit: false,
                },
                (RegimeExpr::Or(_, inner_r), _)
                    if matches!(inner_r.as_ref(), RegimeExpr::EmitMode(_)) =>
                {
                    lhs
                }
                (RegimeExpr::ConstBool(false), _) => rhs,
                (_, RegimeExpr::ConstBool(false)) => lhs,
                (RegimeExpr::ConstBool(true), RegimeExpr::EmitMode(m)) => SimplifiedNode {
                    expr: RegimeExpr::EmitMode(*m),
                    can_emit: true,
                    not_child_can_emit: false,
                },
                (RegimeExpr::ConstBool(true), _) if !rhs.can_emit => SimplifiedNode {
                    expr: RegimeExpr::ConstBool(true),
                    can_emit: false,
                    not_child_can_emit: false,
                },
                (_, RegimeExpr::ConstBool(true)) if !lhs.can_emit => SimplifiedNode {
                    expr: RegimeExpr::ConstBool(true),
                    can_emit: false,
                    not_child_can_emit: false,
                },
                _ => SimplifiedNode {
                    can_emit: lhs.can_emit || rhs.can_emit,
                    expr: lhs.expr | rhs.expr,
                    not_child_can_emit: false,
                },
            },
            RegimeExprFrame::Stage(guard, consequent) => match (&guard.expr, &consequent.expr) {
                (RegimeExpr::ConstBool(true), _) => consequent,
                (RegimeExpr::ConstBool(false), _) | (_, RegimeExpr::ConstBool(false)) => {
                    SimplifiedNode {
                        expr: RegimeExpr::ConstBool(false),
                        can_emit: false,
                        not_child_can_emit: false,
                    }
                }
                _ => SimplifiedNode {
                    can_emit: guard.can_emit || consequent.can_emit,
                    expr: guard.expr >> consequent.expr,
                    not_child_can_emit: false,
                },
            },
            RegimeExprFrame::EmitMode(m) => SimplifiedNode {
                expr: RegimeExpr::EmitMode(m),
                can_emit: true,
                not_child_can_emit: false,
            },
        })
        .expr
    }

    /// Extract the priority-ordered sequence of [`GuardedModeRule`] clauses `(guard, mode)` from
    /// this [`RegimeExpr`] via [`recursion::CollapsibleExt::collapse_frames`], assigning stable
    /// deterministic `rule_id` identifiers (`"rule_00_probe"`, `"rule_01_read"`, ...).
    pub fn extract_guarded_rules(&self) -> Vec<GuardedModeRule> {
        #[inline]
        fn simplify_pure_not(inner: RegimeExpr) -> RegimeExpr {
            match inner.into_frame() {
                RegimeExprFrame::ConstBool(b) => RegimeExpr::ConstBool(!b),
                RegimeExprFrame::Not(double_neg) => double_neg,
                other_frame => !RegimeExpr::from_frame(other_frame),
            }
        }

        #[inline]
        fn simplify_pure_and(lhs: RegimeExpr, rhs: RegimeExpr) -> RegimeExpr {
            match (lhs, rhs) {
                (RegimeExpr::ConstBool(false), _) | (_, RegimeExpr::ConstBool(false)) => {
                    RegimeExpr::ConstBool(false)
                }
                (RegimeExpr::ConstBool(true), r) => r,
                (l, RegimeExpr::ConstBool(true)) => l,
                (l, r) => l & r,
            }
        }

        #[inline]
        fn simplify_pure_or(lhs: RegimeExpr, rhs: RegimeExpr) -> RegimeExpr {
            match (lhs, rhs) {
                (RegimeExpr::ConstBool(true), _) | (_, RegimeExpr::ConstBool(true)) => {
                    RegimeExpr::ConstBool(true)
                }
                (RegimeExpr::ConstBool(false), r) => r,
                (l, RegimeExpr::ConstBool(false)) => l,
                (l, r) => l | r,
            }
        }

        let simplified_root = self.simplify();
        let raw_pairs: Vec<(RegimeExpr, Mode)> = simplified_root
            .collapse_frames(
                |frame: RegimeExprFrame<(RegimeExpr, Vec<(RegimeExpr, Mode)>)>| match frame {
                    RegimeExprFrame::ConstBool(b) => (RegimeExpr::ConstBool(b), Vec::new()),
                    RegimeExprFrame::Atom(a) => (RegimeExpr::Atom(a), Vec::new()),
                    RegimeExprFrame::Not((inner_truth, _)) => {
                        (simplify_pure_not(inner_truth), Vec::new())
                    }
                    RegimeExprFrame::And((l_truth, l_rules), (r_truth, r_rules)) => {
                        let mut combined_rules = Vec::new();
                        for (lg, lm) in l_rules {
                            let guard = simplify_pure_and(lg, r_truth.clone());
                            if !matches!(guard, RegimeExpr::ConstBool(false)) {
                                combined_rules.push((guard, lm));
                            }
                        }
                        for (rg, rm) in r_rules {
                            let guard = simplify_pure_and(l_truth.clone(), rg);
                            if !matches!(guard, RegimeExpr::ConstBool(false)) {
                                combined_rules.push((guard, rm));
                            }
                        }
                        (simplify_pure_and(l_truth, r_truth), combined_rules)
                    }
                    RegimeExprFrame::Or((l_truth, mut l_rules), (r_truth, r_rules)) => {
                        l_rules.extend(r_rules);
                        (simplify_pure_or(l_truth, r_truth), l_rules)
                    }
                    RegimeExprFrame::Stage((g_truth, g_rules), (c_truth, c_rules)) => {
                        let mut combined_rules = Vec::new();
                        for (cg, cm) in c_rules {
                            let guard = simplify_pure_and(g_truth.clone(), cg);
                            if !matches!(guard, RegimeExpr::ConstBool(false)) {
                                combined_rules.push((guard, cm));
                            }
                        }
                        for (gg, gm) in g_rules {
                            let guard = simplify_pure_and(gg, c_truth.clone());
                            if !matches!(guard, RegimeExpr::ConstBool(false)) {
                                combined_rules.push((guard, gm));
                            }
                        }
                        (simplify_pure_and(g_truth, c_truth), combined_rules)
                    }
                    RegimeExprFrame::EmitMode(m) => (
                        RegimeExpr::ConstBool(true),
                        vec![(RegimeExpr::ConstBool(true), m)],
                    ),
                },
            )
            .1;

        raw_pairs
            .into_iter()
            .enumerate()
            .map(|(idx, (guard, mode))| {
                let mode_lower = mode.as_str().to_ascii_lowercase();
                GuardedModeRule {
                    rule_index: idx,
                    rule_id: format!("rule_{idx:02}_{mode_lower}"),
                    guard,
                    mode,
                }
            })
            .collect()
    }

    /// Semantically bind this [`RegimeExpr`]'s extracted priority rules (`extract_guarded_rules`)
    /// against a slice of captured row states (`Some(obs)` for `[MODE:` decision-fork rows, `None`
    /// for non-fork continuation/background rows).
    pub fn bind_capture_row_states(
        &self,
        row_states: &[Option<PocketObsState>],
    ) -> Vec<GuardedRuleStateBinding> {
        let rules = self.extract_guarded_rules();
        let background_prompt_indices: Vec<usize> = row_states
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| if s.is_none() { Some(idx) } else { None })
            .collect();

        let winning_rule_by_row: Vec<Option<usize>> = row_states
            .iter()
            .map(|state_opt| {
                state_opt.as_ref().and_then(|obs| {
                    rules
                        .iter()
                        .find(|r| r.guard.evaluate(obs).truth)
                        .map(|r| r.rule_index)
                })
            })
            .collect();

        rules
            .into_iter()
            .map(|rule| {
                let mut guard_satisfied_indices = Vec::new();
                let mut positive_prompt_indices = Vec::new();
                let mut negative_prompt_indices = Vec::new();
                for (row_idx, state_opt) in row_states.iter().enumerate() {
                    if let Some(obs) = state_opt {
                        if rule.guard.evaluate(obs).truth {
                            guard_satisfied_indices.push(row_idx);
                        }
                        if winning_rule_by_row[row_idx] == Some(rule.rule_index) {
                            positive_prompt_indices.push(row_idx);
                        } else {
                            negative_prompt_indices.push(row_idx);
                        }
                    }
                }
                GuardedRuleStateBinding {
                    rule_index: rule.rule_index,
                    rule_id: rule.rule_id,
                    guard: rule.guard,
                    mode: rule.mode,
                    guard_satisfied_indices,
                    positive_prompt_indices,
                    negative_prompt_indices,
                    background_prompt_indices: background_prompt_indices.clone(),
                }
            })
            .collect()
    }
}
