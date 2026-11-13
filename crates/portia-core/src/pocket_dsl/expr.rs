//! Compositional Epistemic Regime AST (`RegimeExprFrame`, `RegimeExpr`), stack-safe
//! `Collapsible` / `Expandable` / `Drop` instances, and catamorphic `evaluate`.

use super::{EpistemicAtom, PocketObsState};
use crate::hylo::{Collapsible, CollapsibleExt, Expandable, MappableFrame, PartiallyApplied};
use crate::Mode;
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitOr, Not, Shr};
use std::sync::{Arc, OnceLock};

/// Base functor frame `RegimeExprFrame<Child>` for the compositional Epistemic Regime &
/// Trailhead Combinator eDSL, implementing [`recursion::MappableFrame`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegimeExprFrame<Child> {
    /// Constant boolean literal (`true` / `false`) or no-op rule fallback.
    ConstBool(bool),
    /// Atomic observable predicate on [`PocketObsState`].
    Atom(EpistemicAtom),
    /// Logical negation (`!child`).
    Not(Child),
    /// Logical conjunction (`lhs & rhs`).
    And(Child, Child),
    /// Logical disjunction on predicates or priority-inhibition fallback (`lhs | rhs`)
    /// on guarded mode rules.
    Or(Child, Child),
    /// Guarded transition / sequential staging (`guard >> consequent`, via [`std::ops::Shr`]).
    Stage(Child, Child),
    /// Terminal mode emitter (`emit_mode(Mode::...)`).
    EmitMode(Mode),
}

impl MappableFrame for RegimeExprFrame<PartiallyApplied> {
    type Frame<X> = RegimeExprFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            RegimeExprFrame::ConstBool(b) => RegimeExprFrame::ConstBool(b),
            RegimeExprFrame::Atom(a) => RegimeExprFrame::Atom(a),
            RegimeExprFrame::Not(c) => RegimeExprFrame::Not(f(c)),
            RegimeExprFrame::And(l, r) => RegimeExprFrame::And(f(l), f(r)),
            RegimeExprFrame::Or(l, r) => RegimeExprFrame::Or(f(l), f(r)),
            RegimeExprFrame::Stage(l, r) => RegimeExprFrame::Stage(f(l), f(r)),
            RegimeExprFrame::EmitMode(m) => RegimeExprFrame::EmitMode(m),
        }
    }
}

mod arc_regime_expr_serde {
    use super::RegimeExpr;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub fn serialize<S: Serializer>(
        val: &Arc<RegimeExpr>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        val.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<RegimeExpr>, D::Error> {
        RegimeExpr::deserialize(deserializer).map(Arc::new)
    }
}

fn dummy_regime_leaf_arc() -> &'static Arc<RegimeExpr> {
    static LEAF: OnceLock<Arc<RegimeExpr>> = OnceLock::new();
    LEAF.get_or_init(|| Arc::new(RegimeExpr::ConstBool(false)))
}

/// Compositional Epistemic Regime & Trailhead Combinator AST (`RegimeExpr`), supporting native
/// Rust operators (`!`, `&`, `|`, `>>`), `O(1)` `Arc`-shared cloning, and stack-safe evaluation /
/// simplification / dropping via [`recursion::CollapsibleExt::collapse_frames`] catamorphisms.
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub enum RegimeExpr {
    ConstBool(bool),
    Atom(EpistemicAtom),
    Not(#[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>),
    And(
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
    ),
    Or(
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
    ),
    Stage(
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
        #[serde(with = "arc_regime_expr_serde")] Arc<RegimeExpr>,
    ),
    EmitMode(Mode),
}

impl PartialEq for RegimeExpr {
    fn eq(&self, other: &Self) -> bool {
        let mut stack = vec![(self, other)];
        while let Some((lhs, rhs)) = stack.pop() {
            match (lhs, rhs) {
                (Self::ConstBool(a), Self::ConstBool(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (Self::Atom(a), Self::Atom(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (Self::EmitMode(a), Self::EmitMode(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (Self::Not(a), Self::Not(b)) => {
                    if !Arc::ptr_eq(a, b) {
                        stack.push((a.as_ref(), b.as_ref()));
                    }
                }
                (Self::And(al, ar), Self::And(bl, br))
                | (Self::Or(al, ar), Self::Or(bl, br))
                | (Self::Stage(al, ar), Self::Stage(bl, br)) => {
                    if !Arc::ptr_eq(al, bl) {
                        stack.push((al.as_ref(), bl.as_ref()));
                    }
                    if !Arc::ptr_eq(ar, br) {
                        stack.push((ar.as_ref(), br.as_ref()));
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

impl Expandable for RegimeExpr {
    type FrameToken = RegimeExprFrame<PartiallyApplied>;

    fn from_frame(val: RegimeExprFrame<Self>) -> Self {
        match val {
            RegimeExprFrame::ConstBool(b) => RegimeExpr::ConstBool(b),
            RegimeExprFrame::Atom(a) => RegimeExpr::Atom(a),
            RegimeExprFrame::Not(c) => RegimeExpr::Not(Arc::new(c)),
            RegimeExprFrame::And(l, r) => RegimeExpr::And(Arc::new(l), Arc::new(r)),
            RegimeExprFrame::Or(l, r) => RegimeExpr::Or(Arc::new(l), Arc::new(r)),
            RegimeExprFrame::Stage(l, r) => RegimeExpr::Stage(Arc::new(l), Arc::new(r)),
            RegimeExprFrame::EmitMode(m) => RegimeExpr::EmitMode(m),
        }
    }
}

impl Collapsible for RegimeExpr {
    type FrameToken = RegimeExprFrame<PartiallyApplied>;

    fn into_frame(mut self) -> RegimeExprFrame<Self> {
        let dummy = dummy_regime_leaf_arc();
        match &mut self {
            RegimeExpr::ConstBool(b) => RegimeExprFrame::ConstBool(*b),
            RegimeExpr::Atom(a) => RegimeExprFrame::Atom(a.clone()),
            RegimeExpr::EmitMode(m) => RegimeExprFrame::EmitMode(*m),
            RegimeExpr::Not(c) => {
                let arc_c = std::mem::replace(c, Arc::clone(dummy));
                RegimeExprFrame::Not(Arc::unwrap_or_clone(arc_c))
            }
            RegimeExpr::And(l, r) => {
                let arc_l = std::mem::replace(l, Arc::clone(dummy));
                let arc_r = std::mem::replace(r, Arc::clone(dummy));
                RegimeExprFrame::And(Arc::unwrap_or_clone(arc_l), Arc::unwrap_or_clone(arc_r))
            }
            RegimeExpr::Or(l, r) => {
                let arc_l = std::mem::replace(l, Arc::clone(dummy));
                let arc_r = std::mem::replace(r, Arc::clone(dummy));
                RegimeExprFrame::Or(Arc::unwrap_or_clone(arc_l), Arc::unwrap_or_clone(arc_r))
            }
            RegimeExpr::Stage(l, r) => {
                let arc_l = std::mem::replace(l, Arc::clone(dummy));
                let arc_r = std::mem::replace(r, Arc::clone(dummy));
                RegimeExprFrame::Stage(Arc::unwrap_or_clone(arc_l), Arc::unwrap_or_clone(arc_r))
            }
        }
    }
}

impl Drop for RegimeExpr {
    fn drop(&mut self) {
        if matches!(
            self,
            RegimeExpr::ConstBool(_) | RegimeExpr::Atom(_) | RegimeExpr::EmitMode(_)
        ) {
            return;
        }
        let dummy = dummy_regime_leaf_arc();
        let mut stack = Vec::new();
        let push_if_unique = |stack: &mut Vec<RegimeExpr>, arc_child: Arc<RegimeExpr>| {
            if let Ok(unique_child) = Arc::try_unwrap(arc_child) {
                if !matches!(
                    unique_child,
                    RegimeExpr::ConstBool(_) | RegimeExpr::Atom(_) | RegimeExpr::EmitMode(_)
                ) {
                    stack.push(unique_child);
                }
            }
        };

        match self {
            RegimeExpr::ConstBool(_) | RegimeExpr::Atom(_) | RegimeExpr::EmitMode(_) => return,
            RegimeExpr::Not(c) => {
                let arc_c = std::mem::replace(c, Arc::clone(dummy));
                push_if_unique(&mut stack, arc_c);
            }
            RegimeExpr::And(l, r) | RegimeExpr::Or(l, r) | RegimeExpr::Stage(l, r) => {
                let arc_l = std::mem::replace(l, Arc::clone(dummy));
                let arc_r = std::mem::replace(r, Arc::clone(dummy));
                push_if_unique(&mut stack, arc_l);
                push_if_unique(&mut stack, arc_r);
            }
        }

        while let Some(mut node) = stack.pop() {
            match &mut node {
                RegimeExpr::ConstBool(_) | RegimeExpr::Atom(_) | RegimeExpr::EmitMode(_) => {}
                RegimeExpr::Not(c) => {
                    let arc_c = std::mem::replace(c, Arc::clone(dummy));
                    push_if_unique(&mut stack, arc_c);
                }
                RegimeExpr::And(l, r) | RegimeExpr::Or(l, r) | RegimeExpr::Stage(l, r) => {
                    let arc_l = std::mem::replace(l, Arc::clone(dummy));
                    let arc_r = std::mem::replace(r, Arc::clone(dummy));
                    push_if_unique(&mut stack, arc_l);
                    push_if_unique(&mut stack, arc_r);
                }
            }
        }
    }
}

impl Collapsible for &RegimeExpr {
    type FrameToken = RegimeExprFrame<PartiallyApplied>;

    fn into_frame(self) -> RegimeExprFrame<Self> {
        match self {
            RegimeExpr::ConstBool(b) => RegimeExprFrame::ConstBool(*b),
            RegimeExpr::Atom(a) => RegimeExprFrame::Atom(a.clone()),
            RegimeExpr::Not(c) => RegimeExprFrame::Not(c.as_ref()),
            RegimeExpr::And(l, r) => RegimeExprFrame::And(l.as_ref(), r.as_ref()),
            RegimeExpr::Or(l, r) => RegimeExprFrame::Or(l.as_ref(), r.as_ref()),
            RegimeExpr::Stage(l, r) => RegimeExprFrame::Stage(l.as_ref(), r.as_ref()),
            RegimeExpr::EmitMode(m) => RegimeExprFrame::EmitMode(*m),
        }
    }
}

impl Not for RegimeExpr {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        RegimeExpr::Not(Arc::new(self))
    }
}

impl BitAnd for RegimeExpr {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        RegimeExpr::And(Arc::new(self), Arc::new(rhs))
    }
}

impl BitOr for RegimeExpr {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        RegimeExpr::Or(Arc::new(self), Arc::new(rhs))
    }
}

impl Shr for RegimeExpr {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Self) -> Self::Output {
        RegimeExpr::Stage(Arc::new(self), Arc::new(rhs))
    }
}

impl Shr<Mode> for RegimeExpr {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Mode) -> Self::Output {
        RegimeExpr::Stage(Arc::new(self), Arc::new(RegimeExpr::EmitMode(rhs)))
    }
}

/// Result of evaluating a [`RegimeExpr`] against a [`PocketObsState`] via [`RegimeExpr::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegimeEvaluation {
    /// Boolean truth value when this sub-expression is interpreted as a guard predicate.
    pub truth: bool,
    /// Selected epistemic [`Mode`] if a guarded rule clause fired (`Some(mode)`).
    pub selected_mode: Option<Mode>,
    /// Minimal sequence of `(EpistemicAtom, observed_polarity)` pairs that witnessed the firing branch.
    pub active_implicants: Vec<(EpistemicAtom, bool)>,
}

impl RegimeEvaluation {
    /// Compare the operational decision semantics (`truth` and `selected_mode`) of two
    /// [`RegimeEvaluation`] results, independent of diagnostic [`Self::active_implicants`] traces
    /// (which omit atoms that have been algebraically folded into `ConstBool` constants during
    /// [`RegimeExpr::simplify`], [`RegimeExpr::specialize_monotonic_invariants`], or
    /// [`RegimeExpr::specialize_for_current_turn`]).
    #[inline]
    pub fn decision_eq(&self, other: &Self) -> bool {
        self.truth == other.truth && self.selected_mode == other.selected_mode
    }
}

impl RegimeExpr {
    /// Return `true` if this expression subtree can emit a [`Mode`] (`selected_mode.is_some()`), or
    /// `false` if it is a pure Boolean guard predicate (`selected_mode` is guaranteed `None`),
    /// computed via [`recursion::CollapsibleExt::collapse_frames`].
    pub fn can_emit_mode(&self) -> bool {
        self.collapse_frames(|frame: RegimeExprFrame<bool>| match frame {
            RegimeExprFrame::ConstBool(_) | RegimeExprFrame::Atom(_) | RegimeExprFrame::Not(_) => {
                false
            }
            RegimeExprFrame::EmitMode(_) => true,
            RegimeExprFrame::And(l, r)
            | RegimeExprFrame::Or(l, r)
            | RegimeExprFrame::Stage(l, r) => l || r,
        })
    }

    /// Evaluate this compositional regime expression against `state` in a single stack-safe
    /// [`recursion::CollapsibleExt::collapse_frames`] catamorphism, returning the selected
    /// [`Mode`] and the exact prime-implicant witness trace `(EpistemicAtom, bool)`.
    pub fn evaluate(&self, state: &PocketObsState) -> RegimeEvaluation {
        self.collapse_frames(|frame: RegimeExprFrame<RegimeEvaluation>| match frame {
            RegimeExprFrame::ConstBool(b) => RegimeEvaluation {
                truth: b,
                selected_mode: None,
                active_implicants: Vec::new(),
            },
            RegimeExprFrame::Atom(atom) => {
                let val = atom.evaluate(state);
                RegimeEvaluation {
                    truth: val,
                    selected_mode: None,
                    active_implicants: vec![(atom, val)],
                }
            }
            RegimeExprFrame::Not(mut inner) => {
                inner.truth = !inner.truth;
                inner.selected_mode = None;
                inner
            }
            RegimeExprFrame::And(mut lhs, mut rhs) => {
                if !lhs.truth {
                    RegimeEvaluation {
                        truth: false,
                        selected_mode: None,
                        active_implicants: lhs.active_implicants,
                    }
                } else {
                    lhs.active_implicants.append(&mut rhs.active_implicants);
                    RegimeEvaluation {
                        truth: rhs.truth,
                        selected_mode: if rhs.truth {
                            lhs.selected_mode.or(rhs.selected_mode)
                        } else {
                            None
                        },
                        active_implicants: lhs.active_implicants,
                    }
                }
            }
            RegimeExprFrame::Or(mut lhs, mut rhs) => {
                if lhs.selected_mode.is_some() {
                    lhs
                } else if rhs.selected_mode.is_some() {
                    rhs
                } else if lhs.truth {
                    lhs
                } else {
                    lhs.active_implicants.append(&mut rhs.active_implicants);
                    RegimeEvaluation {
                        truth: rhs.truth,
                        selected_mode: None,
                        active_implicants: lhs.active_implicants,
                    }
                }
            }
            RegimeExprFrame::Stage(mut guard, mut consequent) => {
                if guard.truth {
                    guard
                        .active_implicants
                        .append(&mut consequent.active_implicants);
                    RegimeEvaluation {
                        truth: consequent.truth,
                        selected_mode: if consequent.truth {
                            consequent.selected_mode.or(guard.selected_mode)
                        } else {
                            None
                        },
                        active_implicants: guard.active_implicants,
                    }
                } else {
                    RegimeEvaluation {
                        truth: false,
                        selected_mode: None,
                        active_implicants: Vec::new(),
                    }
                }
            }
            RegimeExprFrame::EmitMode(mode) => RegimeEvaluation {
                truth: true,
                selected_mode: Some(mode),
                active_implicants: Vec::new(),
            },
        })
    }
}
