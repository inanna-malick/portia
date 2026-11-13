//! Free-monad sequential agent programs (`FreeAgentProgram<A>`) over [`super::AgentActionStep`].

use super::{unfold_regime_into_free_agent, AgentActionStep};
use crate::hylo::{
    Collapsible, CollapsibleExt, Expandable, ExpandableExt, MappableFrame, PartiallyApplied,
};
use crate::pocket_dsl::{PocketObsState, RegimeExpr};
use crate::Mode;
use serde::{Deserialize, Serialize};
use std::ops::Shr;

/// Type alias connecting [`FreeAgentProgram<A>`] to the canonical `FreeAgent<A>` coalgebraic name.
pub type FreeAgent<A> = FreeAgentProgram<A>;

/// Base functor frame $\text{FreeAgentFrame}\langle A, X \rangle = A + (\text{AgentActionStep} \times X)$
/// for [`FreeAgentProgram<A>`], implementing [`recursion::MappableFrame`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreeAgentFrame<A, X> {
    Pure(A),
    Step { action: AgentActionStep, next: X },
}

impl<A> MappableFrame for FreeAgentFrame<A, PartiallyApplied> {
    type Frame<X> = FreeAgentFrame<A, X>;

    fn map_frame<In, Out>(
        input: Self::Frame<In>,
        mut f: impl FnMut(In) -> Out,
    ) -> Self::Frame<Out> {
        match input {
            FreeAgentFrame::Pure(a) => FreeAgentFrame::Pure(a),
            FreeAgentFrame::Step { action, next } => FreeAgentFrame::Step {
                action,
                next: f(next),
            },
        }
    }
}

/// Free monad $\text{Free } \text{AgentActionF } A$ representing a sequential agent action program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreeAgentProgram<A> {
    Pure(A),
    Step {
        action: AgentActionStep,
        next: Box<FreeAgentProgram<A>>,
    },
}

impl<A> Expandable for FreeAgentProgram<A> {
    type FrameToken = FreeAgentFrame<A, PartiallyApplied>;

    fn from_frame(val: FreeAgentFrame<A, Self>) -> Self {
        match val {
            FreeAgentFrame::Pure(a) => FreeAgentProgram::Pure(a),
            FreeAgentFrame::Step { action, next } => FreeAgentProgram::Step {
                action,
                next: Box::new(next),
            },
        }
    }
}

impl<A> Collapsible for FreeAgentProgram<A> {
    type FrameToken = FreeAgentFrame<A, PartiallyApplied>;

    fn into_frame(self) -> FreeAgentFrame<A, Self> {
        match self {
            FreeAgentProgram::Pure(a) => FreeAgentFrame::Pure(a),
            FreeAgentProgram::Step { action, next } => FreeAgentFrame::Step {
                action,
                next: *next,
            },
        }
    }
}

impl<A: Clone> Collapsible for &FreeAgentProgram<A> {
    type FrameToken = FreeAgentFrame<A, PartiallyApplied>;

    fn into_frame(self) -> FreeAgentFrame<A, Self> {
        match self {
            FreeAgentProgram::Pure(a) => FreeAgentFrame::Pure(a.clone()),
            FreeAgentProgram::Step { action, next } => FreeAgentFrame::Step {
                action: action.clone(),
                next: next.as_ref(),
            },
        }
    }
}

impl<A> FreeAgentProgram<A> {
    /// Construct a `FreeAgentProgram` from an iterator of `AgentActionStep`s ending in `Pure(terminal)`
    /// via [`recursion::ExpandableExt::expand_frames`].
    pub fn from_steps(steps: impl IntoIterator<Item = AgentActionStep>, terminal: A) -> Self {
        let mut iter = steps.into_iter();
        Self::expand_frames(Some(terminal), |mut term_slot| match iter.next() {
            Some(action) => FreeAgentFrame::Step {
                action,
                next: term_slot,
            },
            None => FreeAgentFrame::Pure(
                term_slot
                    .take()
                    .expect("terminal value consumed exactly once at Pure leaf"),
            ),
        })
    }

    /// Count the number of `Step` transitions in this `FreeAgentProgram` via
    /// [`recursion::CollapsibleExt::collapse_frames`].
    pub fn step_count(&self) -> usize
    where
        A: Clone,
    {
        self.collapse_frames(|frame| match frame {
            FreeAgentFrame::Pure(_) => 0,
            FreeAgentFrame::Step { next, .. } => 1 + next,
        })
    }

    /// Collect all [`AgentActionStep`]s in execution order alongside the terminal value `A`
    /// via [`recursion::CollapsibleExt::collapse_frames`].
    pub fn into_steps_and_terminal(self) -> (Vec<AgentActionStep>, A) {
        let (mut rev_steps, term) = self.collapse_frames(|frame| match frame {
            FreeAgentFrame::Pure(a) => (Vec::new(), a),
            FreeAgentFrame::Step {
                action,
                next: (mut tail_rev, a),
            } => {
                tail_rev.push(action);
                (tail_rev, a)
            }
        });
        rev_steps.reverse();
        (rev_steps, term)
    }

    /// Free-monad bind (`self >>= f`) implemented stack-safely via
    /// [`recursion::CollapsibleExt::collapse_frames`]: substitutes every `Pure(a)` leaf with `f(a)`.
    pub fn bind<B>(self, mut f: impl FnMut(A) -> FreeAgentProgram<B>) -> FreeAgentProgram<B> {
        self.collapse_frames(|frame| match frame {
            FreeAgentFrame::Pure(a) => f(a),
            FreeAgentFrame::Step { action, next } => FreeAgentProgram::Step {
                action,
                next: Box::new(next),
            },
        })
    }

    /// Sequentially compose two [`FreeAgentProgram`]s (`self >> rhs`), discarding the `Pure` output
    /// of `self` and continuing into `rhs` via [`Self::bind`].
    pub fn then_program<B>(self, rhs: FreeAgentProgram<B>) -> FreeAgentProgram<B> {
        let mut slot = Some(rhs);
        self.bind(|_| {
            slot.take()
                .expect("FreeAgentProgram::Pure reached at most once in sequential chain")
        })
    }
}

impl FreeAgentProgram<Mode> {
    /// Stack-safely unfold any authored [`RegimeExpr`] into a [`FreeAgentProgram<Mode>`] strategy
    /// tree starting from `initial_obs` up to `max_turns` steps.
    pub fn from_regime_expr(
        regime: &RegimeExpr,
        initial_obs: PocketObsState,
        max_turns: usize,
    ) -> Self {
        unfold_regime_into_free_agent(regime, initial_obs, max_turns)
    }
}

impl<A, B> Shr<FreeAgentProgram<B>> for FreeAgentProgram<A> {
    type Output = FreeAgentProgram<B>;

    #[inline]
    fn shr(self, rhs: FreeAgentProgram<B>) -> Self::Output {
        self.then_program(rhs)
    }
}

impl AgentActionStep {
    /// Lift a single [`AgentActionStep`] into a 1-step [`FreeAgentProgram<()>`].
    #[inline]
    pub fn into_program(self) -> FreeAgentProgram<()> {
        FreeAgentProgram::Step {
            action: self,
            next: Box::new(FreeAgentProgram::Pure(())),
        }
    }
}

impl Shr<AgentActionStep> for AgentActionStep {
    type Output = FreeAgentProgram<()>;

    #[inline]
    fn shr(self, rhs: AgentActionStep) -> Self::Output {
        self.into_program() >> rhs.into_program()
    }
}

impl<A> Shr<AgentActionStep> for FreeAgentProgram<A> {
    type Output = FreeAgentProgram<()>;

    #[inline]
    fn shr(self, rhs: AgentActionStep) -> Self::Output {
        self >> rhs.into_program()
    }
}
