//! LLM-Facing Declarative Intent DSL (`ir`), parameterized via `recursion::MappableFrame`.
//!
//! Eliminates manual `SlotPairOffset(48, 49)` neural register juggling and `nalgebra` leaks:
//! authors specify high-level behavioral & pointer-bus intents (`StanceRoute`, `SignedTernaryFork`,
//! `SubmergedPointerRoute`, `AsidSegmentColor`, `DeltaNetRollbackScrub`) composed with:
//! - `a >> b` (`Seq`: sequential span staging),
//! - `high | low` (`Inhibit`: Rodney Brooks asymmetric priority inhibition — `high`'s active
//!   cursors automatically clamp `low`'s gate `<= -8.0` at matching `(bank, layer)`), and
//! - `a & b` (`Parallel`: simultaneous orthogonal composition).

pub mod regalloc;
pub mod scope_types;

use crate::discovery::gqa_bus::MemorySegmentAsid;
use crate::numerics::sod::ChordalAngleRad;
use portia_core::hylo::{Collapsible, Expandable, MappableFrame, PartiallyApplied};
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{GrammarCartridgeSpec, Mode, MountPolicy};
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitOr, Shr};

/// Hardware mechanism targeted by a declarative `IntentRule`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HardwareIntentKind {
    /// Late-band stance/mode rotation (`2` slots: radial + tangential) with optional anti-echo discovery.
    StanceRoute {
        layer_idx: usize,
        raw_target_dir: Vec<f64>,
        angle: ChordalAngleRad,
        expected_mode: Option<Mode>,
    },
    /// Convex-feasible 2-slot bilinear `SwiGLU` fork (`+1` Branch A, `0` Neutral/ASK midpoint, `-1` Branch B).
    SignedTernaryFork {
        layer_idx: usize,
        branch_axis_dir: Vec<f64>,
        angle: ChordalAngleRad,
    },
    /// Submerged GQA pointer-bus query+head-gate steering (`∈ ker(W_V) ∩ ker(W_U_top)`).
    SubmergedPointerRoute {
        upstream_mlp_layer: usize,
        target_gqa_layer: usize,
        copy_heads: Vec<usize>,
        distractor_heads: Vec<usize>,
        query_passthrough_delta: Vec<f64>,
        angle: ChordalAngleRad,
    },
    /// Write-time (`W_K`) or Read-time (`W_Q`) hardware `ASID` symbol-table coloring (`∈ ker(W_V) ∩ ker(W_U_top)`).
    AsidSegmentColor {
        upstream_mlp_layer: usize,
        target_gqa_layer: usize,
        copy_heads: Vec<usize>,
        segment: MemorySegmentAsid,
        is_write_side_key_stamp: bool,
        angle: ChordalAngleRad,
    },
    /// `DeltaNet-1` `conv1d`-bypassing `in_proj_a` rollback scrubber (`1` slot: `DecayFlush` + `RetentionLock`).
    DeltaNetRollbackScrub {
        upstream_mlp_layer: usize,
        target_deltanet_layer: usize,
        flush_heads: Vec<usize>,
        lock_heads: Vec<usize>,
        a_drive: f64,
    },
}

impl HardwareIntentKind {
    /// Physical MLP layer hosting the compiled SwiGLU slot(s).
    #[inline]
    #[must_use]
    pub const fn host_mlp_layer(&self) -> usize {
        match self {
            Self::StanceRoute { layer_idx, .. } | Self::SignedTernaryFork { layer_idx, .. } => {
                *layer_idx
            }
            Self::SubmergedPointerRoute {
                upstream_mlp_layer, ..
            }
            | Self::AsidSegmentColor {
                upstream_mlp_layer, ..
            }
            | Self::DeltaNetRollbackScrub {
                upstream_mlp_layer, ..
            } => *upstream_mlp_layer,
        }
    }

    /// Number of physical SwiGLU slots required (`1` for `DeltaNetRollbackScrub`, `2` for `SO(d)` pairs).
    #[inline]
    #[must_use]
    pub const fn required_slot_span(&self) -> usize {
        match self {
            Self::DeltaNetRollbackScrub { .. } => 1,
            Self::StanceRoute { .. }
            | Self::SignedTernaryFork { .. }
            | Self::SubmergedPointerRoute { .. }
            | Self::AsidSegmentColor { .. } => 2,
        }
    }
}

/// Declarative leaf rule in an [`IntentExpr`] program (zero manual slot indices!).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentRule {
    pub name: String,
    pub guard: RegimeExpr,
    pub mount_policy: MountPolicy,
    pub grammar: GrammarCartridgeSpec,
    pub kind: HardwareIntentKind,
}

/// Functor frame `IntentExprFrame<Child>` for stack-safe catamorphisms via `recursion`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntentExprFrame<Child> {
    Rule(Box<IntentRule>),
    Seq(Child, Child),
    Inhibit { high: Child, low: Child },
    Parallel(Child, Child),
}

impl MappableFrame for IntentExprFrame<PartiallyApplied> {
    type Frame<X> = IntentExprFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            IntentExprFrame::Rule(r) => IntentExprFrame::Rule(r),
            IntentExprFrame::Seq(a, b) => IntentExprFrame::Seq(f(a), f(b)),
            IntentExprFrame::Inhibit { high, low } => IntentExprFrame::Inhibit {
                high: f(high),
                low: f(low),
            },
            IntentExprFrame::Parallel(a, b) => IntentExprFrame::Parallel(f(a), f(b)),
        }
    }
}

/// Compositional Intent AST (`IntentExpr`) with stack-safe `Drop` and `recursion` instances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntentExpr {
    Rule(Box<IntentRule>),
    Seq(Box<IntentExpr>, Box<IntentExpr>),
    Inhibit {
        high: Box<IntentExpr>,
        low: Box<IntentExpr>,
    },
    Parallel(Box<IntentExpr>, Box<IntentExpr>),
}

impl IntentExpr {
    #[must_use]
    pub fn rule(rule: IntentRule) -> Self {
        Self::Rule(Box::new(rule))
    }
}

impl Expandable for IntentExpr {
    type FrameToken = IntentExprFrame<PartiallyApplied>;

    fn from_frame(val: IntentExprFrame<Self>) -> Self {
        match val {
            IntentExprFrame::Rule(r) => Self::Rule(r),
            IntentExprFrame::Seq(a, b) => Self::Seq(Box::new(a), Box::new(b)),
            IntentExprFrame::Inhibit { high, low } => Self::Inhibit {
                high: Box::new(high),
                low: Box::new(low),
            },
            IntentExprFrame::Parallel(a, b) => Self::Parallel(Box::new(a), Box::new(b)),
        }
    }
}

impl Collapsible for &IntentExpr {
    type FrameToken = IntentExprFrame<PartiallyApplied>;

    fn into_frame(self) -> IntentExprFrame<Self> {
        match self {
            IntentExpr::Rule(r) => IntentExprFrame::Rule(r.clone()),
            IntentExpr::Seq(a, b) => IntentExprFrame::Seq(a.as_ref(), b.as_ref()),
            IntentExpr::Inhibit { high, low } => IntentExprFrame::Inhibit {
                high: high.as_ref(),
                low: low.as_ref(),
            },
            IntentExpr::Parallel(a, b) => IntentExprFrame::Parallel(a.as_ref(), b.as_ref()),
        }
    }
}

impl Shr for IntentExpr {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        Self::Seq(Box::new(self), Box::new(rhs))
    }
}

impl BitOr for IntentExpr {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Inhibit {
            high: Box::new(self),
            low: Box::new(rhs),
        }
    }
}

impl BitAnd for IntentExpr {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Parallel(Box::new(self), Box::new(rhs))
    }
}
