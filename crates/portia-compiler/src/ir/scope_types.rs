//! Type-level scope and causal intra-token depth witnesses (`ir::scope_types`).
//!
//! Enforces two core architectural invariants in Rust's type system:
//! 1. **Static vs. Dynamic Scope Separation:** [`StaticPermanentScope`] (harness-free `.safetensors`
//!    patches) rejects constrained runtime `.lark` / JSON-Schema grammars at compile time, while
//!    [`DynamicSpanScope`] permits paired `(ΔW_slots × GrammarCartridgeSpec)` mounting.
//! 2. **Causal Intra-Token Depth Monotonicity:** [`TokenDepthChain<Start, End>`] enforces strict
//!    forward progression `EarlyPreGqa -> MidDeltaNet -> LateStance` so upstream GQA pointer or
//!    DeltaNet gate writes can never be chained *after* downstream late-band stance rotation.

use portia_core::GrammarCartridgeSpec;
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait indexing whether a cartridge specification targets harness-free static weights
/// ([`StaticPermanentScope`]) or phase-scoped runtime spans ([`DynamicSpanScope`]).
pub trait CartridgeScope: sealed::Sealed + Copy + Clone {}

/// Harness-free static `.safetensors` scope (`0` runtime hooks, `0` `llguidance` grammar masks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticPermanentScope {
    pub max_bg_rss_leakage: f64,
}

/// Phase-scoped runtime `SpanCartridge` scope (`ΔW_slots × GrammarCartridgeSpec`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DynamicSpanScope;

impl sealed::Sealed for StaticPermanentScope {}
impl sealed::Sealed for DynamicSpanScope {}
impl CartridgeScope for StaticPermanentScope {}
impl CartridgeScope for DynamicSpanScope {}

/// Scope-indexed wrapper around [`super::IntentExpr`] preventing runtime grammar attachment on
/// [`StaticPermanentScope`].
///
/// ```compile_fail
/// use portia_compiler::ir::scope_types::{ScopedIntentProgram, StaticPermanentScope};
/// use portia_core::GrammarCartridgeSpec;
///
/// let static_prog: ScopedIntentProgram<StaticPermanentScope> = todo!();
/// // Compile error: `with_runtime_grammar` is only implemented for `DynamicSpanScope`!
/// let _ = static_prog.with_runtime_grammar(GrammarCartridgeSpec::LarkGrammar("start: \"a\"".into()));
/// ```
#[derive(Debug, Clone)]
pub struct ScopedIntentProgram<S: CartridgeScope> {
    pub scope: S,
    pub expr: super::IntentExpr,
}

impl ScopedIntentProgram<DynamicSpanScope> {
    #[must_use]
    pub fn new_dynamic(expr: super::IntentExpr) -> Self {
        Self {
            scope: DynamicSpanScope,
            expr,
        }
    }

    #[must_use]
    pub fn with_runtime_grammar(self, _grammar: GrammarCartridgeSpec) -> Self {
        self
    }
}

impl ScopedIntentProgram<StaticPermanentScope> {
    #[must_use]
    pub fn new_static(expr: super::IntentExpr, max_bg_rss_leakage: f64) -> Self {
        Self {
            scope: StaticPermanentScope { max_bg_rss_leakage },
            expr,
        }
    }
}

/// Early upstream MLP depth stage (`L14` / `L18`) feeding a downstream GQA pointer/head-gate layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyPreGqa;

/// Mid-network MLP depth stage (`L20`) feeding a downstream Gated-DeltaNet fast-weight layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidDeltaNet;

/// Late commitment-band MLP depth stage (`L22` on `0.8B`; `L30` on `4B`/`9B`) driving stance rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateStance;

impl sealed::Sealed for EarlyPreGqa {}
impl sealed::Sealed for MidDeltaNet {}
impl sealed::Sealed for LateStance {}

/// Sealed causal ordering relation `Self -> Next` across transformer layer bands.
pub trait CausalDepthAdvance<Next>: sealed::Sealed {}
impl CausalDepthAdvance<MidDeltaNet> for EarlyPreGqa {}
impl CausalDepthAdvance<LateStance> for EarlyPreGqa {}
impl CausalDepthAdvance<LateStance> for MidDeltaNet {}

/// Type-witnessed intra-token depth chain enforcing forward-only `EarlyPreGqa -> MidDeltaNet -> LateStance`
/// composition.
///
/// ```compile_fail
/// use portia_compiler::ir::scope_types::{EarlyPreGqa, LateStance, TokenDepthChain};
///
/// let late: TokenDepthChain<LateStance, LateStance> = TokenDepthChain::singleton(22);
/// let early: TokenDepthChain<EarlyPreGqa, EarlyPreGqa> = TokenDepthChain::singleton(14);
/// // Compile error: `LateStance` cannot causally precede `EarlyPreGqa`!
/// let _ = late.then(early);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDepthChain<Start, End> {
    layers: Vec<usize>,
    _phantom: PhantomData<(Start, End)>,
}

impl<Stage: sealed::Sealed> TokenDepthChain<Stage, Stage> {
    #[must_use]
    pub fn singleton(layer_idx: usize) -> Self {
        Self {
            layers: vec![layer_idx],
            _phantom: PhantomData,
        }
    }
}

impl<Start, Mid> TokenDepthChain<Start, Mid> {
    #[must_use]
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    #[must_use]
    pub fn then<End>(mut self, next: TokenDepthChain<Mid, End>) -> TokenDepthChain<Start, End>
    where
        Mid: CausalDepthAdvance<End>,
    {
        self.layers.extend(next.layers);
        TokenDepthChain {
            layers: self.layers,
            _phantom: PhantomData,
        }
    }
}
