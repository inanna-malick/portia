//! `HarnessPen (H=0) -> RouterPen -> PayloadPen` deterministic relay state machine.
//!
//! Enforces Single-Pen handoff invariants across every assistant turn:
//! - `[MODE:` is prefilled by `HarnessPen` (`H = 0`) under `CartridgeBank::OrientTrunk`.
//! - `RouterPen` emits the mode token(s) (`" READ"`, `" PRO" + "BE"`, etc.) and `"]"` (`tid = 60`),
//!   all consumed under `CartridgeBank::OrientTrunk`.
//! - At the newline `"\n"` (`tid = 198`) after `"]"`, `HarnessPen` fast-forwards `"\nPlan:"` under
//!   `CartridgeBank::Base` and `PayloadPen` emits the single-line `Plan:` body up to `"\n"`
//!   (healing merged `"\n\n"` = `271` via [`crate::bpe::heal_delimiter_crossing`]).
//! - `HarnessPen` fast-forwards `"</think>\n<sql>\n"` or `"</think>\n<to_user>\n"` under
//!   `CartridgeBank::ModePayload(selected_mode)`, and `PayloadPen` emits the SQL / question / JSON
//!   body until the closing route tag (`"</sql>"` or `"</to_user>"` outside SQL/JSON literals),
//!   detaching back to `CartridgeBank::Base` before `HarnessPen` seals `<|im_end|>\n`.

mod machine;
mod scanners;
mod types;
mod typestate;
mod wire_spec;

#[cfg(test)]
mod tests;

pub const TURN_OPEN_PREFILL: &str = "<|im_start|>assistant\n<think>\n[MODE:";
pub const LEGACY_RESEARCHER_TURN_OPEN_PREFILL: &str = "<|im_start|>researcher\n<think>\n[MODE:";
pub const POST_MODE_FRAMING: &str = "\nPlan:";
pub const TURN_CLOSE_SUFFIX: &str = "<|im_end|>\n";
pub const PLAN_BODY_SINGLE_LINE_LARK: &str = "start: / [^\\r\\n]{3,}\\n/";

pub use machine::RelayMachine;
pub use scanners::{
    find_unquoted_sql_char, has_completed_close_tag_outside_literals, json_has_unclosed_string,
    normalize_lotl_wire_chunk, sql_has_unclosed_quote, sql_has_unquoted_where,
};
pub use types::{
    CompletedRelayTurn, PenSpanRecord, PlannedTrunkStep, RelayPhase, RelayStreamEvent,
    TypedTurnAction,
};
pub use typestate::{
    verify_single_pen_schedule, TurnAtOpenPrefill, TurnAtPayloadBody, TurnAtPlanBody,
    TurnAtPostModePrefill, TurnAtPostPlanPrefill, TurnAtRouterDecision,
};
pub use wire_spec::{
    GrammarCartridgeSpec, KvPrefillDisposition, LotlNormalizedWireChunk, PayloadGrammarAutomaton,
    SpanCartridge, UnconstrainedGrammarCartridge,
};
