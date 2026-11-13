//! `portia-core` — Canonical Wire Protocol, Single-Pen Automaton,
//! Hylomorphism Frame (`MappableFrame`), & Pocket/Steward Coalgebras.
//!
//! Enforces zero vocabulary drift across all crates:
//! - One canonical 6-mode epistemic action vocabulary (`Mode`: `Probe`, `Read`, `Ask`, `Write`, `Result`, `Abstain`)
//! - Exact Qwen3.5 BPE first-token IDs (`5140`, `18104`, `5574`, `37272`, `27604`, `13732`)
//! - Canonical route tags (`RouteTag`: `<sql>` vs `<to_user>`)
//! - Single-Pen handoff law (`|ActivePen(t)| == 1`: `HarnessPen(H=0) -> RouterPen -> PayloadPen`)

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

pub mod bpe;
pub mod domain;
pub mod hylo;
mod mount_policy;
pub mod pocket_dsl;
pub mod relay;
mod slot_types;
pub mod steward_coalgebra;

pub use domain::{ColumnSchema, TableSchema};
pub use mount_policy::{MountPolicy, WireBankKey};
pub use relay::{
    GrammarCartridgeSpec, KvPrefillDisposition, LotlNormalizedWireChunk, PayloadGrammarAutomaton,
    PlannedTrunkStep, SpanCartridge, UnconstrainedGrammarCartridge,
};
pub use slot_types::{
    BaseIntermediateSize, EnvelopeKind, LayerIndex, PhysicalSlotIndex, RelativeSlotOffset,
    SlotPairOffset, TokenId,
};

/// The canonical 6-mode epistemic action vocabulary.
/// Normalizes mode labels to `Mode::Result` while accepting `"REPORT"` as a
/// backwards-compatible parse alias at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Mode {
    Probe,
    Read,
    Ask,
    Write,
    Result,
    Abstain,
}

/// Coupled 2-slot `[radial_slot, tangential_slot]` index pair inside the `[0 .. 32)` router bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouterSlotPair {
    pub radial_slot: usize,
    pub tangential_slot: usize,
}

impl Mode {
    pub const ALL: [Mode; 6] = [
        Mode::Probe,
        Mode::Read,
        Mode::Ask,
        Mode::Write,
        Mode::Result,
        Mode::Abstain,
    ];

    /// Canonical wire string emitted inside `[MODE: <as_str>]`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Mode::Probe => "PROBE",
            Mode::Read => "READ",
            Mode::Ask => "ASK",
            Mode::Write => "WRITE",
            Mode::Result => "RESULT",
            Mode::Abstain => "ABSTAIN",
        }
    }

    /// Parse a mode label (with optional leading/trailing whitespace and `"REPORT"` alias).
    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim() {
            "PROBE" => Some(Mode::Probe),
            "READ" => Some(Mode::Read),
            "ASK" => Some(Mode::Ask),
            "WRITE" => Some(Mode::Write),
            "RESULT" | "REPORT" => Some(Mode::Result),
            "ABSTAIN" => Some(Mode::Abstain),
            _ => None,
        }
    }

    /// Exact first-token BPE ID in the shared 248,320-token Qwen3.5 (0.8B / 4B / 27B) vocabulary
    /// when prefill ends at `[MODE:` (no trailing space).
    pub const fn qwen35_first_token_id(self) -> u32 {
        match self {
            Mode::Probe => 5140,    // " PRO"
            Mode::Read => 18104,    // " READ"
            Mode::Ask => 5574,      // " AS"
            Mode::Write => 37272,   // " WRITE"
            Mode::Result => 27604,  // " RESULT"
            Mode::Abstain => 13732, // " AB"
        }
    }

    /// Remaining BPE continuation token IDs (before `]` = 60) in Qwen3.5.
    pub const fn qwen35_continuation_token_ids(self) -> &'static [u32] {
        match self {
            Mode::Probe => &[11261],       // "BE"
            Mode::Read => &[],             // single token " READ"
            Mode::Ask => &[42],            // "K"
            Mode::Write => &[],            // single token " WRITE"
            Mode::Result => &[],           // single token " RESULT"
            Mode::Abstain => &[759, 6622], // "ST", "AIN"
        }
    }

    /// Canonical outbound route tag for this mode's payload block.
    pub const fn route_tag(self) -> RouteTag {
        match self {
            Mode::Probe | Mode::Read | Mode::Write => RouteTag::ToDatabase,
            Mode::Ask | Mode::Result | Mode::Abstain => RouteTag::ToInquirer,
        }
    }

    /// Strongly-typed coupled 2-slot (`radial_slot`, `tangential_slot`) pair inside `[0 .. 32)`.
    pub const fn router_slot_pair(self) -> RouterSlotPair {
        let idx = match self {
            Mode::Probe => 0,
            Mode::Read => 1,
            Mode::Ask => 2,
            Mode::Write => 3,
            Mode::Result => 4,
            Mode::Abstain => 5,
        };
        RouterSlotPair {
            radial_slot: 2 * idx,
            tangential_slot: 2 * idx + 1,
        }
    }

    /// Reserved 2-slot coupled `[rad, tan]` index pair inside `[0 .. 32)` router bank.
    pub const fn router_slot_pair_offset(self) -> (usize, usize) {
        let pair = self.router_slot_pair();
        (pair.radial_slot, pair.tangential_slot)
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Canonical outbound/inbound XML channel tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouteTag {
    ToDatabase,
    ToInquirer,
}

impl RouteTag {
    pub const fn open_tag(self) -> &'static str {
        match self {
            RouteTag::ToDatabase => "<sql>",
            RouteTag::ToInquirer => "<to_user>",
        }
    }

    pub const fn close_tag(self) -> &'static str {
        match self {
            RouteTag::ToDatabase => "</sql>",
            RouteTag::ToInquirer => "</to_user>",
        }
    }

    /// Pretrain-LOTL single-token Qwen3.5 `<tool_call>` open tag (`248058`).
    pub const fn lotl_open_tag(self) -> &'static str {
        "<tool_call>"
    }

    /// Pretrain-LOTL single-token Qwen3.5 `</tool_call>` close tag (`248059`).
    pub const fn lotl_close_tag(self) -> &'static str {
        "</tool_call>"
    }

    /// Accepted closing tags for this route (`</sql>` and `</to_user>` canonical, with aliases
    /// accepted when parsing historical transcripts).
    pub const fn accepted_close_tags(self) -> &'static [&'static str] {
        match self {
            RouteTag::ToDatabase => &["</sql>", "</tool_call>", "</to_database>", "</to_sql>"],
            RouteTag::ToInquirer => &[
                "</to_user>",
                "</report>",
                "</tool_call>",
                "</to_inquirer>",
                "</to_report>",
            ],
        }
    }
}

/// Single-Pen ownership at any byte/token step.
/// Enforces `|ActivePen(t)| == 1`: deterministic `H = 0` framing belongs strictly to `HarnessPen`,
/// mode selection at `[MODE:` belongs to `RouterPen`, and semantic plan/payload generation belongs to `PayloadPen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PenOwner {
    HarnessPen,
    RouterPen,
    PayloadPen,
}

/// Active cartridge bank mounted during a given span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CartridgeBank {
    Base,
    OrientTrunk,
    ModePayload(Mode),
}

impl CartridgeBank {
    /// Canonical runtime wire key (`"Base"`, `"OrientTrunk"`, `"ModePayload:<Mode>"`).
    #[must_use]
    pub const fn wire_key(self) -> &'static str {
        match self {
            CartridgeBank::Base => "Base",
            CartridgeBank::OrientTrunk => "OrientTrunk",
            CartridgeBank::ModePayload(Mode::Probe) => "ModePayload:Probe",
            CartridgeBank::ModePayload(Mode::Read) => "ModePayload:Read",
            CartridgeBank::ModePayload(Mode::Ask) => "ModePayload:Ask",
            CartridgeBank::ModePayload(Mode::Write) => "ModePayload:Write",
            CartridgeBank::ModePayload(Mode::Result) => "ModePayload:Result",
            CartridgeBank::ModePayload(Mode::Abstain) => "ModePayload:Abstain",
        }
    }

    /// Default single-pen owner associated with this dynamic bank.
    #[must_use]
    pub const fn default_pen_owner(self) -> PenOwner {
        match self {
            CartridgeBank::Base => PenOwner::HarnessPen,
            CartridgeBank::OrientTrunk => PenOwner::RouterPen,
            CartridgeBank::ModePayload(_) => PenOwner::PayloadPen,
        }
    }
}

/// Deterministic `HarnessPen` prefill segment (`H = 0` host-known context region) mounted via
/// [`MountPolicy::PrefillSegment`] so `ASID` memory tags are stamped by bank switching with zero
/// gate-classifier false positives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrefillSegmentKind {
    /// `<schema>CREATE TABLE ...</schema>` authoritative symbol-table prefill (`"Prefill:Schema"`).
    Schema,
    /// `<user>...</user>` untrusted prompt / presupposition prefill (`"Prefill:User"`).
    User,
    /// `<tool_response>...</tool_response>` execution-verified observation prefill (`"Prefill:ToolObs"`).
    ToolObs,
}

impl PrefillSegmentKind {
    #[must_use]
    pub const fn wire_key(self) -> &'static str {
        match self {
            Self::Schema => "Prefill:Schema",
            Self::User => "Prefill:User",
            Self::ToolObs => "Prefill:ToolObs",
        }
    }

    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "SCHEMA" | "AUTHORITATIVESCHEMA" => Some(Self::Schema),
            "USER" | "PROMPT" => Some(Self::User),
            "TOOLOBS" | "TOOL_OBS" | "OBSERVATION" | "TOOLRESPONSE" => Some(Self::ToolObs),
            _ => None,
        }
    }
}

/// Monotonic per-task Steward governance state machine (`NotRequired -> PendingAsk -> {Granted | Denied}`).
///
/// Eliminates illegal `(observed=false, approved=true)` boolean combinations and prevents subsequent
/// `Mode::Read` turns from accidentally clearing a latched `PendingAsk` requirement after an `"ASK"`
/// row observation or `SAVEPOINT` rollback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StewardAuthState {
    #[default]
    NotRequired,
    PendingAsk,
    Granted,
    Denied,
}

impl StewardAuthState {
    #[inline]
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::PendingAsk => "pending_ask",
            Self::Granted => "granted",
            Self::Denied => "denied",
        }
    }

    #[inline]
    #[must_use]
    pub const fn requires_ask(self) -> bool {
        matches!(self, Self::PendingAsk)
    }

    #[inline]
    #[must_use]
    pub const fn blocks_write(self) -> bool {
        matches!(self, Self::PendingAsk | Self::Denied)
    }

    #[inline]
    #[must_use]
    pub const fn is_granted(self) -> bool {
        matches!(self, Self::Granted)
    }

    /// Monotonically latch a discovered Steward `ASK` requirement (from either a `Mode::Read`
    /// row or a `Mode::Write` `StewardAuthorizationViolation` rollback) without clobbering an
    /// already-resolved `Granted` or `Denied` decision for the current task.
    #[inline]
    #[must_use]
    pub const fn latch_ask_requirement(self) -> Self {
        match self {
            Self::NotRequired | Self::PendingAsk => Self::PendingAsk,
            Self::Granted => Self::Granted,
            Self::Denied => Self::Denied,
        }
    }

    /// Record an explicit Steward decision (`ALLOW` -> `Granted`, `DENY` -> `Denied`).
    #[inline]
    #[must_use]
    pub const fn record_decision(self, granted: bool) -> Self {
        if granted {
            Self::Granted
        } else {
            Self::Denied
        }
    }
}

#[derive(Debug, Error)]
pub enum CoreProtocolError {
    #[error("invalid mode label: {0:?}")]
    InvalidModeLabel(String),
    #[error("single-pen handoff violation at step {step}: {detail}")]
    SinglePenViolation { step: usize, detail: String },
    #[error("unexpected framing or boundary token: {0}")]
    BoundaryViolation(String),
}
