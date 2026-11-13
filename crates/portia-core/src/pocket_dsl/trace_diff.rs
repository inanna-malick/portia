//! Differential student-vs-reference divergence classifications and extracted compiler obligations.

use crate::{CartridgeBank, EnvelopeKind, Mode, PenOwner};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Semantic sum type classifying how a student turn relates to a coalgebraic reference step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceClassification {
    /// 100% agreement between `W_0`/student and the Pocket State Machine (`0` obligations emitted).
    LaminarAgreement,
    /// Student failed to emit `'['` (`58`) at cursor `0` of the turn (`TurnStartProseEscape`).
    TurnStartProseEscape {
        cursor_offset: usize,
        observed_token_id: u32,
    },
    /// Student emitted the wrong mode token at `[MODE:` (`ModeDecisionFork`).
    ModeDecisionFork {
        cursor_offset: usize,
        expected_mode: Mode,
        observed_mode: Option<Mode>,
    },
    /// Student emitted the right mode token at `[MODE:`, then drifted mid-`Plan:` or into the
    /// wrong `<to_*>` route (`WholeSpanStanceDrift`).
    WholeSpanStanceDrift {
        active_mode: Mode,
        span_range: Range<usize>,
        trigger_marker: String,
    },
    /// Mode, `Plan:`, and `<to_*>` route matched, but payload diverged at a specific SQL/JSON
    /// lexeme offset inside `payload_span` (`PayloadLexemeFork`).
    PayloadLexemeFork {
        cursor_offset: usize,
        expected_lexeme: String,
        observed_lexeme: String,
    },
}

/// Compiler obligation extracted from differential student-vs-reference trace comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedDslObligation {
    pub obligation_id: String,
    pub world_id: String,
    pub contrasting_worlds: Vec<String>,
    pub turn_index: usize,
    pub bank: CartridgeBank,
    pub pen: PenOwner,
    pub envelope: EnvelopeKind,
    pub target_mode: Mode,
    pub positive_token_id: Option<u32>,
    pub negative_token_id: Option<u32>,
    pub positive_cursor_offsets: Vec<usize>,
    /// Intra-turn continuation offsets clamped $\le -8.0$ so impulse diodes never re-fire mid-turn.
    pub negative_continuation_offsets: Vec<usize>,
    pub classification: DivergenceClassification,
}
