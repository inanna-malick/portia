//! Phase states, single-pen span records, typed turn actions, and completed turn outputs.

use crate::{CartridgeBank, CoreProtocolError, Mode, PenOwner, RouteTag};
use serde::{Deserialize, Serialize};

/// Exhaustive phase state of a single assistant turn inside [`super::RelayMachine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelayPhase {
    /// Waiting for `HarnessPen` to prefill `<|im_start|>assistant\n<think>\n[MODE:`.
    TurnOpenPrefill,
    /// `RouterPen` is emitting the mode label and closing `]` under `CartridgeBank::OrientTrunk`.
    ModeLabel,
    /// `]` has been consumed under `OrientTrunk`; `HarnessPen` now fast-forwards `\nPlan:`
    /// starting at `\n` (`tid = 198`) under `CartridgeBank::Base`.
    PostModePrefill,
    /// `PayloadPen` is generating the single-line `Plan:` text up to `\n` under `CartridgeBank::Base`.
    PlanBody,
    /// `Plan:` newline reached; `HarnessPen` fast-forwards `</think>\n<sql>\n` or `</think>\n<to_user>\n` under `ModePayload(mode)`.
    PostPlanPrefill,
    /// `PayloadPen` is generating the SQL/ASK/JSON payload up to `</sql>` or `</to_user>` under `ModePayload(mode)`.
    PayloadBody,
    /// Closing `</sql>` or `</to_user>` reached; `HarnessPen` appends `<|im_end|>\n` under `CartridgeBank::Base`.
    TurnComplete,
}

/// Recorded step in a turn's Single-Pen schedule trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PenSpanRecord {
    pub step_index: usize,
    pub phase: RelayPhase,
    pub pen: PenOwner,
    /// Cartridge bank under which the model produced this chunk (for `RouterPen`/`PayloadPen`)
    /// or `None` for deterministic `H = 0` `HarnessPen` prefill.
    pub producer_bank: Option<CartridgeBank>,
    /// Cartridge bank mounted when consuming these tokens into the KV / DeltaNet state.
    pub consumer_bank: CartridgeBank,
    pub text: String,
    pub token_ids: Vec<u32>,
}

/// Executable two-phase handoff decision produced by [`super::RelayMachine::plan_sampled_trunk_token`]
/// **after** a trunk token is sampled from `last_logits` (under `producer_bank`) and **before**
/// that token is forwarded into the runtime KV / DeltaNet cache (`consume_bank`).
///
/// When `token_piece` completes a closing route tag (`</sql>` or `</to_user>`), `consume_bank`
/// transitions immediately to [`CartridgeBank::Base`] so the worker detaches `ModePayload(mode)`
/// *before* running the forward pass that consumes the closing token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedTrunkStep {
    pub producer_bank: CartridgeBank,
    pub consume_bank: CartridgeBank,
    pub phase_after: RelayPhase,
    pub span: PenSpanRecord,
}

/// Semantic sum type (`TypedTurnAction`) binding each [`Mode`] to its parsed domain payload
/// so invalid JSON on `Mode::Result`/`Mode::Abstain` or empty SQL on `Probe`/`Read`/`Write`
/// is rejected at the protocol boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypedTurnAction {
    ProbeSchema { sql: String },
    ReadQuery { sql: String },
    AskSteward { question: String },
    WriteMutation { sql: String },
    SubmitResult { report_json: serde_json::Value },
    AbstainTask { abstain_json: serde_json::Value },
}

impl TypedTurnAction {
    pub fn parse_for_mode(mode: Mode, raw_payload: &str) -> Result<Self, CoreProtocolError> {
        let trimmed = raw_payload.trim();
        if trimmed.is_empty() {
            return Err(CoreProtocolError::BoundaryViolation(format!(
                "empty payload for mode {}",
                mode
            )));
        }
        match mode {
            Mode::Probe => Ok(TypedTurnAction::ProbeSchema {
                sql: trimmed.to_string(),
            }),
            Mode::Read => Ok(TypedTurnAction::ReadQuery {
                sql: trimmed.to_string(),
            }),
            Mode::Ask => Ok(TypedTurnAction::AskSteward {
                question: trimmed.to_string(),
            }),
            Mode::Write => Ok(TypedTurnAction::WriteMutation {
                sql: trimmed.to_string(),
            }),
            Mode::Result => {
                if trimmed.starts_with('{') {
                    let report_json: serde_json::Value =
                        serde_json::from_str(trimmed).map_err(|e| {
                            CoreProtocolError::BoundaryViolation(format!(
                                "invalid JSON in RESULT payload: {}",
                                e
                            ))
                        })?;
                    if !report_json.is_object() {
                        return Err(CoreProtocolError::BoundaryViolation(
                            "RESULT payload must be a JSON object".to_string(),
                        ));
                    }
                    Ok(TypedTurnAction::SubmitResult { report_json })
                } else {
                    Ok(TypedTurnAction::SubmitResult {
                        report_json: serde_json::json!({ "summary": trimmed }),
                    })
                }
            }
            Mode::Abstain => {
                if trimmed.starts_with('{') {
                    let abstain_json: serde_json::Value =
                        serde_json::from_str(trimmed).map_err(|e| {
                            CoreProtocolError::BoundaryViolation(format!(
                                "invalid JSON in ABSTAIN payload: {}",
                                e
                            ))
                        })?;
                    if !abstain_json.is_object() {
                        return Err(CoreProtocolError::BoundaryViolation(
                            "ABSTAIN payload must be a JSON object".to_string(),
                        ));
                    }
                    Ok(TypedTurnAction::AbstainTask { abstain_json })
                } else {
                    Ok(TypedTurnAction::AbstainTask {
                        abstain_json: serde_json::json!({ "reason": trimmed }),
                    })
                }
            }
        }
    }

    pub const fn mode(&self) -> Mode {
        match self {
            TypedTurnAction::ProbeSchema { .. } => Mode::Probe,
            TypedTurnAction::ReadQuery { .. } => Mode::Read,
            TypedTurnAction::AskSteward { .. } => Mode::Ask,
            TypedTurnAction::WriteMutation { .. } => Mode::Write,
            TypedTurnAction::SubmitResult { .. } => Mode::Result,
            TypedTurnAction::AbstainTask { .. } => Mode::Abstain,
        }
    }
}

/// Fine-grained event emitted during [`super::RelayMachine`] execution so interactive TUI and CLI
/// frontends can render every prefill chunk, mode selection, and generated token in real time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelayStreamEvent {
    /// Deterministic `H = 0` `HarnessPen` prefill chunk consumed (`[MODE:`, `\nPlan:`,
    /// `</think>\n<sql>\n` / `</think>\n<to_user>\n`, or `<|im_end|>\n`).
    HarnessPrefillSpan { span: PenSpanRecord },
    /// Single token (or healed piece) stepped by `RouterPen` (`ModeLabel`) or `PayloadPen`
    /// (`PlanBody` / `PayloadBody`).
    TrunkTokenStepped {
        phase_after: RelayPhase,
        selected_mode: Option<Mode>,
        piece: String,
        span: PenSpanRecord,
        plan_so_far: String,
        payload_so_far: String,
    },
}

/// Parsed, verified turn produced once [`super::RelayMachine`] reaches [`RelayPhase::TurnComplete`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedRelayTurn {
    pub mode: Mode,
    pub route: RouteTag,
    pub plan: String,
    pub payload: String,
    pub action: TypedTurnAction,
    pub full_wire_text: String,
    pub spans: Vec<PenSpanRecord>,
}
