//! Session outcome accounting (`SessionSummaryReceipt`, `SessionOutcomeStatus`,
//! `SessionRunError`), cycle detection (`TurnCycleDetector`), and Pretrain-LOTL envelopes.

use crate::driver::{AskInteractionMode, DriverObservation, LiveSqliteDailyDriver};
use crate::model_identity::RuntimeProvenance;
use anyhow::Result;
use portia_core::relay::TypedTurnAction;
use portia_core::{Mode, RouteTag, StewardAuthState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Strongly-typed single-turn execution record kept intact until boundary serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEvent {
    pub mode: Mode,
    pub route: RouteTag,
    pub plan: String,
    pub payload: String,
    pub action: TypedTurnAction,
    pub observation: DriverObservation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_interaction_mode: Option<AskInteractionMode>,
}

/// First-class completion status for a multi-turn `portia-cli` session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionOutcomeStatus {
    Completed,
    Abstained,
    IncompleteTurnLimit,
    FailedAfterChanges,
    FailedNoChanges,
}

impl SessionOutcomeStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Abstained => "abstained",
            Self::IncompleteTurnLimit => "incomplete_turn_limit",
            Self::FailedAfterChanges => "failed_after_changes",
            Self::FailedNoChanges => "failed_no_changes",
        }
    }
}

impl std::fmt::Display for SessionOutcomeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Compact outcome summary receipt separating the final answer and database write accounting
/// from the full multi-turn execution transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummaryReceipt {
    #[serde(default)]
    pub provenance: RuntimeProvenance,
    pub status: SessionOutcomeStatus,
    pub turns_executed: usize,
    pub writes_attempted: usize,
    pub writes_committed: usize,
    pub final_answer: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_reason: Option<SessionRunError>,
}

/// A repeated action is an observable no-progress failure, not a proof that model KV is fixed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionRunError {
    #[error("TokenBudgetExceeded: emitted {tokens_generated}/{token_budget} tokens in {phase_at_exhaustion}")]
    TokenBudgetExceeded {
        tokens_generated: usize,
        token_budget: usize,
        phase_at_exhaustion: String,
        partial_text: String,
    },
    #[error("RepeatedTurn: turn index {repeated_turn} repeats index {first_turn} ({mode:?}) at mutation version {state_version}: {payload}")]
    RepeatedTurn {
        first_turn: usize,
        repeated_turn: usize,
        mode: Mode,
        payload: String,
        state_version: u64,
    },
    #[error("Cancelled: interrupted after {tokens_generated} tokens in {phase_at_cancel}")]
    Cancelled {
        tokens_generated: usize,
        phase_at_cancel: String,
    },
}

#[derive(Default)]
pub(super) struct TurnCycleDetector {
    epoch: Option<(u64, StewardAuthState)>,
    seen: BTreeMap<(Mode, String), usize>,
}

impl TurnCycleDetector {
    pub(super) fn new_input(&mut self) {
        self.seen.clear();
    }

    pub(super) fn allow_post_write_verification_reads(&mut self) {
        self.seen.retain(|(mode, _), _| *mode != Mode::Read);
    }

    pub(super) fn observe(
        &mut self,
        mode: Mode,
        payload: &str,
        version: u64,
        steward_auth: StewardAuthState,
        turn: usize,
    ) -> std::result::Result<(), SessionRunError> {
        let current_epoch = (version, steward_auth);
        if self.epoch != Some(current_epoch) {
            self.seen.clear();
            self.epoch = Some(current_epoch);
        }
        let key = (mode, payload.trim().to_string());
        if let Some(&first_turn) = self.seen.get(&key) {
            return Err(SessionRunError::RepeatedTurn {
                first_turn,
                repeated_turn: turn,
                mode,
                payload: key.1,
                state_version: version,
            });
        }
        self.seen.insert(key, turn);
        Ok(())
    }
}

pub fn attach_session_error(receipt: &mut SessionSummaryReceipt, error: Option<&anyhow::Error>) {
    receipt.termination_reason = error
        .and_then(|e| e.downcast_ref::<SessionRunError>())
        .cloned();
}

/// Compute a [`SessionSummaryReceipt`] directly from a typed [`SessionEvent`] slice without
/// `serde_json::Value` string-key recovery.
pub fn summarize_session_outcome(
    events: &[SessionEvent],
    had_error: bool,
    db_version_before: u64,
    db_version_after: u64,
) -> SessionSummaryReceipt {
    let turns_executed = events.len();
    let mut writes_attempted = 0usize;
    let mut writes_committed = 0usize;

    for ev in events {
        if ev.mode == Mode::Write {
            writes_attempted += 1;
            if ev.observation.ok && ev.observation.rows_affected > 0 {
                writes_committed += 1;
            }
        }
    }

    let last_event = events.last();
    let is_result =
        last_event.is_some_and(|ev| ev.mode == Mode::Result || ev.observation.is_chat_continuation);
    let is_abstain = last_event.is_some_and(|ev| ev.mode == Mode::Abstain);
    let reached_terminal = is_result || is_abstain;
    // SQLite counts matched rows even for an idempotent UPDATE. A committed
    // statement alone therefore does not establish that database contents changed.
    let db_mutated = db_version_after > db_version_before;

    let status = if !had_error && is_result {
        SessionOutcomeStatus::Completed
    } else if !had_error && is_abstain {
        SessionOutcomeStatus::Abstained
    } else if (had_error || !reached_terminal) && db_mutated {
        SessionOutcomeStatus::FailedAfterChanges
    } else if !reached_terminal && !had_error {
        SessionOutcomeStatus::IncompleteTurnLimit
    } else {
        SessionOutcomeStatus::FailedNoChanges
    };

    let final_answer = if reached_terminal {
        last_event.map(|ev| match &ev.action {
            TypedTurnAction::SubmitResult { report_json } => report_json.clone(),
            TypedTurnAction::AbstainTask { abstain_json } => abstain_json.clone(),
            _ => serde_json::from_str::<serde_json::Value>(&ev.payload)
                .unwrap_or_else(|_| serde_json::Value::String(ev.payload.clone())),
        })
    } else {
        None
    };

    SessionSummaryReceipt {
        provenance: RuntimeProvenance::default(),
        status,
        turns_executed,
        writes_attempted,
        writes_committed,
        final_answer,
        termination_reason: None,
    }
}

/// Emit a single-line human-legible outcome summary to `stderr`.
pub fn emit_session_outcome_stderr(receipt: &SessionSummaryReceipt) {
    let answer_compact = receipt
        .final_answer
        .as_ref()
        .and_then(|v| serde_json::to_string(v).ok())
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "[portia-cli] outcome={} turns={} writes_attempted={} writes_committed={} answer={}",
        receipt.status,
        receipt.turns_executed,
        receipt.writes_attempted,
        receipt.writes_committed,
        answer_compact
    );
}

/// Format a user prompt into the canonical Pretrain-LOTL ChatML envelope
/// (`<|im_start|>user\n{user_prompt}<|im_end|>\n`), with zero `<|im_start|>system` turn,
/// zero upfront schema/AST dump, and zero English behavioral rules.
pub fn format_lotl_user_prompt_envelope(user_prompt: &str) -> String {
    format!("<|im_start|>user\n{user_prompt}<|im_end|>\n")
}

/// Build the initial Pretrain-LOTL ChatML prompt (`<|im_start|>user\n{user_prompt}<|im_end|>\n`),
/// with zero `<|im_start|>system` turn, zero upfront `[Table ...]` schema dump, and zero English behavioral rules!
pub fn build_lotl_initial_prompt(
    _driver: &LiveSqliteDailyDriver,
    user_prompt: &str,
) -> Result<String> {
    Ok(format_lotl_user_prompt_envelope(user_prompt))
}

/// Build a follow-up ChatML user prompt (`<|im_start|>user\n{user_prompt}<|im_end|>\n`)
/// for subsequent turns in a stateful multi-prompt REPL or TUI session.
pub fn build_lotl_followup_prompt(
    _driver: &LiveSqliteDailyDriver,
    user_prompt: &str,
) -> Result<String> {
    Ok(format_lotl_user_prompt_envelope(user_prompt))
}
