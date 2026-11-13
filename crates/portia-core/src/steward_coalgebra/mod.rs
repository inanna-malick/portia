//! Comonadic Steward Negotiation Coalgebra (`FreeAgentProgram<A>` & `EnvObservation`)
//! and Monotonic `PocketObsState` Observation Reducer (`apply_env_observation_to_pocket_state`).

mod free_agent;
mod unfold;

#[cfg(test)]
mod tests;

use crate::pocket_dsl::PocketObsState;
use serde::{Deserialize, Serialize};

pub use free_agent::{FreeAgent, FreeAgentFrame, FreeAgentProgram};
pub use unfold::unfold_regime_into_free_agent;

/// Outcome of a comonadic steward negotiation step inside `<tool_response>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StewardNegotiationOutcome {
    /// Steward authorizes the Turn-0 user prompt's original parameters unchanged.
    UnconditionalAllow { token: String },
    /// Steward authorizes the mutation **only with counterfactual overridden parameters**
    /// (`ALLOW_WITH_OVERRIDE`).
    ConditionalAllowWithOverride {
        token: String,
        overridden_value: Option<i64>,
        overridden_note: Option<String>,
        overridden_target_id: Option<i64>,
        reason: String,
    },
    /// Multi-step steward negotiation: steward requires the agent to execute an additional
    /// `READ` verifying `required_predicate` before granting authorization.
    RequestAdditionalRead { required_predicate: String },
    /// Disambiguation clarification response (`[CLARIFY]`): narrows multi-candidate ambiguity to
    /// `selected_target_id` **without** granting mutation authorization or minting an auth token.
    ClarifiedCandidateSelection {
        selected_target_id: i64,
        note: String,
    },
    /// Steward explicitly denies the mutation (`DENY` -> agent must `ABSTAIN` with zero writes).
    ExplicitDeny { token: String, reason: String },
}

impl StewardNegotiationOutcome {
    /// Render the canonical `<tool_response>` wire text emitted by the steward.
    pub fn render_wire_response(
        &self,
        steward_code: &str,
        value_col: &str,
        note_col: &str,
        prompt_value: i64,
    ) -> String {
        match self {
            StewardNegotiationOutcome::UnconditionalAllow { token } => format!(
                "{{\"decision\": \"ALLOW\", \"steward\": \"{steward_code}\", \"token\": \"{token}\"}}"
            ),
            StewardNegotiationOutcome::ConditionalAllowWithOverride {
                token,
                overridden_value,
                overridden_note,
                overridden_target_id,
                reason,
            } => {
                let mut clauses = Vec::new();
                if let Some(id_ov) = overridden_target_id {
                    clauses.push(format!("target_id = {id_ov}"));
                }
                if let Some(val_ov) = overridden_value {
                    clauses.push(format!(
                        "`{value_col}` = {val_ov} (capped from {prompt_value})"
                    ));
                }
                if let Some(note_ov) = overridden_note {
                    clauses.push(format!("`{note_col}` = '{note_ov}'"));
                }
                let joined = clauses.join(" and ");
                format!(
                    "{{\"decision\": \"ALLOW_WITH_OVERRIDE\", \"steward\": \"{steward_code}\", \
                     \"token\": \"{token}\", \"reason\": \"{reason}\", \
                     \"directive\": \"Steward {steward_code} authorizes the update under token \
                     {token} ONLY WITH MODIFIED PARAMETERS: set {joined}.\"}}"
                )
            }
            StewardNegotiationOutcome::RequestAdditionalRead { required_predicate } => format!(
                "{{\"decision\": \"REQUIRE_ADDITIONAL_READ\", \"steward\": \"{steward_code}\", \
                 \"required_predicate\": \"{required_predicate}\"}}"
            ),
            StewardNegotiationOutcome::ClarifiedCandidateSelection {
                selected_target_id,
                note,
            } => format!(
                "{{\"decision\": \"CLARIFIED_CANDIDATE\", \"steward\": \"{steward_code}\", \
                 \"selected_target_id\": {selected_target_id}, \"note\": \"{note}\"}}"
            ),
            StewardNegotiationOutcome::ExplicitDeny { token, reason } => format!(
                "{{\"decision\": \"DENY\", \"steward\": \"{steward_code}\", \
                 \"token\": \"{token}\", \"reason\": \"{reason}\"}}"
            ),
        }
    }
}

/// Functor $F = \text{AgentActionF}$ of agent actions in a multi-turn SQL-RW episode.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentActionStep {
    /// Turn-0 candidate discovery `SELECT` (`READ`).
    ReadCandidate { predicate: String },
    /// Secondary `SELECT` (`READ`) requested by a `RequestAdditionalRead` steward reply.
    ReadAdditional { predicate: String },
    /// Steward inquiry (`ASK`) proposing `(proposed_target_id, proposed_value, proposed_note)`.
    AskSteward {
        question: String,
        proposed_target_id: i64,
        proposed_value: i64,
        proposed_note: String,
    },
    /// Database mutation (`WRITE`: `UPDATE` / `DELETE` / `INSERT`).
    WriteMutation {
        target_id: i64,
        mutated_value: i64,
        mutated_note: String,
        auth_token: Option<String>,
    },
    /// Post-write verification `SELECT` (`READ_VERIFY`).
    ReadVerify { target_id: i64 },
    /// Terminal `RESULT` / `REPORT` submission.
    TerminalResult {
        reported_target_id: i64,
        reported_value: i64,
        reported_note: String,
        auth_token: Option<String>,
    },
    /// Terminal `ABSTAIN` submission.
    TerminalAbstain { reason: String },
}

/// Observation emitted by the environment functor $G = \text{EnvResponseF}$ at each step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvObservation {
    CandidateRow {
        target_id: i64,
        current_value: i64,
        current_note: String,
        policy_requires_ask: bool,
        steward_code: String,
    },
    AdditionalRow {
        predicate: String,
        satisfied: bool,
    },
    StewardReply {
        outcome: StewardNegotiationOutcome,
        wire_message: String,
    },
    WriteAck {
        rows_affected: usize,
        applied_target_id: i64,
        applied_value: i64,
        applied_note: String,
    },
    VerifyRow {
        target_id: i64,
        verified_value: i64,
        verified_note: String,
    },
    TerminalAck,
}

/// Apply an [`EnvObservation`] emitted by the environment coalgebra to a [`PocketObsState`],
/// producing the successor epistemic state for the next turn's [`crate::pocket_dsl::RegimeExpr`] evaluation.
pub fn apply_env_observation_to_pocket_state(
    state: &PocketObsState,
    obs: &EnvObservation,
) -> PocketObsState {
    let mut next = state.clone();
    next.turn_index += 1;
    match obs {
        EnvObservation::CandidateRow {
            policy_requires_ask,
            ..
        } => {
            next.has_schema_probe = true;
            if next.candidate_count.is_none() {
                next.candidate_count = Some(1);
            }
            next.policy_requires_ask = *policy_requires_ask;
        }
        EnvObservation::AdditionalRow {
            predicate,
            satisfied,
        } => {
            if predicate.trim().to_ascii_uppercase().starts_with("PRAGMA ") {
                if *satisfied {
                    next.has_schema_probe = true;
                }
            } else if *satisfied && next.candidate_count.is_none() {
                next.candidate_count = Some(1);
            }
        }
        EnvObservation::StewardReply { outcome, .. } => match outcome {
            StewardNegotiationOutcome::ClarifiedCandidateSelection { .. } => {
                next.candidate_count = Some(1);
            }
            StewardNegotiationOutcome::UnconditionalAllow { .. }
            | StewardNegotiationOutcome::ConditionalAllowWithOverride { .. } => {
                if next.steward_grant_approved.is_none() {
                    next.steward_grant_approved = Some(true);
                    next.is_locked_or_denied = false;
                }
            }
            StewardNegotiationOutcome::RequestAdditionalRead { .. } => {
                if next.applied_write_count == 0 {
                    next.candidate_count = None;
                }
            }
            StewardNegotiationOutcome::ExplicitDeny { .. } => {
                if next.steward_grant_approved.is_none() {
                    next.steward_grant_approved = Some(false);
                    next.is_locked_or_denied = true;
                }
            }
        },
        EnvObservation::WriteAck { rows_affected, .. } => {
            next.applied_write_count += 1;
            if *rows_affected > 0 {
                next.has_post_write_verification = false;
                next.zero_row_toctou_race = false;
            } else {
                next.zero_row_toctou_race = true;
            }
        }
        EnvObservation::VerifyRow { .. } => {
            next.has_post_write_verification = true;
        }
        EnvObservation::TerminalAck => {}
    }
    next
}
