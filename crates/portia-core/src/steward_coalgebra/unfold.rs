//! Stack-safe hylomorphic unfolding (`unfold_regime_into_free_agent`) from [`RegimeExpr`]
//! into [`FreeAgent<Mode>`].

use super::{
    apply_env_observation_to_pocket_state, AgentActionStep, EnvObservation, FreeAgent,
    FreeAgentFrame, FreeAgentProgram, StewardNegotiationOutcome,
};
use crate::hylo::{expand_and_collapse, Expandable, PartiallyApplied};
use crate::pocket_dsl::{PocketObsState, RegimeExpr};
use crate::Mode;

#[derive(Debug, Clone)]
struct SimpleRegimeUnfoldSeed<'a> {
    regime: &'a RegimeExpr,
    obs: PocketObsState,
    remaining_turns: usize,
    halted_mode: Option<Mode>,
}

/// Unfold any authored [`RegimeExpr`] from `initial_obs` into a [`FreeAgent<Mode>`] (`FreeAgentProgram<Mode>`)
/// via a stack-safe [`expand_and_collapse`] hylomorphism over [`FreeAgentFrame<Mode, PartiallyApplied>`].
pub fn unfold_regime_into_free_agent(
    regime: &RegimeExpr,
    initial_obs: PocketObsState,
    max_turns: usize,
) -> FreeAgent<Mode> {
    let seed = SimpleRegimeUnfoldSeed {
        regime,
        obs: initial_obs,
        remaining_turns: max_turns,
        halted_mode: None,
    };
    expand_and_collapse::<
        FreeAgentFrame<Mode, PartiallyApplied>,
        SimpleRegimeUnfoldSeed<'_>,
        FreeAgent<Mode>,
    >(
        seed,
        |mut st| {
            if let Some(done_mode) = st.halted_mode {
                return FreeAgentFrame::Pure(done_mode);
            }
            if st.remaining_turns == 0 {
                let fallback = st
                    .regime
                    .specialize_for_current_turn(&st.obs)
                    .evaluate(&st.obs)
                    .selected_mode
                    .unwrap_or(Mode::Abstain);
                return FreeAgentFrame::Pure(fallback);
            }
            st.remaining_turns -= 1;

            let residual_regime = st.regime.specialize_for_current_turn(&st.obs);
            let mode = residual_regime
                .evaluate(&st.obs)
                .selected_mode
                .unwrap_or(Mode::Result);

            let (action, synthetic_obs) = match mode {
                Mode::Probe => (
                    AgentActionStep::ReadAdditional {
                        predicate: format!("PRAGMA table_info({})", st.obs.table_name),
                    },
                    EnvObservation::AdditionalRow {
                        predicate: format!("PRAGMA table_info({})", st.obs.table_name),
                        satisfied: true,
                    },
                ),
                Mode::Read => {
                    if st.obs.applied_write_count == 0 {
                        (
                            AgentActionStep::ReadCandidate {
                                predicate: format!(
                                    "SELECT * FROM {} WHERE entity='{}'",
                                    st.obs.table_name, st.obs.entity_name
                                ),
                            },
                            EnvObservation::CandidateRow {
                                target_id: 101,
                                current_value: 100,
                                current_note: "INIT".to_string(),
                                policy_requires_ask: st.obs.policy_requires_ask,
                                steward_code: "STEWARD".to_string(),
                            },
                        )
                    } else {
                        (
                            AgentActionStep::ReadVerify { target_id: 101 },
                            EnvObservation::VerifyRow {
                                target_id: 101,
                                verified_value: 200,
                                verified_note: "UPDATED".to_string(),
                            },
                        )
                    }
                }
                Mode::Ask => (
                    AgentActionStep::AskSteward {
                        question: format!(
                            "May we {} {} in {}?",
                            st.obs.operation, st.obs.entity_name, st.obs.table_name
                        ),
                        proposed_target_id: 101,
                        proposed_value: 200,
                        proposed_note: "UPDATED".to_string(),
                    },
                    EnvObservation::StewardReply {
                        outcome: StewardNegotiationOutcome::UnconditionalAllow {
                            token: "TOK-ALLOW".to_string(),
                        },
                        wire_message: "{\"decision\":\"ALLOW\"}".to_string(),
                    },
                ),
                Mode::Write => (
                    AgentActionStep::WriteMutation {
                        target_id: 101,
                        mutated_value: 200,
                        mutated_note: "UPDATED".to_string(),
                        auth_token: Some("TOK-ALLOW".to_string()),
                    },
                    EnvObservation::WriteAck {
                        rows_affected: 1,
                        applied_target_id: 101,
                        applied_value: 200,
                        applied_note: "UPDATED".to_string(),
                    },
                ),
                Mode::Result => {
                    st.halted_mode = Some(Mode::Result);
                    (
                        AgentActionStep::TerminalResult {
                            reported_target_id: 101,
                            reported_value: 200,
                            reported_note: "UPDATED".to_string(),
                            auth_token: None,
                        },
                        EnvObservation::TerminalAck,
                    )
                }
                Mode::Abstain => {
                    st.halted_mode = Some(Mode::Abstain);
                    (
                        AgentActionStep::TerminalAbstain {
                            reason: format!(
                                "Regime selected ABSTAIN for {} in {}",
                                st.obs.entity_name, st.obs.table_name
                            ),
                        },
                        EnvObservation::TerminalAck,
                    )
                }
            };

            st.obs = apply_env_observation_to_pocket_state(&st.obs, &synthetic_obs);
            FreeAgentFrame::Step { action, next: st }
        },
        FreeAgentProgram::from_frame,
    )
}
