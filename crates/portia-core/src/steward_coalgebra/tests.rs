use super::*;
use crate::pocket_dsl::{candidates_observed, emit_mode, has_schema_probe, PocketObsState};
use crate::Mode;

#[test]
fn unfold_regime_into_free_agent_and_apply_env_observation_preserve_monotonic_invariants() {
    let regime = (!has_schema_probe() >> Mode::Probe)
        | (!candidates_observed() >> Mode::Read)
        | emit_mode(Mode::Result);
    let init = PocketObsState {
        world_id: "W0".to_string(),
        turn_index: 0,
        table_name: "incidents".to_string(),
        entity_name: "INC-1".to_string(),
        operation: "SELECT".to_string(),
        has_schema_probe: false,
        candidate_count: None,
        policy_requires_ask: false,
        is_locked_or_denied: false,
        steward_grant_approved: None,
        applied_write_count: 0,
        has_post_write_verification: false,
        zero_row_toctou_race: false,
    };
    let prog = FreeAgent::from_regime_expr(&regime, init.clone(), 6);
    let (steps, term) = prog.into_steps_and_terminal();
    assert_eq!(term, Mode::Result);
    assert_eq!(steps.len(), 3);

    let s1 = apply_env_observation_to_pocket_state(
        &init,
        &EnvObservation::StewardReply {
            outcome: StewardNegotiationOutcome::UnconditionalAllow {
                token: "TOK-1".to_string(),
            },
            wire_message: "ALLOW".to_string(),
        },
    );
    assert_eq!(s1.steward_grant_approved, Some(true));
    let s2 = apply_env_observation_to_pocket_state(
        &s1,
        &EnvObservation::StewardReply {
            outcome: StewardNegotiationOutcome::ExplicitDeny {
                token: "DENY".to_string(),
                reason: "late".to_string(),
            },
            wire_message: "DENY".to_string(),
        },
    );
    assert_eq!(s2.steward_grant_approved, Some(true));
}
