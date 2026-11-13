use super::*;
use crate::hylo::CollapsibleExt;
use crate::Mode;
use proptest::prelude::*;

fn sample_obs_state() -> PocketObsState {
    PocketObsState {
        world_id: "B".to_string(),
        turn_index: 2,
        table_name: "client_ledgers".to_string(),
        entity_name: "Mira1".to_string(),
        operation: "UPDATE".to_string(),
        has_schema_probe: true,
        candidate_count: Some(2),
        policy_requires_ask: true,
        is_locked_or_denied: true,
        steward_grant_approved: None,
        applied_write_count: 0,
        has_post_write_verification: false,
        zero_row_toctou_race: false,
    }
}

#[test]
fn compositional_regime_edsl_evaluates_implicants_and_futamura_specializes_ast() {
    let dsl = canonical_pocket_regime_dsl();
    let obs = sample_obs_state();
    let eval = dsl.evaluate(&obs);
    assert_eq!(eval.selected_mode, Some(Mode::Ask));
    assert!(eval
        .active_implicants
        .contains(&(EpistemicAtom::PolicyRequiresAsk, true)));
    assert!(eval
        .active_implicants
        .contains(&(EpistemicAtom::StewardDecisionIs(None), true)));

    let insert_specialized = dsl.specialize_for_operation("INSERT");
    assert!(insert_specialized.ast_node_count() < dsl.ast_node_count());

    let mut insert_zero_cands = obs.clone();
    insert_zero_cands.operation = "INSERT".to_string();
    insert_zero_cands.candidate_count = Some(0);
    insert_zero_cands.policy_requires_ask = false;
    insert_zero_cands.is_locked_or_denied = false;
    assert_eq!(
        insert_specialized
            .evaluate(&insert_zero_cands)
            .selected_mode,
        Some(Mode::Write)
    );
    assert_eq!(
        dsl.evaluate(&insert_zero_cands).selected_mode,
        Some(Mode::Write)
    );

    assert_eq!(
        dsl.active_modes(),
        vec![
            Mode::Probe,
            Mode::Read,
            Mode::Abstain,
            Mode::Ask,
            Mode::Write,
            Mode::Result
        ]
    );
    let ro_disclosed = dsl.specialize_for_schema_invariants(true, false, false, true);
    assert_eq!(
        ro_disclosed.active_modes(),
        vec![Mode::Read, Mode::Abstain, Mode::Ask, Mode::Result]
    );
    let rw_no_steward = dsl.specialize_for_schema_invariants(false, false, false, true);
    let mut ambiguous_obs = obs.clone();
    ambiguous_obs.has_schema_probe = true;
    ambiguous_obs.candidate_count = Some(2);
    ambiguous_obs.policy_requires_ask = false;
    ambiguous_obs.is_locked_or_denied = false;
    ambiguous_obs.steward_grant_approved = None;
    assert_eq!(
        rw_no_steward.evaluate(&ambiguous_obs).selected_mode,
        Some(Mode::Ask)
    );
    assert_eq!(
        ro_disclosed.evaluate(&ambiguous_obs).selected_mode,
        Some(Mode::Ask)
    );

    let redundant = (!!has_schema_probe() & RegimeExpr::ConstBool(true)) >> Mode::Read;
    assert_eq!(redundant.simplify(), has_schema_probe() >> Mode::Read);

    let bool_or_mode = RegimeExpr::ConstBool(true) | emit_mode(Mode::Read);
    assert!(bool_or_mode.can_emit_mode());
    assert_eq!(bool_or_mode.evaluate(&obs).selected_mode, Some(Mode::Read));
    assert_eq!(
        bool_or_mode.simplify().evaluate(&obs).selected_mode,
        Some(Mode::Read)
    );

    let rules = dsl.extract_guarded_rules();
    assert_eq!(rules.len(), 11);
    assert_eq!(rules[0].rule_id, "rule_00_probe");
    assert_eq!(rules[0].mode, Mode::Probe);
    assert_eq!(rules[0].guard, !has_schema_probe());
    assert_eq!(rules[10].rule_id, "rule_10_result");
    assert_eq!(rules[10].mode, Mode::Result);
    assert_eq!(rules[10].guard, RegimeExpr::ConstBool(true));

    let neg_mode = RegimeExpr::ConstBool(true) & !emit_mode(Mode::Read);
    assert!(!neg_mode.can_emit_mode());
    assert_eq!(
        neg_mode.evaluate(&obs),
        RegimeEvaluation {
            truth: false,
            selected_mode: None,
            active_implicants: vec![],
        }
    );
    assert_eq!(neg_mode.simplify(), RegimeExpr::ConstBool(false));
    assert_eq!(neg_mode.simplify().evaluate(&obs), neg_mode.evaluate(&obs));

    let right_and = (has_schema_probe() >> Mode::Read) & policy_requires_ask();
    let right_rules = right_and.extract_guarded_rules();
    assert_eq!(right_rules.len(), 1);
    assert_eq!(
        right_rules[0].guard,
        has_schema_probe() & policy_requires_ask()
    );
    assert_eq!(right_rules[0].mode, Mode::Read);

    let clarify_ro_dsl = (!candidates_observed() >> Mode::Read)
        | (candidate_count_eq(2) >> Mode::Ask)
        | ((policy_requires_ask() & steward_decision_is(None)) >> Mode::Ask)
        | (writes_applied_eq(0) >> Mode::Write)
        | emit_mode(Mode::Result);
    let clarify_ro_spec = clarify_ro_dsl.specialize_for_schema_invariants(true, false, false, true);
    assert_eq!(
        clarify_ro_spec.active_modes(),
        vec![Mode::Read, Mode::Ask, Mode::Result]
    );
    let mut two_cand_obs = obs.clone();
    two_cand_obs.candidate_count = Some(2);
    two_cand_obs.policy_requires_ask = false;
    assert_eq!(
        clarify_ro_spec.evaluate(&two_cand_obs).selected_mode,
        Some(Mode::Ask)
    );

    let two_write_dsl = (writes_applied_eq(1) >> Mode::Write) | emit_mode(Mode::Result);
    let mut turn0 = obs.clone();
    turn0.has_schema_probe = false;
    turn0.candidate_count = None;
    turn0.policy_requires_ask = false;
    turn0.is_locked_or_denied = false;
    turn0.steward_grant_approved = None;
    turn0.applied_write_count = 0;
    turn0.has_post_write_verification = false;
    turn0.zero_row_toctou_race = false;

    let mut turn1 = turn0.clone();
    turn1.has_schema_probe = true;
    turn1.candidate_count = Some(1);

    let mut turn2 = turn1.clone();
    turn2.applied_write_count = 1;

    let mut turn2_second_write = turn2.clone();
    turn2_second_write.applied_write_count = 2;

    let residual_at_w1 = two_write_dsl.specialize_monotonic_invariants(&turn2);
    assert_eq!(
        residual_at_w1.evaluate(&turn2).selected_mode,
        Some(Mode::Write)
    );
    assert_eq!(
        residual_at_w1.evaluate(&turn2_second_write).selected_mode,
        two_write_dsl.evaluate(&turn2_second_write).selected_mode
    );
    assert_eq!(
        residual_at_w1.evaluate(&turn2_second_write).selected_mode,
        Some(Mode::Result)
    );

    let mut turn3 = turn2.clone();
    turn3.has_post_write_verification = true;

    let ask_on_cand_dsl = (candidates_observed() >> Mode::Ask) | emit_mode(Mode::Result);
    let residual_pre_write = ask_on_cand_dsl.specialize_monotonic_invariants(&turn1);
    let after_req_read = crate::steward_coalgebra::apply_env_observation_to_pocket_state(
        &turn1,
        &crate::steward_coalgebra::EnvObservation::StewardReply {
            outcome: crate::steward_coalgebra::StewardNegotiationOutcome::RequestAdditionalRead {
                required_predicate: "SELECT * FROM client_ledgers WHERE active=1".to_string(),
            },
            wire_message: "{}".to_string(),
        },
    );
    assert_eq!(after_req_read.candidate_count, None);
    assert!(residual_pre_write
        .evaluate(&after_req_read)
        .decision_eq(&ask_on_cand_dsl.evaluate(&after_req_read)));
    assert_eq!(
        residual_pre_write.evaluate(&after_req_read).selected_mode,
        Some(Mode::Result)
    );

    let ask_after_allow_dsl =
        (steward_decision_is(Some(true)) >> Mode::Ask) | emit_mode(Mode::Result);
    let mut allowed_state = turn1.clone();
    allowed_state.steward_grant_approved = Some(true);
    let residual_after_allow = ask_after_allow_dsl.specialize_monotonic_invariants(&allowed_state);
    let after_second_reply = crate::steward_coalgebra::apply_env_observation_to_pocket_state(
        &allowed_state,
        &crate::steward_coalgebra::EnvObservation::StewardReply {
            outcome: crate::steward_coalgebra::StewardNegotiationOutcome::ExplicitDeny {
                token: "LATE-DENY".to_string(),
                reason: "Late deny after grant".to_string(),
            },
            wire_message: "{}".to_string(),
        },
    );
    assert_eq!(after_second_reply.steward_grant_approved, Some(true));
    assert!(residual_after_allow
        .evaluate(&after_second_reply)
        .decision_eq(&ask_after_allow_dsl.evaluate(&after_second_reply)));

    let m0 = dsl
        .specialize_monotonic_invariants(&turn0)
        .extract_guarded_rules()
        .len();
    let m1 = dsl
        .specialize_monotonic_invariants(&turn1)
        .extract_guarded_rules()
        .len();
    let m2 = dsl
        .specialize_monotonic_invariants(&turn2)
        .extract_guarded_rules()
        .len();
    let m3 = dsl
        .specialize_monotonic_invariants(&turn3)
        .extract_guarded_rules()
        .len();
    assert_eq!(m0, 11);
    assert!(m0 > m1 && m1 > m2 && m2 >= m3);

    let r0 = dsl
        .specialize_for_current_turn(&turn0)
        .extract_guarded_rules()
        .len();
    let r1 = dsl
        .specialize_for_current_turn(&turn1)
        .extract_guarded_rules()
        .len();
    let r2 = dsl
        .specialize_for_current_turn(&turn2)
        .extract_guarded_rules()
        .len();
    let r3 = dsl
        .specialize_for_current_turn(&turn3)
        .extract_guarded_rules()
        .len();
    assert_eq!(r0, 11);
    assert!(r0 > r1 && r1 >= r2 && r2 > r3);
    assert_eq!(r3, 1);
    assert_eq!(
        dsl.specialize_for_current_turn(&turn3)
            .evaluate(&turn3)
            .selected_mode,
        Some(Mode::Result)
    );
}

#[test]
fn deep_regime_expr_25000_clone_simplify_extract_rules_and_drop_are_stack_safe() {
    let handle = std::thread::Builder::new()
        .name("deep-regime-25k".to_string())
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let depth = 25_000usize;

            let mut guard_chain = has_schema_probe();
            for i in 0..depth {
                let atom = candidate_count_eq(i);
                guard_chain = match i % 4 {
                    0 => !guard_chain,
                    1 => guard_chain & atom,
                    2 => guard_chain | atom,
                    _ => atom >> guard_chain,
                };
            }

            let mut distinct_identical_chain = has_schema_probe();
            for i in 0..depth {
                let atom = candidate_count_eq(i);
                distinct_identical_chain = match i % 4 {
                    0 => !distinct_identical_chain,
                    1 => distinct_identical_chain & atom,
                    2 => distinct_identical_chain | atom,
                    _ => atom >> distinct_identical_chain,
                };
            }
            assert_eq!(guard_chain, distinct_identical_chain);
            drop(distinct_identical_chain);

            let deep_regime = (guard_chain >> (candidates_observed() >> Mode::Write))
                | (policy_requires_ask() >> Mode::Ask)
                | emit_mode(Mode::Result);

            let cloned = deep_regime.clone();
            assert_eq!(cloned, deep_regime);

            let simplified = cloned.simplify();
            assert!(simplified.ast_node_count() > depth);

            let rules = deep_regime.extract_guarded_rules();
            assert_eq!(rules.len(), 3);
            assert_eq!(rules[0].mode, Mode::Write);
            assert_eq!(rules[1].mode, Mode::Ask);
            assert_eq!(rules[2].mode, Mode::Result);

            let owned_node_count: usize = simplified.collapse_frames(|frame| match frame {
                RegimeExprFrame::ConstBool(_)
                | RegimeExprFrame::Atom(_)
                | RegimeExprFrame::EmitMode(_) => 1,
                RegimeExprFrame::Not(c) => 1 + c,
                RegimeExprFrame::And(l, r)
                | RegimeExprFrame::Or(l, r)
                | RegimeExprFrame::Stage(l, r) => 1 + l + r,
            });
            assert!(owned_node_count > depth);

            drop(rules);
            drop(cloned);
            drop(deep_regime);
        })
        .expect("spawn 2 MiB thread");
    handle
        .join()
        .expect("depth-25,000 RegimeExpr clone/simplify/extract_guarded_rules/drop must succeed without stack overflow");
}

fn arb_regime_expr() -> impl Strategy<Value = RegimeExpr> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(RegimeExpr::ConstBool),
        Just(has_schema_probe()),
        Just(candidates_observed()),
        (0usize..3usize).prop_map(candidate_count_eq),
        prop_oneof![Just("INSERT"), Just("UPDATE")].prop_map(operation_eq),
        Just(policy_requires_ask()),
        Just(is_locked_or_denied()),
        proptest::option::of(any::<bool>()).prop_map(steward_decision_is),
        (0usize..2usize).prop_map(writes_applied_eq),
        (0usize..2usize).prop_map(writes_applied_gt),
        Just(post_write_verified()),
        Just(zero_row_toctou_race()),
        (0usize..6usize).prop_map(|idx| emit_mode(Mode::ALL[idx])),
    ];
    leaf.prop_recursive(3, 16, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|x| !x),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| l & r),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| l | r),
            (inner.clone(), inner).prop_map(|(l, r)| l >> r),
        ]
    })
}

fn arb_env_observation() -> impl Strategy<Value = crate::steward_coalgebra::EnvObservation> {
    use crate::steward_coalgebra::{EnvObservation, StewardNegotiationOutcome};
    prop_oneof![
        any::<bool>().prop_map(|policy_requires_ask| EnvObservation::CandidateRow {
            target_id: 1,
            current_value: 100,
            current_note: "ROW".to_string(),
            policy_requires_ask,
            steward_code: "STEWARD".to_string(),
        }),
        (any::<bool>(), any::<bool>()).prop_map(|(is_pragma, satisfied)| {
            EnvObservation::AdditionalRow {
                predicate: if is_pragma {
                    "PRAGMA table_info(tbl)".to_string()
                } else {
                    "SELECT * FROM tbl WHERE id=1".to_string()
                },
                satisfied,
            }
        }),
        prop_oneof![
            Just(StewardNegotiationOutcome::UnconditionalAllow {
                token: "ALLOW-TOK".to_string(),
            }),
            Just(StewardNegotiationOutcome::ConditionalAllowWithOverride {
                token: "OV-TOK".to_string(),
                overridden_value: Some(250),
                overridden_note: Some("OV_NOTE".to_string()),
                overridden_target_id: Some(2),
                reason: "Override".to_string(),
            }),
            Just(StewardNegotiationOutcome::RequestAdditionalRead {
                required_predicate: "SELECT * FROM tbl WHERE extra=1".to_string(),
            }),
            Just(StewardNegotiationOutcome::ClarifiedCandidateSelection {
                selected_target_id: 1,
                note: "CLARIFIED".to_string(),
            }),
            Just(StewardNegotiationOutcome::ExplicitDeny {
                token: "DENY-TOK".to_string(),
                reason: "Denied".to_string(),
            }),
        ]
        .prop_map(|outcome| EnvObservation::StewardReply {
            outcome,
            wire_message: "{}".to_string(),
        }),
        (0usize..2usize).prop_map(|rows_affected| EnvObservation::WriteAck {
            rows_affected,
            applied_target_id: 1,
            applied_value: 200,
            applied_note: "WRITTEN".to_string(),
        }),
        Just(EnvObservation::VerifyRow {
            target_id: 1,
            verified_value: 200,
            verified_note: "WRITTEN".to_string(),
        }),
        Just(EnvObservation::TerminalAck),
    ]
}

proptest! {
    #[test]
    fn prop_regime_expr_simplify_preserves_truth_and_selected_mode(
        has_schema in any::<bool>(),
        cand_opt in proptest::option::of(0usize..3usize),
        is_insert in any::<bool>(),
        requires_ask in any::<bool>(),
        locked in any::<bool>(),
        steward_opt in proptest::option::of(any::<bool>()),
        writes in 0usize..2usize,
        verified in any::<bool>(),
        toctou in any::<bool>(),
        b1 in any::<bool>(),
        b2 in any::<bool>(),
    ) {
        let state = PocketObsState {
            world_id: "W".to_string(),
            turn_index: 1,
            table_name: "tbl".to_string(),
            entity_name: "Ent".to_string(),
            operation: if is_insert { "INSERT".to_string() } else { "UPDATE".to_string() },
            has_schema_probe: has_schema,
            candidate_count: cand_opt,
            policy_requires_ask: requires_ask,
            is_locked_or_denied: locked,
            steward_grant_approved: steward_opt,
            applied_write_count: writes,
            has_post_write_verification: verified,
            zero_row_toctou_race: toctou,
        };

        let dsl = canonical_pocket_regime_dsl();
        let raw_eval = dsl.evaluate(&state);
        let simp_eval = dsl.simplify().evaluate(&state);
        prop_assert_eq!(simp_eval.truth, raw_eval.truth);
        prop_assert_eq!(simp_eval.selected_mode, raw_eval.selected_mode);

        let mixed = (RegimeExpr::ConstBool(b1) | (has_schema_probe() >> Mode::Read))
            | (RegimeExpr::ConstBool(b2) >> emit_mode(Mode::Ask));
        let mixed_raw = mixed.evaluate(&state);
        let mixed_simp = mixed.simplify().evaluate(&state);
        prop_assert_eq!(mixed_simp.truth, mixed_raw.truth);
        prop_assert_eq!(mixed_simp.selected_mode, mixed_raw.selected_mode);
    }

    #[test]
    fn prop_arbitrary_regime_expr_ast_simplify_and_extract_guarded_rules_are_exact(
        expr in arb_regime_expr(),
        has_schema in any::<bool>(),
        cand_opt in proptest::option::of(0usize..3usize),
        is_insert in any::<bool>(),
        requires_ask in any::<bool>(),
        locked in any::<bool>(),
        steward_opt in proptest::option::of(any::<bool>()),
        writes in 0usize..2usize,
        verified in any::<bool>(),
        toctou in any::<bool>(),
    ) {
        let obs = PocketObsState {
            world_id: "W".to_string(),
            turn_index: 1,
            table_name: "tbl".to_string(),
            entity_name: "Ent".to_string(),
            operation: if is_insert { "INSERT".to_string() } else { "UPDATE".to_string() },
            has_schema_probe: has_schema,
            candidate_count: cand_opt,
            policy_requires_ask: requires_ask,
            is_locked_or_denied: locked,
            steward_grant_approved: steward_opt,
            applied_write_count: writes,
            has_post_write_verification: verified,
            zero_row_toctou_race: toctou,
        };

        let raw = expr.evaluate(&obs);
        let simp = expr.simplify().evaluate(&obs);
        prop_assert_eq!(raw.truth, simp.truth);
        prop_assert_eq!(raw.selected_mode, simp.selected_mode);

        let turn_spec = expr.specialize_for_turn_state(&obs).evaluate(&obs);
        prop_assert_eq!(raw.truth, turn_spec.truth);
        prop_assert_eq!(raw.selected_mode, turn_spec.selected_mode);

        let snap_spec = expr.specialize_for_current_turn(&obs).evaluate(&obs);
        prop_assert_eq!(raw.truth, snap_spec.truth);
        prop_assert_eq!(raw.selected_mode, snap_spec.selected_mode);

        let extracted_mode = expr
            .extract_guarded_rules()
            .iter()
            .find(|r| r.guard.evaluate(&obs).truth)
            .map(|r| r.mode);
        prop_assert_eq!(extracted_mode, raw.selected_mode);

        let row_states = vec![Some(obs.clone()), None];
        let bindings = expr.bind_capture_row_states(&row_states);
        let winning_modes: Vec<Mode> = bindings
            .iter()
            .filter(|b| b.positive_prompt_indices.contains(&0))
            .map(|b| b.mode)
            .collect();
        prop_assert!(winning_modes.len() <= 1);
        prop_assert_eq!(winning_modes.first().copied(), raw.selected_mode);
        for b in &bindings {
            prop_assert_eq!(&b.background_prompt_indices, &vec![1usize]);
            prop_assert_eq!(
                b.guard_satisfied_indices.contains(&0),
                b.guard.evaluate(&obs).truth
            );
        }
    }

    #[test]
    fn prop_specialize_monotonic_invariants_valid_across_all_future_transitions(
        expr in arb_regime_expr(),
        is_insert in any::<bool>(),
        init_schema in any::<bool>(),
        init_cand in proptest::option::of(0usize..3usize),
        init_ask in any::<bool>(),
        init_lock in any::<bool>(),
        init_steward in proptest::option::of(any::<bool>()),
        init_writes in 0usize..2usize,
        init_verified in any::<bool>(),
        init_toctou in any::<bool>(),
        observations in proptest::collection::vec(arb_env_observation(), 1..8),
    ) {
        let op = if is_insert { "INSERT".to_string() } else { "UPDATE".to_string() };
        let s0 = PocketObsState {
            world_id: "W_CHAIN".to_string(),
            turn_index: 0,
            table_name: "tbl".to_string(),
            entity_name: "Ent".to_string(),
            operation: op,
            has_schema_probe: init_schema,
            candidate_count: init_cand,
            policy_requires_ask: init_ask,
            is_locked_or_denied: init_lock,
            steward_grant_approved: init_steward,
            applied_write_count: init_writes,
            has_post_write_verification: init_verified,
            zero_row_toctou_race: init_toctou,
        };

        let mut chain = Vec::with_capacity(observations.len() + 1);
        chain.push(s0);
        for obs in &observations {
            let next_state = crate::steward_coalgebra::apply_env_observation_to_pocket_state(
                chain.last().unwrap(),
                obs,
            );
            chain.push(next_state);
        }

        for i in 0..chain.len() {
            let residual_i = expr.specialize_monotonic_invariants(&chain[i]);
            for s_j in &chain[i..] {
                let raw_eval = expr.evaluate(s_j);
                let res_eval = residual_i.evaluate(s_j);
                prop_assert!(res_eval.decision_eq(&raw_eval));
            }
        }
    }
}
