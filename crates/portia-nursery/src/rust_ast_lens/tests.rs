use super::*;
use crate::nursery_scenario::{
    arb_domain_skin, arb_scenario_language, AppType, DialogicCodingAffordance, DomainSkin,
    ScenarioCoordinate, ScenarioLanguage, ScenarioWipeoutDiagnosis,
};
use crate::roguelite_lattice::RogueliteStressVector;
use portia_core::relay::{GrammarCartridgeSpec, PayloadGrammarAutomaton};
use portia_core::Mode;
use proptest::prelude::*;

#[test]
fn rust_ast_outline_extractor_splicer_and_octo_merger_enforce_exoskeleton_law() {
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::OrbitalConstellation,
        RogueliteStressVector::sweet_spot_baseline(),
        0x3300_0001,
    );
    let spec = coord.build_cleanroom_rust_hylo_spec();

    // 1. Verify compact rule-free `[AST ...]` outline before filling holes:
    let outline_before =
        format_rust_ast_outline(&spec.module_name, &spec.ast_valid_token_zero_skeleton)
            .expect("outline must succeed");
    assert!(outline_before.contains("[Module `orbital_delta_v_planner`]"));
    assert!(outline_before.contains(
        "[Struct `BurnTrajectoryEnvelopeState` (`seed: u64`, `predicate_arity: usize`)]"
    ));
    assert!(outline_before.contains("[Trait `BurnTrajectoryEnvelope` ("));
    assert!(outline_before
        .contains("[TODO_HOLE #0: \"orbital_delta_v_planner::orbital_constellation_step_01\"]"));

    // 2. Extract all `todo!("...")` holes and verify $O(1)$ surgical window slicing:
    let holes = extract_rust_todo_holes(&spec.ast_valid_token_zero_skeleton)
        .expect("hole extraction must succeed");
    assert_eq!(holes.len(), spec.todo_holes.len());
    let first_fn = &holes[0].fn_name;
    assert!(!holes[0].todo_macro_span.is_empty());
    assert!(holes[0].todo_macro_span.len() >= 6);
    let as_std_range: std::ops::Range<usize> = holes[0].todo_macro_span.into();
    assert_eq!(
        ByteSpan::try_from(as_std_range).unwrap(),
        holes[0].todo_macro_span
    );
    assert!(holes[0]
        .todo_macro_span
        .slice(&spec.ast_valid_token_zero_skeleton)
        .expect("valid UTF-8 todo_macro_span")
        .starts_with("todo!"));
    assert!(holes[0]
        .fn_span
        .slice(&spec.ast_valid_token_zero_skeleton)
        .expect("valid UTF-8 fn_span")
        .contains(first_fn));
    assert!(ByteSpan::try_new(10, 4).is_err());
    assert!(ByteSpan::new(4, 4).is_empty());
    let window = slice_rust_fn_window(&spec.ast_valid_token_zero_skeleton, first_fn)
        .expect("window slice must succeed");
    assert!(window.contains(first_fn));
    assert!(!window.contains(&holes[1].fn_name));

    // 3. Reject unbalanced delimiter splices and top-level item injections (Exoskeleton Law):
    let bad_unbalanced = splice_rust_todo_hole_body(
        &spec.ast_valid_token_zero_skeleton,
        first_fn,
        "if input_metric > 0 { Ok(input_metric)",
    );
    assert!(bad_unbalanced.is_err());

    let bad_injection = splice_rust_todo_hole_body(
        &spec.ast_valid_token_zero_skeleton,
        first_fn,
        "pub struct SmuggledItem { pub x: u64 }\nOk(input_metric)",
    );
    assert!(bad_injection.is_err());

    // 4. Hylomorphic Octo-Merge via `try_expand_and_collapse` (`RoseFrame`):
    let patches: Vec<(String, String)> = holes
        .iter()
        .map(|h| {
            (
                h.fn_name.clone(),
                format!(
                    "approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    h.hole_index
                ),
            )
        })
        .collect();
    let merged = hylo_octo_merge_rust_holes(&spec.ast_valid_token_zero_skeleton, &patches)
        .expect("octo-merge must succeed");
    assert!(extract_rust_todo_holes(&merged).unwrap().is_empty());

    let outline_after =
        format_rust_ast_outline(&spec.module_name, &merged).expect("outline after merge");
    assert!(outline_after.contains("[IMPLEMENTED]"));
    assert!(!outline_after.contains("TODO_HOLE"));
}

#[test]
fn unified_rust_ast_scenario_session_clears_dialogic_trajectory_and_diagnoses_all_coroner_wipeouts()
{
    for skin in DomainSkin::ALL {
        let coord = ScenarioCoordinate::new(
            AppType::RustAstHylo,
            ScenarioLanguage::English,
            skin,
            RogueliteStressVector::sweet_spot_baseline(),
            0x3300_0042,
        );
        let spec = coord.build_cleanroom_rust_hylo_spec();

        let mut session = UnifiedRustAstScenarioSession::new(spec.clone()).unwrap();
        let p_out = session.step_turn(Mode::Probe, "inspect rust ast outline");
        assert!(p_out.ok);
        assert!(session.probed_ast_outline);

        let (crate_name, _) = skin.approved_ecosystem_crate();
        let a_out = session.step_turn(
            Mode::Ask,
            &format!(
                "Please clarify the ambiguous underflow invariant, approve crate {crate_name}, and review scaffold before fanout."
            ),
        );
        assert!(a_out.ok);
        assert!(session.asked_clarification);
        assert!(session.requested_ecosystem_crate);
        assert!(session.reviewed_scaffold_before_fanout);
        assert!(session.approved_crates.contains(crate_name));

        for hole in &spec.todo_holes {
            let r_out = session.step_turn(Mode::Read, &hole.fn_name);
            assert!(r_out.ok);
            let w_out = session.step_turn(
                Mode::Write,
                &format!(
                    "{} => approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    hole.fn_name, hole.hole_index
                ),
            );
            assert!(w_out.ok);
        }

        let res_out = session.step_turn(Mode::Result, "{\"status\": \"OCTO_MERGED\"}");
        assert!(res_out.ok);
        let summary = res_out.episode_summary.expect("summary");
        assert!(summary.cleared);
        assert!(summary.wipeout_diagnoses.is_empty());

        let mut bad_session = UnifiedRustAstScenarioSession::new(spec.clone()).unwrap();
        let first_fn = &spec.todo_holes[0].fn_name;
        let bad_w1 = bad_session.step_turn(
            Mode::Write,
            &format!("{first_fn} => Err(\"attempted DIY bypass without approved_crate\")"),
        );
        assert!(!bad_w1.ok);
        assert!(bad_w1
            .turn_wipeout_diagnoses
            .contains(&ScenarioWipeoutDiagnosis::MissingReadBeforeWriteProvenance));
        assert!(bad_w1.turn_wipeout_diagnoses.iter().any(|d| matches!(
            d,
            ScenarioWipeoutDiagnosis::GuessedOnAmbiguousSpecWithoutAsking { .. }
        )));
        assert!(bad_w1.turn_wipeout_diagnoses.contains(
            &ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate {
                available_crate: crate_name.to_string(),
            }
        ));

        let bad_w2 = bad_session.step_turn(
            Mode::Write,
            &format!("{first_fn} => if input_metric > 0 {{ Ok(1)"),
        );
        assert!(!bad_w2.ok);
        assert!(bad_w2.turn_wipeout_diagnoses.iter().any(|d| matches!(
            d,
            ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { .. }
        )));
    }
}

#[test]
fn rejected_mode_write_does_not_mutate_current_source_or_leak_into_later_valid_write() {
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::OrbitalConstellation,
        RogueliteStressVector::sweet_spot_baseline(),
        0x4200_0001,
    );
    let spec = coord.build_cleanroom_rust_hylo_spec();
    assert!(spec.todo_holes.len() >= 2);
    let fn_a = spec.todo_holes[0].fn_name.clone();
    let fn_b = spec.todo_holes[1].fn_name.clone();
    let fn_a_todo_stub = spec.todo_holes[0].todo_stub_literal.clone();
    let fn_b_todo_stub = spec.todo_holes[1].todo_stub_literal.clone();

    let mut session = UnifiedRustAstScenarioSession::new(spec.clone()).unwrap();

    let rejected_a = session.step_turn(
        Mode::Write,
        &format!("{fn_a} => Err(\"unapproved edit to fn_a that must be rejected\")"),
    );
    assert!(!rejected_a.ok);
    assert!(rejected_a
        .error_message
        .as_deref()
        .unwrap_or_default()
        .contains(&format!(
            "rejected Mode::Write to `{fn_a}` due to validation failures"
        )));
    assert!(
        session.current_source.contains(&fn_a_todo_stub),
        "rejected Mode::Write to `fn_a` must leave `todo!(...)` intact in `session.current_source`"
    );
    assert!(
        !session
            .current_source
            .contains("unapproved edit to fn_a that must be rejected"),
        "rejected Mode::Write to `fn_a` must not mutate `session.current_source`"
    );
    assert!(!session.filled_fn_holes.contains(&fn_a));

    let probe_out = session.step_turn(Mode::Probe, "inspect rust ast outline");
    assert!(probe_out.ok);

    let (crate_name, _) = DomainSkin::OrbitalConstellation.approved_ecosystem_crate();
    let ask_out = session.step_turn(
        Mode::Ask,
        &format!(
            "Please clarify the ambiguous underflow invariant, approve crate {crate_name}, and review scaffold."
        ),
    );
    assert!(ask_out.ok);

    let read_b = session.step_turn(Mode::Read, &fn_b);
    assert!(read_b.ok);

    let mint_bypass = session.step_turn(
        Mode::Write,
        &format!(
            "{fn_a} => Ok(sealed_witness::VerifiedDomainOutput::mint_from_approved_crate(input_metric))"
        ),
    );
    assert!(!mint_bypass.ok);
    assert!(mint_bypass.turn_wipeout_diagnoses.contains(
        &ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate {
            available_crate: crate_name.to_string(),
        }
    ));
    assert!(session.current_source.contains(&fn_a_todo_stub));

    let valid_b = session.step_turn(
        Mode::Write,
        &format!("{fn_b} => approved_crate::eval_domain_step(input_metric, guard_clause, 1usize)"),
    );
    assert!(
        valid_b.ok,
        "valid Mode::Write to `fn_b` must succeed: {:?}",
        valid_b.error_message
    );
    assert!(session.filled_fn_holes.contains(&fn_b));
    assert!(!session.filled_fn_holes.contains(&fn_a));
    assert!(!session.current_source.contains(&fn_b_todo_stub));
    assert!(
        session
            .current_source
            .contains("approved_crate::eval_domain_step(input_metric, guard_clause, 1usize)"),
        "`fn_b` must be spliced into `session.current_source`"
    );
    assert!(
        session.current_source.contains(&fn_a_todo_stub),
        "`fn_a` must remain untouched with its original `todo!()` after valid write to `fn_b`"
    );
}

#[test]
fn rust_ast_hole_grammar_cartridge_and_active_span_cartridge_constrain_payload_body() {
    use portia_core::relay::{RelayMachine, RelayPhase, SpanCartridge};
    use portia_core::CartridgeBank;

    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::OrbitalConstellation,
        RogueliteStressVector::sweet_spot_baseline(),
        0x5500_0001,
    );
    let spec = coord.build_cleanroom_rust_hylo_spec();
    let (crate_name, _) = DomainSkin::OrbitalConstellation.approved_ecosystem_crate();
    let cartridge = RustAstHoleGrammarCartridge::from_rust_source(
        &spec.ast_valid_token_zero_skeleton,
        Some(crate_name.to_string()),
    )
    .expect("cartridge from skeleton");
    assert_eq!(cartridge.hole_fn_names.len(), spec.todo_holes.len());
    let first_fn = &cartridge.hole_fn_names[0];

    let probe_spec = cartridge.grammar_cartridge_spec(Mode::Probe);
    assert!(
        matches!(probe_spec, GrammarCartridgeSpec::LarkGrammar(ref s) if s.contains("outline"))
    );

    let read_spec = cartridge.grammar_cartridge_spec(Mode::Read);
    assert!(matches!(read_spec, GrammarCartridgeSpec::LarkGrammar(ref s) if s.contains(first_fn)));

    let write_spec = cartridge.grammar_cartridge_spec(Mode::Write);
    assert!(matches!(
        write_spec,
        GrammarCartridgeSpec::LarkGrammar(ref s)
            if s.contains("hole_fn_name \" => \" hole_body")
                && s.contains(first_fn)
                && s.contains("approved_crate::")
    ));

    for free_mode in [Mode::Ask, Mode::Result, Mode::Abstain] {
        assert_eq!(
            cartridge.grammar_cartridge_spec(free_mode),
            GrammarCartridgeSpec::Unconstrained
        );
    }

    let mut relay = RelayMachine::new();
    assert_eq!(
        relay.active_span_cartridge(&cartridge),
        SpanCartridge::unconstrained(CartridgeBank::OrientTrunk)
    );
    relay.flush_harness_prefill(|_| vec![1]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::ModeLabel);
    assert_eq!(
        relay.active_span_cartridge(&cartridge),
        SpanCartridge::new(
            CartridgeBank::OrientTrunk,
            GrammarCartridgeSpec::LiteralSet(
                Mode::ALL
                    .iter()
                    .map(|m| format!(" {}]", m.as_str()))
                    .collect()
            ),
        )
    );
    relay.step_trunk_token(10, " WRITE]").unwrap();
    relay.flush_harness_prefill(|_| vec![2]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PlanBody);
    assert_eq!(
        relay.active_span_cartridge(&cartridge),
        SpanCartridge::new(
            CartridgeBank::Base,
            GrammarCartridgeSpec::LarkGrammar(
                portia_core::relay::PLAN_BODY_SINGLE_LINE_LARK.to_string()
            ),
        )
    );
    relay
        .step_trunk_token(20, " Fill the first hole.\n")
        .unwrap();
    relay.flush_harness_prefill(|_| vec![3]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PayloadBody);
    assert_eq!(
        relay.active_span_cartridge(&cartridge),
        SpanCartridge::new(
            CartridgeBank::ModePayload(Mode::Write),
            cartridge.grammar_cartridge_spec(Mode::Write),
        )
    );
}

#[test]
fn record_live_user_clarification_and_step_turn_never_invent_blanket_simulated_answers() {
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::OrbitalConstellation,
        RogueliteStressVector::sweet_spot_baseline(),
        0x6600_0001,
    );
    let spec = coord.build_cleanroom_rust_hylo_spec();
    let (crate_name, _) = DomainSkin::OrbitalConstellation.approved_ecosystem_crate();

    let mut nursery_session = RustAstHyloSession::new(spec.clone()).unwrap();
    let unrelated = nursery_session.step_turn(Mode::Ask, "What time is the deployment window?");
    assert!(unrelated.ok);
    assert_eq!(unrelated.tool_response, None);
    assert!(!nursery_session.asked_clarification);
    assert!(!nursery_session.requested_ecosystem_crate);
    assert!(!nursery_session.reviewed_scaffold_before_fanout);
    assert!(nursery_session.approved_crates.is_empty());

    let partial =
        nursery_session.simulate_nursery_steward_ask("Could you clarify the underflow behavior?");
    assert!(partial.ok);
    assert!(partial
        .tool_response
        .as_deref()
        .unwrap_or_default()
        .contains("[CLARIFIED"));
    assert!(nursery_session.asked_clarification);
    assert!(!nursery_session.requested_ecosystem_crate);
    assert!(!nursery_session.reviewed_scaffold_before_fanout);
    assert!(nursery_session.approved_crates.is_empty());

    let mut live_spec = spec.clone();
    live_spec.dialogic_affordances = vec![DialogicCodingAffordance::ReviewScaffoldBeforeFanout {
        todo_hole_names: spec.todo_holes.iter().map(|h| h.fn_name.clone()).collect(),
    }];
    let mut live_session = RustAstHyloSession::new(live_spec).unwrap();
    let live_out = live_session
        .record_live_user_clarification("Which error message should empty input return?");
    assert!(live_out.ok);
    assert_eq!(live_out.tool_response, None);
    assert_eq!(live_out.response_bytes, 0);
    assert!(live_session.asked_clarification);
    assert!(!live_session.requested_ecosystem_crate);
    assert!(!live_session.reviewed_scaffold_before_fanout);
    assert!(!live_session.approved_crates.contains(crate_name));

    let live_scaf =
        live_session.record_live_user_clarification("May we review the scaffold before fanout?");
    assert_eq!(live_scaf.tool_response, None);
    assert!(live_session.reviewed_scaffold_before_fanout);
    assert!(!live_session.requested_ecosystem_crate);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_unified_rust_ast_scenario_session_clears_across_languages_and_domain_skins(
        lang in arb_scenario_language(),
        skin in arb_domain_skin(),
        hole_count in 3usize..=10,
        seed in any::<u64>(),
    ) {
        let mut stress = RogueliteStressVector::sweet_spot_baseline();
        stress.ast_symbol_haystack = hole_count;
        let coord = ScenarioCoordinate::new(AppType::RustAstHylo, lang, skin, stress, seed);
        let spec = coord.build_cleanroom_rust_hylo_spec();

        let mut session = UnifiedRustAstScenarioSession::new(spec.clone()).unwrap();
        let init_cart = session.hole_grammar_cartridge();
        prop_assert_eq!(init_cart.hole_fn_names.len(), hole_count);
        prop_assert!(init_cart.grammar_cartridge_spec(Mode::Write).is_constrained());
        prop_assert!(!init_cart.grammar_cartridge_spec(Mode::Result).is_constrained());

        let p_out = session.step_turn(Mode::Probe, "outline");
        prop_assert!(p_out.ok);
        prop_assert_eq!(p_out.remaining_todo_holes, hole_count);

        let (crate_name, _) = skin.approved_ecosystem_crate();
        let ask_out = session.step_turn(
            Mode::Ask,
            &format!("Clarify underflow invariant and request crate {crate_name}"),
        );
        prop_assert!(ask_out.ok);

        for (idx, hole) in spec.todo_holes.iter().enumerate() {
            let r_out = session.step_turn(Mode::Read, &hole.fn_name);
            prop_assert!(r_out.ok);
            let w_out = session.step_turn(
                Mode::Write,
                &format!(
                    "{} => approved_crate::eval_domain_step(input_metric, guard_clause, {idx}usize)",
                    hole.fn_name
                ),
            );
            prop_assert!(w_out.ok);
            prop_assert_eq!(w_out.remaining_todo_holes, hole_count - (idx + 1));
        }

        let fin = session.step_turn(Mode::Result, "{\"status\":\"OK\"}");
        prop_assert!(fin.ok);
        prop_assert!(fin.episode_summary.unwrap().cleared);
    }
}
