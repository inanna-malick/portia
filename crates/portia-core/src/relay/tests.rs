use super::*;
use crate::bpe::{
    canonical_mode_header_tokens_under_orient, TOKEN_CLOSE_BRACKET, TOKEN_DOUBLE_NEWLINE,
    TOKEN_SPACE_READ,
};
use crate::{CartridgeBank, Mode, PenOwner, RouteTag};
use proptest::prelude::*;

#[test]
fn relay_machine_consumes_close_bracket_under_orient_and_switches_on_newline() {
    let mut relay = RelayMachine::new();
    assert_eq!(relay.active_pen(), PenOwner::HarnessPen);
    assert_eq!(relay.active_cartridge_bank(), CartridgeBank::OrientTrunk);

    let open_span = relay
        .flush_harness_prefill(|s| vec![1; s.len()])
        .unwrap()
        .unwrap();
    assert_eq!(open_span.consumer_bank, CartridgeBank::OrientTrunk);
    assert_eq!(relay.phase(), RelayPhase::ModeLabel);
    assert_eq!(relay.active_pen(), PenOwner::RouterPen);

    let m1 = relay.step_trunk_token(TOKEN_SPACE_READ, " READ").unwrap();
    assert_eq!(m1.consumer_bank, CartridgeBank::OrientTrunk);
    let m2 = relay.step_trunk_token(TOKEN_CLOSE_BRACKET, "]").unwrap();
    assert_eq!(m2.consumer_bank, CartridgeBank::OrientTrunk);

    assert_eq!(relay.phase(), RelayPhase::PostModePrefill);
    assert_eq!(relay.active_cartridge_bank(), CartridgeBank::Base);
    let post_mode = relay
        .flush_harness_prefill(|_| vec![198, 19860, 25])
        .unwrap()
        .unwrap();
    assert_eq!(post_mode.consumer_bank, CartridgeBank::Base);
    assert_eq!(post_mode.text, "\nPlan:");

    relay
        .step_trunk_token(TOKEN_DOUBLE_NEWLINE, " Read the candidate rows.\n\n")
        .unwrap();
    assert_eq!(relay.phase(), RelayPhase::PostPlanPrefill);

    let post_plan = relay
        .flush_harness_prefill(|s| vec![2; s.len()])
        .unwrap()
        .unwrap();
    assert_eq!(post_plan.text, "</think>\n<sql>\n");

    relay
        .step_trunk_token(900, "SELECT '</sql>' FROM t")
        .unwrap();
    assert!(relay.phase() == RelayPhase::PayloadBody);
    relay.step_trunk_token(901, "\n</sql>").unwrap();
    assert_eq!(relay.phase(), RelayPhase::TurnComplete);

    let completed = relay.into_completed_turn(|_| vec![248046, 198]).unwrap();
    assert_eq!(completed.mode, Mode::Read);
    assert_eq!(completed.route, RouteTag::ToDatabase);
    assert_eq!(completed.plan, "Read the candidate rows.");
    assert_eq!(completed.payload, "SELECT '</sql>' FROM t");
    assert!(completed.full_wire_text.ends_with("\n</sql><|im_end|>\n"));
    assert_eq!(
        completed.action,
        TypedTurnAction::ReadQuery {
            sql: "SELECT '</sql>' FROM t".to_string()
        }
    );
}

#[test]
fn typestate_single_pen_turn_enforces_phase_order_and_parses_typed_action() {
    let completed = TurnAtOpenPrefill::new()
        .harness_open(|_| vec![58, 22098, 25])
        .unwrap()
        .router_select(Mode::Result)
        .unwrap()
        .harness_plan_header(|_| vec![198, 19860, 25])
        .unwrap()
        .payload_plan(1000, " Report the verified balance.")
        .unwrap()
        .harness_route_open(|_| vec![248069, 198])
        .unwrap()
        .payload_body(
            2000,
            r#"{"status":"ok","evidence_ids":["obs-0001"]}"#,
            |_| vec![248046, 198],
        )
        .unwrap();

    assert_eq!(completed.mode, Mode::Result);
    assert_eq!(completed.route, RouteTag::ToInquirer);
    assert_eq!(completed.plan, "Report the verified balance.");
    assert!(completed
        .full_wire_text
        .contains("\nPlan: Report the verified balance.\n</think>\n<to_user>\n"));
    assert!(completed
        .full_wire_text
        .ends_with("\n</to_user><|im_end|>\n"));
    assert!(matches!(
        completed.action,
        TypedTurnAction::SubmitResult { .. }
    ));

    let bad_result = TurnAtOpenPrefill::new()
        .harness_open(|_| vec![58, 22098, 25])
        .unwrap()
        .router_select(Mode::Result)
        .unwrap()
        .harness_plan_header(|_| vec![198, 19860, 25])
        .unwrap()
        .payload_plan(1000, " Bad report.")
        .unwrap()
        .harness_route_open(|_| vec![248069, 198])
        .unwrap()
        .payload_body(2000, "{not-valid-json", |_| vec![248046, 198]);
    assert!(bad_result.is_err());

    assert_eq!(TURN_OPEN_PREFILL, "<|im_start|>assistant\n<think>\n[MODE:");
    let mut lotl_relay = RelayMachine::new();
    lotl_relay
        .flush_harness_prefill(|_| vec![248045, 74455, 198, 248068, 198, 58, 22098, 25])
        .unwrap();
    lotl_relay.step_trunk_token(18104, " READ]").unwrap();
    lotl_relay
        .flush_harness_prefill(|_| vec![198, 19860, 25])
        .unwrap();
    lotl_relay
        .step_trunk_token(1000, " Read bgp_incidents row.\n")
        .unwrap();
    lotl_relay
        .flush_harness_prefill(|_| vec![248069, 198, 248058, 198])
        .unwrap();
    lotl_relay
        .step_trunk_token(
            2000,
            "SELECT * FROM bgp_incidents WHERE pop_code = 'NRT-TOKYO-04'\n</tool_call>",
        )
        .unwrap();
    let lotl_completed = lotl_relay
        .into_completed_turn(|_| vec![248046, 198])
        .unwrap();
    assert_eq!(lotl_completed.mode, Mode::Read);
    assert_eq!(
        lotl_completed.payload,
        "SELECT * FROM bgp_incidents WHERE pop_code = 'NRT-TOKYO-04'"
    );
}

#[test]
fn active_span_cartridge_mounts_universal_three_phase_grammars_and_lowers_to_lark() {
    struct EpistemicMockGrammar {
        modes: Vec<Mode>,
    }
    impl PayloadGrammarAutomaton for EpistemicMockGrammar {
        fn legal_modes(&self) -> Vec<Mode> {
            self.modes.clone()
        }
        fn grammar_cartridge_spec(&self, mode: Mode) -> GrammarCartridgeSpec {
            match mode {
                Mode::Probe => GrammarCartridgeSpec::LarkGrammar(
                    "start: \"SELECT name FROM sqlite_master WHERE type = 'table';\\n</sql>\""
                        .to_string(),
                ),
                Mode::Ask => GrammarCartridgeSpec::JsonSchema(serde_json::json!({
                    "type": "object",
                    "required": ["select"]
                })),
                _ => GrammarCartridgeSpec::Unconstrained,
            }
        }
    }

    assert!(!GrammarCartridgeSpec::Unconstrained.is_constrained());
    assert_eq!(
        GrammarCartridgeSpec::Unconstrained.to_llguidance_lark(),
        None
    );

    let empty_lits = GrammarCartridgeSpec::LiteralSet(vec![]);
    assert!(empty_lits.is_constrained());
    assert_eq!(empty_lits.to_llguidance_lark(), None);

    let mode_lits = GrammarCartridgeSpec::LiteralSet(vec![
        " PROBE]".to_string(),
        " ASK]".to_string(),
        "quote\"slash\\nl\ncr\r".to_string(),
    ]);
    assert_eq!(
        mode_lits.to_llguidance_lark().as_deref(),
        Some("start: \" PROBE]\" | \" ASK]\" | \"quote\\\"slash\\\\nl\\ncr\\r\"")
    );

    let lark_spec = GrammarCartridgeSpec::LarkGrammar(PLAN_BODY_SINGLE_LINE_LARK.to_string());
    assert_eq!(
        lark_spec.to_llguidance_lark().as_deref(),
        Some(PLAN_BODY_SINGLE_LINE_LARK)
    );

    let json_spec = GrammarCartridgeSpec::JsonSchema(serde_json::json!({"type": "object"}));
    assert_eq!(
        json_spec.to_llguidance_lark().as_deref(),
        Some("start: %json {\"type\":\"object\"}")
    );

    let mock = EpistemicMockGrammar {
        modes: vec![Mode::Probe, Mode::Ask, Mode::Abstain],
    };
    let mut relay = RelayMachine::new();

    assert_eq!(relay.phase(), RelayPhase::TurnOpenPrefill);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::unconstrained(CartridgeBank::OrientTrunk)
    );

    relay.flush_harness_prefill(|_| vec![1]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::ModeLabel);
    let mode_cart = relay.active_span_cartridge(&mock);
    assert_eq!(
        mode_cart,
        SpanCartridge::new(
            CartridgeBank::OrientTrunk,
            GrammarCartridgeSpec::LiteralSet(vec![
                " PROBE]".to_string(),
                " ASK]".to_string(),
                " ABSTAIN]".to_string(),
            ]),
        )
    );
    assert_eq!(
        mode_cart.grammar.to_llguidance_lark().as_deref(),
        Some("start: \" PROBE]\" | \" ASK]\" | \" ABSTAIN]\"")
    );

    let empty_mock = EpistemicMockGrammar { modes: vec![] };
    assert_eq!(
        relay.active_span_cartridge(&empty_mock),
        SpanCartridge::unconstrained(CartridgeBank::OrientTrunk)
    );

    relay.step_trunk_token(5140, " PROBE]").unwrap();
    assert_eq!(relay.phase(), RelayPhase::PostModePrefill);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::unconstrained(CartridgeBank::Base)
    );

    relay.flush_harness_prefill(|_| vec![2]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PlanBody);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::new(
            CartridgeBank::Base,
            GrammarCartridgeSpec::LarkGrammar(PLAN_BODY_SINGLE_LINE_LARK.to_string()),
        )
    );

    relay
        .step_trunk_token(100, " Discover tables in SQLite catalog.\n")
        .unwrap();
    assert_eq!(relay.phase(), RelayPhase::PostPlanPrefill);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::unconstrained(CartridgeBank::ModePayload(Mode::Probe))
    );

    relay.flush_harness_prefill(|_| vec![3]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PayloadBody);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::new(
            CartridgeBank::ModePayload(Mode::Probe),
            mock.grammar_cartridge_spec(Mode::Probe),
        )
    );

    relay
        .step_trunk_token(
            200,
            "SELECT name FROM sqlite_master WHERE type = 'table';\n</sql>",
        )
        .unwrap();
    assert_eq!(relay.phase(), RelayPhase::TurnComplete);
    assert_eq!(
        relay.active_span_cartridge(&mock),
        SpanCartridge::unconstrained(CartridgeBank::Base)
    );
}

#[test]
fn sql_and_json_scanners_and_lotl_wire_normalizer_handle_quoted_literals_and_tags() {
    assert!(!has_completed_close_tag_outside_literals(
        "SELECT '</sql>' FROM t",
        Mode::Read
    ));
    assert!(has_completed_close_tag_outside_literals(
        "SELECT '</sql>' FROM t;\n</sql>",
        Mode::Read
    ));
    assert!(has_completed_close_tag_outside_literals(
        "{\"note\": \"</to_user>\"}\n</to_user>",
        Mode::Result
    ));
    assert!(!has_completed_close_tag_outside_literals(
        "{\"note\": \"</to_user>",
        Mode::Result
    ));

    assert_eq!(find_unquoted_sql_char("SELECT 'a;b'", ';'), None);
    assert_eq!(find_unquoted_sql_char("SELECT \"a;b\" FROM t", ';'), None);
    assert_eq!(
        find_unquoted_sql_char("SELECT 'a;b' FROM t;", ';'),
        Some("SELECT 'a;b' FROM t".len())
    );
    assert_eq!(
        find_unquoted_sql_char("PRAGMA table_info(\"a)b\"", ')'),
        None
    );
    assert_eq!(
        find_unquoted_sql_char("PRAGMA table_info(\"a)b\")", ')'),
        Some("PRAGMA table_info(\"a)b\"".len())
    );

    assert!(!sql_has_unquoted_where("SELECT 'WHERE' FROM t;"));
    assert!(!sql_has_unquoted_where("SELECT * FROM nowhere_table;"));
    assert!(sql_has_unquoted_where("SELECT * FROM t WHERE id = 1;"));

    assert!(sql_has_unclosed_quote("SELECT 'unclosed"));
    assert!(!sql_has_unclosed_quote("SELECT 'escaped '' quote'"));
    assert!(sql_has_unclosed_quote("SELECT \"unclosed"));
    assert!(!sql_has_unclosed_quote("SELECT \"escaped \"\" quote\""));
    assert!(json_has_unclosed_string("{\"k\": \"unclosed"));
    assert!(json_has_unclosed_string(
        "{\"k\": \"escaped \\\" still open"
    ));
    assert!(!json_has_unclosed_string(
        "{\"k\": \"escaped \\\" closed\"}"
    ));

    assert_eq!(
        normalize_lotl_wire_chunk("<|im_end|>\n", Some(Mode::Read)),
        "<|im_end|>\n"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("\nPlan:", Some(Mode::Read)),
        "\nPlan:"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("</think>\n<sql>\n", Some(Mode::Probe)),
        "</think>\n<sql>\n"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("</think>\n<to_user>\n", Some(Mode::Abstain)),
        "</think>\n<to_user>\n"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("\n<|im_end|>\n", Some(Mode::Read)),
        "<|im_end|>\n"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("\nPlan: ", Some(Mode::Read)),
        "\nPlan:"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("SELECT 1;\n</sql>", Some(Mode::Read)),
        "SELECT 1;\n</sql>"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("{\"ok\":true}\n</to_user>", Some(Mode::Result)),
        "{\"ok\":true}\n</to_user>"
    );
    assert_eq!(
        normalize_lotl_wire_chunk("Cannot modify.\n</to_user>", Some(Mode::Abstain)),
        "Cannot modify.\n</to_user>"
    );
    let unsorted_tr = "<|im_start|>user\n<tool_response>\n{\"z\": 1, \"a\": 2, \"m\": 3}\n</tool_response><|im_end|>\n";
    assert_eq!(
        normalize_lotl_wire_chunk(unsorted_tr, None),
        "<|im_start|>user\n<tool_response>\n{\"a\":2,\"m\":3,\"z\":1}\n</tool_response><|im_end|>\n"
    );
    let surrounded_tr =
        "prefix <to_database>\n<tool_response>\n{\"b\":2,\"a\":1}\n</tool_response>\nsuffix <to_inquirer>";
    assert_eq!(
        normalize_lotl_wire_chunk(surrounded_tr, None),
        "prefix <to_database>\n<tool_response>\n{\"a\":1,\"b\":2}\n</tool_response>\nsuffix <to_inquirer>"
    );
}

#[test]
fn lotl_normalized_wire_chunk_preserves_user_prompt_envelopes_and_canonical_wire_strings() {
    let user_prompt =
        "<|im_start|>user\nWhy did legacy code emit <to_database> and \nPlan: ?<|im_end|>\n";
    let wrapped = LotlNormalizedWireChunk::from_user_prompt_envelope(user_prompt);
    assert_eq!(wrapped.as_str(), user_prompt);
    assert_eq!(
        LotlNormalizedWireChunk::normalize(user_prompt, None).as_str(),
        user_prompt
    );
    let encoded_prompt = serde_json::to_string(&wrapped).unwrap();
    let roundtripped_prompt: LotlNormalizedWireChunk =
        serde_json::from_str(&encoded_prompt).unwrap();
    assert_eq!(roundtripped_prompt.as_str(), user_prompt);

    let user_prompt_with_quoted_tr = "<|im_start|>user\nExplain this log:\n<tool_response>\n{\"z\": 1, \"a\": 2}\n</tool_response>\nand <to_inquirer><|im_end|>\n";
    let wrapped_tr_prompt =
        LotlNormalizedWireChunk::from_user_prompt_envelope(user_prompt_with_quoted_tr);
    let encoded_tr_prompt = serde_json::to_string(&wrapped_tr_prompt).unwrap();
    let roundtripped_tr_prompt: LotlNormalizedWireChunk =
        serde_json::from_str(&encoded_tr_prompt).unwrap();
    assert_eq!(roundtripped_tr_prompt.as_str(), user_prompt_with_quoted_tr);
    assert_eq!(
        LotlNormalizedWireChunk::normalize(user_prompt_with_quoted_tr, None).as_str(),
        user_prompt_with_quoted_tr
    );

    let canonical_turn = "<|im_start|>assistant\n<think>\n[MODE: READ]\nPlan: x\n</think>\n<sql>\nSELECT 1;\n</sql><|im_end|>\n";
    let canonical_json = serde_json::to_string(canonical_turn).unwrap();
    let deserialized: LotlNormalizedWireChunk = serde_json::from_str(&canonical_json).unwrap();
    assert_eq!(deserialized.as_str(), canonical_turn);
    assert_eq!(
        deserialized,
        LotlNormalizedWireChunk::normalize(canonical_turn, None)
    );

    for raw in [
        "\nPlan: ",
        "\n<|im_end|>\n",
        "<|im_start|>assistant\n<think>\n[MODE: ASK]\nPlan: ask\n</think>\n<to_user>\nProceed?\n</to_user><|im_end|>\n",
        "<|im_start|>user\n<tool_response>\n{\"z\": 9, \"a\": 1}\n</tool_response><|im_end|>\n",
    ] {
        let encoded = serde_json::to_string(raw).unwrap();
        let decoded: LotlNormalizedWireChunk = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, LotlNormalizedWireChunk::normalize(raw, None));
    }
}

#[test]
fn plan_sampled_trunk_token_tracks_two_phase_banks_across_split_close_tags_and_utf8_literals() {
    let mut relay = RelayMachine::new();
    relay.flush_harness_prefill(|_| vec![1, 2, 3]).unwrap();

    let p_pro = relay.plan_sampled_trunk_token(5140, " PRO").unwrap();
    assert_eq!(p_pro.producer_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_pro.consume_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_pro.phase_after, RelayPhase::ModeLabel);

    let p_be = relay.plan_sampled_trunk_token(11261, "BE").unwrap();
    assert_eq!(p_be.producer_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_be.consume_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_be.phase_after, RelayPhase::ModeLabel);

    let p_close_bracket = relay.plan_sampled_trunk_token(60, "]").unwrap();
    assert_eq!(p_close_bracket.producer_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_close_bracket.consume_bank, CartridgeBank::OrientTrunk);
    assert_eq!(p_close_bracket.phase_after, RelayPhase::PostModePrefill);

    let post_mode = relay
        .flush_harness_prefill(|_| vec![198, 19860, 25])
        .unwrap()
        .unwrap();
    assert_eq!(post_mode.consumer_bank, CartridgeBank::Base);
    assert_eq!(post_mode.text, "\nPlan:");

    let p_plan_1 = relay.plan_sampled_trunk_token(300, " Prüfen ").unwrap();
    assert_eq!(p_plan_1.producer_bank, CartridgeBank::Base);
    assert_eq!(p_plan_1.consume_bank, CartridgeBank::Base);
    assert_eq!(p_plan_1.phase_after, RelayPhase::PlanBody);

    let p_plan_2 = relay.plan_sampled_trunk_token(301, "α-β catalog").unwrap();
    assert_eq!(p_plan_2.producer_bank, CartridgeBank::Base);
    assert_eq!(p_plan_2.consume_bank, CartridgeBank::Base);
    assert_eq!(p_plan_2.phase_after, RelayPhase::PlanBody);

    let p_plan_nl = relay.plan_sampled_trunk_token(198, "\n").unwrap();
    assert_eq!(p_plan_nl.producer_bank, CartridgeBank::Base);
    assert_eq!(p_plan_nl.consume_bank, CartridgeBank::Base);
    assert_eq!(p_plan_nl.phase_after, RelayPhase::PostPlanPrefill);

    relay.flush_harness_prefill(|_| vec![248069, 198]).unwrap();

    let mut read_relay = RelayMachine::new();
    read_relay.flush_harness_prefill(|_| vec![1]).unwrap();
    let m_read = read_relay
        .plan_sampled_trunk_token(TOKEN_SPACE_READ, " READ")
        .unwrap();
    assert_eq!(m_read.consume_bank, CartridgeBank::OrientTrunk);
    let m_br = read_relay
        .plan_sampled_trunk_token(TOKEN_CLOSE_BRACKET, "]")
        .unwrap();
    assert_eq!(m_br.producer_bank, CartridgeBank::OrientTrunk);
    assert_eq!(m_br.consume_bank, CartridgeBank::OrientTrunk);
    assert_eq!(m_br.phase_after, RelayPhase::PostModePrefill);

    read_relay.flush_harness_prefill(|_| vec![2]).unwrap();
    let pl_step = read_relay
        .plan_sampled_trunk_token(400, " Inspect literal </sql> safely.\n")
        .unwrap();
    assert_eq!(pl_step.producer_bank, CartridgeBank::Base);
    assert_eq!(pl_step.consume_bank, CartridgeBank::Base);
    assert_eq!(pl_step.phase_after, RelayPhase::PostPlanPrefill);
    read_relay.flush_harness_prefill(|_| vec![3]).unwrap();

    let body_pieces = [
        "SELECT '",
        "</",
        "sql",
        ">' AS tag_lit, 'λ🚀' AS utf8_lit;\n",
        "</",
        "sql",
    ];
    for (i, piece) in body_pieces.iter().enumerate() {
        let step = read_relay
            .plan_sampled_trunk_token(500 + i as u32, piece)
            .unwrap();
        assert_eq!(step.producer_bank, CartridgeBank::ModePayload(Mode::Read));
        assert_eq!(
            step.consume_bank,
            CartridgeBank::ModePayload(Mode::Read),
            "intermediate or quoted-literal piece {piece:?} at index {i} must consume under ModePayload(Read)"
        );
        assert_eq!(step.phase_after, RelayPhase::PayloadBody);
    }

    let close_step = read_relay.plan_sampled_trunk_token(599, ">").unwrap();
    assert_eq!(
        close_step.producer_bank,
        CartridgeBank::ModePayload(Mode::Read)
    );
    assert_eq!(close_step.consume_bank, CartridgeBank::Base);
    assert_eq!(close_step.phase_after, RelayPhase::TurnComplete);

    let completed = read_relay
        .into_completed_turn(|_| vec![248046, 198])
        .unwrap();
    assert_eq!(
        completed.payload,
        "SELECT '</sql>' AS tag_lit, 'λ🚀' AS utf8_lit;"
    );
}

proptest! {
    #[test]
    fn prop_relay_single_pen_schedule_holds_across_all_modes_and_space_variants(
        mode_idx in 0usize..6usize,
        use_leading_space in any::<bool>(),
        use_leading_plan_space in any::<bool>(),
        use_double_newline in any::<bool>(),
        plan_word in "[A-Za-z0-9_][A-Za-z0-9_ ]{2,23}",
        sql_col in "[a-z_]{2,10}",
    ) {
        let mode = Mode::ALL[mode_idx];
        let mut relay = RelayMachine::new();
        relay.flush_harness_prefill(|_| vec![58, 22098, 25]).unwrap();

        let header_ids = canonical_mode_header_tokens_under_orient(mode);
        let mode_str = if use_leading_space {
            format!(" {}]", mode.as_str())
        } else {
            format!("{}]", mode.as_str())
        };
        let m_span = relay.step_trunk_token(header_ids[0], &mode_str).unwrap();
        prop_assert_eq!(m_span.consumer_bank, CartridgeBank::OrientTrunk);

        let post_mode = relay.flush_harness_prefill(|_| vec![198, 19860, 25]).unwrap().unwrap();
        prop_assert_eq!(post_mode.consumer_bank, CartridgeBank::Base);
        prop_assert_eq!(post_mode.text, "\nPlan:");

        let plan_prefix = if use_leading_plan_space { " " } else { "" };
        let plan_emit = if use_double_newline {
            format!("{}{}\n\n", plan_prefix, plan_word.trim())
        } else {
            format!("{}{}\n", plan_prefix, plan_word.trim())
        };
        relay.step_trunk_token(1000, &plan_emit).unwrap();
        let post_plan = relay.flush_harness_prefill(|_| vec![248069, 198]).unwrap().unwrap();
        prop_assert_eq!(post_plan.text, format!("</think>\n{}\n", mode.route_tag().open_tag()));

        let close_tag = mode.route_tag().close_tag();
        let payload_body = match mode {
            Mode::Probe | Mode::Read | Mode::Write => format!("SELECT {} FROM items;", sql_col),
            Mode::Ask => format!("May we inspect {}?", sql_col),
            Mode::Result | Mode::Abstain => format!("{{\"field\": \"{}\"}}", sql_col),
        };
        relay.step_trunk_token(2000, &format!("{}\n{}", payload_body, close_tag)).unwrap();
        let completed = relay.into_completed_turn(|_| vec![248046, 198]).unwrap();
        prop_assert_eq!(completed.mode, mode);
        prop_assert_eq!(completed.plan, plan_word.trim());
        prop_assert_eq!(completed.payload, payload_body);
        prop_assert_eq!(completed.spans.last().unwrap().text.as_str(), "<|im_end|>\n");
    }

    #[test]
    fn prop_sql_and_json_scanners_and_wire_normalizer_invariants(
        prefix_ident in "[a-z_]{2,10}",
        quoted_mid in "[a-zA-Z0-9_ ]{1,12}",
        val_a in 0i64..1000i64,
        val_z in 0i64..1000i64,
    ) {
        let s_single = format!("SELECT '{quoted_mid};WHERE)</sql>' FROM {prefix_ident}");
        prop_assert!(!sql_has_unclosed_quote(&s_single));
        prop_assert_eq!(find_unquoted_sql_char(&s_single, ';'), None);
        prop_assert!(!sql_has_unquoted_where(&s_single));
        prop_assert!(!has_completed_close_tag_outside_literals(&s_single, Mode::Read));

        let s_with_semi_and_where = format!("{s_single} WHERE {prefix_ident} = 1;\n</sql>");
        prop_assert_eq!(
            find_unquoted_sql_char(&s_with_semi_and_where, ';'),
            Some(s_single.len() + format!(" WHERE {prefix_ident} = 1").len())
        );
        prop_assert!(sql_has_unquoted_where(&s_with_semi_and_where));
        prop_assert!(has_completed_close_tag_outside_literals(&s_with_semi_and_where, Mode::Read));

        let json_unclosed = format!("{{\"k\": \"{quoted_mid}</to_user>");
        prop_assert!(json_has_unclosed_string(&json_unclosed));
        prop_assert!(!has_completed_close_tag_outside_literals(&json_unclosed, Mode::Result));

        let json_closed = format!("{json_unclosed}\"}}\n</to_user>");
        prop_assert!(!json_has_unclosed_string(&json_closed));
        prop_assert!(has_completed_close_tag_outside_literals(&json_closed, Mode::Result));

        let raw_tr = format!(
            "<|im_start|>user\n<tool_response>\n{{\"z_key\": {val_z}, \"a_key\": {val_a}}}\n</tool_response><|im_end|>\n"
        );
        let norm_tr = normalize_lotl_wire_chunk(&raw_tr, None);
        let expected_tr = format!(
            "<|im_start|>user\n<tool_response>\n{{\"a_key\":{val_a},\"z_key\":{val_z}}}\n</tool_response><|im_end|>\n"
        );
        prop_assert_eq!(norm_tr, expected_tr);
    }

    #[test]
    fn prop_lotl_normalized_wire_chunk_is_idempotent_for_arbitrary_strings(
        raw in any::<String>(),
        mode_choice in 0usize..7usize,
    ) {
        let selected_mode = if mode_choice < 6 {
            Some(Mode::ALL[mode_choice])
        } else {
            None
        };
        let once = LotlNormalizedWireChunk::normalize(&raw, selected_mode);
        let twice = LotlNormalizedWireChunk::normalize(once.as_str(), selected_mode);
        prop_assert_eq!(twice.as_str(), once.as_str());
        prop_assert_eq!(
            serde_json::to_value(&once).unwrap(),
            serde_json::Value::String(once.as_str().to_string())
        );
        let raw_json = serde_json::to_string(&raw).unwrap();
        let via_serde: LotlNormalizedWireChunk = serde_json::from_str(&raw_json).unwrap();
        prop_assert_eq!(via_serde, once);
    }

    #[test]
    fn prop_plan_sampled_trunk_token_handoff_across_split_tags_and_quoted_literals(
        mode_idx in 0usize..6usize,
        split_mode_bracket in any::<bool>(),
        split_plan_newline in any::<bool>(),
        plan_stem in "[A-Za-z0-9_]{3,16}",
        unicode_mid in "[α-ωΑ-Ωデータ🚀]{1,6}",
        close_split_1 in 1usize..4usize,
        close_split_2 in 4usize..6usize,
    ) {
        let mode = Mode::ALL[mode_idx];
        let payload_bank = CartridgeBank::ModePayload(mode);
        let mut relay = RelayMachine::new();
        relay.flush_harness_prefill(|_| vec![1]).unwrap();

        if split_mode_bracket {
            let s1 = relay
                .plan_sampled_trunk_token(10, &format!(" {}", mode.as_str()))
                .unwrap();
            prop_assert_eq!(s1.producer_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s1.consume_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s1.phase_after, RelayPhase::ModeLabel);

            let s2 = relay.plan_sampled_trunk_token(60, "]").unwrap();
            prop_assert_eq!(s2.producer_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s2.consume_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s2.phase_after, RelayPhase::PostModePrefill);
        } else {
            let s = relay
                .plan_sampled_trunk_token(10, &format!(" {}]", mode.as_str()))
                .unwrap();
            prop_assert_eq!(s.producer_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s.consume_bank, CartridgeBank::OrientTrunk);
            prop_assert_eq!(s.phase_after, RelayPhase::PostModePrefill);
        }

        let post_mode = relay.flush_harness_prefill(|_| vec![198, 19860, 25]).unwrap().unwrap();
        prop_assert_eq!(post_mode.consumer_bank, CartridgeBank::Base);

        if split_plan_newline {
            let p1 = relay
                .plan_sampled_trunk_token(20, &format!(" {plan_stem} {unicode_mid}"))
                .unwrap();
            prop_assert_eq!(p1.producer_bank, CartridgeBank::Base);
            prop_assert_eq!(p1.consume_bank, CartridgeBank::Base);
            prop_assert_eq!(p1.phase_after, RelayPhase::PlanBody);

            let p2 = relay.plan_sampled_trunk_token(198, "\n").unwrap();
            prop_assert_eq!(p2.producer_bank, CartridgeBank::Base);
            prop_assert_eq!(p2.consume_bank, CartridgeBank::Base);
            prop_assert_eq!(p2.phase_after, RelayPhase::PostPlanPrefill);
        } else {
            let p = relay
                .plan_sampled_trunk_token(20, &format!(" {plan_stem} {unicode_mid}\n"))
                .unwrap();
            prop_assert_eq!(p.producer_bank, CartridgeBank::Base);
            prop_assert_eq!(p.consume_bank, CartridgeBank::Base);
            prop_assert_eq!(p.phase_after, RelayPhase::PostPlanPrefill);
        }

        relay.flush_harness_prefill(|_| vec![248069, 198]).unwrap();

        let close_tag = mode.route_tag().close_tag();
        let c1 = close_split_1.min(close_tag.len() - 2);
        let c2 = close_split_2.clamp(c1 + 1, close_tag.len() - 1);
        let tag_p1 = &close_tag[..c1];
        let tag_p2 = &close_tag[c1..c2];
        let tag_p3 = &close_tag[c2..];

        let body_tokens: Vec<String> = match mode {
            Mode::Probe | Mode::Read | Mode::Write => vec![
                "SELECT '".to_string(),
                tag_p1.to_string(),
                format!("{tag_p2}{tag_p3}"),
                format!(" {unicode_mid}' FROM {plan_stem};\n"),
                tag_p1.to_string(),
                tag_p2.to_string(),
            ],
            Mode::Result | Mode::Abstain => vec![
                "{\"lit\": \"".to_string(),
                tag_p1.to_string(),
                format!("{tag_p2}{tag_p3}"),
                format!(" {unicode_mid}\"}}\n"),
                tag_p1.to_string(),
                tag_p2.to_string(),
            ],
            Mode::Ask => vec![
                format!("Clarify {plan_stem} ({unicode_mid})?\n"),
                tag_p1.to_string(),
                tag_p2.to_string(),
            ],
        };

        for (idx, piece) in body_tokens.iter().enumerate() {
            let step = relay.plan_sampled_trunk_token(100 + idx as u32, piece).unwrap();
            prop_assert_eq!(step.producer_bank, payload_bank);
            prop_assert_eq!(step.consume_bank, payload_bank);
            prop_assert_eq!(step.phase_after, RelayPhase::PayloadBody);
        }

        let final_step = relay.plan_sampled_trunk_token(199, tag_p3).unwrap();
        prop_assert_eq!(final_step.producer_bank, payload_bank);
        prop_assert_eq!(final_step.consume_bank, CartridgeBank::Base);
        prop_assert_eq!(final_step.phase_after, RelayPhase::TurnComplete);

        let completed = relay.into_completed_turn(|_| vec![248046, 198]).unwrap();
        prop_assert_eq!(completed.mode, mode);
    }

    #[test]
    fn prop_lotl_normalized_wire_chunk_is_idempotent_on_adversarial_tag_and_json_mixes(
        fragments in proptest::collection::vec(
            prop_oneof![
                Just("\n<|im_end|>\n".to_string()),
                Just("\nPlan: ".to_string()),
                Just("<|im_start|>assistant\n".to_string()),
                Just("<|im_start|>user\n".to_string()),
                Just("<sql>".to_string()),
                Just("</sql>".to_string()),
                Just("<to_user>".to_string()),
                Just("</to_user>".to_string()),
                Just("<to_database>".to_string()),
                Just("<to_inquirer>".to_string()),
                Just("<tool_response>\n{\"z\":1,\"\\u003cto_database\\u003e\":2,\"<t\":3}\n</tool_response>".to_string()),
                Just("<tool_response>\n\"\\u003cto_inquirer\\u003e\"\n</tool_response>".to_string()),
                "[A-Za-z0-9_{}:,\"'<>\\n ]{0,24}",
            ],
            0..8
        ),
        mode_choice in 0usize..7usize,
    ) {
        let selected_mode = if mode_choice < 6 {
            Some(Mode::ALL[mode_choice])
        } else {
            None
        };
        let raw = fragments.concat();
        let once = LotlNormalizedWireChunk::normalize(&raw, selected_mode);
        let twice = LotlNormalizedWireChunk::normalize(once.as_str(), selected_mode);
        prop_assert_eq!(&twice, &once);
        let raw_json = serde_json::to_string(&raw).unwrap();
        let via_serde: LotlNormalizedWireChunk = serde_json::from_str(&raw_json).unwrap();
        prop_assert_eq!(via_serde, once);
    }
}
