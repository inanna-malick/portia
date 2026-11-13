use crate::{
    GrammarAuditSuite, GrammarHarnessError, GrammarUnderTest, GrammarVerifier, TokenizerHarness,
    TraceVerdict,
};
use serde_json::json;

fn harness(merges: &[&[u8]]) -> TokenizerHarness {
    let merges = merges
        .iter()
        .map(|bytes| bytes.to_vec())
        .collect::<Vec<_>>();
    TokenizerHarness::synthetic_byte_bpe(&merges, b"<eos>").unwrap()
}

fn verifier(grammar: GrammarUnderTest) -> GrammarVerifier {
    GrammarVerifier::new(harness(&[]), grammar).unwrap()
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn assert_accepted(verdict: TraceVerdict, expected_steps: usize) {
    match verdict {
        TraceVerdict::Accepted {
            steps,
            max_step_micros,
            total_micros,
        } => {
            assert_eq!(steps, expected_steps);
            assert!(max_step_micros <= total_micros);
        }
        other => panic!("expected acceptance, got {other:?}"),
    }
}

fn object_schema() -> GrammarUnderTest {
    GrammarUnderTest::JsonSchema(json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": false
    }))
}

fn suite(grammar: GrammarUnderTest, accept: &[&str], reject: &[&str]) -> GrammarAuditSuite {
    GrammarAuditSuite {
        name: "contract".into(),
        grammar,
        must_accept: accept.iter().map(|text| (*text).into()).collect(),
        must_reject: reject.iter().map(|text| (*text).into()).collect(),
        max_step_micros_budget: u64::MAX,
    }
}

#[test]
fn all_grammar_kinds_distinguish_complete_prefix_and_trailing_input() {
    for (grammar, valid, prefix) in [
        (GrammarUnderTest::Lark("start: \"ok\"".into()), "ok", "o"),
        (GrammarUnderTest::Regex("ok".into()), "ok", "o"),
        (object_schema(), "{\"a\":1}", "{\"a\":1"),
    ] {
        let verifier = verifier(grammar);
        assert_accepted(verifier.verify_string_greedy(valid).unwrap(), valid.len());
        assert_eq!(
            verifier.verify_string_greedy(prefix).unwrap(),
            TraceVerdict::IncompleteAtEnd {
                steps: prefix.len()
            }
        );
        let trailing = format!("{valid}x");
        assert!(matches!(
            verifier.verify_string_greedy(&trailing).unwrap(),
            TraceVerdict::MaskRejectedAtStep { step_index, token_id, .. }
                if step_index == valid.len() && token_id == u32::from(b'x')
        ));
    }
}

#[test]
fn nullable_grammar_and_empty_prefix_are_distinct() {
    let nullable = verifier(GrammarUnderTest::Regex("a*".into()));
    assert_accepted(nullable.verify_token_sequence(&[]).unwrap(), 0);
    assert_accepted(nullable.verify_string_under_partition("", &[]).unwrap(), 0);
    let nonnullable = verifier(GrammarUnderTest::Regex("a+".into()));
    assert_eq!(
        nonnullable.verify_token_sequence(&[]).unwrap(),
        TraceVerdict::IncompleteAtEnd { steps: 0 }
    );
}

#[test]
fn terminal_eos_is_allowed_once_including_after_automatic_stop() {
    for (pattern, text) in [("a*", ""), ("ok", "ok"), ("[a-z]+", "ok")] {
        let verifier = verifier(GrammarUnderTest::Regex(pattern.into()));
        let eos = verifier.harness().tok_env().eos_token();
        let mut tokens = bytes(text);
        tokens.push(eos);
        assert_accepted(
            verifier.verify_token_sequence(&tokens).unwrap(),
            tokens.len(),
        );
        assert_accepted(
            verifier.verify_token_sequence_canonical(&tokens).unwrap(),
            tokens.len(),
        );
        tokens.push(eos);
        assert!(matches!(
            verifier.verify_token_sequence(&tokens).unwrap(),
            TraceVerdict::MaskRejectedAtStep { step_index, token_id, .. }
                if step_index == text.len() + 1 && token_id == eos
        ));
        *tokens.last_mut().unwrap() = u32::from(b'a');
        assert!(matches!(
            verifier.verify_token_sequence(&tokens).unwrap(),
            TraceVerdict::MaskRejectedAtStep { step_index, token_id, .. }
                if step_index == text.len() + 1 && token_id == u32::from(b'a')
        ));
    }
}

#[test]
fn premature_eos_is_rejected_at_its_position() {
    let verifier = verifier(GrammarUnderTest::Regex("ok".into()));
    let eos = verifier.harness().tok_env().eos_token();
    for tokens in [vec![eos], vec![u32::from(b'o'), eos]] {
        assert!(matches!(
            verifier.verify_token_sequence(&tokens).unwrap(),
            TraceVerdict::MaskRejectedAtStep { step_index, token_id, .. }
                if step_index == tokens.len() - 1 && token_id == eos
        ));
    }
}

#[test]
fn invalid_ids_are_rejected_without_becoming_empty_payload() {
    let verifier = verifier(GrammarUnderTest::Regex("ok".into()));
    let vocab_size = verifier.harness().tok_env().tok_trie().vocab_size() as u32;
    for invalid in [vocab_size, u32::MAX] {
        assert_eq!(
            verifier.verify_token_sequence(&[invalid]).unwrap(),
            TraceVerdict::MaskRejectedAtStep {
                step_index: 0,
                token_id: invalid,
                token_bytes: Vec::new(),
            }
        );
        let mut partition = bytes("ok");
        partition.push(invalid);
        assert!(matches!(
            verifier.verify_string_under_partition("ok", &partition),
            Err(GrammarHarnessError::PartitionBytesMismatch)
        ));
    }
}

#[test]
fn partitions_require_exact_payload_bytes_and_exclude_eos() {
    let verifier = verifier(GrammarUnderTest::Regex("ok".into()));
    assert_accepted(
        verifier
            .verify_string_under_partition("ok", &bytes("ok"))
            .unwrap(),
        2,
    );
    let eos = verifier.harness().tok_env().eos_token();
    for partition in [
        bytes("o"),
        bytes("ox"),
        vec![eos],
        vec![b'o' as u32, b'k' as u32, eos],
    ] {
        assert!(matches!(
            verifier.verify_string_under_partition("ok", &partition),
            Err(GrammarHarnessError::PartitionBytesMismatch)
        ));
    }
}

#[test]
fn canonical_forcing_and_unforced_partitions_have_distinct_contracts() {
    let harness = harness(&[b"SELECT"]);
    assert!(harness.tok_env().tokenize_is_canonical());
    let grammar = GrammarUnderTest::Lark("start: \"SELECT \" /[a-z]+/ \";\"".into());
    let verifier = GrammarVerifier::new(harness, grammar.clone()).unwrap();
    let text = "SELECT x;";
    let byte_partition = bytes(text);
    assert_accepted(
        verifier.verify_token_sequence(&byte_partition).unwrap(),
        text.len(),
    );
    assert_accepted(
        verifier
            .verify_string_under_partition(text, &byte_partition)
            .unwrap(),
        text.len(),
    );
    assert!(matches!(
        verifier
            .verify_token_sequence_canonical(&byte_partition)
            .unwrap(),
        TraceVerdict::MaskRejectedAtStep { step_index: 0, .. }
    ));
    let canonical = verifier.harness().tokenize_str(text);
    assert!(canonical.len() < text.len());
    assert_accepted(
        verifier.verify_string_greedy(text).unwrap(),
        canonical.len(),
    );
    let receipt = verifier
        .verify_batch_suite(&suite(grammar, &[text], &[]))
        .unwrap();
    assert_eq!(receipt.total_mask_steps, canonical.len());
}

#[test]
fn straddled_sql_delimiters_are_accepted_greedily_and_bytewise() {
    let harness = harness(&[b";\n</sql>"]);
    let grammar =
        GrammarUnderTest::Lark("start: \"<sql>\" /[^<>\\r\\n;]+/ \";\" \"\\n\" \"</sql>\"".into());
    let verifier = GrammarVerifier::new(harness, grammar).unwrap();
    let text = "<sql>SELECT x;\n</sql>";
    let greedy = verifier.harness().tokenize_str(text);
    assert!(greedy.len() < text.len());
    assert_accepted(verifier.verify_string_greedy(text).unwrap(), greedy.len());
    assert_accepted(
        verifier
            .verify_string_under_partition(text, &bytes(text))
            .unwrap(),
        text.len(),
    );
    assert!(!verifier
        .verify_string_greedy("<sql>SELECT x;\n</sql>x")
        .unwrap()
        .is_accepted());
}

#[test]
fn straddled_json_delimiters_and_framed_newline_are_accepted() {
    let schema = GrammarUnderTest::JsonSchema(json!({
        "type": "object", "properties": {"a": {"type": "string"}},
        "required": ["a"], "additionalProperties": false
    }));
    let json_line = GrammarUnderTest::Lark(r#"start: "{\"a\":\"" /[a-z]+/ "\"}" "\n""#.into());
    for (grammar, text) in [(schema, "{\"a\":\"x\"}"), (json_line, "{\"a\":\"x\"}\n")] {
        let harness = harness(&[b"\"}\n", b"\"}", b"\":\""]);
        let verifier = GrammarVerifier::new(harness, grammar).unwrap();
        let greedy = verifier.harness().tokenize_str(text);
        assert!(greedy.len() < text.len());
        assert_accepted(verifier.verify_string_greedy(text).unwrap(), greedy.len());
        assert_accepted(
            verifier
                .verify_string_under_partition(text, &bytes(text))
                .unwrap(),
            text.len(),
        );
    }
}

#[test]
fn utf8_byte_splits_require_complete_codepoints() {
    let verifier = verifier(GrammarUnderTest::Lark(
        "start: \"<s>\" /[^\\r\\n<>]+/ \"</s>\"".into(),
    ));
    let text = "<s>€🦀無</s>";
    assert_accepted(
        verifier
            .verify_string_under_partition(text, &bytes(text))
            .unwrap(),
        text.len(),
    );
    let mut truncated = bytes("<s>");
    truncated.extend([0xf0, 0x9f]);
    assert_eq!(
        verifier.verify_token_sequence(&truncated).unwrap(),
        TraceVerdict::IncompleteAtEnd { steps: 5 }
    );
    truncated.extend(bytes("</s>"));
    assert!(matches!(
        verifier.verify_token_sequence(&truncated).unwrap(),
        TraceVerdict::MaskRejectedAtStep { step_index: 5, .. }
    ));
}

#[test]
fn malformed_and_unsupported_grammars_return_structured_errors() {
    for grammar in [
        GrammarUnderTest::Lark("start: (".into()),
        GrammarUnderTest::Regex("[".into()),
        GrammarUnderTest::Regex("a\\".into()),
        GrammarUnderTest::JsonSchema(json!(42)),
        GrammarUnderTest::JsonSchema(json!({"type": "array", "uniqueItems": true})),
        GrammarUnderTest::JsonSchema(json!({
            "type": "array", "uniqueItems": true, "x-guidance": {"lenient": true}
        })),
    ] {
        assert!(matches!(
            GrammarVerifier::new(harness(&[]), grammar),
            Err(GrammarHarnessError::InvalidGrammar(_))
        ));
    }
}

#[test]
fn supported_regex_escapes_and_schema_extensions_remain_valid() {
    for (grammar, text) in [
        (GrammarUnderTest::Regex(r"a\\".into()), "a\\"),
        (GrammarUnderTest::Regex("a/b".into()), "a/b"),
        (
            GrammarUnderTest::JsonSchema(json!({
                "type": "integer", "description": "count", "x-example": true
            })),
            "1",
        ),
    ] {
        assert_accepted(
            verifier(grammar).verify_string_greedy(text).unwrap(),
            text.len(),
        );
    }
}

#[test]
fn suite_counts_inspected_masks_across_accepted_and_rejected_traces() {
    let grammar = GrammarUnderTest::Regex("ok".into());
    let verifier = verifier(grammar.clone());
    let receipt = verifier
        .verify_batch_suite(&suite(grammar, &["ok"], &["o", "okx", "x"]))
        .unwrap();
    assert_eq!(receipt.name, "contract");
    assert_eq!(receipt.accepted_verified, 1);
    assert_eq!(receipt.rejected_verified, 3);
    assert_eq!(receipt.total_mask_steps, 2 + 1 + 3 + 1);
}

#[test]
fn suites_reject_grammar_mismatch_and_incorrect_expectations() {
    let grammar = GrammarUnderTest::Regex("ok".into());
    let verifier = verifier(grammar.clone());
    let different = GrammarUnderTest::Regex("no".into());
    assert!(matches!(
        verifier.verify_batch_suite(&suite(different, &[], &[])),
        Err(GrammarHarnessError::InvalidGrammar(_))
    ));
    assert!(matches!(
        verifier.verify_batch_suite(&suite(grammar.clone(), &["o"], &[])),
        Err(GrammarHarnessError::UnexpectedRejection { input, verdict })
            if input == "o" && verdict == TraceVerdict::IncompleteAtEnd { steps: 1 }
    ));
    assert!(matches!(
        verifier.verify_batch_suite(&suite(grammar, &[], &["ok"])),
        Err(GrammarHarnessError::UnexpectedAcceptance { input }) if input == "ok"
    ));
}

#[test]
fn empty_suite_has_no_masks_or_timing() {
    let grammar = GrammarUnderTest::Regex("a*".into());
    let verifier = verifier(grammar.clone());
    let receipt = verifier
        .verify_batch_suite(&suite(grammar.clone(), &[], &[]))
        .unwrap();
    assert_eq!(receipt.total_mask_steps, 0);
    assert_eq!(receipt.worst_step_micros, 0);
    let nullable = verifier
        .verify_batch_suite(&suite(grammar, &[""], &[]))
        .unwrap();
    assert_eq!(nullable.accepted_verified, 1);
    assert_eq!(nullable.total_mask_steps, 0);
    assert_eq!(nullable.worst_step_micros, 0);
}

#[test]
fn repeated_calls_start_from_fresh_parser_state() {
    let verifier = verifier(GrammarUnderTest::Regex("ok".into()));
    assert!(!verifier.verify_string_greedy("x").unwrap().is_accepted());
    assert_accepted(verifier.verify_string_greedy("ok").unwrap(), 2);
    assert_eq!(
        verifier.verify_string_greedy("o").unwrap(),
        TraceVerdict::IncompleteAtEnd { steps: 1 }
    );
    assert_accepted(verifier.verify_string_greedy("ok").unwrap(), 2);
}

#[test]
fn matcher_failure_cannot_satisfy_a_negative_suite_case() {
    let grammar = GrammarUnderTest::Regex("ok".into());
    let mut verifier = verifier(grammar.clone());
    verifier.canonical = verifier.canonical.clone().into_cancellable();
    verifier.canonical.cancellation_handle().unwrap().cancel();
    assert!(matches!(
        verifier.verify_batch_suite(&suite(grammar, &[], &["x"])),
        Err(GrammarHarnessError::InvalidGrammar(_))
    ));
}
