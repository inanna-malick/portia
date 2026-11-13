use crate::strategies::{arb_adversarial_bpe_merges, arb_token_partition_for_bytes};
use crate::*;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{TestError, TestRunner};
use serde_json::json;

const SQL: &str = r#"start: "<sql>" statement "\n" close
statement: "SELECT " identifier " FROM " identifier ";"
identifier: /[a-z]{1,8}/
close: "</sql>""#;
const TEXT: &str = r#"start: "<text>" /[^\r\n<>]+/ "</text>""#;

fn env(merges: &[Vec<u8>]) -> TokenizerHarness {
    TokenizerHarness::synthetic_byte_bpe(merges, b"<eos>").expect("synthetic vocabulary")
}

fn raw(env: &TokenizerHarness, tokens: &[u32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|&id| env.tok_env().tok_trie().token(id))
        .copied()
        .collect()
}

fn byte_tokens(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn json_grammar(kind: &str) -> GrammarUnderTest {
    GrammarUnderTest::JsonSchema(json!({
        "type": "object", "properties": {"a": {"type": kind}},
        "required": ["a"], "additionalProperties": false
    }))
}

fn unicode() -> BoxedStrategy<String> {
    proptest::collection::vec(any::<char>(), 0..12)
        .prop_map(|chars| chars.into_iter().collect())
        .boxed()
}

fn json_string_object(body: String) -> String {
    // llguidance 1.9 rejects a literal DEL in JSON strings; its escaped spelling
    // preserves the generated value while staying in the native JSON language.
    json!({"a": body}).to_string().replace('\u{7f}', "\\u007f")
}

fn sql_text() -> BoxedStrategy<String> {
    ("[a-z]{1,8}", "[a-z]{1,8}")
        .prop_map(|(column, table)| format!("<sql>SELECT {column} FROM {table};\n</sql>"))
        .boxed()
}

fn sql_oracle(text: &str) -> bool {
    let Some(body) = text
        .strip_prefix("<sql>SELECT ")
        .and_then(|s| s.strip_suffix(";\n</sql>"))
    else {
        return false;
    };
    let Some((column, table)) = body.split_once(" FROM ") else {
        return false;
    };
    [column, table]
        .iter()
        .all(|s| (1..=8).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_lowercase()))
}

type PartitionCase = (String, Vec<Vec<u8>>, Vec<u32>);

fn case(text: impl Strategy<Value = String> + 'static) -> BoxedStrategy<PartitionCase> {
    (
        text,
        arb_adversarial_bpe_merges(vec![b";\n</sql>".to_vec()], 12),
    )
        .prop_flat_map(|(text, merges)| {
            let partitions = arb_token_partition_for_bytes(env(&merges), text.as_bytes().to_vec());
            partitions.prop_map(move |tokens| (text.clone(), merges.clone(), tokens))
        })
        .boxed()
}

fn crosses(env: &TokenizerHarness, tokens: &[u32], boundary: usize) -> bool {
    let mut offset = 0;
    tokens.iter().any(|&id| {
        let end = offset + env.tok_env().tok_trie().token(id).len();
        let crossing = offset < boundary && boundary < end;
        offset = end;
        crossing
    })
}

fn force_merge(text: &str, merge: &[u8], env: &TokenizerHarness) -> Vec<u32> {
    let at = text
        .as_bytes()
        .windows(merge.len())
        .position(|w| w == merge)
        .unwrap();
    let id = env
        .tok_env()
        .tok_trie()
        .token_id(merge)
        .expect("merge token");
    text.as_bytes()[..at]
        .iter()
        .copied()
        .map(u32::from)
        .chain(std::iter::once(id))
        .chain(
            text.as_bytes()[at + merge.len()..]
                .iter()
                .copied()
                .map(u32::from),
        )
        .collect()
}

fn unicode_partitions() -> BoxedStrategy<PartitionCase> {
    (
        proptest::collection::vec(proptest::sample::select(vec!['é', '€', '🦀', '無']), 1..10),
        arb_adversarial_bpe_merges(vec![], 12),
    )
        .prop_flat_map(|(symbols, mut merges)| {
            // A leading crab guarantees both an incomplete two-byte prefix and
            // a pair crossing from its final byte into the next codepoint.
            let body = format!("🦀{}", symbols.into_iter().collect::<String>());
            merges.extend(body.as_bytes().windows(2).map(<[u8]>::to_vec));
            let text = format!("<text>{body}</text>");
            arb_token_partition_for_bytes(env(&merges), text.as_bytes().to_vec())
                .prop_map(move |tokens| (text.clone(), merges.clone(), tokens))
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn synthetic_greedy_roundtrip(text in unicode(), merges in arb_adversarial_bpe_merges(vec![], 16)) {
        let env = env(&merges);
        let tokens = env.tokenize_str(&text);
        prop_assert_eq!(raw(&env, &tokens), text.as_bytes());
        prop_assert!(!tokens.contains(&env.tok_env().eos_token()));
        prop_assert!(merges.iter().all(|m| m.len() >= 2));
    }

    #[test]
    fn partition_generation_and_shrinking_preserve_bytes((text, merges, tokens) in case(unicode())) {
        let env = env(&merges);
        prop_assert_eq!(raw(&env, &tokens), text.as_bytes());
        let mut runner = TestRunner::new(ProptestConfig { failure_persistence: None, ..ProptestConfig::default() });
        let mut tree = arb_token_partition_for_bytes(env.clone(), text.as_bytes().to_vec()).new_tree(&mut runner).expect("byte-fallback coverage");
        for _ in 0..64 {
            prop_assert_eq!(raw(&env, &tree.current()), text.as_bytes());
            if !tree.simplify() { break; }
        }
    }

    #[test]
    fn json_objects_accept_partitions((text, merges, tokens) in case((0u32..10000).prop_map(|n| json!({"a": n}).to_string()))) {
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        prop_assert!(parsed["a"].is_u64());
        let verifier = GrammarVerifier::new(env(&merges), json_grammar("integer")).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
        prop_assert!(verifier.verify_token_sequence(&byte_tokens(&text))?.is_accepted());
        prop_assert!(verifier.verify_string_greedy(&text)?.is_accepted());
    }

    #[test]
    fn json_escaped_strings_accept_partitions((text, merges, tokens) in case(unicode().prop_map(json_string_object))) {
        prop_assert!(serde_json::from_str::<serde_json::Value>(&text).unwrap()["a"].is_string());
        let verifier = GrammarVerifier::new(env(&merges), json_grammar("string")).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
        prop_assert!(verifier.verify_token_sequence(&byte_tokens(&text))?.is_accepted());
    }

    #[test]
    fn json_truncated_objects_are_incomplete(n in 0u32..10000, merges in arb_adversarial_bpe_merges(vec![], 12)) {
        let mut text = json!({"a": n}).to_string();
        text.pop();
        prop_assert!(serde_json::from_str::<serde_json::Value>(&text).is_err());
        let verifier = GrammarVerifier::new(env(&merges), json_grammar("integer")).unwrap();
        prop_assert!(matches!(verifier.verify_token_sequence(&byte_tokens(&text))?, TraceVerdict::IncompleteAtEnd { .. }), "truncated object must remain incomplete");
    }

    #[test]
    fn json_trailing_garbage_is_rejected((text, merges, tokens) in case((0u32..10000).prop_map(|n| format!("{}!", json!({"a": n}))))) {
        prop_assert!(serde_json::from_str::<serde_json::Value>(&text).is_err());
        let verifier = GrammarVerifier::new(env(&merges), json_grammar("integer")).unwrap();
        prop_assert!(!verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
    }

    #[test]
    fn json_schema_type_mutations_are_rejected((text, merges, tokens) in case(unicode().prop_map(|s| json!({"a": s}).to_string()))) {
        prop_assert!(!serde_json::from_str::<serde_json::Value>(&text).unwrap()["a"].is_number());
        let verifier = GrammarVerifier::new(env(&merges), json_grammar("integer")).unwrap();
        prop_assert!(!verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
    }

    #[test]
    fn sql_strings_accept_partitions((text, merges, tokens) in case(sql_text())) {
        prop_assert!(sql_oracle(&text));
        let verifier = GrammarVerifier::new(env(&merges), GrammarUnderTest::Lark(SQL.into())).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
        prop_assert!(verifier.verify_token_sequence(&byte_tokens(&text))?.is_accepted());
        prop_assert!(verifier.verify_string_greedy(&text)?.is_accepted());
    }

    #[test]
    fn sql_closing_boundary_merge_is_exercised(text in sql_text(), mut merges in arb_adversarial_bpe_merges(vec![], 12)) {
        let merge = b";\n</sql>";
        merges.push(merge.to_vec());
        let env = env(&merges);
        let tokens = force_merge(&text, merge, &env);
        let boundary = text.find(';').unwrap() + 1;
        prop_assert!(crosses(&env, &tokens, boundary));
        prop_assert!(!crosses(&env, &byte_tokens(&text), boundary));
        prop_assert!(sql_oracle(&text));
        let verifier = GrammarVerifier::new(env, GrammarUnderTest::Lark(SQL.into())).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
    }

    #[test]
    fn json_delimiter_merge_is_exercised(body in "[a-z]{0,12}", mut merges in arb_adversarial_bpe_merges(vec![], 12)) {
        let text = json!({"a": body}).to_string();
        let merge = b"\":\"";
        merges.push(merge.to_vec());
        let env = env(&merges);
        let tokens = force_merge(&text, merge, &env);
        prop_assert!(crosses(&env, &tokens, 4));
        prop_assert!(!crosses(&env, &byte_tokens(&text), 4));
        prop_assert!(serde_json::from_str::<serde_json::Value>(&text).is_ok());
        let verifier = GrammarVerifier::new(env, json_grammar("string")).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
    }

    #[test]
    fn sql_forbidden_delimiter_injection_is_rejected((text, merges, tokens) in case(sql_text().prop_map(|s| s.replacen(" FROM ", "< FROM ", 1)))) {
        prop_assert!(!sql_oracle(&text));
        let verifier = GrammarVerifier::new(env(&merges), GrammarUnderTest::Lark(SQL.into())).unwrap();
        prop_assert!(!verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
    }

    #[test]
    fn sql_missing_closing_tag_is_incomplete((text, merges, tokens) in case(sql_text().prop_map(|mut s| { s.truncate(s.len() - "</sql>".len()); s }))) {
        prop_assert!(!sql_oracle(&text));
        let verifier = GrammarVerifier::new(env(&merges), GrammarUnderTest::Lark(SQL.into())).unwrap();
        prop_assert!(matches!(verifier.verify_string_under_partition(&text, &tokens)?, TraceVerdict::IncompleteAtEnd { .. }), "missing closing tag must remain incomplete");
    }

    #[test]
    fn unicode_byte_splits_accept((text, merges, tokens) in unicode_partitions()) {
        let harness = env(&merges);
        let verifier = GrammarVerifier::new(harness.clone(), GrammarUnderTest::Lark(TEXT.into())).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &byte_tokens(&text))?.is_accepted());
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
        prop_assert!(verifier.verify_string_greedy(&text)?.is_accepted());
        let body_start = "<text>".len();
        for offset in [body_start, body_start + '🦀'.len_utf8() - 1] {
            let merge = &text.as_bytes()[offset..offset + 2];
            prop_assert!(std::str::from_utf8(merge).is_err());
            let forced = force_merge(&text, merge, &harness);
            prop_assert!(forced.iter().any(|&id| harness.tok_env().tok_trie().token(id) == merge));
            prop_assert!(verifier.verify_string_under_partition(&text, &forced)?.is_accepted());
        }
    }

    #[test]
    fn partial_codepoints_cannot_terminate(symbol in proptest::sample::select(vec!['é', '€', '🦀', '無']), cut in 0usize..3, merges in arb_adversarial_bpe_merges(vec![], 12)) {
        let body = symbol.to_string();
        let cut = 1 + cut % (body.len() - 1);
        let mut bytes = b"<text>".to_vec();
        bytes.extend_from_slice(&body.as_bytes()[..cut]);
        prop_assert!(std::str::from_utf8(&bytes).is_err());
        let verifier = GrammarVerifier::new(env(&merges), GrammarUnderTest::Lark(TEXT.into())).unwrap();
        let tokens: Vec<_> = bytes.iter().copied().map(u32::from).collect();
        prop_assert!(matches!(verifier.verify_token_sequence(&tokens)?, TraceVerdict::IncompleteAtEnd { .. }), "partial UTF-8 must remain incomplete");
        bytes.extend_from_slice(b"</text>");
        let broken: Vec<_> = bytes.into_iter().map(u32::from).collect();
        prop_assert!(!verifier.verify_token_sequence(&broken)?.is_accepted());
    }

    #[test]
    fn regex_acceptance_and_mutation_match_independent_oracle((text, merges, tokens) in case(("[a-z]{1,8}", 0u32..10000).prop_map(|(name, n)| format!("{name}:{n};")))) {
        let oracle = regex::Regex::new(r"\A[a-z]{1,8}:[0-9]{1,4};\z").unwrap();
        prop_assert!(oracle.is_match(&text));
        let verifier = GrammarVerifier::new(env(&merges), GrammarUnderTest::Regex("[a-z]{1,8}:[0-9]{1,4};".into())).unwrap();
        prop_assert!(verifier.verify_string_under_partition(&text, &tokens)?.is_accepted());
        prop_assert!(verifier.verify_string_greedy(&text)?.is_accepted());
        let invalid = text.replacen(':', "!", 1);
        prop_assert!(!oracle.is_match(&invalid));
        prop_assert!(!verifier.verify_token_sequence(&byte_tokens(&invalid))?.is_accepted());
    }
}

#[test]
fn json_unicode_scalar_regression() {
    let verifier = GrammarVerifier::new(env(&[]), json_grammar("string")).unwrap();
    for scalar in [
        "\u{7f}",
        "�",
        "\0",
        "\u{348dd}",
        "\u{ff63e}",
        "\u{b2705}",
        "%",
    ] {
        let text = json_string_object(scalar.into());
        let verdict = verifier.verify_token_sequence(&byte_tokens(&text)).unwrap();
        assert!(
            verdict.is_accepted(),
            "scalar {scalar:?}, verdict {verdict:?}"
        );
    }
}

#[test]
fn native_json_rejects_literal_del_despite_serde_acceptance() {
    let text = json!({"a": "\u{7f}"}).to_string();
    assert!(serde_json::from_str::<serde_json::Value>(&text).is_ok());
    let verifier = GrammarVerifier::new(env(&[]), json_grammar("string")).unwrap();
    assert_eq!(
        verifier.verify_token_sequence(&byte_tokens(&text)).unwrap(),
        TraceVerdict::MaskRejectedAtStep {
            step_index: 6,
            token_id: 127,
            token_bytes: vec![127]
        }
    );
}

#[test]
fn byte_only_generator_fails_boundary_coverage_check() {
    let text = "<sql>SELECT x FROM y;\n</sql>";
    let env = env(&[b";\n</sql>".to_vec()]);
    let boundary = text.find(';').unwrap() + 1;
    let mut runner = TestRunner::new(ProptestConfig {
        cases: 128,
        failure_persistence: None,
        ..ProptestConfig::default()
    });
    let result = runner.run(&Just(byte_tokens(text)), |tokens| {
        prop_assert!(crosses(&env, &tokens, boundary));
        Ok(())
    });
    assert!(matches!(result, Err(TestError::Fail(_, _))));
}
