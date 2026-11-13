//! Straddled-BPE `llguidance` grammar verification suite for `portia-sql` hex-obfuscated schemas
//! powered by [`lark_bpe_proptest`].

use lark_bpe_proptest::{
    strategies::{arb_adversarial_bpe_merges, arb_token_partition_for_bytes},
    GrammarAuditSuite, GrammarUnderTest, GrammarVerifier, TokenizerHarness,
};
use portia_core::{GrammarCartridgeSpec, Mode, TableSchema};
use portia_sql::grammar_trie::compile_sqlite_lark_grammar_for_schemas;
use portia_sql::relational_ast::RelationalSchemaGraph;
use proptest::prelude::*;
use rusqlite::Connection;

fn hex_schema_lark_grammar() -> String {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE t_9f2 (c_19a INTEGER PRIMARY KEY, c_88d INTEGER NOT NULL);
         CREATE TABLE t_4c1 (c_77b INTEGER NOT NULL REFERENCES t_9f2(c_19a), c_3e1 INTEGER NOT NULL);",
    )
    .unwrap();
    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
    let tables: Vec<TableSchema> = graph.tables.values().cloned().collect();
    match compile_sqlite_lark_grammar_for_schemas(&tables, Mode::Read) {
        GrammarCartridgeSpec::LarkGrammar(g) => g,
        other => panic!("expected LarkGrammar, got {other:?}"),
    }
}

#[test]
fn hex_sqlite_lark_grammar_accepts_valid_join_and_rejects_hallucinated_think_columns() {
    let lark = hex_schema_lark_grammar();
    let harness = TokenizerHarness::synthetic_byte_bpe(
        &[
            b"77b;".to_vec(),
            b"t_4c1.c_".to_vec(),
            b"c_77b".to_vec(),
            b" = ".to_vec(),
        ],
        b"<eos>",
    )
    .unwrap();
    let verifier = GrammarVerifier::new(harness, GrammarUnderTest::Lark(lark.clone())).unwrap();

    let suite = GrammarAuditSuite {
        name: "hex_fk_join_vs_think_decoy".to_string(),
        grammar: GrammarUnderTest::Lark(lark),
        must_accept: vec![
            "SELECT SUM(t_9f2.c_88d) FROM t_9f2 JOIN t_4c1 ON t_9f2.c_19a = t_4c1.c_77b;"
                .to_string(),
        ],
        must_reject: vec![
            "SELECT SUM(t_9f2.c_88d) FROM t_9f2 JOIN t_4c1 ON t_9f2.c_19a = t_4c1.c_dead;"
                .to_string(),
            "DROP TABLE t_9f2;".to_string(),
        ],
        max_step_micros_budget: 250_000,
    };
    let receipt = verifier.verify_batch_suite(&suite).unwrap();
    assert_eq!(receipt.accepted_verified, 1);
    assert_eq!(receipt.rejected_verified, 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_hex_sql_lark_grammar_invariant_across_straddled_bpe_partitions(
        merges in arb_adversarial_bpe_merges(
            vec![b".c_".to_vec(), b"77b;".to_vec(), b" = t_".to_vec()],
            12,
        )
    ) {
        let lark = hex_schema_lark_grammar();
        let valid_sql = "SELECT t_9f2.c_88d FROM t_9f2 JOIN t_4c1 ON t_9f2.c_19a = t_4c1.c_77b;";
        let harness = TokenizerHarness::synthetic_byte_bpe(&merges, b"<eos>").unwrap();
        let part_strat = arb_token_partition_for_bytes(harness.clone(), valid_sql.as_bytes().to_vec());
        let mut runner = proptest::test_runner::TestRunner::deterministic();
        let partition = part_strat.new_tree(&mut runner).unwrap().current();
        let verifier = GrammarVerifier::new(harness, GrammarUnderTest::Lark(lark)).unwrap();
        let verdict = verifier.verify_string_under_partition(valid_sql, &partition).unwrap();
        prop_assert!(verdict.is_accepted(), "straddled BPE partition rejected valid SQL: {verdict:?}");
    }
}
