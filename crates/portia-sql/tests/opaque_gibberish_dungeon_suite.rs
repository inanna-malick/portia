//! Property-test suite for `portia_sql::roguelite_dungeon::opaque_gibberish` (`<= 350` LOC):
//! verifies pairwise mutant-sum separation, `RelationalSchemaGraph` foreign-key recovery across
//! both `TableConstraint` and `InlineColumnReferences` DDL styles, `llguidance` grammar acceptance
//! of all `N` same-type columns (proving grammar alone cannot solve FK selection!), and `v2`
//! `IntentExpr` lowering.

use lark_bpe_proptest::{GrammarUnderTest, GrammarVerifier, TokenizerHarness};
use portia_compiler::ir::regalloc::allocate_neural_registers;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{GrammarCartridgeSpec, Mode, TableSchema};
use portia_sql::grammar_trie::compile_sqlite_lark_grammar_for_schemas;
use portia_sql::relational_ast::{ForeignKeyEdge, RelationalSchemaGraph};
use portia_sql::roguelite_dungeon::{arb_opaque_gibberish_dungeon_spec, OpaqueGibberishVerdict};
use portia_sql::v2_pointer_bridge::{
    build_relational_schema_pointer_program, SqlPointerBridgeConfig,
};
use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[test]
fn opaque_gibberish_files_stay_under_350_loc_limit() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "src/roguelite_dungeon/opaque_gibberish.rs",
        "tests/opaque_gibberish_dungeon_suite.rs",
    ] {
        let p = manifest.join(rel);
        let n = std::fs::read_to_string(&p).unwrap().lines().count();
        assert!(n <= 350, "{} has {} lines (> 350)", p.display(), n);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn prop_opaque_gibberish_dungeon_mutant_oracle_and_schema_pointer_lowering(
        spec in arb_opaque_gibberish_dungeon_spec()
    ) {
        let inst = spec.build().expect("OpaqueGibberishDungeonSpec::build must succeed");
        prop_assert_eq!(
            inst.child_columns_in_ddl_order.len(),
            spec.child_candidate_columns.clamp(4, 10)
        );

        // 1. Strict bijective mutant separation: `gold_sum` and every decoy column's mutant sum
        //    must be 100% pairwise distinct so `evaluate_candidate_sql` never misattributes!
        let mut all_sums = BTreeSet::new();
        prop_assert!(all_sums.insert(inst.gold_sum));
        for (col, &msum) in &inst.decoy_mutant_sums {
            prop_assert!(
                all_sums.insert(msum),
                "mutant sum collision on {col}: {msum} collided with existing {all_sums:?}"
            );
        }

        // 2. Verify self-diagnosing mutant oracle classifies gold + all 3 decoy classes + bad SQL:
        prop_assert_eq!(
            inst.evaluate_candidate_sql(&inst.gold_sql),
            OpaqueGibberishVerdict::ExactGoldMatch { sum: inst.gold_sum }
        );

        let mk_join = |col: &str| {
            format!(
                "SELECT SUM({0}.{1}) FROM {0} JOIN {2} ON {0}.{3} = {2}.{col};",
                inst.parent_table, inst.parent_metric_col, inst.child_table, inst.parent_pk_col
            )
        };
        let think_sql = mk_join(&inst.think_scratchpad_decoy_col);
        prop_assert_eq!(
            inst.evaluate_candidate_sql(&think_sql),
            OpaqueGibberishVerdict::CopiedThinkScratchpadDecoy {
                got_sum: inst.decoy_mutant_sums[&inst.think_scratchpad_decoy_col],
                expected_sum: inst.gold_sum,
            }
        );

        let user_sql = mk_join(&inst.user_prose_decoy_col);
        prop_assert_eq!(
            inst.evaluate_candidate_sql(&user_sql),
            OpaqueGibberishVerdict::CopiedUserProseDecoy {
                got_sum: inst.decoy_mutant_sums[&inst.user_prose_decoy_col],
                expected_sum: inst.gold_sum,
            }
        );

        for silent_col in &inst.silent_schema_decoy_cols {
            let s_sql = mk_join(silent_col);
            prop_assert_eq!(
                inst.evaluate_candidate_sql(&s_sql),
                OpaqueGibberishVerdict::PickedSilentSchemaDecoy {
                    decoy_column: silent_col.clone(),
                    got_sum: inst.decoy_mutant_sums[silent_col],
                    expected_sum: inst.gold_sum,
                }
            );
        }

        // 3. Verify `RelationalSchemaGraph` recovers the exact gibberish FK edge under both
        //    `TableConstraint` and `InlineColumnReferences`, and lowers into `v2` Pointer-ISA!
        let conn = inst.open_connection().unwrap();
        let schema_graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
        prop_assert_eq!(schema_graph.foreign_keys.len(), 1);
        let edge = &schema_graph.foreign_keys[0];
        prop_assert_eq!(&edge.child_table, &inst.child_table);
        prop_assert_eq!(&edge.child_col, &inst.true_fk_col);
        prop_assert_eq!(&edge.parent_table, &inst.parent_table);
        prop_assert_eq!(&edge.parent_col, &inst.parent_pk_col);

        let cfg = SqlPointerBridgeConfig::qwen35_08b_l18_to_l19(vec![1, 5, 7, 3], vec![0, 2, 4, 6]);
        let mut fk_deltas: BTreeMap<ForeignKeyEdge, Vec<f64>> = BTreeMap::new();
        fk_deltas.insert(edge.clone(), vec![0.25; 192]);
        let prog = build_relational_schema_pointer_program(
            &schema_graph,
            &cfg,
            &fk_deltas,
            RegimeExpr::ConstBool(true),
        )
        .unwrap();
        let allocated = allocate_neural_registers(&prog, 64).unwrap();
        prop_assert_eq!(allocated.allocated_rules.len(), 4);

        // 4. Verify `llguidance` SQLite schema grammar accepts BOTH `gold_sql` AND `think_sql`
        //    (proving why grammar alone cannot disambiguate same-table integer columns!), while
        //    rejecting non-existent hallucinated columns!
        let tables: Vec<TableSchema> = schema_graph.tables.values().cloned().collect();
        if let GrammarCartridgeSpec::LarkGrammar(lark) =
            compile_sqlite_lark_grammar_for_schemas(&tables, Mode::Read)
        {
            let harness = TokenizerHarness::synthetic_byte_bpe(&[b" = ".to_vec()], b"<eos>").unwrap();
            let verifier = GrammarVerifier::new(harness, GrammarUnderTest::Lark(lark)).unwrap();
            prop_assert!(verifier.verify_string_greedy(&inst.gold_sql).unwrap().is_accepted());
            prop_assert!(verifier.verify_string_greedy(&think_sql).unwrap().is_accepted());
            let bogus_sql = mk_join("c_nonexistent_hallucinated");
            prop_assert!(!verifier.verify_string_greedy(&bogus_sql).unwrap().is_accepted());
        }
    }
}
