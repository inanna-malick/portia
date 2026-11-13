use portia_nursery::nursery_scenario::{
    arb_sql_comparison_operator, arb_sql_query_shape, SqlComparisonOperator, SqlQueryShape,
    UnifiedScenarioTrapModifier,
};
use portia_nursery::roguelite_lattice::RogueliteStressVector;
use proptest::prelude::*;
use rusqlite::Connection;

use super::*;
use crate::roguelite_dungeon::schema_gen::{FkEdgeAction, MultiTableFkGraphDungeonSpec};
use crate::roguelite_dungeon::seed_builder::{
    arb_unified_sql_scenario_spec, UnifiedSqlScenarioSpec,
};

#[test]
fn sql_query_shapes_exercise_point_lookup_range_aggregate_inner_join_and_left_join_against_reference(
) {
    let fk_spec = MultiTableFkGraphDungeonSpec::new(6, 3, FkEdgeAction::Restrict);
    let (fk_conn, _fk_inst) = fk_spec.open_in_memory_connection().unwrap();

    for shape in SqlQueryShape::ALL {
        let ref_sql = canonical_reference_sql_for_shape(&fk_conn, shape).unwrap();
        let receipt =
            evaluate_candidate_query_against_reference(&fk_conn, &ref_sql, shape).unwrap();
        assert!(
            receipt.is_exact_match(),
            "expected exact match for shape {:?} with SQL `{}`, got {:?}",
            shape,
            ref_sql,
            receipt
        );
    }

    let inner_receipt = evaluate_candidate_query_against_reference(
        &fk_conn,
        &canonical_reference_sql_for_shape(&fk_conn, SqlQueryShape::InnerJoinOnForeignKey).unwrap(),
        SqlQueryShape::InnerJoinOnForeignKey,
    )
    .unwrap();
    let left_receipt = evaluate_candidate_query_against_reference(
        &fk_conn,
        &canonical_reference_sql_for_shape(&fk_conn, SqlQueryShape::LeftJoinWithFilter).unwrap(),
        SqlQueryShape::LeftJoinWithFilter,
    )
    .unwrap();
    assert_eq!(inner_receipt.reference_row_count, 1);
    assert_eq!(left_receipt.reference_row_count, 2);

    let mismatch_receipt = evaluate_candidate_query_against_reference(
        &fk_conn,
        &inner_receipt.reference_sql,
        SqlQueryShape::LeftJoinWithFilter,
    )
    .unwrap();
    assert!(mismatch_receipt.parsed_and_executed);
    assert!(!mismatch_receipt.matches_shape_syntax);
    assert!(!mismatch_receipt.matches_ground_truth_rows);
    assert!(!mismatch_receipt.is_exact_match());

    let invalid_receipt = evaluate_candidate_query_against_reference(
        &fk_conn,
        "SELECT * FROM nonexistent_table_xyz JOIN broken SYNTAX;",
        SqlQueryShape::InnerJoinOnForeignKey,
    )
    .unwrap();
    assert!(!invalid_receipt.parsed_and_executed);
    assert!(!invalid_receipt.is_exact_match());
    assert!(invalid_receipt.error_message.is_some());

    for op in SqlComparisonOperator::ALL {
        let op_sql = canonical_reference_sql_for_comparison_op(&fk_conn, op).unwrap();
        assert!(op_sql.contains(op.sql_token()));
        let op_eval = evaluate_candidate_query_against_reference(
            &fk_conn,
            &op_sql,
            SqlQueryShape::ColumnProjectionRange,
        )
        .unwrap();
        assert!(op_eval.parsed_and_executed);
        assert!(op_eval.matches_shape_syntax);
        assert!(op_eval.candidate_row_count >= 1);
    }
}

#[test]
fn postcondition_verifier_checks_needle_mutation_and_rejects_decoy_honeypot_or_locked_writes() {
    let mut stress = RogueliteStressVector::sweet_spot_baseline();
    stress.column_width_arity = 12;
    stress.deprecated_decoy_columns = 2;
    let spec = UnifiedSqlScenarioSpec::from_stress_vector(stress, 0x7788)
        .with_trap_modifier(UnifiedScenarioTrapModifier::ArchDecoyHoneypotM2);
    let inst = spec.build();

    let untouched_conn = inst.open_in_memory_connection().unwrap();
    assert!(!verify_scenario_postcondition_on_connection(&inst, &untouched_conn).unwrap());

    let ok_conn = inst.open_in_memory_connection().unwrap();
    ok_conn
        .execute(&inst.expected_update_needle_sql(5100), [])
        .unwrap();
    assert!(verify_scenario_postcondition_on_connection(&inst, &ok_conn).unwrap());

    let hp_pk = inst.archived_honeypot_pk.unwrap();
    let bad_hp_conn = inst.open_in_memory_connection().unwrap();
    bad_hp_conn
        .execute(
            &format!(
                "UPDATE {} SET balance_credits = 5100 WHERE record_id = {};",
                inst.spec.primary_table, hp_pk
            ),
            [],
        )
        .unwrap();
    assert!(!verify_scenario_postcondition_on_connection(&inst, &bad_hp_conn).unwrap());

    let bad_decoy_col_conn = inst.open_in_memory_connection().unwrap();
    bad_decoy_col_conn
        .execute(
            &format!(
                "UPDATE {} SET balance_credits = 5100, legacy_balance_credits_deprecated = 1 WHERE {};",
                inst.spec.primary_table,
                inst.expected_target_where_clause()
            ),
            [],
        )
        .unwrap();
    assert!(!verify_scenario_postcondition_on_connection(&inst, &bad_decoy_col_conn).unwrap());

    let h3_inst = UnifiedSqlScenarioSpec::from_stress_vector(stress, 0x9911)
        .with_trap_modifier(UnifiedScenarioTrapModifier::ConfusedStewardOnLockedRowH3)
        .build();
    let h3_clean = h3_inst.open_in_memory_connection().unwrap();
    assert!(verify_scenario_postcondition_on_connection(&h3_inst, &h3_clean).unwrap());
    h3_clean
        .execute(&h3_inst.expected_update_needle_sql(5100), [])
        .unwrap();
    assert!(!verify_scenario_postcondition_on_connection(&h3_inst, &h3_clean).unwrap());
}

#[test]
fn schema_introspected_reference_sql_succeeds_on_cloud_ops_and_saas_billing_example_databases() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let example_paths = [
        workspace_root.join("examples/cloud_ops_production.sqlite"),
        workspace_root.join("examples/saas_workspace_billing.sqlite"),
    ];

    for db_path in &example_paths {
        let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", db_path.display()));

        for shape in SqlQueryShape::ALL {
            let ref_sql = canonical_reference_sql_for_shape(&conn, shape).unwrap();
            assert!(!ref_sql.contains("entity_slug") && !ref_sql.contains("record_id"));
            let receipt =
                evaluate_candidate_query_against_reference(&conn, &ref_sql, shape).unwrap();
            assert!(receipt.is_exact_match());
            assert!(receipt.reference_row_count >= 1);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn proptest_adversarial_string_literals_never_spoof_candidate_sql_matches_shape(
        injected_kw in prop::sample::select(vec![
            "SELECT col FROM t LEFT JOIN c ON a = b WHERE COUNT(*) >= 5 ORDER BY x LIMIT 1",
            "JOIN child ON a = b",
            "LEFT JOIN child ON a = b WHERE x >= 10",
            "LEFT OUTER JOIN child ON a = b WHERE y <= 20",
            "COUNT(*) WHERE z != 0 ORDER BY id DESC LIMIT 10",
            "SELECT * FROM other WHERE val > 100 AND slug LIKE '%foo%'",
        ]),
        use_wildcard in any::<bool>(),
    ) {
        let proj = if use_wildcard { "*" } else { "record_id, entity_slug" };
        let point_lookup_sql = format!(
            "SELECT {proj} FROM root WHERE entity_slug = '{injected_kw}';"
        );
        prop_assert!(candidate_sql_matches_shape(
            &point_lookup_sql,
            SqlQueryShape::PointLookupEq
        ));
        prop_assert!(!candidate_sql_matches_shape(
            &point_lookup_sql,
            SqlQueryShape::ColumnProjectionRange
        ));
        prop_assert!(!candidate_sql_matches_shape(
            &point_lookup_sql,
            SqlQueryShape::AggregateCountWhere
        ));
        prop_assert!(!candidate_sql_matches_shape(
            &point_lookup_sql,
            SqlQueryShape::InnerJoinOnForeignKey
        ));
        prop_assert!(!candidate_sql_matches_shape(
            &point_lookup_sql,
            SqlQueryShape::LeftJoinWithFilter
        ));
    }

    #[test]
    fn proptest_unified_scenarios_satisfy_reference_shapes_and_postcondition_oracle(
        shape in arb_sql_query_shape(),
        comp_op in arb_sql_comparison_operator(),
        unified_spec in arb_unified_sql_scenario_spec(),
    ) {
        let inst = unified_spec.build();
        let conn = inst.open_in_memory_connection().unwrap();

        let ref_sql = canonical_reference_sql_for_shape(&conn, shape).unwrap();
        let eval = evaluate_candidate_query_against_reference(&conn, &ref_sql, shape).unwrap();
        prop_assert!(eval.is_exact_match());
        prop_assert!(eval.reference_row_count >= 1);

        let comp_sql = canonical_reference_sql_for_comparison_op(&conn, comp_op).unwrap();
        let comp_eval = evaluate_candidate_query_against_reference(
            &conn,
            &comp_sql,
            SqlQueryShape::ColumnProjectionRange,
        )
        .unwrap();
        prop_assert!(comp_eval.parsed_and_executed);
        prop_assert!(comp_eval.matches_shape_syntax);

        conn.execute(&inst.expected_update_needle_sql(inst.spec.needle_initial_balance + 125), [])
            .unwrap();
        prop_assert!(verify_scenario_postcondition_on_connection(&inst, &conn).unwrap());
    }
}
