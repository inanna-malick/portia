//! Comprehensive Cleanroom Test Suite for `portia-sql`:
//! 1. SQLite Bytecode `authorizer` Provenance (`collect_read_tables_via_sqlite_authorizer`) &
//!    `sqlparser::ast` Write Target Validation (`parse_single_write_statement_target_and_op`).
//! 2. `SqlPredicateAst` & `RelationalSchemaGraph`:
//!    - 3-valued Kleene SQL predicate evaluation, single-bug AST mutant paramorphism, and `CHECK (...)`
//!      constraint extraction (`ExactNormalizedAst` vs `SampleRowEquivalence` diagnosis).
//! 3. `UnifiedSqlScenarioSpec` -> `open_in_memory_connection()`, reference query shape evaluation,
//!    and `verify_scenario_postcondition_on_connection`.

use portia_sql::{
    col, collect_read_tables_via_sqlite_authorizer, configure_hardened_sqlite,
    parse_read_or_probe_tables, parse_single_write_statement_target_and_op, MutantMatchBasis,
    RelationalSchemaGraph, SqlColumnRef, SqlPredicateAst, SqlPredicateFrame, SqlWriteOp,
};
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn test_sqlite_bytecode_authorizer_table_provenance_and_write_guard() {
    let conn = Connection::open_in_memory().expect("in-memory connection");
    configure_hardened_sqlite(&conn).expect("harden sqlite");
    conn.execute_batch(
        "CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT NOT NULL);
         CREATE TABLE instruments (inst_id INTEGER PRIMARY KEY, dept_id INTEGER NOT NULL, calibration_kpa INTEGER NOT NULL, status TEXT NOT NULL, FOREIGN KEY(dept_id) REFERENCES departments(dept_id));
         INSERT INTO departments VALUES (1, 'Chronometry'), (2, 'Barometry');
         INSERT INTO instruments VALUES (101, 1, 300, 'ACTIVE'), (102, 1, 320, 'ACTIVE'), (201, 2, 150, 'STANDBY');",
    )
    .unwrap();

    // 1. Complex multi-table JOIN + CTE + GROUP BY + COUNT(*) + ORDER BY + LIMIT table provenance via bytecode authorizer
    let complex_sql = "
        WITH active_inst AS (
            SELECT dept_id, calibration_kpa FROM instruments WHERE status = 'ACTIVE'
        )
        SELECT d.dept_name, COUNT(*) AS active_count, SUM(a.calibration_kpa) AS total_kpa
        FROM departments d
        JOIN active_inst a ON d.dept_id = a.dept_id
        GROUP BY d.dept_name
        ORDER BY total_kpa DESC
        LIMIT 5;
    ";
    let read_tables =
        collect_read_tables_via_sqlite_authorizer(&conn, complex_sql).expect("authorizer tables");
    assert_eq!(
        read_tables,
        BTreeSet::from(["departments".to_string(), "instruments".to_string()])
    );

    // 2. Write target parser extracts single table and rejects multi-statement SQL or DDL
    assert_eq!(
        parse_single_write_statement_target_and_op(
            "UPDATE instruments SET calibration_kpa = 350 WHERE inst_id = 101;"
        )
        .unwrap(),
        ("instruments".to_string(), SqlWriteOp::Update)
    );
    assert!(parse_single_write_statement_target_and_op(
        "UPDATE instruments SET calibration_kpa = 0; DROP TABLE instruments;"
    )
    .is_err());
    assert!(parse_single_write_statement_target_and_op("DROP TABLE instruments;").is_err());

    // 3. `parse_read_or_probe_tables` handles leading comments & quoted/schema-qualified PRAGMA
    assert_eq!(
        parse_read_or_probe_tables("-- inspect schema\nPRAGMA main.table_info(\"a;b\");").unwrap(),
        BTreeSet::from(["a;b".to_string()])
    );
    assert_eq!(
        parse_read_or_probe_tables("/* block */ PRAGMA table_info = [Instruments];").unwrap(),
        BTreeSet::from(["instruments".to_string()])
    );

    // 4. `SqlColumnRef::parse_sql_binding` distinguishes `"\"a.b\""` from `t."a.b"`
    let dotted_unqual = SqlColumnRef::parse_sql_binding("\"a.b\"").unwrap();
    assert!(dotted_unqual.table_qualifier.is_none());
    assert!(dotted_unqual.matches_strict_table_and_column("instruments", "a.b"));
    assert!(!dotted_unqual.matches_strict_table_and_column("a", "b"));
    let dotted_qual = SqlColumnRef::parse_sql_binding("instruments.\"a.b\"").unwrap();
    assert!(dotted_qual.matches_strict_table_and_column("instruments", "a.b"));
    assert!(!dotted_qual.matches_strict_table_and_column("departments", "a.b"));
}

#[test]
fn test_compositional_sql_predicate_edsl_mutants_and_check_constraints() {
    // 1. Build a compositional SQL predicate using `col(...)` and `BitAnd` / `BitOr` / `Not`
    let pred = col("delay_minutes").ge(30)
        & col("review_state").eq_str("OPEN")
        & !col("change_freeze").eq_str("LOCKED");

    assert_eq!(
        pred.render_sql(),
        "delay_minutes >= 30 AND review_state = 'OPEN' AND NOT (change_freeze = 'LOCKED')"
    );
    assert_eq!(
        pred.referenced_columns(),
        BTreeSet::from([
            "delay_minutes".to_string(),
            "review_state".to_string(),
            "change_freeze".to_string(),
        ])
    );

    // 2. Verify row evaluation and single-pass paramorphism (`single_bug_mutants`)
    let matching_row = BTreeMap::from([
        ("delay_minutes".to_string(), serde_json::json!(30)),
        ("review_state".to_string(), serde_json::json!("OPEN")),
        ("change_freeze".to_string(), serde_json::json!("CLEAR")),
    ]);
    let boundary_row = BTreeMap::from([
        ("delay_minutes".to_string(), serde_json::json!(29)),
        ("review_state".to_string(), serde_json::json!("OPEN")),
        ("change_freeze".to_string(), serde_json::json!("CLEAR")),
    ]);
    assert!(pred.evaluate_row(&matching_row));
    assert!(!pred.evaluate_row(&boundary_row));

    let mutants = pred.single_bug_mutants();
    assert!(!mutants.is_empty());
    let wrong_exclusive = col("delay_minutes").gt(30)
        & col("review_state").eq_str("OPEN")
        & !col("change_freeze").eq_str("LOCKED");
    assert_eq!(
        pred.match_single_bug_mutant(&wrong_exclusive, &[matching_row.clone(), boundary_row]),
        Some("exclusive_boundary".to_string())
    );

    // Also verify `InList` and `IsNull` inside `SqlPredicateAst`:
    let in_and_null =
        col("review_state").in_strs(["OPEN", "ESCALATED"]) & col("override_note").is_not_null();
    assert_eq!(
        in_and_null.render_sql(),
        "review_state IN ('OPEN', 'ESCALATED') AND override_note IS NOT NULL"
    );
    let mut row_with_note = matching_row;
    row_with_note.insert("override_note".to_string(), serde_json::json!("APPROVED"));
    assert!(in_and_null.evaluate_row(&row_with_note));

    // Verify `SqlPredicateAst::from_frame` and `SqlPredicateFrame`
    let manual_ast = SqlPredicateAst::new(SqlPredicateFrame::Not(col("flag").eq_int(0)));
    assert_eq!(manual_ast.render_sql(), "NOT (flag = 0)");

    // 3. Verify live SQLite `CHECK (...)` constraint extraction and observable `skipped_check_diagnostics`
    //    inside `RelationalSchemaGraph::inspect_from_connection`
    let check_conn = Connection::open_in_memory().unwrap();
    check_conn
        .execute_batch(
            "CREATE TABLE cryo_ballast_reservoirs (
                reservoir_id INTEGER PRIMARY KEY,
                cell_code TEXT NOT NULL CHECK (cell_code LIKE 'CELL-%'),
                reserve_liters INTEGER NOT NULL CHECK (reserve_liters >= 0),
                CONSTRAINT chk_cap CHECK (reserve_liters <= 5000 AND cell_code != 'DECOMMISSIONED')
            );",
        )
        .unwrap();
    let check_graph = RelationalSchemaGraph::inspect_from_connection(&check_conn).unwrap();
    let checks = check_graph.check_constraints_for("cryo_ballast_reservoirs");
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0], col("reserve_liters").ge(0));
    assert_eq!(
        checks[1],
        col("reserve_liters").le(5000) & col("cell_code").ne_str("DECOMMISSIONED")
    );
    let skipped = check_graph.skipped_checks_for("cryo_ballast_reservoirs");
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].constraint_name.as_deref(), Some("cell_code"));
    assert!(skipped[0].raw_expr_sql.contains("LIKE"));
    assert!(check_graph.evaluate_row_check_constraints(
        "cryo_ballast_reservoirs",
        &BTreeMap::from([
            ("reserve_liters".to_string(), serde_json::json!(180)),
            ("cell_code".to_string(), serde_json::json!("CELL-NORTH")),
        ])
    ));
    assert!(!check_graph.evaluate_row_check_constraints(
        "cryo_ballast_reservoirs",
        &BTreeMap::from([
            ("reserve_liters".to_string(), serde_json::json!(-5)),
            ("cell_code".to_string(), serde_json::json!("CELL-NORTH")),
        ])
    ));

    // 4. Verify `diagnose_single_bug_mutant` distinguishes structured `MutantMatchDiagnosis`
    //    (`ExactNormalizedAst` vs `SampleRowEquivalence`):
    let expected_where = col("folio_id").eq_int(101) & col("courier_code").eq_str("SELDEN");
    let unscoped_mutant = col("courier_code").eq_str("SELDEN");
    let diagnosis = expected_where
        .diagnose_single_bug_mutant(&unscoped_mutant, &[])
        .expect("matches dropped_left_conjunct single-bug mutant");
    assert_eq!(diagnosis.mutant_name, "dropped_left_conjunct");
    assert_eq!(diagnosis.match_basis, MutantMatchBasis::ExactNormalizedAst);
}

#[test]
fn test_cleanroom_11_axis_nursery_and_7d_roguelite_dungeon_parity() {
    use portia_sql::engine::{clone_connection_to_memory, diff_shadow_databases};
    use portia_sql::roguelite_dungeon::{
        canonical_reference_sql_for_shape, evaluate_candidate_query_against_reference,
        shrink_stress_vector_to_irreducible_cliff, verify_scenario_postcondition_on_connection,
        CollisionLocus, IntentTrajectoryKind, NeedleDepth, PolicyLockSurfaceIso,
        PrimaryKeyIsomorphism, RogueliteStressVector, SqlQueryShape, StewardAgendaStep,
        UnifiedScenarioTrapModifier, UnifiedSqlScenarioSpec, WavePermutation,
    };

    // 1. Verify hylomorphic multi-axis irreducible cliff shrinker isolates a 2-way SQL stress cliff
    //    (column_width_arity >= 12 && schema_graph_topology >= 4)
    let failing_seed = RogueliteStressVector {
        row_haystack_size: 40,
        needle_depth: NeedleDepth::Bot90Pct,
        identifier_swarm_collision: 12,
        collision_locus: CollisionLocus::MiddleHexDiff,
        column_width_arity: 24,
        deprecated_decoy_columns: 4,
        schema_graph_topology: 8,
        fk_chain_depth: 3,
        policy_predicate_arity: 5,
        episode_turn_horizon: 10,
        chitchat_carry_distance: 3,
        steward_negotiation_rounds: 3,
        ast_symbol_haystack: 20,
        homonym_overload_count: 3,
    };
    let receipt = shrink_stress_vector_to_irreducible_cliff(failing_seed, |v| {
        v.column_width_arity >= 12 && v.schema_graph_topology >= 4
    });
    assert_eq!(receipt.irreducible_failing.column_width_arity, 12);
    assert_eq!(receipt.irreducible_failing.schema_graph_topology, 4);
    assert_eq!(receipt.irreducible_failing.row_haystack_size, 5);
    assert_eq!(receipt.irreducible_failing.identifier_swarm_collision, 3);
    assert_eq!(receipt.irreducible_failing.policy_predicate_arity, 2);
    assert_eq!(receipt.irreducible_failing.episode_turn_horizon, 3);
    assert_eq!(receipt.irreducible_failing.ast_symbol_haystack, 8);
    assert_eq!(receipt.contrastive_lower_covers.len(), 2);

    // 2. Materialize the irreducible cliff into a live SQLite scenario with surface isomorphisms,
    //    conversational preamble turns, multi-round steward overrides, and wave permutation:
    let spec = UnifiedSqlScenarioSpec::from_stress_vector(receipt.irreducible_failing, 2026)
        .with_primary_key_iso(PrimaryKeyIsomorphism::CompositeTenantKey {
            tenant_code: "ACME".to_string(),
            base_offset: 7000,
        })
        .with_policy_lock_iso(PolicyLockSurfaceIso::DomainClearanceHold)
        .with_permute_core_columns_sk(true)
        .with_upfront_schema(false)
        .with_conversational_preamble_turns(2)
        .with_wave_permutation(WavePermutation::Reverse210)
        .with_intent_trajectory(IntentTrajectoryKind::ValueConditionalBranch {
            min_balance_threshold: 100,
        })
        .with_steward_agenda(vec![
            StewardAgendaStep::RequestAdditionalRead {
                required_table: "roguelite_primary_ledger".to_string(),
                required_column: "balance_credits".to_string(),
            },
            StewardAgendaStep::ConditionalAllowWithOverride {
                token: "STEWARD_TOK_A".to_string(),
                overridden_post_balance: Some(8800),
                overridden_target_pk: None,
                overridden_status_note: Some("CAP_8800".to_string()),
            },
            StewardAgendaStep::RevisedOverride {
                token: "STEWARD_TOK_B".to_string(),
                revised_post_balance: 9200,
            },
        ])
        .with_trap_modifiers(vec![
            UnifiedScenarioTrapModifier::BlindSchemaProbeFirstH2,
            UnifiedScenarioTrapModifier::ArchDecoyHoneypotM2,
            UnifiedScenarioTrapModifier::CrossWaveValueCarryH1a { wave2_delta: 50 },
            UnifiedScenarioTrapModifier::CrossWaveOutcomeBranchH1b,
        ]);

    let instance = spec.build();
    let conn = instance
        .open_in_memory_connection()
        .expect("open seeded in-memory sqlite");
    let pk = instance.needle_primary_key;
    let slug = instance.needle_entity_slug.clone();

    // Before mutation, postcondition verifier rejects unmodified write-trajectory state:
    assert!(!verify_scenario_postcondition_on_connection(&instance, &conn).unwrap());

    // Reference query shapes evaluate cleanly against the seeded scenario connection:
    for shape in SqlQueryShape::ALL {
        let ref_sql = canonical_reference_sql_for_shape(&conn, shape).unwrap();
        let eval = evaluate_candidate_query_against_reference(&conn, &ref_sql, shape).unwrap();
        assert!(eval.is_exact_match());
    }

    // Candidate needle read (excluding ARCHIVED honeypot decoy row) returns exactly 1 ACTIVE row:
    let active_rows: i64 = conn
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM roguelite_primary_ledger WHERE entity_slug = '{}' AND lifecycle_status = 'ACTIVE';",
                slug
            ),
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active_rows, 1);

    // Clone reference DB before mutation and apply the exact expected final update
    // (including composite key `tenant_code = 'ACME' AND record_id = ... AND lifecycle_status = 'ACTIVE'`):
    let expected_shadow_conn = clone_connection_to_memory(&conn).expect("clone memory db");
    let expected_update_sql = instance.expected_update_needle_sql(9200);
    assert!(expected_update_sql.contains("tenant_code = 'ACME' AND record_id = "));
    assert!(expected_update_sql.contains("lifecycle_status = 'ACTIVE'"));
    expected_shadow_conn
        .execute(&expected_update_sql, [])
        .unwrap();

    // Execute scoped write on live connection and verify both shadow equivalence and postcondition oracle:
    let affected = conn.execute(&expected_update_sql, []).unwrap();
    assert_eq!(affected, 1);

    let updated_bal: i64 = conn
        .query_row(
            &format!(
                "SELECT balance_credits FROM roguelite_primary_ledger WHERE record_id = {} AND lifecycle_status = 'ACTIVE';",
                pk
            ),
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(updated_bal, 9200);

    let pre_wave_diffs = diff_shadow_databases(&expected_shadow_conn, &conn).unwrap();
    assert!(
        pre_wave_diffs.is_equivalent(),
        "expected 0 shadow diffs, got: {:?}",
        pre_wave_diffs
    );
    assert!(verify_scenario_postcondition_on_connection(&instance, &conn).unwrap());
}
