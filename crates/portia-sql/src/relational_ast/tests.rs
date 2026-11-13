use std::collections::{BTreeMap, BTreeSet};

use rusqlite::Connection;

use super::*;

fn setup_alpine_fk_connection() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "
        PRAGMA foreign_keys = ON;
        CREATE TABLE alpine_levain_campaigns (
            campaign_id INTEGER PRIMARY KEY,
            campaign_title TEXT NOT NULL,
            review_status TEXT NOT NULL,
            poster_altitude_m INTEGER NOT NULL,
            condition_note TEXT,
            policy_decision TEXT NOT NULL,
            steward_code TEXT NOT NULL,
            glacier_clamp TEXT NOT NULL
        );
        CREATE TABLE alpine_poster_links (
            link_id INTEGER PRIMARY KEY,
            campaign_id INTEGER NOT NULL,
            attachment_label TEXT NOT NULL,
            FOREIGN KEY (campaign_id) REFERENCES alpine_levain_campaigns(campaign_id) ON DELETE CASCADE
        );
        INSERT INTO alpine_levain_campaigns VALUES
            (101, 'ridge_poster', 'ACTIVE', 1800, NULL, 'ALLOW', 'LUNA', 'CLEAR'),
            (102, 'valley_banner', 'ACTIVE', 950, 'WEATHERED', 'ALLOW', 'LUNA', 'CLEAR'),
            (103, 'summit_beacon', 'RESERVE', 2400, 'CALIBRATED', 'ASK', 'LUNA', 'CLEAR');
        INSERT INTO alpine_poster_links VALUES
            (901, 101, 'CASCADE_SAFE_CHILD_LINK'),
            (902, 102, 'VALLEY_RIGGING_LINK');
        ",
    )
    .unwrap();
    conn
}

#[test]
fn relational_schema_graph_discovers_fk_edges_nullability_and_numeric_types() {
    let conn = setup_alpine_fk_connection();
    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();

    assert_eq!(graph.tables.len(), 2);
    assert!(graph.tables.contains_key("alpine_levain_campaigns"));
    assert!(graph.tables.contains_key("alpine_poster_links"));

    assert_eq!(
        graph.foreign_keys,
        vec![ForeignKeyEdge::new(
            "alpine_poster_links",
            "campaign_id",
            "alpine_levain_campaigns",
            "campaign_id",
        )]
    );
    assert_eq!(
        graph.joinable_tables_for("alpine_levain_campaigns"),
        BTreeSet::from(["alpine_poster_links".to_string()])
    );
    assert!(graph.is_numeric_column("alpine_levain_campaigns", "poster_altitude_m"));
    assert!(!graph.is_numeric_column("alpine_levain_campaigns", "campaign_title"));
    assert!(graph.is_nullable_column("alpine_levain_campaigns", "condition_note"));
    assert!(!graph.is_nullable_column("alpine_levain_campaigns", "campaign_title"));
}

#[test]
fn compositional_sql_predicate_edsl_runs_all_four_collapse_frames_algebras() {
    let conn = setup_alpine_fk_connection();
    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();

    // 1. Ergonomic Rust eDSL combinators (`BitAnd`, `BitOr`, `Not`, `col(...)`):
    let pred = col("poster_altitude_m").ge(1500)
        & col("review_status").eq_str("ACTIVE")
        & !col("glacier_clamp").eq_str("LOCKED")
        & col("condition_note").is_null();

    // Algebra 1: SQL Renderer (`render_sql`)
    assert_eq!(
        pred.render_sql(),
        "poster_altitude_m >= 1500 AND review_status = 'ACTIVE' AND NOT (glacier_clamp = 'LOCKED') AND condition_note IS NULL"
    );

    // Automatic parenthesization of nested `OR` inside `AND`:
    let disjunctive_inside_and = (col("review_status").eq_str("ACTIVE")
        | col("review_status").eq_str("RESERVE"))
        & col("poster_altitude_m").ge(1000);
    assert_eq!(
        disjunctive_inside_and.render_sql(),
        "(review_status = 'ACTIVE' OR review_status = 'RESERVE') AND poster_altitude_m >= 1000"
    );

    // Algebra 2: Schema Type-Checker & Column-Reference Verifier (`validate_against_graph` via `try_collapse_frames`)
    let cols = pred
        .validate_against_graph(&graph, "alpine_levain_campaigns")
        .expect("valid predicate over alpine_levain_campaigns");
    assert_eq!(
        cols,
        BTreeSet::from([
            "poster_altitude_m".to_string(),
            "review_status".to_string(),
            "glacier_clamp".to_string(),
            "condition_note".to_string(),
        ])
    );
    assert_eq!(pred.referenced_columns(), cols);

    // Type-checker rejects unknown columns:
    let bad_col = col("non_existent_column").eq_int(42);
    assert_eq!(
        bad_col.validate_against_graph(&graph, "alpine_levain_campaigns"),
        Err(SqlTypeError::UnknownColumn {
            table: "alpine_levain_campaigns".to_string(),
            column: "non_existent_column".to_string(),
        })
    );

    // Type-checker rejects numeric range operators (`>=`) on `TEXT` columns:
    let bad_range = col("campaign_title").ge(100);
    assert_eq!(
        bad_range.validate_against_graph(&graph, "alpine_levain_campaigns"),
        Err(SqlTypeError::NonNumericRangeComparison {
            table: "alpine_levain_campaigns".to_string(),
            column: "campaign_title".to_string(),
            col_type: "TEXT".to_string(),
            op: ">=".to_string(),
        })
    );

    // Type-checker rejects `IS NULL` on `NOT NULL` columns:
    let bad_null = col("campaign_title").is_null();
    assert_eq!(
        bad_null.validate_against_graph(&graph, "alpine_levain_campaigns"),
        Err(SqlTypeError::NonNullableColumnNullCheck {
            table: "alpine_levain_campaigns".to_string(),
            column: "campaign_title".to_string(),
        })
    );

    // Algebra 3: Relational Row Interpreter (`evaluate_row`)
    let row_101 = BTreeMap::from([
        ("campaign_id".to_string(), serde_json::json!(101)),
        ("poster_altitude_m".to_string(), serde_json::json!(1800)),
        ("review_status".to_string(), serde_json::json!("ACTIVE")),
        ("glacier_clamp".to_string(), serde_json::json!("CLEAR")),
        ("condition_note".to_string(), serde_json::Value::Null),
    ]);
    let row_102 = BTreeMap::from([
        ("campaign_id".to_string(), serde_json::json!(102)),
        ("poster_altitude_m".to_string(), serde_json::json!(1500)),
        ("review_status".to_string(), serde_json::json!("ACTIVE")),
        ("glacier_clamp".to_string(), serde_json::json!("CLEAR")),
        ("condition_note".to_string(), serde_json::json!("WEATHERED")),
    ]);
    let row_103 = BTreeMap::from([
        ("campaign_id".to_string(), serde_json::json!(103)),
        ("poster_altitude_m".to_string(), serde_json::json!(1500)),
        ("review_status".to_string(), serde_json::json!("RESERVE")),
        ("glacier_clamp".to_string(), serde_json::json!("CLEAR")),
        ("condition_note".to_string(), serde_json::Value::Null),
    ]);
    assert!(pred.evaluate_row(&row_101));
    assert!(!pred.evaluate_row(&row_102));
    assert!(!pred.evaluate_row(&row_103));

    // Algebra 4: Single-Pass Paramorphism (`single_bug_mutants`)
    let target_filter = col("poster_altitude_m").ge(1500) & col("condition_note").is_null();
    let mutants = target_filter.single_bug_mutants();
    let mutant_names: BTreeSet<String> = mutants.iter().map(|(n, _)| n.clone()).collect();
    assert!(mutant_names.contains("dropped_left_conjunct"));
    assert!(mutant_names.contains("dropped_right_conjunct"));
    assert!(mutant_names.contains("and_to_or"));
    assert!(mutant_names.contains("exclusive_boundary"));
    assert!(mutant_names.contains("negated_null_check"));

    let sample_rows = vec![row_101, row_102, row_103];
    let bug_exclusive = col("poster_altitude_m").gt(1500) & col("condition_note").is_null();
    assert_eq!(
        target_filter.match_single_bug_mutant(&bug_exclusive, &sample_rows),
        Some("exclusive_boundary".to_string())
    );

    let bug_null = col("poster_altitude_m").ge(1500) & col("condition_note").is_not_null();
    assert_eq!(
        target_filter.match_single_bug_mutant(&bug_null, &sample_rows),
        Some("negated_null_check".to_string())
    );

    let bug_or = col("poster_altitude_m").ge(1500) | col("condition_note").is_null();
    assert_eq!(
        target_filter.match_single_bug_mutant(&bug_or, &sample_rows),
        Some("and_to_or".to_string())
    );

    // Structured diagnosis distinguishes `ExactNormalizedAst` from `SampleRowEquivalence`:
    assert_eq!(
        target_filter.diagnose_single_bug_mutant(&bug_exclusive, &sample_rows),
        Some(MutantMatchDiagnosis {
            mutant_name: "exclusive_boundary".to_string(),
            mutant_sql: "poster_altitude_m > 1500 AND condition_note IS NULL".to_string(),
            match_basis: MutantMatchBasis::ExactNormalizedAst,
        })
    );

    let sample_equiv_candidate =
        col("poster_altitude_m").ge(1600) & col("condition_note").is_null();
    assert_eq!(
        target_filter.diagnose_single_bug_mutant(&sample_equiv_candidate, &[]),
        None
    );
    assert_eq!(
        target_filter.diagnose_single_bug_mutant(&sample_equiv_candidate, &sample_rows),
        Some(MutantMatchDiagnosis {
            mutant_name: "exclusive_boundary".to_string(),
            mutant_sql: "poster_altitude_m > 1500 AND condition_note IS NULL".to_string(),
            match_basis: MutantMatchBasis::SampleRowEquivalence {
                sample_row_count: 3,
            },
        })
    );
}

#[test]
fn relational_schema_graph_records_skipped_check_diagnostics_for_unsupported_check_syntax() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "
        CREATE TABLE cryo_ballast_reservoirs (
            reservoir_id INTEGER PRIMARY KEY,
            cell_code TEXT NOT NULL CHECK (cell_code LIKE 'CELL-%'),
            reserve_liters INTEGER NOT NULL CHECK (reserve_liters >= 0),
            CONSTRAINT chk_code_len CHECK (length(cell_code) >= 6),
            CONSTRAINT chk_cap CHECK (reserve_liters <= 5000 AND cell_code != 'DECOMMISSIONED')
        );
        ",
    )
    .unwrap();

    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
    let imported = graph.check_constraints_for("cryo_ballast_reservoirs");
    assert_eq!(imported.len(), 2);
    assert_eq!(imported[0], col("reserve_liters").ge(0));
    assert_eq!(
        imported[1],
        col("reserve_liters").le(5000) & col("cell_code").ne_str("DECOMMISSIONED")
    );

    let skipped = graph.skipped_checks_for("cryo_ballast_reservoirs");
    assert_eq!(skipped.len(), 2);
    assert_eq!(skipped[0].table_name, "cryo_ballast_reservoirs");
    assert_eq!(skipped[0].constraint_name.as_deref(), Some("cell_code"));
    assert!(skipped[0].raw_expr_sql.contains("LIKE"));
    assert_eq!(skipped[1].table_name, "cryo_ballast_reservoirs");
    assert_eq!(skipped[1].constraint_name.as_deref(), Some("chk_code_len"));
    assert!(skipped[1].raw_expr_sql.contains("length"));
}

#[test]
fn relational_schema_graph_fk_reachable_subtree_and_depth_handle_chains_and_cycles() {
    let conn = setup_alpine_fk_connection();
    let mut graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();

    let tree = graph.fk_reachable_subtree("alpine_levain_campaigns");
    assert_eq!(tree.label, "alpine_levain_campaigns");
    assert_eq!(tree.children.len(), 1);
    assert_eq!(tree.children[0].label, "alpine_poster_links");
    assert!(tree.children[0].children.is_empty());
    assert_eq!(graph.fk_subtree_depth("alpine_levain_campaigns"), 2);
    assert_eq!(graph.fk_subtree_depth("alpine_poster_links"), 1);

    graph.foreign_keys.push(ForeignKeyEdge::new(
        "alpine_levain_campaigns",
        "campaign_id",
        "alpine_poster_links",
        "campaign_id",
    ));
    let cyclic_tree = graph.fk_reachable_subtree("alpine_levain_campaigns");
    assert_eq!(cyclic_tree.children.len(), 1);
    assert!(cyclic_tree.children[0].children.is_empty());
    assert_eq!(graph.fk_subtree_depth("alpine_levain_campaigns"), 2);
}

#[test]
fn from_sqlparser_expr_normalizes_reversed_binary_comparisons_and_between() {
    assert_eq!(
        SqlPredicateAst::parse_sql_where("0 <= reserve_liters").unwrap(),
        col("reserve_liters").ge(0)
    );
    assert_eq!(
        SqlPredicateAst::parse_sql_where("100 > reserve_liters").unwrap(),
        col("reserve_liters").lt(100)
    );
    assert_eq!(
        SqlPredicateAst::parse_sql_where("'OPEN' = status").unwrap(),
        col("status").eq_str("OPEN")
    );
    assert_eq!(
        SqlPredicateAst::parse_sql_where("reserve_liters BETWEEN 10 AND 50").unwrap(),
        col("reserve_liters").ge(10) & col("reserve_liters").le(50)
    );
    assert_eq!(
        SqlPredicateAst::parse_sql_where("reserve_liters NOT BETWEEN 10 AND 50").unwrap(),
        !(col("reserve_liters").ge(10) & col("reserve_liters").le(50))
    );
}

#[test]
fn three_valued_sql_truth_semantics_distinguishes_where_selection_from_check_satisfaction() {
    let null_row = BTreeMap::from([("x".to_string(), serde_json::Value::Null)]);
    let missing_row: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    let eq_one = col("x").eq_int(1);
    assert_eq!(
        eq_one.evaluate_row_three_valued(&null_row),
        SqlTruthValue::Unknown
    );
    assert!(!eq_one.evaluate_row(&null_row));

    let not_eq_one = !col("x").eq_int(1);
    assert_eq!(
        not_eq_one.evaluate_row_three_valued(&null_row),
        SqlTruthValue::Unknown
    );
    assert!(!not_eq_one.evaluate_row(&null_row));

    let gt_zero = col("x").gt(0);
    assert_eq!(
        gt_zero.evaluate_row_three_valued(&null_row),
        SqlTruthValue::Unknown
    );
    assert!(!gt_zero.evaluate_row(&null_row));
    assert!(gt_zero.evaluate_row_for_check(&null_row));

    let neg_row = BTreeMap::from([("x".to_string(), serde_json::json!(-5))]);
    assert_eq!(
        gt_zero.evaluate_row_three_valued(&neg_row),
        SqlTruthValue::False
    );
    assert!(!gt_zero.evaluate_row_for_check(&neg_row));

    assert_eq!(
        col("x").is_null().evaluate_row_three_valued(&null_row),
        SqlTruthValue::True
    );
    assert_eq!(
        col("x").is_not_null().evaluate_row_three_valued(&null_row),
        SqlTruthValue::False
    );
    assert_eq!(
        col("x").is_null().evaluate_row_three_valued(&missing_row),
        SqlTruthValue::Unknown
    );
}

#[test]
fn strongly_typed_predicate_frame_and_sqlparser_where_and_tokenizer_cursor_handle_keywords_in_strings_and_not_in(
) {
    let parsed = SqlPredicateAst::parse_sql_where(
        "WHERE (review_status = 'ACTIVE AND READY' OR poster_altitude_m >= -50) AND glacier_clamp NOT IN ('LOCKED', 'JAMMED') AND condition_note IS NOT NULL;",
    )
    .expect("sqlparser should parse complex WHERE with NOT IN and embedded keywords");
    let expected = (col("review_status").eq_str("ACTIVE AND READY")
        | col("poster_altitude_m").ge(-50))
        & !col("glacier_clamp").in_strs(["LOCKED", "JAMMED"])
        & col("condition_note").is_not_null();
    assert_eq!(parsed, expected);

    let leaf = col("poster_altitude_m").ge(1500);
    assert_eq!(
        *leaf.0,
        SqlPredicateFrame::Comparison {
            column: SqlColumnRef::from("poster_altitude_m"),
            op: SqlComparisonOp::Ge,
            literal: SqlScalarValue::Int(1500),
        }
    );
}

#[test]
fn quoted_identifiers_in_scalar_position_fail_lowering_instead_of_rewriting_as_string_literals() {
    let single_quoted = SqlPredicateAst::parse_sql_where("a = 'b'")
        .expect("single-quoted string literal must lower cleanly");
    assert_eq!(single_quoted, col("a").eq_str("b"));

    for col_cmp in ["a = \"b\"", "a = `b`", "a = [b]"] {
        let res = SqlPredicateAst::parse_sql_where(col_cmp);
        assert!(
            res.is_err(),
            "expected `{col_cmp}` to fail lowering as an unsupported column-to-column comparison, got: {res:?}"
        );
    }

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (a TEXT, b TEXT, CHECK(a = \"b\"));")
        .unwrap();
    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
    assert!(graph.check_constraints_for("t").is_empty());
    assert_eq!(graph.skipped_checks_for("t").len(), 1);
}

#[test]
fn inspect_from_connection_resolves_implicit_foreign_key_parent_primary_key_column() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, name TEXT);\n\
         CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p);",
    )
    .unwrap();

    let graph = RelationalSchemaGraph::inspect_from_connection(&conn)
        .expect("schema inspection with shorthand REFERENCES p must succeed");
    assert_eq!(graph.foreign_keys.len(), 1);
    let fk = &graph.foreign_keys[0];
    assert_eq!(fk.child_table, "c");
    assert_eq!(fk.child_col, "pid");
    assert_eq!(fk.parent_table, "p");
    assert_eq!(fk.parent_col, "id");
}

#[test]
fn parse_sql_where_handles_multibyte_utf8_without_panic_and_rejects_trailing_garbage() {
    let utf8_pred = SqlPredicateAst::parse_sql_where("a你你 > 1")
        .expect("multibyte UTF-8 identifier must parse without slicing panic");
    assert_eq!(utf8_pred, col("a你你").gt(1));

    let trailing_err = SqlPredicateAst::parse_sql_where("x = 1 garbage");
    assert!(
        trailing_err.is_err(),
        "expected `x = 1 garbage` to fail with trailing token error, got: {trailing_err:?}"
    );

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE t_text_num (x TEXT CHECK(x >= 2));\n\
         CREATE TABLE t_nocase (x TEXT COLLATE NOCASE CHECK(x = 'abc'));",
    )
    .unwrap();
    let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
    assert!(graph.check_constraints_for("t_text_num").is_empty());
    assert_eq!(graph.skipped_checks_for("t_text_num").len(), 1);
    assert!(graph.check_constraints_for("t_nocase").is_empty());
    assert_eq!(graph.skipped_checks_for("t_nocase").len(), 1);
}

#[test]
fn structured_sql_ident_and_column_ref_preserve_quotes_whitespace_dots_keywords_and_roundtrip() {
    use proptest::prelude::*;

    let whitespace_pred = SqlPredicateAst::from_sql_str("\" x \" >= 10")
        .expect("parse quoted padded-whitespace identifier");
    assert_eq!(whitespace_pred.render_sql(), "\" x \" >= 10");
    assert_eq!(
        SqlPredicateAst::from_sql_str(&whitespace_pred.render_sql()),
        Ok(whitespace_pred.clone())
    );

    let all_spaces_pred =
        SqlPredicateAst::from_sql_str("\"   \" = 1").expect("parse all-spaces identifier");
    assert_eq!(all_spaces_pred.render_sql(), "\"   \" = 1");
    assert_eq!(
        SqlPredicateAst::from_sql_str(&all_spaces_pred.render_sql()),
        Ok(all_spaces_pred)
    );

    let space_and_dot_pred = col("a b").eq_int(7)
        & col("a.b").ge(3)
        & qualified_col("t", "a b").ne_str("LOCKED")
        & qualified_col("my.tbl", "a\"b").is_not_null()
        & col("order").in_ints([1, 2])
        & col("123col").lt(99);
    assert_eq!(
        space_and_dot_pred.render_sql(),
        "\"a b\" = 7 AND \"a.b\" >= 3 AND t.\"a b\" != 'LOCKED' AND \"my.tbl\".\"a\"\"b\" IS NOT NULL AND \"order\" IN (1, 2) AND \"123col\" < 99"
    );
    let reparsed = SqlPredicateAst::from_sql_str(&space_and_dot_pred.render_sql())
        .expect("roundtrip complex quoted/qualified SQL predicate");
    assert_eq!(reparsed, space_and_dot_pred);

    let unqual_dot = SqlPredicateAst::from_sql_str("\"a.b\" = 5").unwrap();
    let qual_dot = SqlPredicateAst::from_sql_str("a.b = 5").unwrap();
    assert_ne!(unqual_dot, qual_dot);
    assert_eq!(
        unqual_dot.referenced_column_refs(),
        BTreeSet::from([SqlColumnRef::unqualified(SqlIdent::quoted("a.b"))])
    );
    assert_eq!(
        qual_dot.referenced_column_refs(),
        BTreeSet::from([SqlColumnRef::qualified(
            SqlIdent::unquoted("a"),
            SqlIdent::unquoted("b")
        )])
    );

    let padded_row = BTreeMap::from([
        (" x ".to_string(), serde_json::json!(15)),
        ("x".to_string(), serde_json::json!(-5)),
    ]);
    let unpadded_pred = col("x").ge(10);
    assert!(whitespace_pred.evaluate_row(&padded_row));
    assert!(!unpadded_pred.evaluate_row(&padded_row));

    let only_b_row = BTreeMap::from([("b".to_string(), serde_json::json!(5))]);
    let literal_a_dot_b_row = BTreeMap::from([("a.b".to_string(), serde_json::json!(5))]);
    assert_eq!(
        unqual_dot.evaluate_row_three_valued(&only_b_row),
        SqlTruthValue::Unknown
    );
    assert!(unqual_dot.evaluate_row(&literal_a_dot_b_row));
    assert!(qual_dot.evaluate_row(&only_b_row));
    assert!(qual_dot.evaluate_row(&literal_a_dot_b_row));

    let schema_conn = Connection::open_in_memory().unwrap();
    schema_conn
        .execute_batch("CREATE TABLE t (b INTEGER, \"a b\" TEXT);")
        .unwrap();
    let schema_graph = RelationalSchemaGraph::inspect_from_connection(&schema_conn).unwrap();
    let schema_t = schema_graph.get_table("t").expect("table t exists");
    assert!(
        unqual_dot.validate_against_schema(schema_t, None).is_err(),
        "unqualified `\"a.b\"` must not match column `b`"
    );
    assert!(qualified_col("t", "a b")
        .eq_str("OK")
        .validate_against_schema(schema_t, None)
        .is_ok());
    assert!(
        qualified_col("wrong_table", "a b")
            .eq_str("OK")
            .validate_against_schema(schema_t, None)
            .is_err(),
        "mismatched table qualifier must fail schema validation"
    );

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE t (
            id INTEGER PRIMARY KEY,
            \" x \" INTEGER,
            x INTEGER,
            \"a b\" INTEGER,
            \"a.b\" INTEGER,
            \"order\" TEXT,
            \"a\"\"b\" INTEGER
        );
        INSERT INTO t VALUES (1, 12, -5, 7, 3, 'OPEN', 42);
        INSERT INTO t VALUES (2, -5, 12, 7, 3, 'OPEN', 42);
        INSERT INTO t VALUES (3, 12, 12, 0, 3, 'LOCKED', NULL);",
    )
    .unwrap();

    let live_pred = col(" x ").ge(10)
        & qualified_col("t", "a b").eq_int(7)
        & col("a.b").ge(3)
        & col("order").eq_str("OPEN")
        & col("a\"b").is_not_null();
    let query = format!(
        "SELECT id FROM t WHERE {} ORDER BY id;",
        live_pred.render_sql()
    );
    let selected_ids: Vec<i64> = conn
        .prepare(&query)
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(selected_ids, vec![1]);

    let col_strategy = prop_oneof![
        Just(SqlColumnRef::from("x")),
        Just(SqlColumnRef::from(" x ")),
        Just(SqlColumnRef::from("a b")),
        Just(SqlColumnRef::from("a.b")),
        Just(SqlColumnRef::from("order")),
        Just(SqlColumnRef::from("select")),
        Just(SqlColumnRef::from("true")),
        Just(SqlColumnRef::from("123col")),
        Just(SqlColumnRef::from("a\"b")),
        Just(SqlColumnRef::qualified("t", "a b")),
        Just(SqlColumnRef::qualified("my.tbl", " x ")),
    ];
    let leaf_strategy = prop_oneof![
        (
            col_strategy.clone(),
            prop_oneof![
                Just(SqlComparisonOp::Eq),
                Just(SqlComparisonOp::Ne),
                Just(SqlComparisonOp::Lt),
                Just(SqlComparisonOp::Le),
                Just(SqlComparisonOp::Gt),
                Just(SqlComparisonOp::Ge),
            ],
            prop_oneof![
                (-5i64..=5i64).prop_map(SqlScalarValue::Int),
                prop_oneof![Just(-2.5), Just(0.0), Just(2.0), Just(3.25)]
                    .prop_map(SqlScalarValue::Float),
                prop_oneof![Just("OPEN"), Just("A'B"), Just("SELECT WHERE")]
                    .prop_map(|s| SqlScalarValue::Text(s.to_string())),
                any::<bool>().prop_map(SqlScalarValue::Bool),
            ],
        )
            .prop_map(|(c, op, lit)| SqlPredicateAst::comparison(c, op, lit)),
        (
            col_strategy.clone(),
            prop::collection::vec(-2i64..=3i64, 1..=3),
        )
            .prop_map(|(c, ns)| col(c).in_ints(ns)),
        (col_strategy, any::<bool>()).prop_map(|(c, neg)| SqlPredicateAst::is_null(c, neg)),
    ];
    let tree_strategy = leaf_strategy.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|p| !p),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| l & r),
            (inner.clone(), inner).prop_map(|(l, r)| l | r),
        ]
    });

    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(96));
    runner
        .run(&tree_strategy, |ast| {
            let sql = ast.render_sql();
            let parsed = SqlPredicateAst::from_sql_str(&sql)
                .map_err(|e| TestCaseError::fail(format!("failed to parse `{sql}`: {e}")))?;
            prop_assert_eq!(parsed, ast, "roundtrip mismatch for rendered SQL `{}`", sql);
            Ok(())
        })
        .expect("arbitrary SqlPredicateAst trees must roundtrip losslessly via render_sql -> from_sql_str");
}

#[test]
fn check_constraint_domain_errors_fail_closed_while_sql_null_passes_in_ast_and_schema() {
    let check_pred = SqlPredicateAst::from_sql_str("balance >= 0").unwrap();

    let null_row = BTreeMap::from([("balance".to_string(), serde_json::Value::Null)]);
    assert_eq!(
        check_pred.try_evaluate_check_constraint(&null_row),
        Ok(true)
    );
    assert!(check_pred.evaluate_check_constraint(&null_row));
    assert!(check_pred.evaluate_row_for_check(&null_row));
    assert!(!check_pred.evaluate_row(&null_row));

    let neg_row = BTreeMap::from([("balance".to_string(), serde_json::json!(-5))]);
    assert_eq!(
        check_pred.try_evaluate_check_constraint(&neg_row),
        Ok(false)
    );
    assert!(!check_pred.evaluate_check_constraint(&neg_row));
    assert!(!check_pred.evaluate_row_for_check(&neg_row));

    for bad_val in [
        serde_json::json!("OPEN"),
        serde_json::json!([1, 2]),
        serde_json::json!({"nested": 1}),
    ] {
        let bad_row = BTreeMap::from([("balance".to_string(), bad_val.clone())]);
        let err = check_pred
            .try_evaluate_check_constraint(&bad_row)
            .expect_err("incompatible non-null domain value must return Err(SqlRowDomainError)");
        assert_eq!(err.column, "balance");
        assert!(
            !check_pred.evaluate_check_constraint(&bad_row),
            "evaluate_check_constraint must fail closed (false) on {bad_val:?}"
        );
        assert!(
            !check_pred.evaluate_row_for_check(&bad_row),
            "evaluate_row_for_check must fail closed (false) on {bad_val:?}"
        );
    }
}
