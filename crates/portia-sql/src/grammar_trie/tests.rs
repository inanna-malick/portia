use proptest::prelude::*;
use rusqlite::Connection;
use serde::Serialize;

use super::*;

fn sample_bgp_schema() -> TableSchema {
    TableSchema::new(
        "bgp_incidents",
        vec![
            ColumnSchema::new("incident_id", "INTEGER", true),
            ColumnSchema::new("pop_code", "TEXT", false),
            ColumnSchema::new("lifecycle_state", "TEXT", false),
            ColumnSchema::new("drop_rate_bps", "INTEGER", false),
            ColumnSchema::new("mitigation_rule", "TEXT", false),
            ColumnSchema::new("policy_decision", "TEXT", false),
            ColumnSchema::new("oncall_sre", "TEXT", false),
            ColumnSchema::new("change_freeze", "TEXT", false),
        ],
    )
}

#[test]
fn live_database_engine_inspect_schema_reads_pragma_table_info_and_catalog() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE bgp_incidents (
            incident_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL,
            drop_rate_bps INTEGER NOT NULL
        );
        CREATE TABLE pop_facilities (
            facility_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL
        );",
    )
    .unwrap();
    let schema = inspect_table_schema(&conn, "bgp_incidents").unwrap();
    assert_eq!(schema.table_name, "bgp_incidents");
    assert_eq!(schema.primary_key_column(), Some("incident_id"));
    assert_eq!(
        schema.column_names(),
        vec![
            "incident_id".to_string(),
            "pop_code".to_string(),
            "lifecycle_state".to_string(),
            "drop_rate_bps".to_string()
        ]
    );
    assert_eq!(schema.columns[0].cid, 0);
    assert!(schema.columns[0].is_primary_key);
    assert_eq!(schema.columns[1].cid, 1);
    assert!(schema.columns[1].notnull);
    assert!(!schema.columns[1].nullable());
    let all = inspect_all_table_schemas(&conn).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].table_name, "bgp_incidents");
    assert_eq!(all[1].table_name, "pop_facilities");
}

#[test]
fn orbital_25col_schema_strips_5_quarantined_decoys_while_preserving_archive_pk() {
    let mut cols: Vec<ColumnSchema> = [
        ("maneuver_id", "INTEGER", true),
        ("vehicle_code", "TEXT", false),
        ("lifecycle_state", "TEXT", false),
        ("delta_v_mm_s", "INTEGER", false),
        ("burn_rule", "TEXT", false),
        ("policy_decision", "TEXT", false),
        ("flight_director", "TEXT", false),
        ("change_freeze", "TEXT", false),
        ("orbit_regime", "TEXT", false),
        ("perigee_km", "INTEGER", false),
        ("apogee_km", "INTEGER", false),
        ("inclination_deg", "INTEGER", false),
        ("propellant_kg", "INTEGER", false),
        ("thruster_bank", "TEXT", false),
        ("conjunction_risk", "TEXT", false),
        ("ground_station_code", "TEXT", false),
        ("telemetry_band", "TEXT", false),
        ("attitude_mode", "TEXT", false),
        ("solar_beta_deg", "INTEGER", false),
        ("burn_duration_sec", "INTEGER", false),
    ]
    .into_iter()
    .map(|(n, t, pk)| ColumnSchema::new(n, t, pk))
    .collect();

    let decoys = [
        ("legacy_delta_v_deprecated", "INTEGER"),
        ("archive_delta_v_v1", "INTEGER"),
        ("shadow_delta_v_decoy", "INTEGER"),
        ("legacy_propellant_deprecated", "INTEGER"),
        ("shadow_burn_rule_decoy", "TEXT"),
    ];
    for (n, t) in decoys {
        assert!(is_quarantined_grammar_trie_decoy_column(n));
        cols.push(ColumnSchema::new(n, t, false));
    }
    assert_eq!(cols.len(), 25);

    let clean_schema = TableSchema::new("orbital_maneuvers", cols).without_quarantined_decoys();
    assert_eq!(clean_schema.columns.len(), 20);
    let remaining = clean_schema.column_name_set();
    assert!(remaining.contains("delta_v_mm_s"));
    for (decoy, _) in decoys {
        assert!(!remaining.contains(decoy), "decoy column {decoy} leaked");
    }

    let archive_schema = TableSchema::new(
        "orbital_maneuvers_archive",
        vec![
            ColumnSchema::new("archive_id", "INTEGER", true),
            ColumnSchema::new("maneuver_id", "INTEGER", false),
            ColumnSchema::new("archive_maneuver_v1", "INTEGER", false),
        ],
    )
    .without_quarantined_decoys();
    assert_eq!(archive_schema.columns.len(), 2);
    assert_eq!(archive_schema.primary_key_column(), Some("archive_id"));
}

#[test]
fn inspect_top_level_sql_clauses_summarizes_structure_and_ignores_string_literal_keywords() {
    let bgp = sample_bgp_schema();
    let bgp_arch = TableSchema::new(
        "bgp_incidents_archive",
        vec![ColumnSchema::new("archive_id", "INTEGER", true)],
    );
    let pop = TableSchema::new(
        "pop_facilities",
        vec![ColumnSchema::new("pop_code", "TEXT", true)],
    );
    let schemas = vec![bgp, bgp_arch, pop];

    // Ambiguous partial prefix without delimiter withholds primary_table until delimited:
    let undelimited = inspect_top_level_sql_clauses("SELECT * FROM bgp_incidents", &schemas);
    assert_eq!(undelimited.statement_keyword.as_deref(), Some("SELECT"));
    assert!(undelimited.has_select_wildcard);
    assert_eq!(undelimited.primary_table, None);

    let delimited = inspect_top_level_sql_clauses("SELECT * FROM bgp_incidents;", &schemas);
    assert_eq!(delimited.primary_table.as_deref(), Some("bgp_incidents"));

    // String literal containing spoofed SQL clauses does not set structural flags:
    let spoofed = inspect_top_level_sql_clauses(
        "SELECT pop_code FROM bgp_incidents WHERE mitigation_rule = 'LEFT JOIN pop_facilities ON x = y GROUP BY z ORDER BY w LIMIT 5",
        &schemas,
    );
    assert_eq!(spoofed.primary_table.as_deref(), Some("bgp_incidents"));
    assert!(spoofed.has_where_keyword);
    assert!(spoofed.has_where_comparison_or_like);
    assert!(!spoofed.has_join_keyword);
    assert!(!spoofed.has_left_join);
    assert!(!spoofed.has_on_keyword);
    assert!(!spoofed.has_group_by_clause);
    assert!(!spoofed.has_order_by_clause);
    assert!(!spoofed.has_limit_keyword);

    // Full aggregate + LEFT JOIN + ON + WHERE LIKE + GROUP BY + ORDER BY + LIMIT summary:
    let rich = inspect_top_level_sql_clauses(
        "SELECT pop_facilities.pop_code, SUM(drop_rate_bps) FROM bgp_incidents LEFT JOIN pop_facilities ON bgp_incidents.pop_code = pop_facilities.pop_code WHERE lifecycle_state LIKE 'ACT%' GROUP BY pop_facilities.pop_code ORDER BY pop_facilities.pop_code DESC LIMIT 10;",
        &schemas,
    );
    assert_eq!(rich.primary_table.as_deref(), Some("bgp_incidents"));
    assert_eq!(rich.joined_tables, vec!["pop_facilities".to_string()]);
    assert!(rich.has_left_join && rich.has_join_keyword && rich.has_on_keyword);
    assert!(rich.has_aggregate_call && rich.has_where_keyword && rich.has_where_comparison_or_like);
    assert!(rich.has_group_by_clause && rich.has_order_by_clause && rich.has_limit_keyword);

    let pragma = inspect_top_level_sql_clauses("PRAGMA table_info(pop_facilities);", &schemas);
    assert_eq!(pragma.statement_keyword.as_deref(), Some("PRAGMA"));
    assert_eq!(pragma.primary_table.as_deref(), Some("pop_facilities"));
}

#[test]
fn validate_completed_sql_uses_sqlparser_ast_across_dml_joins_ctes_and_literals() {
    let bgp_schema = sample_bgp_schema();
    let pop_schema = TableSchema::new(
        "pop_facilities",
        vec![
            ColumnSchema::new("facility_id", "INTEGER", true),
            ColumnSchema::new("pop_code", "TEXT", false),
            ColumnSchema::new("region_name", "TEXT", false),
            ColumnSchema::new("capacity_gbps", "INTEGER", false),
        ],
    );
    let schemas = vec![bgp_schema, pop_schema];

    for valid_sql in [
        "UPDATE bgp_incidents SET lifecycle_state = 'WHERE SET x = 1', mitigation_rule = 'FROM other_table WHERE bad_col = 2' WHERE incident_id = 501;",
        "WITH recent_bgp AS (SELECT incident_id, pop_code FROM bgp_incidents WHERE lifecycle_state = 'ACTIVE') SELECT * FROM recent_bgp;",
        "PRAGMA table_info(BGP_INCIDENTS);",
        "SELECT COUNT(*) FROM bgp_incidents WHERE drop_rate_bps >= 500;",
        "SELECT incident_id FROM bgp_incidents WHERE EXISTS (SELECT facility_id FROM pop_facilities WHERE pop_facilities.pop_code = bgp_incidents.pop_code);",
        "SELECT bgp_incidents.incident_id, pop_facilities.region_name FROM bgp_incidents JOIN pop_facilities ON bgp_incidents.pop_code = pop_facilities.pop_code WHERE pop_facilities.capacity_gbps >= 100 AND bgp_incidents.lifecycle_state LIKE 'ACT%';",
        "SELECT incident_id, region_name FROM bgp_incidents LEFT JOIN pop_facilities ON bgp_incidents.pop_code = pop_facilities.pop_code ORDER BY capacity_gbps DESC LIMIT 5;",
    ] {
        assert!(
            validate_completed_sql_against_schemas(valid_sql, &schemas).is_ok(),
            "expected valid SQL: {valid_sql}"
        );
    }

    let dup_err = validate_completed_sql_against_schemas(
        "UPDATE bgp_incidents SET lifecycle_state = 'WHERE SET', lifecycle_state = 'ACTIVE' WHERE incident_id = 501;",
        &schemas,
    )
    .unwrap_err();
    assert!(dup_err.contains("Duplicate SET assignment to column `lifecycle_state`"));

    for (invalid_sql, expected_substr) in [
        (
            "UPDATE bgp_incidents SET hallucinated_col = 'SET WHERE' WHERE incident_id = 501;",
            "Hallucinated column `hallucinated_col` in SET clause",
        ),
        (
            "SELECT * FROM bgp_incidents WHERE pop_code = 'WHERE SET' AND non_existent_col = 42;",
            "Hallucinated column `non_existent_col` in WHERE clause",
        ),
        (
            "INSERT INTO bgp_incidents (incident_id, bogus_col) VALUES (501, 'SET WHERE');",
            "Hallucinated column `bogus_col` in INSERT INTO column list",
        ),
        (
            "SELECT * FROM missing_table WHERE incident_id = 501;",
            "Unknown table `missing_table`",
        ),
        (
            "SELECT * FROM bgp_incidents ORDER BY hallucinated_col DESC LIMIT 1;",
            "Hallucinated column `hallucinated_col` in ORDER BY clause",
        ),
        (
            "SELECT * FROM bgp_incidents JOIN unknown_table ON bgp_incidents.pop_code = unknown_table.pop_code;",
            "Unknown table `unknown_table`",
        ),
        (
            "SELECT * FROM bgp_incidents JOIN pop_facilities ON bgp_incidents.pop_code = pop_facilities.bogus_col;",
            "Hallucinated column `bogus_col` in JOIN ON clause",
        ),
        (
            "SELECT ghost_table.pop_code FROM bgp_incidents JOIN pop_facilities ON bgp_incidents.pop_code = pop_facilities.pop_code;",
            "Unknown table or alias qualifier `ghost_table`",
        ),
    ] {
        let err = validate_completed_sql_against_schemas(invalid_sql, &schemas)
            .expect_err(invalid_sql);
        assert!(err.contains(expected_substr), "unexpected err `{err}` for `{invalid_sql}`");
    }

    for invalid_subq in [
        "PRAGMA table_info(missing_table);",
        "SELECT bad_col FROM bad_tbl UNION SELECT 1;",
        "SELECT incident_id FROM bgp_incidents WHERE incident_id IN (SELECT bad_col FROM bgp_incidents);",
        "SELECT incident_id FROM bgp_incidents WHERE EXISTS (SELECT 1 FROM bad_tbl);",
        "INSERT INTO bgp_incidents (incident_id, pop_code) SELECT incident_id, bogus_col FROM bgp_incidents;",
    ] {
        assert!(validate_completed_sql_against_schemas(invalid_subq, &schemas).is_err());
    }
}

#[test]
fn sqlite_schema_grammar_cartridge_compiles_specialized_lark_and_mounts_in_relay() {
    use portia_core::relay::{
        GrammarCartridgeSpec, PayloadGrammarAutomaton, RelayMachine, RelayPhase, SpanCartridge,
    };
    use portia_core::{CartridgeBank, Mode};

    let bgp_schema = TableSchema::new(
        "bgp_incidents",
        vec![
            ColumnSchema::new("incident_id", "INTEGER", true),
            ColumnSchema::new("drop_rate_bps", "INTEGER", false),
            ColumnSchema::new("legacy_drop_rate_v1", "INTEGER", false),
        ],
    );
    let audit_schema = TableSchema::new(
        "bgp_incidents_audit_log",
        vec![
            ColumnSchema::new("incident_id", "INTEGER", true),
            ColumnSchema::new("audit_actor", "TEXT", false),
        ],
    );
    let schemas = vec![bgp_schema, audit_schema];
    let cartridge = SqliteSchemaGrammarCartridge(&schemas);

    assert!(UPSTREAM_SQLITE_LARK_GRAMMAR.contains("expr : or_expr"));
    assert!(UPSTREAM_SQLITE_LARK_GRAMMAR.contains("atom_expr : literal_value"));
    assert!(UPSTREAM_SQLITE_LARK_GRAMMAR.contains("%ignore /[ \\t\\r\\n]+/"));

    for unconstrained_mode in [Mode::Ask, Mode::Result, Mode::Abstain] {
        assert_eq!(
            cartridge.grammar_cartridge_spec(unconstrained_mode),
            GrammarCartridgeSpec::Unconstrained
        );
    }
    assert_eq!(
        compile_sqlite_lark_grammar_for_schemas(&[], Mode::Read),
        GrammarCartridgeSpec::Unconstrained
    );

    let GrammarCartridgeSpec::LarkGrammar(probe_lark) =
        cartridge.grammar_cartridge_spec(Mode::Probe)
    else {
        panic!("expected LarkGrammar for Mode::Probe");
    };
    assert!(probe_lark.contains("start: pragma_stmt \";\"?"));
    assert!(probe_lark.contains("table_name\t\t: SCHEMA_TABLE_IDENT"));
    assert!(probe_lark.contains("bgp_incidents") && probe_lark.contains("bgp_incidents_audit_log"));

    let GrammarCartridgeSpec::LarkGrammar(read_lark) = cartridge.grammar_cartridge_spec(Mode::Read)
    else {
        panic!("expected LarkGrammar for Mode::Read");
    };
    assert!(read_lark
        .contains("start: (select_stmt | compound_select_stmt | simple_select_stmt) \";\"?"));
    assert!(read_lark.contains("column_name\t\t: SCHEMA_COLUMN_IDENT"));
    for col in [
        "incident_id",
        "drop_rate_bps",
        "legacy_drop_rate_v1",
        "audit_actor",
    ] {
        assert!(read_lark.contains(col), "missing column {col}");
    }

    let clean_schemas: Vec<TableSchema> = schemas
        .iter()
        .map(TableSchema::without_quarantined_decoys)
        .collect();
    let GrammarCartridgeSpec::LarkGrammar(clean_read_lark) =
        compile_sqlite_lark_grammar_for_schemas(&clean_schemas, Mode::Read)
    else {
        panic!("expected LarkGrammar for cleaned Mode::Read");
    };
    assert!(clean_read_lark.contains("column_name\t\t: SCHEMA_COLUMN_IDENT"));
    for col in ["incident_id", "drop_rate_bps", "audit_actor"] {
        assert!(clean_read_lark.contains(col), "missing clean column {col}");
    }
    assert!(!clean_read_lark.contains("legacy_drop_rate_v1"));

    let GrammarCartridgeSpec::LarkGrammar(write_lark) =
        cartridge.grammar_cartridge_spec(Mode::Write)
    else {
        panic!("expected LarkGrammar for Mode::Write");
    };
    assert!(write_lark.contains("start: (update_stmt | insert_stmt | delete_stmt) \";\"?"));

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
    relay.step_trunk_token(10, " READ]").unwrap();
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
        .step_trunk_token(20, " Query bgp_incidents.\n")
        .unwrap();
    relay.flush_harness_prefill(|_| vec![3]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PayloadBody);
    assert_eq!(
        relay.active_span_cartridge(&cartridge),
        SpanCartridge::new(
            CartridgeBank::ModePayload(Mode::Read),
            cartridge.grammar_cartridge_spec(Mode::Read),
        )
    );
}

#[test]
fn epistemic_sql_grammar_cartridge_enforces_discovery_steward_and_readonly_laws() {
    use portia_core::relay::{
        GrammarCartridgeSpec, PayloadGrammarAutomaton, RelayMachine, RelayPhase, SpanCartridge,
        PLAN_BODY_SINGLE_LINE_LARK,
    };
    use portia_core::{CartridgeBank, Mode};

    let all_tables = vec![
        "bgp_incidents".to_string(),
        "bgp_incidents_audit_log".to_string(),
    ];
    let bgp_schema = TableSchema::new(
        "bgp_incidents",
        vec![
            ColumnSchema::new("incident_id", "INTEGER", true),
            ColumnSchema::new("drop_rate_bps", "INTEGER", false),
            ColumnSchema::new("legacy_drop_rate_decoy", "INTEGER", false),
        ],
    );
    let audit_schema = TableSchema::new(
        "bgp_incidents_audit_log",
        vec![
            ColumnSchema::new("incident_id", "INTEGER", true),
            ColumnSchema::new("audit_actor", "TEXT", false),
        ],
    );

    let unprobed_unknown_catalog = EpistemicSqlGrammarCartridge {
        all_table_names: &[],
        catalog_discovered: false,
        probed_schemas: &[],
        pending_steward_ask: false,
        steward_grant_approved: false,
        ambiguous_candidates: &[],
        read_only: false,
    };
    assert_eq!(
        unprobed_unknown_catalog.legal_modes(),
        vec![Mode::Probe, Mode::Ask, Mode::Abstain]
    );
    assert_eq!(
        compile_epistemic_span_grammar(&unprobed_unknown_catalog, Mode::Probe),
        GrammarCartridgeSpec::LarkGrammar(
            "start: \"SELECT name FROM sqlite_master WHERE type = \"i \"'table'\" \";\\n</sql>\""
                .to_string()
        )
    );
    assert_eq!(
        compile_epistemic_span_grammar(&unprobed_unknown_catalog, Mode::Read),
        GrammarCartridgeSpec::Unconstrained
    );
    assert_eq!(
        compile_epistemic_span_grammar(&unprobed_unknown_catalog, Mode::Write),
        GrammarCartridgeSpec::Unconstrained
    );

    let unprobed_with_table_hints = EpistemicSqlGrammarCartridge {
        all_table_names: &all_tables,
        catalog_discovered: false,
        probed_schemas: &[],
        pending_steward_ask: false,
        steward_grant_approved: true,
        ambiguous_candidates: &[],
        read_only: false,
    };
    assert_eq!(
        unprobed_with_table_hints.legal_modes(),
        vec![Mode::Probe, Mode::Ask, Mode::Abstain]
    );
    assert_eq!(
        compile_epistemic_span_grammar(&unprobed_with_table_hints, Mode::Probe),
        GrammarCartridgeSpec::LarkGrammar(
            "start: (\"SELECT name FROM sqlite_master WHERE type = \"i \"'table'\" | \"PRAGMA table_info(\"i (PROBE_TABLE_IDENT) \")\") \";\\n</sql>\"\nPROBE_TABLE_IDENT: /(?i-u:(?:(?:bgp_incidents|bgp_incidents_audit_log)|\\\"(?:bgp_incidents|bgp_incidents_audit_log)\\\"|`(?:bgp_incidents|bgp_incidents_audit_log)`))/"
                .to_string()
        )
    );

    let probed_bgp_only = [bgp_schema.clone()];
    let partial_probed_cart = EpistemicSqlGrammarCartridge {
        all_table_names: &all_tables,
        catalog_discovered: true,
        probed_schemas: &probed_bgp_only,
        pending_steward_ask: false,
        steward_grant_approved: false,
        ambiguous_candidates: &[],
        read_only: false,
    };
    assert_eq!(
        partial_probed_cart.legal_modes(),
        vec![
            Mode::Probe,
            Mode::Read,
            Mode::Ask,
            Mode::Write,
            Mode::Result,
            Mode::Abstain,
        ]
    );
    assert_eq!(
        compile_epistemic_span_grammar(&partial_probed_cart, Mode::Probe),
        GrammarCartridgeSpec::LarkGrammar(
            "start: \"PRAGMA table_info(\"i (PROBE_TABLE_IDENT) \")\" \";\\n</sql>\"\nPROBE_TABLE_IDENT: /(?i-u:(?:(?:bgp_incidents_audit_log)|\\\"(?:bgp_incidents_audit_log)\\\"|`(?:bgp_incidents_audit_log)`))/"
                .to_string()
        )
    );

    let GrammarCartridgeSpec::LarkGrammar(read_lark) =
        compile_epistemic_span_grammar(&partial_probed_cart, Mode::Read)
    else {
        panic!("expected LarkGrammar for Mode::Read");
    };
    assert!(read_lark.contains(
        "start: (select_stmt | compound_select_stmt | simple_select_stmt) \";\\n</sql>\""
    ));
    assert!(read_lark.contains("table_name\t\t: SCHEMA_TABLE_IDENT\n"));
    assert!(read_lark.contains("bgp_incidents"));
    assert!(!read_lark.contains("bgp_incidents_audit_log"));
    assert!(read_lark.contains("column_name\t\t: SCHEMA_COLUMN_IDENT\n"));
    for col in ["incident_id", "drop_rate_bps", "legacy_drop_rate_decoy"] {
        assert!(read_lark.contains(col), "missing probed column {col}");
    }

    let GrammarCartridgeSpec::LarkGrammar(write_lark) =
        compile_epistemic_span_grammar(&partial_probed_cart, Mode::Write)
    else {
        panic!("expected LarkGrammar for Mode::Write");
    };
    assert!(write_lark.contains("start: (update_stmt | insert_stmt | delete_stmt) \";\\n</sql>\""));
    assert!(write_lark.contains("table_name\t\t: SCHEMA_TABLE_IDENT\n"));
    assert!(write_lark.contains("bgp_incidents"));
    assert!(!write_lark.contains("bgp_incidents_audit_log"));

    let both_probed = [bgp_schema, audit_schema];
    let all_probed_cart = EpistemicSqlGrammarCartridge {
        probed_schemas: &both_probed,
        ..partial_probed_cart
    };
    assert_eq!(
        compile_epistemic_span_grammar(&all_probed_cart, Mode::Probe),
        GrammarCartridgeSpec::LarkGrammar(
            "start: \"PRAGMA table_info(\"i (PROBE_TABLE_IDENT) \")\" \";\\n</sql>\"\nPROBE_TABLE_IDENT: /(?i-u:(?:(?:bgp_incidents|bgp_incidents_audit_log)|\\\"(?:bgp_incidents|bgp_incidents_audit_log)\\\"|`(?:bgp_incidents|bgp_incidents_audit_log)`))/"
                .to_string()
        )
    );

    let steward_gated_cart = EpistemicSqlGrammarCartridge {
        pending_steward_ask: true,
        steward_grant_approved: false,
        ..partial_probed_cart
    };
    assert_eq!(
        steward_gated_cart.legal_modes(),
        vec![
            Mode::Probe,
            Mode::Read,
            Mode::Ask,
            Mode::Result,
            Mode::Abstain,
        ]
    );
    let GrammarCartridgeSpec::LarkGrammar(steward_ask_lark) =
        compile_epistemic_span_grammar(&steward_gated_cart, Mode::Ask)
    else {
        panic!("expected LarkGrammar for Steward Mode::Ask");
    };
    assert!(steward_ask_lark.starts_with("start: ask_obj \"\\n</to_user>\"\nask_obj: %json "));
    assert!(steward_ask_lark.contains("\"ALLOW\"") && steward_ask_lark.contains("\"DENY\""));
    assert!(steward_ask_lark.contains("\"select\"") && steward_ask_lark.contains("\"options\""));

    let steward_approved_cart = EpistemicSqlGrammarCartridge {
        pending_steward_ask: true,
        steward_grant_approved: true,
        ..partial_probed_cart
    };
    assert!(steward_approved_cart.legal_modes().contains(&Mode::Write));

    let ambig_opts = vec![
        "bgp_incidents".to_string(),
        "bgp_incidents_audit_log".to_string(),
    ];
    let ambiguous_cart = EpistemicSqlGrammarCartridge {
        ambiguous_candidates: &ambig_opts,
        ..partial_probed_cart
    };
    let GrammarCartridgeSpec::LarkGrammar(ambig_ask_lark) =
        compile_epistemic_span_grammar(&ambiguous_cart, Mode::Ask)
    else {
        panic!("expected LarkGrammar for ambiguous Mode::Ask");
    };
    assert!(ambig_ask_lark.starts_with("start: multi_obj \"\\n</to_user>\"\nmulti_obj: %json "));
    assert!(ambig_ask_lark.contains("\"multi\"") && ambig_ask_lark.contains("\"bgp_incidents\""));

    for user_mode in [Mode::Ask, Mode::Result, Mode::Abstain] {
        let GrammarCartridgeSpec::LarkGrammar(prose_lark) =
            compile_epistemic_span_grammar(&partial_probed_cart, user_mode)
        else {
            panic!("expected LarkGrammar for {user_mode:?}");
        };
        assert!(prose_lark
            .lines()
            .next()
            .unwrap()
            .ends_with("\"\\n</to_user>\""));
    }

    for (pending, approved) in [(false, false), (false, true), (true, false), (true, true)] {
        let ro_cart = EpistemicSqlGrammarCartridge {
            read_only: true,
            pending_steward_ask: pending,
            steward_grant_approved: approved,
            ..partial_probed_cart
        };
        assert!(!ro_cart.legal_modes().contains(&Mode::Write));
    }

    let mut relay = RelayMachine::new();
    relay.flush_harness_prefill(|_| vec![1]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::ModeLabel);
    assert_eq!(
        relay.active_span_cartridge(&unprobed_unknown_catalog),
        SpanCartridge::new(
            CartridgeBank::OrientTrunk,
            GrammarCartridgeSpec::LiteralSet(vec![
                " PROBE]".to_string(),
                " ASK]".to_string(),
                " ABSTAIN]".to_string(),
            ]),
        )
    );
    relay.step_trunk_token(5140, " PROBE]").unwrap();
    relay.flush_harness_prefill(|_| vec![2]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PlanBody);
    assert_eq!(
        relay.active_span_cartridge(&unprobed_unknown_catalog),
        SpanCartridge::new(
            CartridgeBank::Base,
            GrammarCartridgeSpec::LarkGrammar(PLAN_BODY_SINGLE_LINE_LARK.to_string()),
        )
    );
    relay
        .step_trunk_token(20, " Discover SQLite catalog tables.\n")
        .unwrap();
    relay.flush_harness_prefill(|_| vec![3]).unwrap();
    assert_eq!(relay.phase(), RelayPhase::PayloadBody);
    assert_eq!(
        relay.active_span_cartridge(&unprobed_unknown_catalog),
        SpanCartridge::new(
            CartridgeBank::ModePayload(Mode::Probe),
            compile_epistemic_span_grammar(&unprobed_unknown_catalog, Mode::Probe),
        )
    );
}

proptest! {
    #[test]
    fn prop_completed_sql_clauses_decoys_and_schema_cartridge_commute(
        t1_name in "t1_[a-z]{4,9}",
        t2_name in "t2_[a-z]{4,9}",
        pk_col in "id_[a-z]{3,7}",
        text_col in "code_[a-z]{3,7}",
        int_col in "val_[a-z]{3,7}",
        t2_col in "cap_[a-z]{3,7}",
        decoy_col in "legacy_[a-z]{3,6}_deprecated",
        row_id in 1i64..1000i64,
        int_val in 0i64..5000i64,
    ) {
        // These columns model the nonquarantined side of the schema. Random suffixes
        // can otherwise produce a deliberately quarantined name such as `val_old`.
        prop_assume!([&pk_col, &text_col, &int_col, &t2_col].iter()
            .all(|name| !is_quarantined_grammar_trie_decoy_column(name)));
        let s1_raw = TableSchema::new(
            t1_name.clone(),
            vec![
                ColumnSchema::new(pk_col.clone(), "INTEGER", true),
                ColumnSchema::new(text_col.clone(), "TEXT", false),
                ColumnSchema::new(int_col.clone(), "INTEGER", false),
                ColumnSchema::new(decoy_col.clone(), "TEXT", false),
            ],
        );
        let s1 = s1_raw.without_quarantined_decoys();
        prop_assert!(is_quarantined_grammar_trie_decoy_column(&decoy_col));
        prop_assert!(!s1.column_name_set().contains(&decoy_col));

        let s2 = TableSchema::new(
            t2_name.clone(),
            vec![
                ColumnSchema::new(pk_col.clone(), "INTEGER", true),
                ColumnSchema::new(t2_col.clone(), "INTEGER", false),
            ],
        );
        let schemas = vec![s1, s2];

        for valid_sql in [
            format!("PRAGMA table_info({t1_name});"),
            format!("SELECT {pk_col}, {text_col}, {int_col} FROM {t1_name} WHERE {pk_col} = {row_id} AND {int_col} >= {int_val};"),
            format!("SELECT {t1_name}.{text_col}, SUM({t2_name}.{t2_col}) FROM {t1_name} JOIN {t2_name} ON {t1_name}.{pk_col} = {t2_name}.{pk_col} WHERE {t1_name}.{text_col} = ' FROM {t2_name} JOIN ' GROUP BY {t1_name}.{text_col} ORDER BY {t1_name}.{text_col} ASC LIMIT 5;"),
            format!("UPDATE {t1_name} SET {int_col} = {int_val} WHERE {pk_col} = {row_id};"),
            format!("INSERT INTO {t1_name} ({pk_col}, {text_col}, {int_col}) VALUES ({row_id}, 'OK', {int_val});"),
            format!("DELETE FROM {t1_name} WHERE {pk_col} = {row_id};"),
        ] {
            prop_assert!(
                validate_completed_sql_against_schemas(&valid_sql, &schemas).is_ok(),
                "valid SQL rejected: {valid_sql}"
            );
        }

        for invalid_sql in [
            format!("PRAGMA table_info(unknown_{t1_name});"),
            format!("SELECT * FROM unknown_{t1_name} WHERE {pk_col} = {row_id};"),
            format!("SELECT {decoy_col} FROM {t1_name} WHERE {pk_col} = {row_id};"),
            format!("SELECT {pk_col} FROM {t1_name} UNION SELECT bogus_col FROM {t1_name};"),
            format!("SELECT * FROM {t1_name} WHERE {pk_col} = {row_id}; DROP TABLE {t1_name};"),
            format!("UPDATE {t1_name} SET bogus_col = {int_val} WHERE {pk_col} = {row_id};"),
        ] {
            prop_assert!(
                validate_completed_sql_against_schemas(&invalid_sql, &schemas).is_err(),
                "invalid SQL accepted: {invalid_sql}"
            );
        }

        let summary = inspect_top_level_sql_clauses(
            &format!("SELECT * FROM {t1_name} WHERE {text_col} = 'LEFT JOIN {t2_name} ON x = y GROUP BY z ORDER BY w LIMIT 9';"),
            &schemas,
        );
        prop_assert_eq!(summary.primary_table.as_deref(), Some(t1_name.as_str()));
        prop_assert!(summary.has_select_wildcard && summary.has_where_keyword);
        prop_assert!(!summary.has_join_keyword && !summary.has_group_by_clause && !summary.has_order_by_clause && !summary.has_limit_keyword);

        let cartridge = SqliteSchemaGrammarCartridge(&schemas);
        let read_spec = portia_core::relay::PayloadGrammarAutomaton::grammar_cartridge_spec(
            &cartridge,
            portia_core::Mode::Read,
        );
        prop_assert!(matches!(
            read_spec,
            portia_core::GrammarCartridgeSpec::LarkGrammar(ref lark)
                if lark.contains(&t1_name)
                    && lark.contains(&t2_name)
                    && lark.contains(&pk_col)
                    && lark.contains(&t2_col)
                    && !lark.contains(&decoy_col)
        ));
    }

    #[test]
    fn prop_epistemic_sql_grammar_cartridge_monotonic_and_provenance_invariants(
        table_suffixes in proptest::collection::btree_set("[a-z]{3,8}", 2..=8),
        probed_mask in proptest::collection::vec(any::<bool>(), 8),
        col_stem in "col_[a-z]{3,7}",
        decoy_col in "shadow_[a-z]{3,6}_decoy",
        catalog_discovered in any::<bool>(),
        pending_steward_ask in any::<bool>(),
        steward_grant_approved in any::<bool>(),
        read_only in any::<bool>(),
        use_ambiguous in any::<bool>(),
    ) {
        use portia_core::relay::PayloadGrammarAutomaton;
        use portia_core::{GrammarCartridgeSpec, Mode};

        let all_table_names: Vec<String> = table_suffixes
            .into_iter()
            .enumerate()
            .map(|(i, suf)| format!("tbl_{i}_{suf}"))
            .collect();

        let mut probed_schemas = Vec::new();
        let mut unprobed_names = Vec::new();
        for (idx, tbl_name) in all_table_names.iter().enumerate() {
            if probed_mask[idx % probed_mask.len()] {
                probed_schemas.push(TableSchema::new(
                    tbl_name.clone(),
                    vec![
                        ColumnSchema::new("id", "INTEGER", true),
                        ColumnSchema::new(format!("{col_stem}_{idx}"), "INTEGER", false),
                        ColumnSchema::new(format!("{decoy_col}_{idx}"), "TEXT", false),
                    ],
                ));
            } else {
                unprobed_names.push(tbl_name.clone());
            }
        }

        let ambig_slice: Vec<String> = if use_ambiguous {
            all_table_names.iter().take(2).cloned().collect()
        } else {
            Vec::new()
        };

        let cart = EpistemicSqlGrammarCartridge {
            all_table_names: &all_table_names,
            catalog_discovered,
            probed_schemas: &probed_schemas,
            pending_steward_ask,
            steward_grant_approved,
            ambiguous_candidates: &ambig_slice,
            read_only,
        };

        let modes = cart.legal_modes();
        if probed_schemas.is_empty() {
            prop_assert_eq!(modes, vec![Mode::Probe, Mode::Ask, Mode::Abstain]);
            prop_assert_eq!(
                compile_epistemic_span_grammar(&cart, Mode::Read),
                GrammarCartridgeSpec::Unconstrained
            );
            prop_assert_eq!(
                compile_epistemic_span_grammar(&cart, Mode::Write),
                GrammarCartridgeSpec::Unconstrained
            );
        } else {
            prop_assert!(modes.contains(&Mode::Probe));
            prop_assert!(modes.contains(&Mode::Read));
            prop_assert!(modes.contains(&Mode::Ask));
            prop_assert!(modes.contains(&Mode::Result));
            prop_assert!(modes.contains(&Mode::Abstain));

            let write_expected =
                !read_only && (!pending_steward_ask || steward_grant_approved);
            prop_assert_eq!(modes.contains(&Mode::Write), write_expected);
        }

        let GrammarCartridgeSpec::LarkGrammar(probe_lark) =
            compile_epistemic_span_grammar(&cart, Mode::Probe)
        else {
            prop_assert!(false, "Mode::Probe must always produce LarkGrammar");
            unreachable!()
        };
        prop_assert!(probe_lark.contains("\";\\n</sql>\""));
        if !catalog_discovered {
            prop_assert!(probe_lark.contains("\"SELECT name FROM sqlite_master WHERE type = \"i \"'table'\""));
        } else {
            prop_assert!(!probe_lark.contains("sqlite_master"));
            if !unprobed_names.is_empty() {
                for unprobed in &unprobed_names {
                    prop_assert!(probe_lark.contains(unprobed));
                }
                for probed in &probed_schemas {
                    prop_assert!(!probe_lark.contains(&probed.table_name));
                }
            }
        }

        if !probed_schemas.is_empty() {
            for sql_mode in [Mode::Read, Mode::Write] {
                let GrammarCartridgeSpec::LarkGrammar(sql_lark) =
                    compile_epistemic_span_grammar(&cart, sql_mode)
                else {
                    prop_assert!(false, "{sql_mode:?} must produce LarkGrammar when probed_schemas non-empty");
                    unreachable!()
                };
                prop_assert!(sql_lark.contains("\";\\n</sql>\""));
                for probed in &probed_schemas {
                    prop_assert!(sql_lark.contains(&probed.table_name));
                    for col in &probed.columns {
                        prop_assert!(sql_lark.contains(&col.name));
                    }
                }
                for unprobed in &unprobed_names {
                    prop_assert!(!sql_lark.contains(unprobed));
                }
            }
        }

        let GrammarCartridgeSpec::LarkGrammar(ask_lark) =
            compile_epistemic_span_grammar(&cart, Mode::Ask)
        else {
            prop_assert!(false, "Mode::Ask must produce LarkGrammar");
            unreachable!()
        };
        prop_assert!(ask_lark.contains("\"\\n</to_user>\""));
        if pending_steward_ask && !steward_grant_approved {
            prop_assert!(ask_lark.contains("%json") && ask_lark.contains("\"ALLOW\"") && ask_lark.contains("\"DENY\""));
        } else if ambig_slice.len() >= 2 {
            prop_assert!(ask_lark.contains("%json") && ask_lark.contains("\"multi\""));
        }
    }

    #[test]
    fn prop_wide_50_table_and_sql_keyword_schemas_preserve_grammar_and_ast_invariants(
        table_count in 1usize..=50usize,
        cols_per_table in 3usize..=15usize,
        kw_table_idx in 0usize..9usize,
        kw_col_idx in 0usize..9usize,
        mixed_case in any::<bool>(),
        catalog_discovered in any::<bool>(),
    ) {
        use portia_core::{GrammarCartridgeSpec, Mode};

        const KW_POOL: [&str; 9] = [
            "order", "group", "select", "where", "key", "table", "index", "limit", "values",
        ];
        let kw_table = KW_POOL[kw_table_idx];
        let kw_col = KW_POOL[kw_col_idx];

        let mut all_table_names = Vec::with_capacity(table_count);
        let mut schemas = Vec::with_capacity(table_count);

        for t_idx in 0..table_count {
            let tbl_name = if t_idx == 0 {
                if mixed_case {
                    let mut chars = kw_table.chars();
                    match chars.next() {
                        Some(c) => format!("{}{}", c.to_ascii_uppercase(), chars.as_str()),
                        None => kw_table.to_string(),
                    }
                } else {
                    kw_table.to_string()
                }
            } else if mixed_case && t_idx % 2 == 1 {
                format!("Acct_Table_{t_idx}")
            } else {
                format!("acct_table_{t_idx}")
            };
            all_table_names.push(tbl_name.clone());

            let mut cols = Vec::with_capacity(cols_per_table);
            cols.push(ColumnSchema::new("id", "INTEGER", true));
            cols.push(ColumnSchema::new(kw_col, "TEXT", false));
            for c_idx in 2..cols_per_table {
                let cname = if mixed_case && c_idx % 2 == 0 {
                    format!("Metric_{t_idx}_{c_idx}")
                } else {
                    format!("metric_{t_idx}_{c_idx}")
                };
                cols.push(ColumnSchema::new(cname, "INTEGER", false));
            }
            schemas.push(TableSchema::new(tbl_name, cols));
        }

        let cart = EpistemicSqlGrammarCartridge {
            all_table_names: &all_table_names,
            catalog_discovered,
            probed_schemas: &schemas,
            pending_steward_ask: false,
            steward_grant_approved: false,
            ambiguous_candidates: &[],
            read_only: false,
        };

        for mode in [Mode::Probe, Mode::Read, Mode::Write, Mode::Ask, Mode::Result, Mode::Abstain] {
            let spec = compile_epistemic_span_grammar(&cart, mode);
            let GrammarCartridgeSpec::LarkGrammar(lark) = spec else {
                prop_assert!(false, "expected LarkGrammar for {mode:?}");
                unreachable!()
            };
            if matches!(mode, Mode::Read | Mode::Write) {
                prop_assert!(lark.contains("SCHEMA_TABLE_IDENT"));
                prop_assert!(lark.contains("SCHEMA_COLUMN_IDENT"));
                if is_sqlite_reserved_keyword(kw_table) {
                    // Reserved SQLite keywords must appear inside quoted branches, never as bare rule literals
                    let bare_lit = format!("\"{kw_table}\"i");
                    prop_assert!(!lark.contains(&bare_lit));
                }
                if is_sqlite_reserved_keyword(kw_col) {
                    let bare_lit = format!("\"{kw_col}\"i");
                    prop_assert!(!lark.contains(&bare_lit));
                }
            }
        }

        let first_tbl = &schemas[0].table_name;
        let dq_tbl = format!("\"{}\"", first_tbl.replace('"', "\"\""));
        let dq_col = format!("\"{}\"", kw_col.replace('"', "\"\""));
        let valid_pragma = format!("PRAGMA table_info({dq_tbl});");
        prop_assert!(
            validate_completed_sql_against_schemas(&valid_pragma, &schemas).is_ok(),
            "quoted PRAGMA rejected: {valid_pragma}"
        );
        if is_sqlite_reserved_keyword(first_tbl) {
            let unquoted_pragma = format!("PRAGMA table_info({first_tbl});");
            prop_assert!(
                validate_completed_sql_against_schemas(&unquoted_pragma, &schemas).is_err(),
                "unquoted reserved keyword PRAGMA should be rejected: {unquoted_pragma}"
            );
        }

        let adv_select = format!(
            "SELECT id, {dq_col} FROM {dq_tbl} WHERE {dq_col} = 'SELECT * FROM bogus;</sql></to_user>' ORDER BY {dq_col} ASC LIMIT 3;"
        );
        prop_assert!(
            validate_completed_sql_against_schemas(&adv_select, &schemas).is_ok(),
            "valid quoted keyword SELECT rejected: {adv_select}"
        );
        let clause = inspect_top_level_sql_clauses(&adv_select, &schemas);
        prop_assert_eq!(clause.primary_table.as_deref(), Some(first_tbl.as_str()));
        prop_assert!(clause.has_where_keyword && clause.has_order_by_clause && clause.has_limit_keyword);
        prop_assert!(!clause.has_join_keyword && !clause.has_group_by_clause);
    }
}

#[test]
fn sql_keyword_and_special_identifier_grammars_align_with_sqlite_parser() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE \"order\" (
            \"index\" INTEGER PRIMARY KEY,
            \"select\" TEXT NOT NULL,
            \"where\" INTEGER NOT NULL,
            \"limit\" INTEGER NOT NULL,
            \"values\" TEXT NOT NULL,
            key TEXT NOT NULL,
            \"a\"\"b\" TEXT NOT NULL
        );
        CREATE TABLE \"group\" (
            \"index\" INTEGER PRIMARY KEY,
            \"table\" TEXT NOT NULL,
            \"order\" INTEGER NOT NULL
        );
        CREATE TABLE view (
            row INTEGER PRIMARY KEY,
            plan TEXT NOT NULL
        );
        INSERT INTO \"order\" VALUES (1, 'alpha', 10, 5, 'v1', 'k1', 'q1');
        INSERT INTO \"group\" VALUES (1, 'tbl_ref', 99);
        INSERT INTO view VALUES (1, 'fast');",
    )
    .unwrap();

    let schemas = inspect_all_table_schemas(&conn).unwrap();
    assert_eq!(schemas.len(), 3);

    // Reserved keyword queries succeed when quoted in BOTH SQLite and `validate_completed_sql_against_schemas`:
    for valid_sql in [
        "PRAGMA table_info(\"order\");",
        "PRAGMA table_info(`group`);",
        "PRAGMA table_info(view);",
        "SELECT \"select\", \"where\", key, \"a\"\"b\" FROM \"order\" JOIN \"group\" ON \"order\".\"index\" = \"group\".\"index\" WHERE \"limit\" >= 1 ORDER BY \"group\".\"order\" DESC LIMIT 5;",
        "SELECT row, plan FROM view WHERE row = 1;",
        "UPDATE \"order\" SET \"values\" = 'v2', key = 'k2' WHERE \"index\" = 1;",
    ] {
        assert!(
            validate_completed_sql_against_schemas(valid_sql, &schemas).is_ok(),
            "expected valid SQL to pass schema validation: {valid_sql}"
        );
        conn.prepare(valid_sql)
            .unwrap_or_else(|e| panic!("SQLite rejected {valid_sql}: {e}"));
    }

    // Unquoted reserved SQLite keywords fail in BOTH SQLite and `validate_completed_sql_against_schemas`:
    for invalid_sql in [
        "PRAGMA table_info(order);",
        "PRAGMA table_info(group);",
        "SELECT select FROM \"order\";",
        "SELECT * FROM order;",
    ] {
        assert!(
            validate_completed_sql_against_schemas(invalid_sql, &schemas).is_err(),
            "expected unquoted reserved keyword SQL to fail validation: {invalid_sql}"
        );
        assert!(
            conn.prepare(invalid_sql).is_err(),
            "expected SQLite to reject unquoted reserved keyword SQL: {invalid_sql}"
        );
    }
}

fn discover_llguidance_python() -> Option<(std::path::PathBuf, bool)> {
    if let Ok(explicit) = std::env::var("PORTIA_PYTHON") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return Some((std::path::PathBuf::from(trimmed), true));
        }
    }
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = Vec::new();
    if let Some(ws_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        candidates.push(ws_root.join(".venv/bin/python"));
    }
    for cand in candidates {
        if cand.is_file() {
            let ok = std::process::Command::new(&cand)
                .args(["-c", "import llguidance"])
                .output()
                .is_ok_and(|o| o.status.success());
            if ok {
                return Some((cand, true));
            }
        }
    }
    let path_py = std::path::PathBuf::from("python3");
    let ok = std::process::Command::new(&path_py)
        .args(["-c", "import llguidance"])
        .output()
        .is_ok_and(|o| o.status.success());
    if ok {
        return Some((path_py, false));
    }
    None
}

#[test]
fn llguidance_batch_fuzzer_validates_26_diverse_keyword_and_50_table_grammars() {
    use portia_core::relay::TypedTurnAction;
    use portia_core::{GrammarCartridgeSpec, Mode};
    use std::io::Write;
    use std::process::{Command, Stdio};

    let Some((python_bin, must_succeed_import)) = discover_llguidance_python() else {
        eprintln!("skipping real llguidance batch test: no python with `llguidance` found");
        return;
    };

    #[derive(Serialize)]
    struct BatchGrammarCase {
        name: String,
        lark: String,
        valid_inputs: Vec<String>,
        invalid_inputs: Vec<String>,
        max_ms_per_tok: f64,
    }

    let mut cases: Vec<BatchGrammarCase> = Vec::with_capacity(28);
    let keyword_names = [
        "order", "group", "select", "where", "key", "table", "index", "limit", "values",
    ];

    // 1..=9: Nine SQL-keyword schemas across Probe, Read, and Write (9 * 2 = 18 diverse schemas + modes)
    for (idx, kw) in keyword_names.iter().enumerate() {
        let other_kw = keyword_names[(idx + 3) % keyword_names.len()];
        let tbl_a = (*kw).to_string();
        let tbl_b = format!("ledger_{idx}");
        let all_tables = vec![tbl_a.clone(), tbl_b.clone()];
        let schemas = vec![
            TableSchema::new(
                tbl_a.clone(),
                vec![
                    ColumnSchema::new("id", "INTEGER", true),
                    ColumnSchema::new(*kw, "TEXT", false),
                    ColumnSchema::new(other_kw, "INTEGER", false),
                    ColumnSchema::new("key", "TEXT", false),
                    ColumnSchema::new("view", "TEXT", false),
                ],
            ),
            TableSchema::new(
                tbl_b.clone(),
                vec![
                    ColumnSchema::new("id", "INTEGER", true),
                    ColumnSchema::new("note_text", "TEXT", false),
                ],
            ),
        ];

        let probe_cart = EpistemicSqlGrammarCartridge {
            all_table_names: &all_tables,
            catalog_discovered: idx % 2 == 0,
            probed_schemas: &[],
            pending_steward_ask: false,
            steward_grant_approved: false,
            ambiguous_candidates: &[],
            read_only: false,
        };
        let GrammarCartridgeSpec::LarkGrammar(probe_lark) =
            compile_epistemic_span_grammar(&probe_cart, Mode::Probe)
        else {
            panic!("expected Probe LarkGrammar");
        };
        let mut probe_valid = vec![
            format!("PRAGMA table_info(\"{tbl_a}\");\n</sql>"),
            format!("PRAGMA table_info(`{tbl_a}`);\n</sql>"),
            format!("PRAGMA table_info({tbl_b});\n</sql>"),
        ];
        if !probe_cart.catalog_discovered {
            probe_valid.push("SELECT name FROM sqlite_master WHERE type = 'table';\n</sql>".into());
            probe_valid.push("SELECT NAME FROM SQLITE_MASTER WHERE TYPE = 'table';\n</sql>".into());
        }
        let mut probe_invalid = vec!["PRAGMA table_info(hallucinated_tbl);\n</sql>".into()];
        if !probe_cart.catalog_discovered {
            probe_invalid
                .push("SELECT name FROM sqlite_master WHERE type = 'TABLE';\n</sql>".into());
            probe_invalid
                .push("SELECT name FROM sqlite_master WHERE type = 'Table';\n</sql>".into());
        }
        if is_sqlite_reserved_keyword(&tbl_a) {
            probe_invalid.push(format!("PRAGMA table_info({tbl_a});\n</sql>"));
        }
        cases.push(BatchGrammarCase {
            name: format!("kw_probe_{idx}_{kw}"),
            lark: probe_lark,
            valid_inputs: probe_valid,
            invalid_inputs: probe_invalid,
            max_ms_per_tok: 5.0,
        });

        let sql_mode = if idx % 2 == 0 {
            Mode::Read
        } else {
            Mode::Write
        };
        let rw_cart = EpistemicSqlGrammarCartridge {
            all_table_names: &all_tables,
            catalog_discovered: true,
            probed_schemas: &schemas,
            pending_steward_ask: false,
            steward_grant_approved: false,
            ambiguous_candidates: &[],
            read_only: false,
        };
        let GrammarCartridgeSpec::LarkGrammar(rw_lark) =
            compile_epistemic_span_grammar(&rw_cart, sql_mode)
        else {
            panic!("expected Read/Write LarkGrammar");
        };
        let (rw_valid, rw_invalid) = if sql_mode == Mode::Read {
            let mut inv = vec![
                format!("SELECT bogus_col FROM \"{tbl_a}\";\n</sql>"),
                "SELECT id FROM bogus_table;\n</sql>".to_string(),
                format!("SELECT id FROM \"{tbl_a}\"; SELECT id FROM {tbl_b};\n</sql>"),
                format!("SELECT id FROM \"{tbl_a}\"; DELETE FROM {tbl_b};\n</sql>"),
            ];
            if is_sqlite_reserved_keyword(kw) {
                inv.push(format!("SELECT {kw} FROM \"{tbl_a}\";\n</sql>"));
                inv.push(format!("SELECT id FROM {tbl_a};\n</sql>"));
            }
            (
                vec![
                    format!(
                        "SELECT id, \"{kw}\", `{other_kw}`, key, \"key\", view FROM \"{tbl_a}\" WHERE \"{other_kw}\" >= 1 ORDER BY \"{kw}\" DESC LIMIT 5;\n</sql>"
                    ),
                    format!(
                        "SELECT \"{tbl_a}\".id, {tbl_b}.note_text FROM \"{tbl_a}\" JOIN {tbl_b} ON \"{tbl_a}\".id = {tbl_b}.id;\n</sql>"
                    ),
                ],
                inv,
            )
        } else {
            let mut inv = vec![
                format!("UPDATE \"{tbl_a}\" SET bogus_col = 1 WHERE id = 1;\n</sql>"),
                "DELETE FROM bogus_table WHERE id = 1;\n</sql>".to_string(),
                format!("DELETE FROM \"{tbl_a}\" WHERE id = 1; DELETE FROM {tbl_b} WHERE id = 2;\n</sql>"),
                format!("DELETE FROM \"{tbl_a}\" WHERE id = 1; SELECT id FROM {tbl_b};\n</sql>"),
            ];
            if is_sqlite_reserved_keyword(kw) {
                inv.push(format!(
                    "UPDATE {tbl_a} SET key = 'x' WHERE id = 1;\n</sql>"
                ));
            }
            (
                vec![
                    format!(
                        "UPDATE \"{tbl_a}\" SET \"{kw}\" = 'updated', key = 'ok' WHERE id = 1;\n</sql>"
                    ),
                    format!("UPDATE \"{tbl_a}\" SET key = 'literal; SELECT is data' WHERE id = 1;\n</sql>"),
                    format!(
                        "INSERT INTO `{tbl_a}` (id, `{kw}`, `{other_kw}`, key, view) VALUES (2, 'v', 7, 'k', 'vw');\n</sql>"
                    ),
                    format!("DELETE FROM {tbl_b} WHERE id = 2;\n</sql>"),
                ],
                inv,
            )
        };
        cases.push(BatchGrammarCase {
            name: format!("kw_{sql_mode:?}_{idx}_{kw}"),
            lark: rw_lark,
            valid_inputs: rw_valid,
            invalid_inputs: rw_invalid,
            max_ms_per_tok: 5.0,
        });
    }

    // Cases 19..=21: 50-table x 15-column wide enterprise schema across Probe, Read, and Write!
    let mut wide_names = Vec::with_capacity(50);
    let mut wide_schemas = Vec::with_capacity(50);
    for t in 0..50 {
        let tname = match t {
            0 => "order".to_string(),
            1 => "group".to_string(),
            2 => "Accounts_Ledger_2".to_string(),
            _ => format!("domain_table_{t:02}"),
        };
        wide_names.push(tname.clone());
        let mut cols = Vec::with_capacity(15);
        cols.push(ColumnSchema::new("record_id", "INTEGER", true));
        cols.push(ColumnSchema::new("order", "INTEGER", false));
        cols.push(ColumnSchema::new("select", "TEXT", false));
        cols.push(ColumnSchema::new("key", "TEXT", false));
        for c in 4..15 {
            cols.push(ColumnSchema::new(
                format!("field_t{t:02}_c{c:02}"),
                "INTEGER",
                false,
            ));
        }
        wide_schemas.push(TableSchema::new(tname, cols));
    }
    let wide_probe_cart = EpistemicSqlGrammarCartridge {
        all_table_names: &wide_names,
        catalog_discovered: true,
        probed_schemas: &wide_schemas[..20],
        pending_steward_ask: false,
        steward_grant_approved: false,
        ambiguous_candidates: &[],
        read_only: false,
    };
    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&wide_probe_cart, Mode::Probe)
    {
        cases.push(BatchGrammarCase {
            name: "wide_50_tables_probe".into(),
            lark,
            valid_inputs: vec![
                "PRAGMA table_info(domain_table_25);\n</sql>".into(),
                "PRAGMA table_info(\"domain_table_49\");\n</sql>".into(),
            ],
            invalid_inputs: vec![
                "PRAGMA table_info(domain_table_05);\n</sql>".into(),
                "PRAGMA table_info(nonexistent_table);\n</sql>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }

    let wide_full_cart = EpistemicSqlGrammarCartridge {
        all_table_names: &wide_names,
        catalog_discovered: true,
        probed_schemas: &wide_schemas,
        pending_steward_ask: false,
        steward_grant_approved: false,
        ambiguous_candidates: &[],
        read_only: false,
    };
    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&wide_full_cart, Mode::Read)
    {
        cases.push(BatchGrammarCase {
            name: "wide_50x15_read".into(),
            lark,
            valid_inputs: vec![
                "SELECT record_id, \"order\", `select`, key, field_t49_c14 FROM domain_table_49 WHERE field_t49_c10 >= 42 ORDER BY \"order\" DESC LIMIT 10;\n</sql>".into(),
                "SELECT Accounts_Ledger_2.record_id, \"group\".field_t01_c05 FROM accounts_ledger_2 JOIN \"group\" ON accounts_ledger_2.record_id = \"group\".record_id;\n</sql>".into(),
            ],
            invalid_inputs: vec![
                "SELECT order FROM domain_table_49;\n</sql>".into(),
                "SELECT a.record_id FROM domain_table_49 a;\n</sql>".into(),
                "SELECT field_t99_c99 FROM domain_table_49;\n</sql>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }
    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&wide_full_cart, Mode::Write)
    {
        cases.push(BatchGrammarCase {
            name: "wide_50x15_write".into(),
            lark,
            valid_inputs: vec![
                "UPDATE \"order\" SET \"select\" = 'approved', field_t00_c14 = 99 WHERE record_id = 1;\n</sql>".into(),
                "DELETE FROM domain_table_33 WHERE record_id = 7;\n</sql>".into(),
            ],
            invalid_inputs: vec![
                "UPDATE order SET key = 'bad' WHERE record_id = 1;\n</sql>".into(),
                "UPDATE domain_table_33 SET missing_col = 1 WHERE record_id = 7;\n</sql>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }

    // Cases 22..=25: Steward `%json` Ask, MultiSelect `%json` Ask, Conversational Ask, and Result/Abstain!
    let steward_cart = EpistemicSqlGrammarCartridge {
        pending_steward_ask: true,
        steward_grant_approved: false,
        ..wide_full_cart
    };
    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&steward_cart, Mode::Ask)
    {
        cases.push(BatchGrammarCase {
            name: "ask_steward_json".into(),
            lark,
            valid_inputs: vec![
                "{\"select\":\"Authorize update on ASK-gated account 102?\",\"options\":[\"ALLOW\",\"DENY\"]}\n</to_user>".into(),
            ],
            invalid_inputs: vec![
                "{\"select\":\"hi\",\"options\":[\"MAYBE\",\"DENY\"]}\n</to_user>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }

    let ambig_candidates = vec![
        "101 (MATTERHORN-7 · ALLOW)".to_string(),
        "102 (CHAMONIX-3 · ASK)".to_string(),
    ];
    let multi_cart = EpistemicSqlGrammarCartridge {
        ambiguous_candidates: &ambig_candidates,
        ..wide_full_cart
    };
    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&multi_cart, Mode::Ask)
    {
        cases.push(BatchGrammarCase {
            name: "ask_multiselect_json".into(),
            lark,
            valid_inputs: vec![
                "{\"multi\":\"Which account should be updated?\",\"options\":[\"101 (MATTERHORN-7 · ALLOW)\",\"102 (CHAMONIX-3 · ASK)\"]}\n</to_user>".into(),
            ],
            invalid_inputs: vec![
                "{\"multi\":\"Which?\",\"options\":[\"999 (BOGUS)\"]}\n</to_user>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }

    if let GrammarCartridgeSpec::LarkGrammar(lark) =
        compile_epistemic_span_grammar(&wide_full_cart, Mode::Ask)
    {
        cases.push(BatchGrammarCase {
            name: "ask_chitchat_prose".into(),
            lark,
            valid_inputs: vec!["Which account slug would you like to inspect?\n</to_user>".into()],
            invalid_inputs: vec!["inject <xml> tag\n</to_user>".into()],
            max_ms_per_tok: 5.0,
        });
    }

    for mode in [Mode::Result, Mode::Abstain] {
        let GrammarCartridgeSpec::LarkGrammar(lark) =
            compile_epistemic_span_grammar(&wide_full_cart, mode)
        else {
            panic!("expected terminal grammar for {mode}");
        };
        let valid_inputs: Vec<String> = vec![
            "0\n</to_user>".into(),
            "7\n</to_user>".into(),
            "The operation completed.\n</to_user>".into(),
            "No matching row was found.\nNo changes were made.\n</to_user>".into(),
            "[These brackets begin plain prose.]\n</to_user>".into(),
            "First line.\nSecond line.\nThird line.\n</to_user>".into(),
            "First\nSecond\nThird\nFourth\n</to_user>".into(),
            "First line.\n7\n</to_user>".into(),
            "  A visible answer.  \n</to_user>".into(),
            "éé\n答复\n</to_user>".into(),
            format!("{}\n</to_user>", "x".repeat(400)),
        ];
        for text in &valid_inputs {
            TypedTurnAction::parse_for_mode(mode, text.strip_suffix("\n</to_user>").unwrap())
                .unwrap();
        }
        cases.push(BatchGrammarCase {
            name: format!("{}_bounded_prose", mode.as_str()),
            lark,
            valid_inputs,
            invalid_inputs: vec![
                "bad </to_user> inline <tag>\n</to_user>".into(),
                "{\"summary\":\"A valid JSON object is not terminal prose.\"}\n</to_user>".into(),
                "  {\"reason\":\"No rows.\"}\n</to_user>".into(),
                "{\n</to_user>".into(),
                "  \nPlain prose after a blank line.\n</to_user>".into(),
                "First line.\n \t\nThird line.\n</to_user>".into(),
                "  \n</to_user>".into(),
                // Pilot case 06 reached the JSON parser with a literal newline
                // inside an ABSTAIN string because the prose branch accepted it.
                "{\"reason\":\"First line.\nSecond line.\"}\n</to_user>".into(),
                "  {\"reason\":\"First line.\nSecond line.\"}\n</to_user>".into(),
                " \t\n {malformed JSON}\n</to_user>".into(),
                "{\"value\":true,}\n</to_user>".into(),
                "{\"value\":true} trailing prose\n</to_user>".into(),
            ],
            max_ms_per_tok: 5.0,
        });
    }

    assert_eq!(cases.len(), 26, "expected exactly 26 diverse grammar cases");

    let py_script = r#"
import glob, json, os, sys, time
import llguidance

home = os.path.expanduser("~")
tok_candidates = sorted(
    glob.glob(os.path.join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/tokenizer.json"))
)
if tok_candidates:
    ll_tok = llguidance.LLTokenizer(tok_candidates[0])
else:
    ll_tok = llguidance.LLTokenizer("byte")

cases = json.load(sys.stdin)
for case in cases:
    name = case["name"]
    g = llguidance.grammar_from("lark", case["lark"])
    m0 = llguidance.LLMatcher(ll_tok, g)
    if m0.is_error():
        raise AssertionError(f"[{name}] grammar failed to compile: {m0.get_error()}")

    for text in case["valid_inputs"]:
        m = llguidance.LLMatcher(ll_tok, g)
        tids = ll_tok.tokenize_str(text)
        t0 = time.perf_counter()
        for tid in tids:
            _ = m.compute_bitmask()
            ok = m.consume_token(tid)
            if m.stop_reason() == "ParserTooComplex":
                raise AssertionError(f"[{name}] ParserTooComplex on valid {text!r}: {m.get_error()}")
            if not ok:
                raise AssertionError(
                    f"[{name}] rejected valid input {text!r} at token {ll_tok.dbg_tokens([tid])}: stop={m.stop_reason()} err={m.get_error()}"
                )
        ms_per_tok = (time.perf_counter() - t0) * 1000.0 / max(1, len(tids))
        accepted = m.is_accepting() or (m.is_stopped() and m.stop_reason() == "NoExtension")
        if not accepted or m.is_error():
            raise AssertionError(
                f"[{name}] valid input did not finish in accepting state: {text!r} stop={m.stop_reason()} err={m.get_error()}"
            )
        if ms_per_tok > case["max_ms_per_tok"]:
            raise AssertionError(
                f"[{name}] mask latency {ms_per_tok:.3f} ms/tok exceeded budget {case['max_ms_per_tok']} ms/tok"
            )

    for bad_text in case["invalid_inputs"]:
        m = llguidance.LLMatcher(ll_tok, g)
        tids = ll_tok.tokenize_str(bad_text)
        rejected = False
        for tid in tids:
            _ = m.compute_bitmask()
            ok = m.consume_token(tid)
            if m.stop_reason() == "ParserTooComplex":
                raise AssertionError(f"[{name}] ParserTooComplex on invalid {bad_text!r}: {m.get_error()}")
            if not ok:
                rejected = True
                break
        if not rejected:
            accepted = m.is_accepting() or (m.is_stopped() and m.stop_reason() == "NoExtension")
            if accepted and not m.is_error():
                raise AssertionError(f"[{name}] grammar erroneously ACCEPTED invalid input: {bad_text!r}")

print(f"VERIFIED_CASES={len(cases)}")
"#;

    let payload = serde_json::to_vec(&cases).unwrap();
    let mut child = Command::new(&python_bin)
        .args(["-c", py_script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {:?}: {e}", python_bin));
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(e) = stdin.write_all(&payload) {
            if e.kind() != std::io::ErrorKind::BrokenPipe {
                panic!("failed to write payload to {:?}: {e}", python_bin);
            }
        }
    }
    let output = child.wait_with_output().unwrap();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !must_succeed_import && stderr.contains("No module named 'llguidance'") {
            return;
        }
        panic!(
            "llguidance batch fuzzer failed on {:?} (status={}):\nSTDOUT:\n{}\nSTDERR:\n{}",
            python_bin, output.status, stdout, stderr
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("VERIFIED_CASES=26"),
        "unexpected llguidance output: {stdout}"
    );
}
