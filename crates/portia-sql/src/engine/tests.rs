use std::collections::BTreeSet;

use rusqlite::Connection;

use super::*;

#[test]
fn hardened_sqlite_authorizer_whitelist_and_savepoint_rollback_work_on_raw_connection() {
    let conn = Connection::open_in_memory().unwrap();
    configure_hardened_sqlite(&conn).unwrap();
    conn.execute_batch(
        "CREATE TABLE alpine_levain_campaigns (
            campaign_id INTEGER PRIMARY KEY,
            station_code TEXT NOT NULL,
            poster_altitude_m INTEGER NOT NULL
        );
        INSERT INTO alpine_levain_campaigns VALUES
            (101, 'CHAMONIX-04', 1800),
            (102, 'ZERMATT-09', 2100),
            (103, 'GRINDEL-02', 1200);",
    )
    .unwrap();

    assert_eq!(
        list_sqlite_user_tables(&conn).unwrap(),
        vec!["alpine_levain_campaigns".to_string()]
    );

    // Introspection pragma on the whitelist succeeds and extracts table provenance:
    let pragma_tables = collect_read_tables_via_sqlite_authorizer(
        &conn,
        "PRAGMA table_info(alpine_levain_campaigns);",
    )
    .unwrap();
    assert_eq!(
        pragma_tables,
        BTreeSet::from(["alpine_levain_campaigns".to_string()])
    );

    // Mutating / non-whitelisted pragma (`PRAGMA user_version=100`) and DML are rejected by the read authorizer:
    assert!(collect_read_tables_via_sqlite_authorizer(&conn, "PRAGMA user_version=100;").is_err());
    assert!(collect_read_tables_via_sqlite_authorizer(
        &conn,
        "UPDATE alpine_levain_campaigns SET poster_altitude_m = 9999;"
    )
    .is_err());

    // SAVEPOINT rollback restores all rows cleanly:
    conn.execute_batch("SAVEPOINT sp_write_guard;").unwrap();
    let affected = conn
        .execute(
            "UPDATE alpine_levain_campaigns SET poster_altitude_m = 9999;",
            [],
        )
        .unwrap();
    assert_eq!(affected, 3);
    rollback_and_release_savepoint(&conn, "sp_write_guard").unwrap();

    let alt_101: i64 = conn
        .query_row(
            "SELECT poster_altitude_m FROM alpine_levain_campaigns WHERE campaign_id = 101;",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(alt_101, 1800);
}

#[test]
fn composite_primary_keys_and_row_multiplicity_are_preserved_in_shadow_diff() {
    let conn = Connection::open_in_memory().unwrap();
    configure_hardened_sqlite(&conn).unwrap();
    conn.execute_batch(
        "CREATE TABLE composite_items (
            a INTEGER NOT NULL,
            b INTEGER NOT NULL,
            val TEXT NOT NULL,
            PRIMARY KEY (a, b)
        );
        INSERT INTO composite_items VALUES (1, 1, 'x'), (1, 2, 'y');",
    )
    .unwrap();

    let ref_conn = clone_connection_to_memory(&conn).unwrap();
    assert!(verify_shadow_equivalence(&conn, &ref_conn).unwrap());

    conn.execute(
        "UPDATE composite_items SET val = 'z' WHERE a = 1 AND b = 1;",
        [],
    )
    .unwrap();

    let diff = diff_shadow_databases(&conn, &ref_conn).unwrap();
    assert!(!diff.is_equivalent());
    let tdiff = diff.table_diffs.get("composite_items").unwrap();
    assert_eq!(tdiff.modified_rows.len(), 1);
    assert_eq!(tdiff.inserted_rows.len(), 0);
    assert_eq!(tdiff.deleted_rows.len(), 0);
    assert_eq!(
        tdiff.modified_rows[0].expected_row.get("val"),
        Some(&SqlValue::Text("x".to_string()))
    );
    assert_eq!(
        tdiff.modified_rows[0].actual_row.get("val"),
        Some(&SqlValue::Text("z".to_string()))
    );
}
