//! Whole-database snapshots, online in-memory cloning, and primary-key-aware relational diffing.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::Connection;

use super::authorizer::{configure_hardened_sqlite, list_sqlite_user_tables};
use super::types::{
    DatabaseSnapshot, EngineError, KeyedRow, RelationalDiff, RowModification, SqlRow, SqlValue,
    TableDiff,
};

/// Dump all user tables and their ordered rows from `conn` for shadow relational equivalence.
pub fn dump_all_tables(conn: &Connection) -> Result<DatabaseSnapshot, EngineError> {
    let tables = list_sqlite_user_tables(conn)?;

    let mut snapshot = BTreeMap::new();
    for table in tables {
        let escaped = table.replace('"', "\"\"");
        let sql = format!("SELECT * FROM \"{escaped}\";");
        let mut stmt = conn.prepare(&sql)?;
        let col_names: Vec<String> = stmt.column_names().into_iter().map(str::to_owned).collect();
        let mut rows_out: Vec<SqlRow> = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let mut map = BTreeMap::new();
            for (idx, col) in col_names.iter().enumerate() {
                let val = SqlValue::from_value_ref(row.get_ref(idx)?);
                map.insert(col.clone(), val);
            }
            rows_out.push(map);
        }
        // Sort rows deterministically by their cached serialized representation so row order never causes false diffs.
        rows_out.sort_by_cached_key(|row| serde_json::to_string(row).unwrap_or_default());
        snapshot.insert(table, rows_out);
    }
    Ok(snapshot)
}

/// Clone `src` into a fresh in-memory `Connection` using SQLite's C-level online `Backup` API.
pub fn clone_connection_to_memory(src: &Connection) -> Result<Connection, EngineError> {
    let mut dst = Connection::open_in_memory()?;
    configure_hardened_sqlite(&dst)?;
    {
        let backup = rusqlite::backup::Backup::new(src, &mut dst)?;
        backup.step(-1)?;
    }
    Ok(dst)
}

/// Compute the exact algebraic `RelationalDiff` ($\Delta R_{\text{actual}} \ominus \Delta R_{\text{expected}}$)
/// between `agent_conn` and `ref_conn` across all user tables.
pub fn diff_shadow_databases(
    agent_conn: &Connection,
    ref_conn: &Connection,
) -> Result<RelationalDiff, EngineError> {
    let agent_dump = dump_all_tables(agent_conn)?;
    let ref_dump = dump_all_tables(ref_conn)?;

    let all_tables: BTreeSet<String> = agent_dump.keys().chain(ref_dump.keys()).cloned().collect();

    let mut table_diffs = BTreeMap::new();
    for table in all_tables {
        let a_rows = agent_dump.get(&table).cloned().unwrap_or_default();
        let r_rows = ref_dump.get(&table).cloned().unwrap_or_default();
        if a_rows == r_rows {
            continue;
        }

        let mut pk_cols = detect_primary_key_columns(agent_conn, &table);
        if pk_cols.is_empty() {
            pk_cols = detect_primary_key_columns(ref_conn, &table);
        }

        let mut a_buckets: BTreeMap<String, Vec<KeyedRow>> = BTreeMap::new();
        for (i, row) in a_rows.into_iter().enumerate() {
            let (k_str, primary_key, _) = extract_row_primary_key(&row, &pk_cols, i);
            a_buckets
                .entry(k_str)
                .or_default()
                .push(KeyedRow { primary_key, row });
        }

        let mut r_buckets: BTreeMap<String, Vec<KeyedRow>> = BTreeMap::new();
        for (i, row) in r_rows.into_iter().enumerate() {
            let (k_str, primary_key, _) = extract_row_primary_key(&row, &pk_cols, i);
            r_buckets
                .entry(k_str)
                .or_default()
                .push(KeyedRow { primary_key, row });
        }

        let all_keys: BTreeSet<String> =
            a_buckets.keys().chain(r_buckets.keys()).cloned().collect();

        let mut tdiff = TableDiff::default();
        for k_str in all_keys {
            let mut a_list = a_buckets.remove(&k_str).unwrap_or_default();
            let mut r_list = r_buckets.remove(&k_str).unwrap_or_default();

            // Cancel identical rows within the same key bucket first (preserving row multiplicity):
            let mut a_idx = 0;
            while a_idx < a_list.len() {
                if let Some(r_pos) = r_list.iter().position(|r| r.row == a_list[a_idx].row) {
                    a_list.remove(a_idx);
                    r_list.remove(r_pos);
                } else {
                    a_idx += 1;
                }
            }

            let common_len = a_list.len().min(r_list.len());
            for idx in 0..common_len {
                tdiff.modified_rows.push(RowModification {
                    primary_key: a_list[idx].primary_key.clone(),
                    expected_row: r_list[idx].row.clone(),
                    actual_row: a_list[idx].row.clone(),
                });
            }
            for extra_a in a_list.into_iter().skip(common_len) {
                tdiff.inserted_rows.push(extra_a);
            }
            for extra_r in r_list.into_iter().skip(common_len) {
                tdiff.deleted_rows.push(extra_r);
            }
        }

        if !tdiff.is_empty() {
            table_diffs.insert(table, tdiff);
        }
    }

    let session_changeset_bytes = serde_json::to_vec(&table_diffs).unwrap_or_default();
    Ok(RelationalDiff {
        table_diffs,
        session_changeset_bytes,
    })
}

fn detect_primary_key_columns(conn: &Connection, table: &str) -> Vec<String> {
    let escaped = table.replace('"', "\"\"");
    let Ok(mut stmt) = conn.prepare(&format!("PRAGMA table_info(\"{escaped}\");")) else {
        return Vec::new();
    };
    let Ok(mut rows) = stmt.query([]) else {
        return Vec::new();
    };
    let mut pk_cols: Vec<(i64, String)> = Vec::new();
    while let Ok(Some(row)) = rows.next() {
        let Ok(name) = row.get::<_, String>(1) else {
            continue;
        };
        let pk: i64 = row.get(5).unwrap_or(0);
        if pk > 0 {
            pk_cols.push((pk, name));
        }
    }
    pk_cols.sort_by_key(|(pk_idx, _)| *pk_idx);
    pk_cols.into_iter().map(|(_, name)| name).collect()
}

/// Whole-database shadow relational equivalence check across all user tables.
pub fn verify_shadow_equivalence(
    agent_conn: &Connection,
    ref_conn: &Connection,
) -> Result<bool, EngineError> {
    Ok(diff_shadow_databases(agent_conn, ref_conn)?.is_equivalent())
}

fn extract_row_primary_key(
    row: &SqlRow,
    pk_cols: &[String],
    fallback_idx: usize,
) -> (String, SqlValue, Option<i64>) {
    if pk_cols.len() == 1 {
        if let Some(val) = row.get(&pk_cols[0]) {
            let k_str =
                serde_json::to_string(val).unwrap_or_else(|_| format!("idx:{fallback_idx}"));
            return (k_str, val.clone(), val.as_i64());
        }
    } else if pk_cols.len() > 1 {
        let pk_vals: Vec<SqlValue> = pk_cols
            .iter()
            .map(|c| row.get(c).cloned().unwrap_or(SqlValue::Null))
            .collect();
        let k_str =
            serde_json::to_string(&pk_vals).unwrap_or_else(|_| format!("idx:{fallback_idx}"));
        let rowid_opt = pk_vals.first().and_then(SqlValue::as_i64);
        return (k_str.clone(), SqlValue::Text(k_str), rowid_opt);
    }
    if let Some((_, first_val)) = row.iter().next() {
        let k_str =
            serde_json::to_string(first_val).unwrap_or_else(|_| format!("idx:{fallback_idx}"));
        return (k_str, first_val.clone(), first_val.as_i64());
    }
    (
        format!("idx:{fallback_idx}"),
        SqlValue::Integer(fallback_idx as i64),
        None,
    )
}
