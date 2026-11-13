//! SQLite `PRAGMA table_info` schema introspection and opt-in decoy-column classifier re-export.

use rusqlite::Connection;

use crate::engine::EngineError;
use portia_core::{ColumnSchema, TableSchema};

/// Return `true` if `name` matches an opt-in synthetic benchmark decoy column (`legacy_*`, `old_*`, `archive_*`, etc.).
#[inline]
pub fn is_quarantined_grammar_trie_decoy_column(name: &str) -> bool {
    portia_core::domain::is_quarantined_decoy_column_name(name)
}

/// Inspect a single table's `TableSchema` from a `rusqlite::Connection` using `PRAGMA table_info`.
pub fn inspect_table_schema(
    conn: &Connection,
    table_name: &str,
) -> Result<TableSchema, EngineError> {
    let escaped = table_name.replace('"', "\"\"");
    let pragma_sql = format!("PRAGMA table_info(\"{escaped}\");");
    let mut stmt = conn.prepare(&pragma_sql)?;
    let cols = stmt
        .query_map([], |row| {
            let cid: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let col_type: String = row.get(2)?;
            let notnull = row.get::<_, i64>(3).unwrap_or(0) != 0;
            let is_primary_key = row.get::<_, i64>(5).unwrap_or(0) > 0;
            Ok(ColumnSchema::with_sqlite_metadata(
                cid,
                name,
                col_type,
                notnull,
                is_primary_key,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TableSchema::new(table_name, cols))
}

/// Inspect all non-system tables in `conn` using `sqlite_master` and `PRAGMA table_info`.
pub fn inspect_all_table_schemas(conn: &Connection) -> Result<Vec<TableSchema>, EngineError> {
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
    )?;
    let table_names: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut schemas = Vec::with_capacity(table_names.len());
    for name in table_names {
        schemas.push(inspect_table_schema(conn, &name)?);
    }
    Ok(schemas)
}
