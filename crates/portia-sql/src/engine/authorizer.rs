//! Hardened SQLite connection setup, `SAVEPOINT` rollback, and C-level bytecode `authorizer` provenance.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use rusqlite::config::DbConfig;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::Connection;

use super::types::EngineError;

/// Hardens a `rusqlite::Connection` with `SQLITE_DBCONFIG_DQS_DML = false`,
/// `SQLITE_DBCONFIG_DEFENSIVE = true`, and `PRAGMA foreign_keys = ON`.
pub fn configure_hardened_sqlite(conn: &Connection) -> Result<(), EngineError> {
    let _ = conn.set_db_config(DbConfig::SQLITE_DBCONFIG_DQS_DML, false);
    let _ = conn.set_db_config(DbConfig::SQLITE_DBCONFIG_DEFENSIVE, true);
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(())
}

/// List all non-system user tables in `conn` ordered by table name.
pub fn list_sqlite_user_tables(conn: &Connection) -> Result<Vec<String>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
    )?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Roll back and release a named SQLite `SAVEPOINT` (`ROLLBACK TO SAVEPOINT <sp>; RELEASE SAVEPOINT <sp>;`).
pub fn rollback_and_release_savepoint(conn: &Connection, sp_name: &str) -> Result<(), String> {
    let sql = format!("ROLLBACK TO SAVEPOINT {sp_name}; RELEASE SAVEPOINT {sp_name};");
    conn.execute_batch(&sql)
        .map_err(|e| format!("failed to roll back SAVEPOINT {sp_name}: {e}"))
}

/// Return `true` iff `p_low` (lowercase pragma name) is on the explicit read-only schema
/// introspection pragma whitelist.
fn is_read_only_introspection_pragma(p_low: &str) -> bool {
    matches!(
        p_low,
        "table_info"
            | "table_xinfo"
            | "index_list"
            | "index_info"
            | "index_xinfo"
            | "foreign_key_list"
            | "database_list"
            | "collation_list"
            | "table_list"
            | "function_list"
            | "module_list"
            | "pragma_list"
            | "compile_options"
            | "integrity_check"
            | "quick_check"
            | "foreign_key_check"
    )
}

/// Extract every user table read by `sql` using SQLite's bytecode `authorizer` hook during
/// `conn.prepare(sql)`, capturing base tables across `JOIN`s, CTEs, subqueries, and
/// `UNION` / `INTERSECT` / `EXCEPT` set operations.
pub fn collect_read_tables_via_sqlite_authorizer(
    conn: &Connection,
    sql: &str,
) -> Result<BTreeSet<String>, String> {
    let read_tables: Arc<Mutex<BTreeSet<String>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let tables_hook = Arc::clone(&read_tables);

    conn.authorizer(Some(move |ctx: AuthContext<'_>| -> Authorization {
        match ctx.action {
            AuthAction::Read { table_name, .. } => {
                let lower = table_name.to_lowercase();
                if !lower.starts_with("sqlite_") {
                    if let Ok(mut guard) = tables_hook.lock() {
                        guard.insert(lower);
                    }
                }
                Authorization::Allow
            }
            AuthAction::Select | AuthAction::Function { .. } | AuthAction::Recursive => {
                Authorization::Allow
            }
            AuthAction::Pragma {
                pragma_name,
                pragma_value,
            } => {
                let p_low = pragma_name.to_lowercase();
                if p_low == "table_info" || p_low == "table_xinfo" {
                    if let Some(val) = pragma_value {
                        let cleaned = val.trim().trim_matches(['\'', '"', '`']).to_lowercase();
                        if !cleaned.is_empty() && !cleaned.starts_with("sqlite_") {
                            if let Ok(mut guard) = tables_hook.lock() {
                                guard.insert(cleaned);
                            }
                        }
                    }
                    Authorization::Allow
                } else if is_read_only_introspection_pragma(&p_low) {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            _ => Authorization::Deny,
        }
    }));

    let prep_res = conn.prepare(sql).map(|_| ());
    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
    prep_res.map_err(|e| e.to_string())?;

    let out = read_tables.lock().map(|g| g.clone()).unwrap_or_default();
    Ok(out)
}
