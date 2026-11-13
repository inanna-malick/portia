//! Shared column-interleaving, DDL synthesis, authorizer query tracing, and Fog-of-War pagination
//! helpers across SQL dungeon presets and unified scenarios.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::Connection;

use crate::engine::{
    configure_hardened_sqlite, ColumnRef, EngineError, FogOfWarPageMetadata, SqlRow, SqlValue,
};

use super::fk_graph::FkEdgeAction;

/// Deterministic row-count cliff points for haystack scaling sweeps (`3 -> 1,000` rows).
pub const HAYSTACK_CLIFF_POINTS: [usize; 9] = [3, 5, 10, 25, 50, 100, 250, 500, 1000];

/// Canonical deceptive column stems shared across [`super::WideTableDungeonSpec`] and
/// [`crate::roguelite_dungeon::UnifiedSqlScenarioSpec`].
pub const DECEPTIVE_COLUMN_STEMS: [&str; 10] = [
    "legacy_balance_credits_deprecated",
    "archive_balance_credits_v1",
    "old_policy_decision_deprecated",
    "shadow_lock_state_legacy",
    "deprecated_entity_slug_v0",
    "legacy_credit_quota_deprecated",
    "archive_policy_override_v1",
    "stale_balance_mirror_deprecated",
    "legacy_steward_handle_v0",
    "deprecated_hold_timer_sec",
];

/// Shared helper to construct `(canonical_columns, deceptive_decoy_columns, noise_telemetry_columns, all_columns_in_order)`
/// without duplicating column-interleaving logic across presets and unified scenarios.
pub(in crate::roguelite_dungeon) fn build_shared_column_layout(
    pk_col: &str,
    total_columns: usize,
    requested_deceptive: usize,
    policy_guard_columns: &[String],
) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let canonical_columns = vec![
        pk_col.to_string(),
        "entity_slug".to_string(),
        "balance_credits".to_string(),
        "policy_decision".to_string(),
        "lock_state".to_string(),
        "steward_handle".to_string(),
    ];

    let min_required = canonical_columns.len() + policy_guard_columns.len();
    let total_cols = total_columns.clamp(min_required, 100);
    let remaining = total_cols.saturating_sub(min_required);
    let deceptive_count = requested_deceptive.min(remaining).min(25);
    let noise_count = remaining.saturating_sub(deceptive_count);

    let mut deceptive_decoy_columns = Vec::with_capacity(deceptive_count);
    for i in 0..deceptive_count {
        let base = DECEPTIVE_COLUMN_STEMS[i % DECEPTIVE_COLUMN_STEMS.len()];
        let col_name = if i < DECEPTIVE_COLUMN_STEMS.len() {
            base.to_string()
        } else {
            format!("{}_{:02}", base, i)
        };
        deceptive_decoy_columns.push(col_name);
    }

    let mut noise_telemetry_columns = Vec::with_capacity(noise_count);
    for i in 0..noise_count {
        noise_telemetry_columns.push(format!("telemetry_sensor_col_{:03}", i + 1));
    }

    let mut all_columns_in_order = Vec::with_capacity(total_cols);
    all_columns_in_order.push(pk_col.to_string());
    all_columns_in_order.push("entity_slug".to_string());
    for d_col in &deceptive_decoy_columns {
        all_columns_in_order.push(d_col.clone());
    }
    all_columns_in_order.push("balance_credits".to_string());
    all_columns_in_order.push("policy_decision".to_string());
    all_columns_in_order.push("lock_state".to_string());
    all_columns_in_order.push("steward_handle".to_string());
    for g_col in policy_guard_columns {
        all_columns_in_order.push(g_col.clone());
    }
    for n_col in &noise_telemetry_columns {
        all_columns_in_order.push(n_col.clone());
    }

    (
        canonical_columns,
        deceptive_decoy_columns,
        noise_telemetry_columns,
        all_columns_in_order,
    )
}

/// Shared helper to format a `CREATE TABLE` DDL statement for any column layout.
pub(super) fn build_shared_create_table_ddl(
    table_name: &str,
    pk_col: &str,
    all_columns_in_order: &[String],
) -> String {
    let col_defs: Vec<String> = all_columns_in_order
        .iter()
        .map(|c| {
            if c == pk_col {
                format!("{} INTEGER PRIMARY KEY", pk_col)
            } else if c == "balance_credits"
                || c.starts_with("policy_guard_c")
                || c.contains("balance")
                || c.contains("quota")
                || c.contains("timer")
            {
                format!("{} INTEGER NOT NULL", c)
            } else {
                format!("{} TEXT NOT NULL", c)
            }
        })
        .collect();
    format!("CREATE TABLE {} ({});", table_name, col_defs.join(", "))
}

/// Shared helper to append child FK chain tables ($T_2 \dots T_d$) and distractor sibling tables into `setup_sql`.
#[allow(clippy::too_many_arguments)]
pub(in crate::roguelite_dungeon) fn append_shared_fk_chain_and_distractors_ddl(
    setup_sql: &mut Vec<String>,
    root_table: &str,
    root_pk_col: &str,
    root_blocked_pk: i64,
    fk_depth: usize,
    child_table_prefix: &str,
    child_table_suffix: &str,
    child_pk_step: i64,
    leaf_action: FkEdgeAction,
    distractor_count: usize,
) -> (Vec<String>, Vec<String>) {
    let mut chain_tables = Vec::with_capacity(fk_depth);
    chain_tables.push(root_table.to_string());

    for k in 1..fk_depth {
        let parent_tbl = chain_tables[k - 1].clone();
        let child_tbl = format!("{}_{:02}_{}", child_table_prefix, k + 1, child_table_suffix);
        let parent_pk_col = if k == 1 {
            root_pk_col.to_string()
        } else {
            format!("stage_{:02}_id", k)
        };
        let child_pk_col = format!("stage_{:02}_id", k + 1);
        let edge_sql = if k + 1 == fk_depth {
            leaf_action.as_sql_clause()
        } else {
            FkEdgeAction::Restrict.as_sql_clause()
        };

        setup_sql.push(format!(
            "CREATE TABLE {} (
                {} INTEGER PRIMARY KEY,
                parent_ref_id INTEGER NOT NULL,
                dependency_status TEXT NOT NULL,
                FOREIGN KEY (parent_ref_id) REFERENCES {}({}) {}
            );",
            child_tbl, child_pk_col, parent_tbl, parent_pk_col, edge_sql
        ));

        let parent_val = if k == 1 {
            root_blocked_pk
        } else {
            (k as i64) * child_pk_step + 1
        };
        let child_val = ((k + 1) as i64) * child_pk_step + 1;
        setup_sql.push(format!(
            "INSERT INTO {} VALUES ({}, {}, 'ACTIVE_HOLD');",
            child_tbl, child_val, parent_val
        ));
        chain_tables.push(child_tbl);
    }

    let mut distractor_tables = Vec::with_capacity(distractor_count);
    for i in 0..distractor_count {
        let d_tbl = format!("distractor_mirror_{:02}_archive", i + 1);
        setup_sql.push(format!(
            "CREATE TABLE {} (
                mirror_id INTEGER PRIMARY KEY,
                entity_slug TEXT NOT NULL,
                shadow_metric INTEGER NOT NULL
            );",
            d_tbl
        ));
        setup_sql.push(format!(
            "INSERT INTO {} VALUES ({}, 'distractor_slug_{:02}', {});",
            d_tbl,
            9000 + i as i64,
            i + 1,
            (i as i64 + 1) * 42
        ));
        distractor_tables.push(d_tbl);
    }

    (chain_tables, distractor_tables)
}

pub(in crate::roguelite_dungeon) fn open_seeded_in_memory_connection(
    setup_sql: &[String],
) -> Result<Connection, EngineError> {
    let conn = Connection::open_in_memory()?;
    configure_hardened_sqlite(&conn)?;
    for stmt in setup_sql {
        conn.execute_batch(stmt)?;
    }
    Ok(conn)
}

pub(super) fn query_rows_with_authorizer(
    conn: &Connection,
    sql: &str,
) -> Result<(Vec<SqlRow>, BTreeSet<ColumnRef>), EngineError> {
    let cols_read: Arc<Mutex<BTreeSet<ColumnRef>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let cols_hook = Arc::clone(&cols_read);

    conn.authorizer(Some(move |ctx: AuthContext<'_>| -> Authorization {
        match ctx.action {
            AuthAction::Select | AuthAction::Function { .. } | AuthAction::Recursive => {
                Authorization::Allow
            }
            AuthAction::Read {
                table_name,
                column_name,
            } => {
                let t_low = table_name.to_lowercase();
                let c_low = column_name.to_lowercase();
                if !t_low.starts_with("sqlite_") {
                    if let Ok(mut guard) = cols_hook.lock() {
                        guard.insert(ColumnRef::new(t_low, c_low));
                    }
                }
                Authorization::Allow
            }
            _ => Authorization::Deny,
        }
    }));

    let run_res = (|| -> Result<Vec<SqlRow>, EngineError> {
        let mut stmt = conn.prepare(sql)?;
        let raw_col_names: Vec<String> =
            stmt.column_names().into_iter().map(str::to_owned).collect();
        let mut col_names: Vec<String> = Vec::with_capacity(raw_col_names.len());
        let mut name_counts: BTreeMap<String, usize> = BTreeMap::new();
        for col in raw_col_names {
            let count = name_counts.entry(col.clone()).or_insert(0);
            *count += 1;
            if *count == 1 && !col_names.contains(&col) {
                col_names.push(col);
            } else {
                let mut dup_idx = *count;
                let mut candidate = format!("{col}#{dup_idx}");
                while col_names.contains(&candidate) {
                    dup_idx += 1;
                    candidate = format!("{col}#{dup_idx}");
                }
                col_names.push(candidate);
            }
        }
        let mut rows_out = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let mut map = BTreeMap::new();
            for (idx, col_key) in col_names.iter().enumerate() {
                let val = SqlValue::from_value_ref(row.get_ref(idx)?);
                map.insert(col_key.clone(), val);
            }
            rows_out.push(map);
        }
        Ok(rows_out)
    })();

    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
    let rows = run_res?;
    let trace_cols = cols_read.lock().map(|g| g.clone()).unwrap_or_default();
    Ok((rows, trace_cols))
}

pub(super) fn build_fog_page_metadata(
    rows: &mut Vec<SqlRow>,
    max_page_rows: usize,
) -> FogOfWarPageMetadata {
    let total_matching_rows = rows.len();
    let has_more = total_matching_rows > max_page_rows;
    if has_more {
        rows.truncate(max_page_rows);
    }
    let returned_rows = rows.len();
    let next_offset = if has_more { Some(returned_rows) } else { None };
    let hint = if has_more {
        Some(format!(
            "Use WHERE predicate or LIMIT {} OFFSET {}",
            max_page_rows, returned_rows
        ))
    } else {
        None
    };
    FogOfWarPageMetadata {
        total_matching_rows,
        returned_rows,
        has_more,
        next_offset,
        hint,
    }
}
