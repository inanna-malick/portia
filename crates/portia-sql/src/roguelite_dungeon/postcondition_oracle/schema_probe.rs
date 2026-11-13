//! Live SQLite column-role, cell-sampling, and foreign-key join-plan discovery for reference SQL.

use portia_core::{Mode, TableSchema};
use rusqlite::Connection;

use crate::engine::EngineError;
use crate::grammar_trie::inspect_all_table_schemas;
use crate::relational_ast::RelationalSchemaGraph;

fn has_text_affinity(col_type: &str) -> bool {
    let up = col_type.to_ascii_uppercase();
    up.contains("TEXT")
        || up.contains("CHAR")
        || up.contains("CLOB")
        || up.contains("STR")
        || up.contains("VARCHAR")
}

fn has_numeric_affinity(col_type: &str) -> bool {
    let up = col_type.to_ascii_uppercase();
    up.contains("INT")
        || up.contains("REAL")
        || up.contains("NUM")
        || up.contains("FLOA")
        || up.contains("DOUB")
        || up.contains("DEC")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredTableColumnRoles {
    pub(super) table_name: String,
    pub(super) pk_col: String,
    pub(super) text_col: String,
    pub(super) num_col: String,
    pub(super) pk_is_numeric: bool,
}

pub(super) fn discover_table_column_roles(raw_sc: &TableSchema) -> DiscoveredTableColumnRoles {
    let sc = raw_sc.without_quarantined_decoys();
    let pk_col = sc
        .primary_key_column()
        .or_else(|| sc.columns.first().map(|c| c.name.as_str()))
        .unwrap_or("rowid")
        .to_string();
    let pk_is_numeric = sc
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(&pk_col))
        .map(|c| has_numeric_affinity(&c.col_type) || c.col_type.trim().is_empty())
        .unwrap_or(true);
    let text_col = sc
        .columns
        .iter()
        .find(|c| {
            !c.is_primary_key
                && !c.name.eq_ignore_ascii_case(&pk_col)
                && has_text_affinity(&c.col_type)
        })
        .or_else(|| sc.columns.iter().find(|c| has_text_affinity(&c.col_type)))
        .map(|c| c.name.as_str())
        .or_else(|| sc.columns.get(1).map(|c| c.name.as_str()))
        .unwrap_or(pk_col.as_str())
        .to_string();
    let num_col = sc
        .columns
        .iter()
        .find(|c| {
            !c.is_primary_key
                && !c.name.eq_ignore_ascii_case(&pk_col)
                && has_numeric_affinity(&c.col_type)
        })
        .map(|c| c.name.as_str())
        .unwrap_or(pk_col.as_str())
        .to_string();
    DiscoveredTableColumnRoles {
        table_name: sc.table_name,
        pk_col,
        text_col,
        num_col,
        pk_is_numeric,
    }
}

pub(super) fn inspect_ordered_schemas_and_graph(
    conn: &Connection,
) -> Result<(Vec<TableSchema>, RelationalSchemaGraph), EngineError> {
    let mut tbl_stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid ASC;",
    )?;
    let rowid_order: Vec<String> = tbl_stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut schemas: Vec<TableSchema> = inspect_all_table_schemas(conn)?
        .into_iter()
        .map(|sc| sc.without_quarantined_decoys())
        .collect();
    schemas.sort_by_key(|sc| {
        rowid_order
            .iter()
            .position(|name| name.eq_ignore_ascii_case(&sc.table_name))
            .unwrap_or(usize::MAX)
    });
    let graph = RelationalSchemaGraph::inspect_from_connection(conn)?.without_quarantined_decoys();
    if schemas.is_empty() {
        return Err(EngineError::ExecutionError {
            mode: Mode::Read,
            detail:
                "cannot synthesize reference SQL on an empty SQLite schema with zero user tables"
                    .to_string(),
        });
    }
    Ok((schemas, graph))
}

#[derive(Debug, Clone)]
pub(super) struct SampledSqlCell {
    pub(super) sql_literal: String,
    pub(super) raw_text: String,
    pub(super) int_val: Option<i64>,
}

pub(super) fn sample_sql_cell(conn: &Connection, query: &str) -> Option<SampledSqlCell> {
    let mut stmt = conn.prepare(query).ok()?;
    let mut rows = stmt.query([]).ok()?;
    let row = rows.next().ok()??;
    let val_ref = row.get_ref(0).ok()?;
    match val_ref {
        rusqlite::types::ValueRef::Integer(i) => Some(SampledSqlCell {
            sql_literal: i.to_string(),
            raw_text: i.to_string(),
            int_val: Some(i),
        }),
        rusqlite::types::ValueRef::Real(f) => Some(SampledSqlCell {
            sql_literal: f.to_string(),
            raw_text: f.to_string(),
            int_val: Some(f as i64),
        }),
        rusqlite::types::ValueRef::Text(t) => {
            let s = String::from_utf8_lossy(t).into_owned();
            let escaped = s.replace('\'', "''");
            let int_val = s.trim().parse::<i64>().ok();
            Some(SampledSqlCell {
                sql_literal: format!("'{escaped}'"),
                raw_text: s,
                int_val,
            })
        }
        rusqlite::types::ValueRef::Null | rusqlite::types::ValueRef::Blob(_) => None,
    }
}

fn pick_child_projection_column(child_sc: &TableSchema, from_col: &str) -> String {
    let clean = child_sc.without_quarantined_decoys();
    let roles = discover_table_column_roles(&clean);
    clean
        .columns
        .iter()
        .find(|c| {
            !c.is_primary_key
                && !c.name.eq_ignore_ascii_case(&roles.pk_col)
                && !c.name.eq_ignore_ascii_case(from_col)
                && has_text_affinity(&c.col_type)
        })
        .or_else(|| {
            clean.columns.iter().find(|c| {
                !c.is_primary_key
                    && !c.name.eq_ignore_ascii_case(&roles.pk_col)
                    && !c.name.eq_ignore_ascii_case(from_col)
            })
        })
        .map(|c| c.name.clone())
        .unwrap_or(roles.text_col)
}

#[derive(Debug, Clone)]
pub(super) enum DiscoveredJoinPlan {
    CrossTable {
        parent_table: String,
        parent_pk: String,
        parent_text: String,
        to_col: String,
        child_table: String,
        from_col: String,
        child_col: String,
        sample_parent_pk_lit: String,
        sample_child_col_lit: String,
    },
    SelfJoin {
        root_table: String,
        pk_col: String,
        text_col: String,
        num_col: String,
        sample_pk_lit: String,
    },
}

fn try_build_cross_table_join_plan(
    conn: &Connection,
    parent_sc: &TableSchema,
    to_col: &str,
    child_sc: &TableSchema,
    from_col: &str,
) -> Option<DiscoveredJoinPlan> {
    let parent_roles = discover_table_column_roles(parent_sc);
    let child_col = pick_child_projection_column(child_sc, from_col);
    let parent_table = &parent_roles.table_name;
    let parent_pk = &parent_roles.pk_col;
    let child_table = &child_sc.table_name;

    let join_sample_sql = format!(
        "SELECT {child_table}.{child_col} FROM {parent_table} JOIN {child_table} ON {parent_table}.{to_col} = {child_table}.{from_col} ORDER BY {parent_table}.{parent_pk} ASC LIMIT 1;"
    );
    let sample_child = sample_sql_cell(conn, &join_sample_sql)?;
    let parent_pk_sql =
        format!("SELECT {parent_pk} FROM {parent_table} ORDER BY {parent_pk} ASC LIMIT 1;");
    let sample_parent_pk = sample_sql_cell(conn, &parent_pk_sql)?;

    Some(DiscoveredJoinPlan::CrossTable {
        parent_table: parent_table.clone(),
        parent_pk: parent_pk.clone(),
        parent_text: parent_roles.text_col,
        to_col: to_col.to_string(),
        child_table: child_table.clone(),
        from_col: from_col.to_string(),
        child_col,
        sample_parent_pk_lit: sample_parent_pk.sql_literal,
        sample_child_col_lit: sample_child.sql_literal,
    })
}

pub(super) fn discover_join_plan(
    conn: &Connection,
    schemas: &[TableSchema],
    graph: &RelationalSchemaGraph,
    root_roles: &DiscoveredTableColumnRoles,
    sample_pk_lit: &str,
) -> DiscoveredJoinPlan {
    let table_index = |name: &str| -> usize {
        schemas
            .iter()
            .position(|s| s.table_name.eq_ignore_ascii_case(name))
            .unwrap_or(usize::MAX)
    };

    let mut fk_edges = graph.foreign_keys.clone();
    fk_edges.sort_by_key(|edge| {
        let root_priority = if edge
            .parent_table
            .eq_ignore_ascii_case(&root_roles.table_name)
        {
            0usize
        } else if edge
            .child_table
            .eq_ignore_ascii_case(&root_roles.table_name)
        {
            1usize
        } else {
            2usize
        };
        (
            root_priority,
            table_index(&edge.parent_table),
            table_index(&edge.child_table),
        )
    });

    for edge in &fk_edges {
        if let (Some(parent_sc), Some(child_sc)) = (
            schemas
                .iter()
                .find(|s| s.table_name.eq_ignore_ascii_case(&edge.parent_table)),
            schemas
                .iter()
                .find(|s| s.table_name.eq_ignore_ascii_case(&edge.child_table)),
        ) {
            if let Some(plan) = try_build_cross_table_join_plan(
                conn,
                parent_sc,
                &edge.parent_col,
                child_sc,
                &edge.child_col,
            ) {
                return plan;
            }
        }
    }

    for t1 in schemas {
        for t2 in schemas {
            if t1.table_name.eq_ignore_ascii_case(&t2.table_name) {
                continue;
            }
            let mut shared_cols: Vec<(&str, bool)> = Vec::new();
            for c1 in &t1.columns {
                if let Some(c2) = t2
                    .columns
                    .iter()
                    .find(|c2| c2.name.eq_ignore_ascii_case(&c1.name))
                {
                    let is_pk_in_either = c1.is_primary_key || c2.is_primary_key;
                    shared_cols.push((c1.name.as_str(), is_pk_in_either));
                }
            }
            shared_cols.sort_by_key(|(_, is_pk_in_either)| !*is_pk_in_either);
            for (shared_col, _) in shared_cols {
                if let Some(plan) =
                    try_build_cross_table_join_plan(conn, t1, shared_col, t2, shared_col)
                {
                    return plan;
                }
            }
        }
    }

    DiscoveredJoinPlan::SelfJoin {
        root_table: root_roles.table_name.clone(),
        pk_col: root_roles.pk_col.clone(),
        text_col: root_roles.text_col.clone(),
        num_col: root_roles.num_col.clone(),
        sample_pk_lit: sample_pk_lit.to_string(),
    }
}
