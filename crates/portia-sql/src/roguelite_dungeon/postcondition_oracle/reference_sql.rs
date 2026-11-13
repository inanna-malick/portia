//! Canonical reference SQL synthesis and candidate result-set comparison.

use portia_nursery::nursery_scenario::{SqlComparisonOperator, SqlQueryShape};
use rusqlite::Connection;

use crate::engine::{EngineError, SqlRow, SqlValue};

use super::schema_probe::{
    discover_join_plan, discover_table_column_roles, inspect_ordered_schemas_and_graph,
    sample_sql_cell, DiscoveredJoinPlan, SampledSqlCell,
};
use super::shape_syntax::{candidate_sql_matches_shape, SqlQueryShapeEvaluationReceipt};

fn execute_readonly_query_rows(conn: &Connection, sql: &str) -> Result<Vec<SqlRow>, EngineError> {
    let mut stmt = conn.prepare(sql)?;
    let col_names: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut map = SqlRow::new();
        for (idx, col_name) in col_names.iter().enumerate() {
            let val_ref = row.get_ref(idx)?;
            let sql_val = match val_ref {
                rusqlite::types::ValueRef::Null => SqlValue::Null,
                rusqlite::types::ValueRef::Integer(i) => SqlValue::Integer(i),
                rusqlite::types::ValueRef::Real(f) => SqlValue::Text(format!("{f:.6}")),
                rusqlite::types::ValueRef::Text(t) => {
                    SqlValue::Text(String::from_utf8_lossy(t).into_owned())
                }
                rusqlite::types::ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
            };
            map.insert(col_name.clone(), sql_val);
        }
        out.push(map);
    }
    Ok(out)
}

fn rows_match_ground_truth(reference_rows: &[SqlRow], candidate_rows: &[SqlRow]) -> bool {
    if reference_rows.len() != candidate_rows.len() {
        return false;
    }
    if reference_rows == candidate_rows {
        return true;
    }
    reference_rows
        .iter()
        .zip(candidate_rows.iter())
        .all(|(r_row, c_row)| {
            r_row.len() == c_row.len()
                && r_row
                    .values()
                    .zip(c_row.values())
                    .all(|(r_val, c_val)| r_val == c_val)
        })
}

/// Inspect `conn` and synthesize the canonical ground-truth reference SQL statement for `shape`.
pub fn canonical_reference_sql_for_shape(
    conn: &Connection,
    shape: SqlQueryShape,
) -> Result<String, EngineError> {
    let (schemas, graph) = inspect_ordered_schemas_and_graph(conn)?;
    let root_sc = &schemas[0];
    let root_roles = discover_table_column_roles(root_sc);
    let root_table = &root_roles.table_name;
    let pk_col = &root_roles.pk_col;
    let text_col = &root_roles.text_col;
    let num_col = &root_roles.num_col;

    let sample_pk = sample_sql_cell(
        conn,
        &format!("SELECT {pk_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
    )
    .unwrap_or_else(|| SampledSqlCell {
        sql_literal: "1".to_string(),
        raw_text: "1".to_string(),
        int_val: Some(1),
    });
    let sample_pk_lit = &sample_pk.sql_literal;

    let sample_text = sample_sql_cell(
        conn,
        &format!("SELECT {text_col} FROM {root_table} WHERE {pk_col} = {sample_pk_lit} LIMIT 1;"),
    )
    .or_else(|| {
        sample_sql_cell(
            conn,
            &format!("SELECT {text_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
        )
    })
    .unwrap_or_else(|| SampledSqlCell {
        sql_literal: "''".to_string(),
        raw_text: String::new(),
        int_val: None,
    });

    let sample_num = sample_sql_cell(
        conn,
        &format!("SELECT {num_col} FROM {root_table} WHERE {pk_col} = {sample_pk_lit} LIMIT 1;"),
    )
    .or_else(|| {
        sample_sql_cell(
            conn,
            &format!("SELECT {num_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
        )
    })
    .unwrap_or_else(|| sample_pk.clone());

    let sql = match shape {
        SqlQueryShape::PointLookupEq => {
            format!("SELECT * FROM {root_table} WHERE {pk_col} = {sample_pk_lit};")
        }
        SqlQueryShape::ColumnProjectionRange => {
            format!(
                "SELECT {pk_col}, {text_col} FROM {root_table} WHERE {num_col} >= {} ORDER BY {num_col} DESC, {pk_col} ASC LIMIT 5;",
                sample_num.sql_literal
            )
        }
        SqlQueryShape::AggregateCountWhere => {
            format!(
                "SELECT COUNT(*) AS matching_count FROM {root_table} WHERE {text_col} = {};",
                sample_text.sql_literal
            )
        }
        SqlQueryShape::InnerJoinOnForeignKey => {
            match discover_join_plan(conn, &schemas, &graph, &root_roles, sample_pk_lit) {
                DiscoveredJoinPlan::CrossTable {
                    parent_table,
                    parent_pk,
                    parent_text,
                    to_col,
                    child_table,
                    from_col,
                    child_col,
                    sample_child_col_lit,
                    ..
                } => {
                    format!(
                        "SELECT {parent_table}.{parent_pk}, {parent_table}.{parent_text}, {child_table}.{child_col} FROM {parent_table} JOIN {child_table} ON {parent_table}.{to_col} = {child_table}.{from_col} WHERE {child_table}.{child_col} = {sample_child_col_lit} ORDER BY {parent_table}.{parent_pk} ASC;"
                    )
                }
                DiscoveredJoinPlan::SelfJoin {
                    root_table,
                    pk_col,
                    text_col,
                    num_col,
                    sample_pk_lit,
                } => {
                    format!(
                        "SELECT t1.{pk_col}, t1.{text_col}, t2.{num_col} FROM {root_table} AS t1 JOIN {root_table} AS t2 ON t1.{pk_col} = t2.{pk_col} WHERE t1.{pk_col} = {sample_pk_lit};"
                    )
                }
            }
        }
        SqlQueryShape::LeftJoinWithFilter => {
            match discover_join_plan(conn, &schemas, &graph, &root_roles, sample_pk_lit) {
                DiscoveredJoinPlan::CrossTable {
                    parent_table,
                    parent_pk,
                    parent_text,
                    to_col,
                    child_table,
                    from_col,
                    child_col,
                    sample_parent_pk_lit,
                    ..
                } => {
                    format!(
                        "SELECT {parent_table}.{parent_pk}, {parent_table}.{parent_text}, {child_table}.{child_col} FROM {parent_table} LEFT JOIN {child_table} ON {parent_table}.{to_col} = {child_table}.{from_col} WHERE {parent_table}.{parent_pk} >= {sample_parent_pk_lit} ORDER BY {parent_table}.{parent_pk} ASC LIMIT 5;"
                    )
                }
                DiscoveredJoinPlan::SelfJoin {
                    root_table,
                    pk_col,
                    text_col,
                    num_col,
                    sample_pk_lit,
                } => {
                    format!(
                        "SELECT t1.{pk_col}, t1.{text_col}, t2.{num_col} FROM {root_table} AS t1 LEFT JOIN {root_table} AS t2 ON t1.{pk_col} = t2.{pk_col} WHERE t1.{pk_col} >= {sample_pk_lit} ORDER BY t1.{pk_col} ASC LIMIT 5;"
                    )
                }
            }
        }
    };
    Ok(sql)
}

/// Synthesize a canonical range/pattern query exercising [`SqlComparisonOperator`] (`>`, `>=`, `<`, `<=`, `!=`, `LIKE`)
/// against the primary table of `conn`.
pub fn canonical_reference_sql_for_comparison_op(
    conn: &Connection,
    op: SqlComparisonOperator,
) -> Result<String, EngineError> {
    let (schemas, _graph) = inspect_ordered_schemas_and_graph(conn)?;
    let root_sc = &schemas[0];
    let root_roles = discover_table_column_roles(root_sc);
    let root_table = &root_roles.table_name;
    let pk_col = &root_roles.pk_col;
    let text_col = &root_roles.text_col;
    let range_col = if root_roles.pk_is_numeric {
        &root_roles.pk_col
    } else {
        &root_roles.num_col
    };

    let sample_pk = sample_sql_cell(
        conn,
        &format!("SELECT {pk_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
    )
    .unwrap_or_else(|| SampledSqlCell {
        sql_literal: "1".to_string(),
        raw_text: "1".to_string(),
        int_val: Some(1),
    });
    let sample_pk_lit = &sample_pk.sql_literal;

    let sample_range_val: i64 = sample_sql_cell(
        conn,
        &format!("SELECT {range_col} FROM {root_table} WHERE {pk_col} = {sample_pk_lit} LIMIT 1;"),
    )
    .or_else(|| {
        sample_sql_cell(
            conn,
            &format!("SELECT {range_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
        )
    })
    .and_then(|c| c.int_val)
    .or(sample_pk.int_val)
    .unwrap_or(1);

    let sample_text = sample_sql_cell(
        conn,
        &format!("SELECT {text_col} FROM {root_table} WHERE {pk_col} = {sample_pk_lit} LIMIT 1;"),
    )
    .or_else(|| {
        sample_sql_cell(
            conn,
            &format!("SELECT {text_col} FROM {root_table} ORDER BY {pk_col} ASC LIMIT 1;"),
        )
    })
    .map(|c| c.raw_text)
    .unwrap_or_default();
    let prefix = sample_text
        .chars()
        .take(4)
        .collect::<String>()
        .replace('\'', "''");
    let ne_sentinel = if sample_range_val == -1 { -2 } else { -1 };

    let where_clause = match op {
        SqlComparisonOperator::Gt => {
            format!("{range_col} > {}", sample_range_val.saturating_sub(1))
        }
        SqlComparisonOperator::Gte => format!("{range_col} >= {sample_range_val}"),
        SqlComparisonOperator::Lt => {
            format!("{range_col} < {}", sample_range_val.saturating_add(10_000))
        }
        SqlComparisonOperator::Lte => {
            format!("{range_col} <= {}", sample_range_val.saturating_add(10_000))
        }
        SqlComparisonOperator::Ne => format!("{range_col} != {ne_sentinel}"),
        SqlComparisonOperator::Like => format!("{text_col} LIKE '{prefix}%'"),
    };
    Ok(format!(
        "SELECT {pk_col}, {text_col} FROM {root_table} WHERE {where_clause} ORDER BY {pk_col} ASC LIMIT 5;"
    ))
}

/// Evaluate a model's `candidate_sql` query against the ground-truth reference result for `expected_shape`
/// on a live SQLite connection (`conn`).
pub fn evaluate_candidate_query_against_reference(
    conn: &Connection,
    candidate_sql: &str,
    expected_shape: SqlQueryShape,
) -> Result<SqlQueryShapeEvaluationReceipt, EngineError> {
    let reference_sql = canonical_reference_sql_for_shape(conn, expected_shape)?;
    let reference_rows = execute_readonly_query_rows(conn, &reference_sql)?;
    let matches_shape_syntax = candidate_sql_matches_shape(candidate_sql, expected_shape);

    match execute_readonly_query_rows(conn, candidate_sql) {
        Ok(candidate_rows) => {
            let matches_ground_truth_rows =
                rows_match_ground_truth(&reference_rows, &candidate_rows);
            Ok(SqlQueryShapeEvaluationReceipt {
                expected_shape,
                reference_sql,
                candidate_sql: candidate_sql.to_string(),
                parsed_and_executed: true,
                matches_shape_syntax,
                matches_ground_truth_rows,
                reference_row_count: reference_rows.len(),
                candidate_row_count: candidate_rows.len(),
                reference_rows,
                candidate_rows,
                error_message: None,
            })
        }
        Err(err) => Ok(SqlQueryShapeEvaluationReceipt {
            expected_shape,
            reference_sql,
            candidate_sql: candidate_sql.to_string(),
            parsed_and_executed: false,
            matches_shape_syntax,
            matches_ground_truth_rows: false,
            reference_row_count: reference_rows.len(),
            candidate_row_count: 0,
            reference_rows,
            candidate_rows: Vec::new(),
            error_message: Some(err.to_string()),
        }),
    }
}
