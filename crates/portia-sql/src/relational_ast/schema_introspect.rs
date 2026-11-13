//! Live SQLite `PRAGMA table_info`, `PRAGMA foreign_key_list`, and `sqlite_master` `CHECK (...)`
//! introspection for [`RelationalSchemaGraph`].

use std::collections::{BTreeMap, BTreeSet};

use portia_core::hylo::CollapsibleExt;
use portia_core::{ColumnSchema, TableSchema};
use rusqlite::Connection;

use crate::engine::EngineError;
use crate::grammar_trie::inspect_table_schema;

use super::ident::SqlColumnRef;
use super::predicate::{SqlPredicateAst, SqlPredicateFrame};
use super::scalar_ops::SqlScalarValue;
use super::schema_graph::{
    ForeignKeyEdge, RelationalSchemaGraph, SkippedCheckConstraintDiagnostic,
};
use super::type_check::is_numeric_sql_type_str;

pub fn validate_check_predicate_against_table_schema(
    ast: &SqlPredicateAst,
    table_schema: &TableSchema,
    non_binary_collated_cols: &BTreeSet<String>,
) -> Result<(), String> {
    let resolve_table_col = |col_ref: &SqlColumnRef| -> Result<&ColumnSchema, String> {
        if let Some(tbl) = &col_ref.table_qualifier {
            if !table_schema.table_name.is_empty()
                && !tbl.matches_ascii_case_insensitive(&table_schema.table_name)
            {
                return Err(format!(
                    "CHECK references unknown table qualifier `{}` in `{col_ref}` on table `{}`",
                    tbl.name, table_schema.table_name
                ));
            }
        }
        table_schema
            .columns
            .iter()
            .find(|c| col_ref.column.matches_ascii_case_insensitive(&c.name))
            .ok_or_else(|| {
                format!(
                    "CHECK references unknown column `{col_ref}` in table `{}`",
                    table_schema.table_name
                )
            })
    };

    ast.try_collapse_frames(|frame: SqlPredicateFrame<()>| match frame {
        SqlPredicateFrame::Comparison {
            column,
            op,
            literal,
        } => {
            let col_sc = resolve_table_col(&column)?;
            if non_binary_collated_cols
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&col_sc.name))
                || col_sc.col_type.to_ascii_uppercase().contains("COLLATE")
            {
                return Err(format!(
                    "column `{}` in table `{}` uses a non-BINARY collation (`{}`), which is unsupported for typed CHECK import",
                    col_sc.name, table_schema.table_name, col_sc.col_type
                ));
            }
            let is_numeric_col = is_numeric_sql_type_str(&col_sc.col_type);
            match &literal {
                SqlScalarValue::Int(_) | SqlScalarValue::Float(_) | SqlScalarValue::Bool(_) => {
                    if !is_numeric_col {
                        return Err(format!(
                            "numeric/boolean literal `{}` compared against non-numeric column `{}` (`{}`) in table `{}` changes SQLite affinity semantics",
                            literal.render_sql(),
                            col_sc.name,
                            col_sc.col_type,
                            table_schema.table_name
                        ));
                    }
                }
                SqlScalarValue::Text(_) => {
                    if is_numeric_col || op.is_range_op() {
                        return Err(format!(
                            "text literal `{}` with `{}` on column `{}` (`{}`) in table `{}` is unsupported for typed CHECK import",
                            literal.render_sql(),
                            op.as_sql_str(),
                            col_sc.name,
                            col_sc.col_type,
                            table_schema.table_name
                        ));
                    }
                }
                SqlScalarValue::Null => {}
            }
            Ok(())
        }
        SqlPredicateFrame::IsNull { column, .. } => {
            let _ = resolve_table_col(&column)?;
            Ok(())
        }
        SqlPredicateFrame::InList { column, literals } => {
            let col_sc = resolve_table_col(&column)?;
            if non_binary_collated_cols
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&col_sc.name))
                || col_sc.col_type.to_ascii_uppercase().contains("COLLATE")
            {
                return Err(format!(
                    "column `{}` in table `{}` uses a non-BINARY collation (`{}`), which is unsupported for typed CHECK import",
                    col_sc.name, table_schema.table_name, col_sc.col_type
                ));
            }
            let is_numeric_col = is_numeric_sql_type_str(&col_sc.col_type);
            for lit in &literals {
                match lit {
                    SqlScalarValue::Int(_) | SqlScalarValue::Float(_) | SqlScalarValue::Bool(_) => {
                        if !is_numeric_col {
                            return Err(format!(
                                "numeric/boolean literal `{}` in IN list on non-numeric column `{}` (`{}`) in table `{}` changes SQLite affinity semantics",
                                lit.render_sql(),
                                col_sc.name,
                                col_sc.col_type,
                                table_schema.table_name
                            ));
                        }
                    }
                    SqlScalarValue::Text(_) => {
                        if is_numeric_col {
                            return Err(format!(
                                "text literal `{}` in IN list on numeric column `{}` (`{}`) in table `{}` changes SQLite affinity semantics",
                                lit.render_sql(),
                                col_sc.name,
                                col_sc.col_type,
                                table_schema.table_name
                            ));
                        }
                    }
                    SqlScalarValue::Null => {}
                }
            }
            Ok(())
        }
        SqlPredicateFrame::And((), ())
        | SqlPredicateFrame::Or((), ())
        | SqlPredicateFrame::Not(()) => Ok(()),
    })
}

fn resolve_implicit_parent_primary_key_column(
    conn: &Connection,
    parent_table: &str,
) -> Result<String, EngineError> {
    let escaped = parent_table.replace('"', "\"\"");
    let pragma_ti = format!("PRAGMA table_info(\"{escaped}\");");
    let mut ti_stmt = conn.prepare(&pragma_ti)?;
    let mut ti_rows = ti_stmt.query([])?;
    let mut pk_cols: Vec<(i64, String)> = Vec::new();
    while let Some(row) = ti_rows.next()? {
        let col_name: String = row.get(1)?;
        let pk_flag: i64 = row.get(5)?;
        if pk_flag > 0 {
            pk_cols.push((pk_flag, col_name));
        }
    }
    pk_cols.sort_by_key(|(pk_idx, _)| *pk_idx);
    Ok(pk_cols
        .into_iter()
        .next()
        .map(|(_, col)| col)
        .unwrap_or_else(|| "rowid".to_string()))
}

impl RelationalSchemaGraph {
    /// Introspect all user tables, column types, nullability (`notnull == 0`), primary keys,
    /// foreign-key edges (`PRAGMA foreign_key_list`), and `CHECK (...)` constraints (parsed from
    /// `sqlite_master` `CREATE TABLE` DDL into [`SqlPredicateAst`], recording any unimported
    /// `CHECK` clauses in `skipped_check_diagnostics`) from a live SQLite [`Connection`].
    pub fn inspect_from_connection(conn: &Connection) -> Result<Self, EngineError> {
        let mut stmt = conn.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
        )?;
        let table_entries: Vec<(String, Option<String>)> = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut tables = Vec::with_capacity(table_entries.len());
        let mut nullable_columns: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut foreign_keys: Vec<ForeignKeyEdge> = Vec::new();
        let mut check_constraints: BTreeMap<String, Vec<SqlPredicateAst>> = BTreeMap::new();
        let mut skipped_check_diagnostics: Vec<SkippedCheckConstraintDiagnostic> = Vec::new();
        let dialect = sqlparser::dialect::SQLiteDialect {};

        for (tname, create_sql_opt) in &table_entries {
            let escaped = tname.replace('"', "\"\"");
            let current_table_schema = inspect_table_schema(conn, tname)?;
            let null_set: BTreeSet<String> = current_table_schema
                .columns
                .iter()
                .filter(|c| c.nullable() && !c.is_primary_key)
                .map(|c| c.name.clone())
                .collect();
            nullable_columns.insert(tname.clone(), null_set);
            tables.push(current_table_schema.clone());

            let pragma_fk = format!("PRAGMA foreign_key_list(\"{escaped}\");");
            let mut fk_stmt = conn.prepare(&pragma_fk)?;
            let mut fk_rows = fk_stmt.query([])?;
            while let Some(row) = fk_rows.next()? {
                let parent_table: String = row.get(2)?;
                let child_col: String = row.get(3)?;
                let parent_col_opt: Option<String> = row.get(4)?;
                let parent_col = match parent_col_opt.filter(|s| !s.is_empty()) {
                    Some(col) => col,
                    None => resolve_implicit_parent_primary_key_column(conn, &parent_table)?,
                };
                foreign_keys.push(ForeignKeyEdge::new(
                    tname.clone(),
                    child_col,
                    parent_table,
                    parent_col,
                ));
            }

            let mut table_checks = Vec::new();
            if let Some(ddl) = create_sql_opt {
                match sqlparser::parser::Parser::parse_sql(&dialect, ddl) {
                    Ok(stmts) => {
                        for parsed_stmt in stmts {
                            if let sqlparser::ast::Statement::CreateTable(create_tbl) = parsed_stmt
                            {
                                let non_binary_collated_cols =
                                    collect_non_binary_collated_columns(&create_tbl);

                                for col_def in &create_tbl.columns {
                                    for opt_def in &col_def.options {
                                        if let sqlparser::ast::ColumnOption::Check(expr) =
                                            &opt_def.option
                                        {
                                            match SqlPredicateAst::from_sqlparser_expr(expr)
                                                .and_then(|ast| {
                                                    validate_check_predicate_against_table_schema(
                                                        &ast,
                                                        &current_table_schema,
                                                        &non_binary_collated_cols,
                                                    )?;
                                                    Ok(ast)
                                                }) {
                                                Ok(ast) => table_checks.push(ast),
                                                Err(reason) => skipped_check_diagnostics.push(
                                                    SkippedCheckConstraintDiagnostic {
                                                        table_name: tname.clone(),
                                                        constraint_name: opt_def
                                                            .name
                                                            .as_ref()
                                                            .map(ToString::to_string)
                                                            .or_else(|| {
                                                                Some(col_def.name.to_string())
                                                            }),
                                                        raw_expr_sql: expr.to_string(),
                                                        reason,
                                                    },
                                                ),
                                            }
                                        }
                                    }
                                }
                                for tc in &create_tbl.constraints {
                                    if let sqlparser::ast::TableConstraint::Check {
                                        name,
                                        expr,
                                        ..
                                    } = tc
                                    {
                                        match SqlPredicateAst::from_sqlparser_expr(expr).and_then(
                                            |ast| {
                                                validate_check_predicate_against_table_schema(
                                                    &ast,
                                                    &current_table_schema,
                                                    &non_binary_collated_cols,
                                                )?;
                                                Ok(ast)
                                            },
                                        ) {
                                            Ok(ast) => table_checks.push(ast),
                                            Err(reason) => skipped_check_diagnostics.push(
                                                SkippedCheckConstraintDiagnostic {
                                                    table_name: tname.clone(),
                                                    constraint_name: name
                                                        .as_ref()
                                                        .map(ToString::to_string),
                                                    raw_expr_sql: expr.to_string(),
                                                    reason,
                                                },
                                            ),
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        if ddl.to_ascii_uppercase().contains("CHECK") {
                            skipped_check_diagnostics.push(SkippedCheckConstraintDiagnostic {
                                table_name: tname.clone(),
                                constraint_name: None,
                                raw_expr_sql: ddl.clone(),
                                reason: format!("DDL parse error: {err}"),
                            });
                        }
                    }
                }
            }
            if !table_checks.is_empty() {
                check_constraints.insert(tname.clone(), table_checks);
            }
        }

        foreign_keys.sort();
        foreign_keys.dedup();
        let mut graph = Self::new(tables, foreign_keys, nullable_columns);
        graph.check_constraints = check_constraints;
        graph.skipped_check_diagnostics = skipped_check_diagnostics;
        Ok(graph)
    }
}

fn collect_non_binary_collated_columns(
    create_tbl: &sqlparser::ast::CreateTable,
) -> BTreeSet<String> {
    let mut non_binary_collated_cols = BTreeSet::new();
    for col_def in &create_tbl.columns {
        let col_type_upper = col_def.data_type.to_string().to_ascii_uppercase();
        if col_type_upper.contains("COLLATE ") && !col_type_upper.contains("COLLATE BINARY") {
            non_binary_collated_cols.insert(col_def.name.value.clone());
        }
        for opt_def in &col_def.options {
            let opt_upper = opt_def.option.to_string().to_ascii_uppercase();
            if opt_upper.starts_with("COLLATE ")
                && opt_upper != "COLLATE BINARY"
                && opt_upper != "COLLATE \"BINARY\""
            {
                non_binary_collated_cols.insert(col_def.name.value.clone());
            }
        }
        if let Some(coll_obj) = &col_def.collation {
            if !coll_obj
                .to_string()
                .trim_matches('"')
                .eq_ignore_ascii_case("BINARY")
            {
                non_binary_collated_cols.insert(col_def.name.value.clone());
            }
        }
    }
    non_binary_collated_cols
}
