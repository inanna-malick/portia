//! Top-level statement (`PRAGMA`, `SELECT`, `UPDATE`, `INSERT`, `DELETE`) `sqlparser` AST
//! validation against discovered [`TableSchema`]s.

use std::collections::BTreeSet;

use sqlparser::dialect::SQLiteDialect;

use portia_core::TableSchema;

use super::lark_specialize::is_sqlite_reserved_keyword;
use super::tokenizer_scan::{
    split_leading_ident, strip_ident_quotes, strip_pragma_table_info_prefix,
    unescape_doubled_ident_body,
};
use super::validate_expr::{
    ast_object_name_leaf, ensure_ident_not_unquoted_reserved,
    ensure_object_name_not_unquoted_reserved, find_table_schema, validate_expr_refs_only,
    validate_expr_subqueries,
};
use super::validate_query::{collect_and_validate_table_factor, validate_query_ast};

fn validate_pragma_table_info_sql(
    trimmed: &str,
    upper: &str,
    schemas: &[TableSchema],
) -> Result<(), String> {
    if let Some(rest) = strip_pragma_table_info_prefix(trimmed, upper) {
        let quote_ch = rest.trim_start().chars().next();
        let is_quoted = matches!(quote_ch, Some('"' | '`' | '\''));
        let (raw_tbl, after_tbl, _) = split_leading_ident(rest);
        let tbl = match quote_ch {
            Some(q @ ('"' | '`' | '\'')) => unescape_doubled_ident_body(raw_tbl, q),
            _ => raw_tbl.to_string(),
        };
        if tbl.is_empty() {
            return Err("Missing table identifier in PRAGMA table_info".to_string());
        }
        if !is_quoted && is_sqlite_reserved_keyword(&tbl) {
            return Err(format!(
                "Reserved SQLite keyword `{tbl}` must be quoted in PRAGMA table_info"
            ));
        }
        let after_ws = after_tbl.trim_start();
        if !after_ws.starts_with(')') {
            return Err(format!(
                "Expected closing `)` after table `{tbl}` in PRAGMA table_info"
            ));
        }
        if !schemas.is_empty() && find_table_schema(schemas, &tbl).is_none() {
            return Err(format!("Unknown table `{tbl}` in PRAGMA table_info"));
        }
    }
    Ok(())
}

/// Validate a completed SQL statement against `schemas` via `sqlparser` AST inspection
/// (verifying that referenced tables exist in `schemas`, every referenced `WHERE` / `SET` / `INSERT` /
/// `SELECT` column belongs to the target table's `PRAGMA table_info` schema, and no column is
/// assigned twice in `UPDATE ... SET`).
pub fn validate_completed_sql_against_schemas(
    sql: &str,
    schemas: &[TableSchema],
) -> Result<(), String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err("Empty SQL statement".to_string());
    }
    if schemas.is_empty() {
        return Ok(());
    }

    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("PRAGMA") {
        return validate_pragma_table_info_sql(trimmed, &upper, schemas);
    }

    let dialect = SQLiteDialect {};
    let stmts = sqlparser::parser::Parser::parse_sql(&dialect, trimmed)
        .map_err(|e| format!("SQL syntax error: {e}"))?;
    if stmts.len() != 1 {
        return Err(format!(
            "Expected exactly 1 SQL statement, got {}",
            stmts.len()
        ));
    }

    let empty_ctes = BTreeSet::new();
    match &stmts[0] {
        sqlparser::ast::Statement::Pragma { name, value, .. } => {
            if ast_object_name_leaf(name).eq_ignore_ascii_case("table_info") {
                if let Some(v) = value {
                    let raw = v.to_string();
                    let tbl = strip_ident_quotes(&raw);
                    if !tbl.is_empty() && find_table_schema(schemas, &tbl).is_none() {
                        return Err(format!("Unknown table `{tbl}` in PRAGMA table_info"));
                    }
                }
            }
            Ok(())
        }
        sqlparser::ast::Statement::Query(query) => validate_query_ast(query, schemas, &empty_ctes),
        sqlparser::ast::Statement::Update {
            table,
            assignments,
            selection,
            ..
        } => {
            let mut in_scope = Vec::new();
            let mut has_untracked = false;
            collect_and_validate_table_factor(
                &table.relation,
                schemas,
                &empty_ctes,
                &mut in_scope,
                &mut has_untracked,
            )?;
            let Some((_, target_sc)) = in_scope.first().map(|(_, sc)| ((), *sc)) else {
                return Ok(());
            };
            if assignments.is_empty() {
                return Err(format!(
                    "Empty SET assignment list in UPDATE for table `{}`",
                    target_sc.table_name
                ));
            }
            let valid_cols: BTreeSet<String> = target_sc
                .columns
                .iter()
                .map(|c| c.name.to_ascii_lowercase())
                .collect();
            let mut seen_set_cols = BTreeSet::new();
            for assign in assignments {
                let col_raw = match &assign.target {
                    sqlparser::ast::AssignmentTarget::ColumnName(obj_name) => {
                        ensure_object_name_not_unquoted_reserved(obj_name, "SET")?;
                        ast_object_name_leaf(obj_name).to_ascii_lowercase()
                    }
                    sqlparser::ast::AssignmentTarget::Tuple(objs) => {
                        for o in objs {
                            ensure_object_name_not_unquoted_reserved(o, "SET")?;
                        }
                        objs.first()
                            .map(|o| ast_object_name_leaf(o).to_ascii_lowercase())
                            .unwrap_or_default()
                    }
                };
                if !valid_cols.contains(&col_raw) {
                    return Err(format!(
                        "Hallucinated column `{col_raw}` in SET clause for table `{}` (valid: {:?})",
                        target_sc.table_name, valid_cols
                    ));
                }
                if !seen_set_cols.insert(col_raw.clone()) {
                    return Err(format!(
                        "Duplicate SET assignment to column `{col_raw}` in table `{}`",
                        target_sc.table_name
                    ));
                }
                validate_expr_subqueries(&assign.value, schemas, &empty_ctes, &in_scope)?;
                validate_expr_refs_only(&assign.value, &in_scope, None, "SET")?;
            }
            if let Some(sel) = selection {
                validate_expr_subqueries(sel, schemas, &empty_ctes, &in_scope)?;
                validate_expr_refs_only(sel, &in_scope, None, "WHERE")?;
            }
            Ok(())
        }
        sqlparser::ast::Statement::Insert(ins) => {
            ensure_object_name_not_unquoted_reserved(&ins.table_name, "INSERT INTO")?;
            let tbl_name = ast_object_name_leaf(&ins.table_name);
            let Some(target_sc) = find_table_schema(schemas, &tbl_name) else {
                return Err(format!("Unknown table `{tbl_name}` in INSERT INTO"));
            };
            let valid_cols: BTreeSet<String> = target_sc
                .columns
                .iter()
                .map(|c| c.name.to_ascii_lowercase())
                .collect();
            let mut seen_insert_cols = BTreeSet::new();
            for col_ident in &ins.columns {
                ensure_ident_not_unquoted_reserved(col_ident, "INSERT INTO")?;
                let col_raw = col_ident.value.to_ascii_lowercase();
                if !valid_cols.contains(&col_raw) {
                    return Err(format!(
                        "Hallucinated column `{col_raw}` in INSERT INTO column list for table `{}` (valid: {:?})",
                        target_sc.table_name, valid_cols
                    ));
                }
                if !seen_insert_cols.insert(col_raw.clone()) {
                    return Err(format!(
                        "Duplicate column `{col_raw}` in INSERT INTO column list for table `{}`",
                        target_sc.table_name
                    ));
                }
            }
            if let Some(source) = &ins.source {
                validate_query_ast(source, schemas, &empty_ctes)?;
            }
            Ok(())
        }
        sqlparser::ast::Statement::Delete(del) => {
            let mut in_scope = Vec::new();
            let mut has_untracked = false;
            match &del.from {
                sqlparser::ast::FromTable::WithFromKeyword(list)
                | sqlparser::ast::FromTable::WithoutKeyword(list) => {
                    for twj in list {
                        collect_and_validate_table_factor(
                            &twj.relation,
                            schemas,
                            &empty_ctes,
                            &mut in_scope,
                            &mut has_untracked,
                        )?;
                    }
                }
            }
            if let Some(sel) = &del.selection {
                validate_expr_subqueries(sel, schemas, &empty_ctes, &in_scope)?;
                validate_expr_refs_only(sel, &in_scope, None, "WHERE")?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
