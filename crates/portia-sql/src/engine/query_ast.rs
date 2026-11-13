//! `sqlparser::ast` table-provenance extraction for `READ`/`PROBE` queries and single-target `WRITE` DML.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sqlparser::ast::{Expr, ObjectName, SetExpr, Statement, TableFactor};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::parser::Parser as SqlParser;

fn object_name_leaf_lower(name: &ObjectName) -> String {
    name.0
        .last()
        .map(|ident| ident.value.to_lowercase())
        .unwrap_or_default()
}

fn collect_tables_from_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Subquery(q) => collect_tables_from_query(q, out),
        Expr::InSubquery {
            expr: inner,
            subquery,
            ..
        } => {
            collect_tables_from_expr(inner, out);
            collect_tables_from_query(subquery, out);
        }
        Expr::Exists { subquery, .. } => collect_tables_from_query(subquery, out),
        Expr::BinaryOp { left, right, .. } => {
            collect_tables_from_expr(left, out);
            collect_tables_from_expr(right, out);
        }
        Expr::UnaryOp { expr: inner, .. } | Expr::Nested(inner) => {
            collect_tables_from_expr(inner, out);
        }
        _ => {}
    }
}

fn collect_tables_from_table_factor(factor: &TableFactor, out: &mut BTreeSet<String>) {
    match factor {
        TableFactor::Table { name, .. } => {
            let leaf = object_name_leaf_lower(name);
            if !leaf.is_empty() && !leaf.starts_with("sqlite_") {
                out.insert(leaf);
            }
        }
        TableFactor::Derived { subquery, .. } => {
            collect_tables_from_query(subquery, out);
        }
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => {
            collect_tables_from_table_factor(&table_with_joins.relation, out);
            for j in &table_with_joins.joins {
                collect_tables_from_table_factor(&j.relation, out);
            }
        }
        _ => {}
    }
}

fn collect_tables_from_select(select: &sqlparser::ast::Select, out: &mut BTreeSet<String>) {
    for twj in &select.from {
        collect_tables_from_table_factor(&twj.relation, out);
        for join in &twj.joins {
            collect_tables_from_table_factor(&join.relation, out);
        }
    }
    if let Some(sel) = &select.selection {
        collect_tables_from_expr(sel, out);
    }
    for item in &select.projection {
        if let sqlparser::ast::SelectItem::UnnamedExpr(e)
        | sqlparser::ast::SelectItem::ExprWithAlias { expr: e, .. } = item
        {
            collect_tables_from_expr(e, out);
        }
    }
}

fn collect_tables_from_set_expr(set_expr: &SetExpr, out: &mut BTreeSet<String>) {
    match set_expr {
        SetExpr::Select(select) => collect_tables_from_select(select, out),
        SetExpr::Query(q) => collect_tables_from_query(q, out),
        SetExpr::SetOperation { left, right, .. } => {
            collect_tables_from_set_expr(left, out);
            collect_tables_from_set_expr(right, out);
        }
        _ => {}
    }
}

fn collect_tables_from_query(query: &sqlparser::ast::Query, out: &mut BTreeSet<String>) {
    let mut ctes = BTreeSet::new();
    if let Some(with) = &query.with {
        for cte in &with.cte_tables {
            ctes.insert(cte.alias.name.value.to_lowercase());
            collect_tables_from_query(&cte.query, out);
        }
    }
    collect_tables_from_set_expr(query.body.as_ref(), out);
    for cte_name in ctes {
        out.remove(&cte_name);
    }
}

/// Recursively extract all referenced base user tables from a parsed `sqlparser::ast::Query`
/// (handling CTEs, `JOIN`s, derived subqueries, `EXISTS`/`IN` subqueries, and `UNION`/`INTERSECT`/`EXCEPT`),
/// excluding CTE alias names and `sqlite_*` system tables.
pub fn collect_tables_from_sql_query_ast(
    query: &sqlparser::ast::Query,
) -> Result<BTreeSet<String>, String> {
    let mut tables = BTreeSet::new();
    collect_tables_from_query(query, &mut tables);
    Ok(tables)
}

/// Strongly-typed SQL `WRITE` DML operation (`INSERT`, `UPDATE`, `DELETE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SqlWriteOp {
    Insert,
    Update,
    Delete,
}

impl SqlWriteOp {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Insert => "INSERT",
            Self::Update => "UPDATE",
            Self::Delete => "DELETE",
        }
    }
}

impl std::fmt::Display for SqlWriteOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn strip_leading_sql_comments(mut s: &str) -> &str {
    loop {
        s = s.trim_start();
        if let Some(rest) = s.strip_prefix("--") {
            match rest.find('\n') {
                Some(nl) => s = &rest[nl + 1..],
                None => return "",
            }
        } else if let Some(rest) = s.strip_prefix("/*") {
            match rest.find("*/") {
                Some(end) => s = &rest[end + 2..],
                None => return s,
            }
        } else {
            return s;
        }
    }
}

fn unquote_sqlite_pragma_ident(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        return trimmed[1..trimmed.len() - 1].replace("\"\"", "\"");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('\'') && trimmed.ends_with('\'') {
        return trimmed[1..trimmed.len() - 1].replace("''", "'");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('`') && trimmed.ends_with('`') {
        return trimmed[1..trimmed.len() - 1].replace("``", "`");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed[1..trimmed.len() - 1].to_string();
    }
    trimmed.to_string()
}

fn extract_pragma_target_table(pragma_sql: &str) -> Option<String> {
    let body = pragma_sql.trim().trim_end_matches(';').trim();
    let after_pragma = body.get("PRAGMA".len()..)?.trim_start();
    // Optional `schema.` prefix before `table_info` / `table_xinfo`:
    let after_name = {
        let lower = after_pragma.to_ascii_lowercase();
        let name_pos = lower
            .find("table_info")
            .map(|i| (i, "table_info".len()))
            .or_else(|| lower.find("table_xinfo").map(|i| (i, "table_xinfo".len())))?;
        let prefix = after_pragma[..name_pos.0].trim();
        if !prefix.is_empty() && !prefix.ends_with('.') {
            return None;
        }
        after_pragma[name_pos.0 + name_pos.1..].trim_start()
    };
    let arg = if let Some(inner) = after_name.strip_prefix('(') {
        let inner = inner.strip_suffix(')')?.trim();
        inner
    } else if let Some(rhs) = after_name.strip_prefix('=') {
        rhs.trim()
    } else {
        return None;
    };
    let unquoted = unquote_sqlite_pragma_ident(arg).to_lowercase();
    if unquoted.is_empty() {
        None
    } else {
        Some(unquoted)
    }
}

/// Parse a `READ` or `PROBE` SQL string (supporting `PRAGMA table_info(...)`, `SELECT` with
/// `JOIN`, `GROUP BY`, `COUNT(*)`, `ORDER BY`, `LIMIT`, CTEs `WITH ... SELECT`, subqueries,
/// `UNION`/`INTERSECT`/`EXCEPT`, and table aliases) and extract the referenced base tables.
pub fn parse_read_or_probe_tables(sql: &str) -> Result<BTreeSet<String>, String> {
    let trimmed = strip_leading_sql_comments(sql).trim();
    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("PRAGMA ")
        || upper.starts_with("PRAGMA\t")
        || upper.starts_with("PRAGMA\n")
    {
        let dialect = SQLiteDialect {};
        if let Ok(stmts) = SqlParser::parse_sql(&dialect, trimmed) {
            if stmts.len() > 1 {
                return Err("InvalidReadSQL: multi-statement PRAGMA is not permitted".to_string());
            }
        } else if trimmed.trim_end_matches(';').contains(';') {
            return Err("InvalidReadSQL: multi-statement PRAGMA is not permitted".to_string());
        }
        let mut out = BTreeSet::new();
        if let Some(tbl) = extract_pragma_target_table(trimmed) {
            out.insert(tbl);
        }
        return Ok(out);
    }

    let dialect = SQLiteDialect {};
    let stmts =
        SqlParser::parse_sql(&dialect, trimmed).map_err(|e| format!("InvalidReadSQL: {}", e))?;
    if stmts.len() != 1 {
        return Err(format!(
            "InvalidReadSQL: expected exactly 1 statement, got {}",
            stmts.len()
        ));
    }
    match &stmts[0] {
        Statement::Query(q) => collect_tables_from_sql_query_ast(q),
        other => Err(format!(
            "InvalidReadSQL: only SELECT or PRAGMA allowed in READ/PROBE, got {:?}",
            other
        )),
    }
}

/// Parse a single `WRITE` statement (`INSERT`, `UPDATE`, `DELETE`) via `sqlparser` and extract
/// both `(target_table_lower, write_op)`.
pub fn parse_single_write_statement_target_and_op(
    sql: &str,
) -> Result<(String, SqlWriteOp), String> {
    let dialect = SQLiteDialect {};
    let stmts = SqlParser::parse_sql(&dialect, sql.trim())
        .map_err(|e| format!("InvalidWriteSQL: {}", e))?;
    if stmts.len() != 1 {
        return Err(format!(
            "InvalidWriteSQL: expected exactly 1 DML statement, got {}",
            stmts.len()
        ));
    }
    match &stmts[0] {
        Statement::Insert(ins) => Ok((object_name_leaf_lower(&ins.table_name), SqlWriteOp::Insert)),
        Statement::Update { table, .. } => {
            let mut set = BTreeSet::new();
            collect_tables_from_table_factor(&table.relation, &mut set);
            let tbl = set
                .into_iter()
                .next()
                .ok_or_else(|| "InvalidWriteSQL: missing UPDATE target table".to_string())?;
            Ok((tbl, SqlWriteOp::Update))
        }
        Statement::Delete(del) => {
            let mut set = BTreeSet::new();
            for twj in &del.tables {
                set.insert(object_name_leaf_lower(twj));
            }
            match &del.from {
                sqlparser::ast::FromTable::WithFromKeyword(list)
                | sqlparser::ast::FromTable::WithoutKeyword(list) => {
                    for twj in list {
                        collect_tables_from_table_factor(&twj.relation, &mut set);
                    }
                }
            }
            let tbl = set
                .into_iter()
                .next()
                .ok_or_else(|| "InvalidWriteSQL: missing DELETE target table".to_string())?;
            Ok((tbl, SqlWriteOp::Delete))
        }
        other => Err(format!(
            "InvalidWriteSQL: only INSERT/UPDATE/DELETE permitted in WRITE mode, got {:?}",
            other
        )),
    }
}

/// Parse a single `WRITE` statement (`INSERT`, `UPDATE`, `DELETE`) via `sqlparser` and extract
/// the target table name in lowercase. Rejects multi-statement queries (`; DROP TABLE ...`) and DDL.
pub fn parse_single_write_target_table(sql: &str) -> Result<String, String> {
    parse_single_write_statement_target_and_op(sql).map(|(tbl, _)| tbl)
}
