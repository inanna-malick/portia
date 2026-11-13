//! Expression-level `sqlparser::ast` walkers, reserved-keyword checks, and column-scope validators.

use std::collections::BTreeSet;

use portia_core::TableSchema;

use super::lark_specialize::is_sqlite_reserved_keyword;
use super::validate_query::validate_query_ast_with_outer_scope;

pub(super) fn ensure_ident_not_unquoted_reserved(
    ident: &sqlparser::ast::Ident,
    ctx: &str,
) -> Result<(), String> {
    if ident.quote_style.is_none() && is_sqlite_reserved_keyword(&ident.value) {
        return Err(format!(
            "Reserved SQLite keyword `{}` must be quoted in {ctx}",
            ident.value
        ));
    }
    Ok(())
}

pub(super) fn ensure_object_name_not_unquoted_reserved(
    name: &sqlparser::ast::ObjectName,
    ctx: &str,
) -> Result<(), String> {
    for ident in &name.0 {
        ensure_ident_not_unquoted_reserved(ident, ctx)?;
    }
    Ok(())
}

pub(super) fn ast_object_name_leaf(name: &sqlparser::ast::ObjectName) -> String {
    name.0
        .last()
        .map(|ident| ident.value.clone())
        .unwrap_or_default()
}

pub(super) fn find_table_schema<'a>(
    schemas: &'a [TableSchema],
    name: &str,
) -> Option<&'a TableSchema> {
    schemas
        .iter()
        .find(|sc| sc.table_name.eq_ignore_ascii_case(name))
}

fn try_walk_expr_shallow<F>(expr: &sqlparser::ast::Expr, f: &mut F) -> Result<(), String>
where
    F: FnMut(&sqlparser::ast::Expr) -> Result<(), String>,
{
    use sqlparser::ast::Expr;
    f(expr)?;
    match expr {
        Expr::BinaryOp { left, right, .. }
        | Expr::AnyOp { left, right, .. }
        | Expr::AllOp { left, right, .. } => {
            try_walk_expr_shallow(left, f)?;
            try_walk_expr_shallow(right, f)
        }
        Expr::Like { expr, pattern, .. }
        | Expr::ILike { expr, pattern, .. }
        | Expr::RLike { expr, pattern, .. } => {
            try_walk_expr_shallow(expr, f)?;
            try_walk_expr_shallow(pattern, f)
        }
        Expr::UnaryOp { expr: inner, .. }
        | Expr::Nested(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::IsTrue(inner)
        | Expr::IsNotTrue(inner)
        | Expr::IsFalse(inner)
        | Expr::IsNotFalse(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::InSubquery { expr: inner, .. } => try_walk_expr_shallow(inner, f),
        Expr::InList {
            expr: target, list, ..
        } => {
            try_walk_expr_shallow(target, f)?;
            for item in list {
                try_walk_expr_shallow(item, f)?;
            }
            Ok(())
        }
        Expr::Between {
            expr: target,
            low,
            high,
            ..
        } => {
            try_walk_expr_shallow(target, f)?;
            try_walk_expr_shallow(low, f)?;
            try_walk_expr_shallow(high, f)
        }
        Expr::Function(func) => {
            if let sqlparser::ast::FunctionArguments::List(arg_list) = &func.args {
                for arg in &arg_list.args {
                    if let sqlparser::ast::FunctionArg::Unnamed(
                        sqlparser::ast::FunctionArgExpr::Expr(e),
                    )
                    | sqlparser::ast::FunctionArg::Named {
                        arg: sqlparser::ast::FunctionArgExpr::Expr(e),
                        ..
                    } = arg
                    {
                        try_walk_expr_shallow(e, f)?;
                    }
                }
            }
            Ok(())
        }
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                try_walk_expr_shallow(op, f)?;
            }
            for cond in conditions {
                try_walk_expr_shallow(cond, f)?;
            }
            for res in results {
                try_walk_expr_shallow(res, f)?;
            }
            if let Some(el) = else_result {
                try_walk_expr_shallow(el, f)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn collect_expr_column_refs(
    expr: &sqlparser::ast::Expr,
    clause_label: &str,
    out: &mut Vec<(Option<String>, String)>,
) -> Result<(), String> {
    use sqlparser::ast::Expr;
    try_walk_expr_shallow(expr, &mut |node| {
        match node {
            Expr::Identifier(ident) => {
                ensure_ident_not_unquoted_reserved(ident, clause_label)?;
                out.push((None, ident.value.clone()));
            }
            Expr::CompoundIdentifier(idents) => {
                for ident in idents {
                    ensure_ident_not_unquoted_reserved(ident, clause_label)?;
                }
                if let Some(col_ident) = idents.last() {
                    let qual = (idents.len() >= 2).then(|| idents[idents.len() - 2].value.clone());
                    out.push((qual, col_ident.value.clone()));
                }
            }
            _ => {}
        }
        Ok(())
    })
}

pub(super) fn validate_expr_subqueries<'a>(
    expr: &sqlparser::ast::Expr,
    schemas: &'a [TableSchema],
    cte_names: &BTreeSet<String>,
    outer_scope: &[(String, &'a TableSchema)],
) -> Result<(), String> {
    use sqlparser::ast::Expr;
    try_walk_expr_shallow(expr, &mut |node| match node {
        Expr::InSubquery { subquery, .. }
        | Expr::Subquery(subquery)
        | Expr::Exists { subquery, .. } => {
            validate_query_ast_with_outer_scope(subquery, schemas, cte_names, outer_scope)
        }
        _ => Ok(()),
    })
}

pub(super) fn validate_column_ref_in_scope(
    qual: Option<&str>,
    col_name: &str,
    in_scope: &[(String, &TableSchema)],
    clause_label: &str,
) -> Result<(), String> {
    if in_scope.is_empty() {
        return Ok(());
    }
    let col_low = col_name.to_ascii_lowercase();
    if let Some(q) = qual {
        let q_low = q.to_ascii_lowercase();
        let Some((_, sc)) = in_scope.iter().find(|(alias, sc)| {
            alias.eq_ignore_ascii_case(&q_low) || sc.table_name.eq_ignore_ascii_case(&q_low)
        }) else {
            return Err(format!(
                "Unknown table or alias qualifier `{q}` for column `{col_name}` in {clause_label} clause"
            ));
        };
        let valid_cols: BTreeSet<String> = sc
            .columns
            .iter()
            .map(|c| c.name.to_ascii_lowercase())
            .collect();
        if !valid_cols.contains(&col_low) {
            return Err(format!(
                "Hallucinated column `{col_name}` in {clause_label} clause for table `{}` (valid: {:?})",
                sc.table_name, valid_cols
            ));
        }
        return Ok(());
    }

    let any_matches = in_scope.iter().any(|(_, sc)| {
        sc.columns
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(&col_low))
    });
    if !any_matches {
        let primary_sc = in_scope[0].1;
        let valid_cols: BTreeSet<String> = primary_sc
            .columns
            .iter()
            .map(|c| c.name.to_ascii_lowercase())
            .collect();
        return Err(format!(
            "Hallucinated column `{col_name}` in {clause_label} clause for table `{}` (valid: {:?})",
            primary_sc.table_name, valid_cols
        ));
    }
    Ok(())
}

pub(super) fn validate_expr_refs_only(
    expr: &sqlparser::ast::Expr,
    scope: &[(String, &TableSchema)],
    select_aliases: Option<&BTreeSet<String>>,
    clause_label: &str,
) -> Result<(), String> {
    let mut refs = Vec::new();
    collect_expr_column_refs(expr, clause_label, &mut refs)?;
    for (qual, col) in refs {
        if qual.is_none() && select_aliases.is_some_and(|al| al.contains(&col.to_ascii_lowercase()))
        {
            continue;
        }
        validate_column_ref_in_scope(qual.as_deref(), &col, scope, clause_label)?;
    }
    Ok(())
}
