//! `SELECT` / `JOIN` / CTE / subquery `sqlparser::ast::Query` scope validation.

use std::collections::BTreeSet;

use portia_core::TableSchema;

use super::validate_expr::{
    ast_object_name_leaf, ensure_ident_not_unquoted_reserved,
    ensure_object_name_not_unquoted_reserved, find_table_schema, validate_column_ref_in_scope,
    validate_expr_refs_only, validate_expr_subqueries,
};

pub(super) fn collect_and_validate_table_factor<'a>(
    tf: &sqlparser::ast::TableFactor,
    schemas: &'a [TableSchema],
    cte_names: &BTreeSet<String>,
    in_scope: &mut Vec<(String, &'a TableSchema)>,
    has_untracked_source: &mut bool,
) -> Result<(), String> {
    use sqlparser::ast::TableFactor;
    match tf {
        TableFactor::Table { name, alias, .. } => {
            ensure_object_name_not_unquoted_reserved(name, "table reference")?;
            let tbl_name = ast_object_name_leaf(name);
            let tbl_low = tbl_name.to_ascii_lowercase();
            if cte_names.contains(&tbl_low) || tbl_low.starts_with("sqlite_") {
                *has_untracked_source = true;
                return Ok(());
            }
            let Some(sc) = find_table_schema(schemas, &tbl_name) else {
                return Err(format!("Unknown table `{tbl_name}`"));
            };
            let scope_key = alias
                .as_ref()
                .map(|a| a.name.value.to_ascii_lowercase())
                .unwrap_or(tbl_low);
            in_scope.push((scope_key, sc));
            Ok(())
        }
        TableFactor::Derived { subquery, .. } => {
            *has_untracked_source = true;
            validate_query_ast(subquery, schemas, cte_names)
        }
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => {
            collect_and_validate_table_factor(
                &table_with_joins.relation,
                schemas,
                cte_names,
                in_scope,
                has_untracked_source,
            )?;
            for join in &table_with_joins.joins {
                collect_and_validate_table_factor(
                    &join.relation,
                    schemas,
                    cte_names,
                    in_scope,
                    has_untracked_source,
                )?;
            }
            Ok(())
        }
        _ => {
            *has_untracked_source = true;
            Ok(())
        }
    }
}

fn validate_set_expr<'a>(
    set_expr: &sqlparser::ast::SetExpr,
    schemas: &'a [TableSchema],
    cte_names: &BTreeSet<String>,
    outer_scope: &[(String, &'a TableSchema)],
) -> Result<(), String> {
    match set_expr {
        sqlparser::ast::SetExpr::Select(select) => {
            let mut in_scope: Vec<(String, &TableSchema)> = Vec::new();
            let mut has_untracked_source = false;
            for twj in &select.from {
                collect_and_validate_table_factor(
                    &twj.relation,
                    schemas,
                    cte_names,
                    &mut in_scope,
                    &mut has_untracked_source,
                )?;
                for join in &twj.joins {
                    collect_and_validate_table_factor(
                        &join.relation,
                        schemas,
                        cte_names,
                        &mut in_scope,
                        &mut has_untracked_source,
                    )?;
                    let mut combined_scope = in_scope.clone();
                    combined_scope.extend_from_slice(outer_scope);
                    if !has_untracked_source {
                        let constraint = match &join.join_operator {
                            sqlparser::ast::JoinOperator::Inner(c)
                            | sqlparser::ast::JoinOperator::LeftOuter(c)
                            | sqlparser::ast::JoinOperator::RightOuter(c)
                            | sqlparser::ast::JoinOperator::FullOuter(c) => Some(c),
                            _ => None,
                        };
                        if let Some(sqlparser::ast::JoinConstraint::On(on_expr)) = constraint {
                            validate_expr_subqueries(on_expr, schemas, cte_names, &combined_scope)?;
                            validate_expr_refs_only(on_expr, &combined_scope, None, "JOIN ON")?;
                        } else if let Some(sqlparser::ast::JoinConstraint::Using(using_idents)) =
                            constraint
                        {
                            for ident in using_idents {
                                ensure_ident_not_unquoted_reserved(ident, "JOIN USING")?;
                                validate_column_ref_in_scope(
                                    None,
                                    &ident.value,
                                    &combined_scope,
                                    "JOIN USING",
                                )?;
                            }
                        }
                    }
                }
            }

            let mut combined_scope = in_scope.clone();
            combined_scope.extend_from_slice(outer_scope);

            for item in &select.projection {
                if let sqlparser::ast::SelectItem::UnnamedExpr(expr)
                | sqlparser::ast::SelectItem::ExprWithAlias { expr, .. } = item
                {
                    validate_expr_subqueries(expr, schemas, cte_names, &combined_scope)?;
                }
            }
            if let Some(sel) = &select.selection {
                validate_expr_subqueries(sel, schemas, cte_names, &combined_scope)?;
            }

            if !has_untracked_source && !in_scope.is_empty() {
                let mut select_aliases = BTreeSet::new();
                for item in &select.projection {
                    match item {
                        sqlparser::ast::SelectItem::UnnamedExpr(expr) => {
                            validate_expr_refs_only(expr, &combined_scope, None, "SELECT")?;
                        }
                        sqlparser::ast::SelectItem::ExprWithAlias { expr, alias } => {
                            select_aliases.insert(alias.value.to_ascii_lowercase());
                            validate_expr_refs_only(expr, &combined_scope, None, "SELECT")?;
                        }
                        _ => {}
                    }
                }
                if let Some(sel) = &select.selection {
                    validate_expr_refs_only(sel, &combined_scope, Some(&select_aliases), "WHERE")?;
                }
                if let sqlparser::ast::GroupByExpr::Expressions(group_exprs, _) = &select.group_by {
                    for g_expr in group_exprs {
                        validate_expr_subqueries(g_expr, schemas, cte_names, &combined_scope)?;
                        validate_expr_refs_only(
                            g_expr,
                            &combined_scope,
                            Some(&select_aliases),
                            "GROUP BY",
                        )?;
                    }
                }
                if let Some(having_expr) = &select.having {
                    validate_expr_subqueries(having_expr, schemas, cte_names, &combined_scope)?;
                    validate_expr_refs_only(
                        having_expr,
                        &combined_scope,
                        Some(&select_aliases),
                        "HAVING",
                    )?;
                }
            }
            Ok(())
        }
        sqlparser::ast::SetExpr::SetOperation { left, right, .. } => {
            validate_set_expr(left, schemas, cte_names, outer_scope)?;
            validate_set_expr(right, schemas, cte_names, outer_scope)
        }
        sqlparser::ast::SetExpr::Query(subquery) => {
            validate_query_ast_with_outer_scope(subquery, schemas, cte_names, outer_scope)
        }
        sqlparser::ast::SetExpr::Values(values) => {
            for row in &values.rows {
                for expr in row {
                    validate_expr_subqueries(expr, schemas, cte_names, outer_scope)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_query_ast_with_outer_scope<'a>(
    query: &sqlparser::ast::Query,
    schemas: &'a [TableSchema],
    outer_ctes: &BTreeSet<String>,
    outer_scope: &[(String, &'a TableSchema)],
) -> Result<(), String> {
    let mut cte_names = outer_ctes.clone();
    if let Some(with) = &query.with {
        for cte in &with.cte_tables {
            cte_names.insert(cte.alias.name.value.to_ascii_lowercase());
            validate_query_ast_with_outer_scope(&cte.query, schemas, &cte_names, outer_scope)?;
        }
    }
    validate_set_expr(query.body.as_ref(), schemas, &cte_names, outer_scope)?;
    if let Some(order_by) = &query.order_by {
        if let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() {
            let mut in_scope = Vec::new();
            let mut has_untracked_source = false;
            for twj in &select.from {
                collect_and_validate_table_factor(
                    &twj.relation,
                    schemas,
                    &cte_names,
                    &mut in_scope,
                    &mut has_untracked_source,
                )?;
                for join in &twj.joins {
                    collect_and_validate_table_factor(
                        &join.relation,
                        schemas,
                        &cte_names,
                        &mut in_scope,
                        &mut has_untracked_source,
                    )?;
                }
            }
            if !has_untracked_source && !in_scope.is_empty() {
                let mut combined_scope = in_scope;
                combined_scope.extend_from_slice(outer_scope);
                let mut select_aliases = BTreeSet::new();
                for item in &select.projection {
                    if let sqlparser::ast::SelectItem::ExprWithAlias { alias, .. } = item {
                        select_aliases.insert(alias.value.to_ascii_lowercase());
                    }
                }
                for ob_expr in &order_by.exprs {
                    validate_expr_subqueries(&ob_expr.expr, schemas, &cte_names, &combined_scope)?;
                    validate_expr_refs_only(
                        &ob_expr.expr,
                        &combined_scope,
                        Some(&select_aliases),
                        "ORDER BY",
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn validate_query_ast(
    query: &sqlparser::ast::Query,
    schemas: &[TableSchema],
    outer_ctes: &BTreeSet<String>,
) -> Result<(), String> {
    validate_query_ast_with_outer_scope(query, schemas, outer_ctes, &[])
}
