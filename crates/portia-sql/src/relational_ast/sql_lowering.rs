//! Stack-safe `sqlparser` expression and `WHERE`-clause lowering into [`SqlPredicateAst`].

use portia_core::hylo::{try_expand_and_collapse, Expandable, PartiallyApplied};
use sqlparser::dialect::SQLiteDialect;

use super::ident::{SqlColumnRef, SqlIdent};
use super::predicate::{SqlPredicateAst, SqlPredicateFrame};
use super::scalar_ops::{SqlComparisonOp, SqlScalarValue};

impl SqlPredicateAst {
    /// Unfold and collapse a [`sqlparser::ast::Expr`] into a [`SqlPredicateAst`] via stack-safe
    /// [`try_expand_and_collapse`], preserving structured [`SqlIdent`] quote metadata and
    /// `table_qualifier` vs. `column` boundaries inside [`SqlColumnRef`] without `.trim()`.
    pub fn from_sqlparser_expr(expr: &sqlparser::ast::Expr) -> Result<Self, String> {
        fn extract_column_ident(mut e: &sqlparser::ast::Expr) -> Result<SqlColumnRef, String> {
            while let sqlparser::ast::Expr::Nested(inner) = e {
                e = inner.as_ref();
            }
            match e {
                sqlparser::ast::Expr::Identifier(ident) if ident.quote_style != Some('\'') => {
                    Ok(SqlColumnRef::unqualified(SqlIdent::from_sqlparser_ident(
                        ident,
                    )))
                }
                sqlparser::ast::Expr::CompoundIdentifier(idents) => match idents.as_slice() {
                    [col_ident] if col_ident.quote_style != Some('\'') => Ok(
                        SqlColumnRef::unqualified(SqlIdent::from_sqlparser_ident(col_ident)),
                    ),
                    [tbl_ident, col_ident]
                        if tbl_ident.quote_style != Some('\'')
                            && col_ident.quote_style != Some('\'') =>
                    {
                        Ok(SqlColumnRef::qualified(
                            SqlIdent::from_sqlparser_ident(tbl_ident),
                            SqlIdent::from_sqlparser_ident(col_ident),
                        ))
                    }
                    _ => Err(format!(
                        "expected `<column>` or `<table>.<column>` identifier in predicate, found compound expression: {e}"
                    )),
                },
                other => Err(format!(
                    "expected column identifier in predicate, found expression: {other}"
                )),
            }
        }

        enum PredicateUnfoldSeed<'a> {
            Expr {
                expr: &'a sqlparser::ast::Expr,
                unwrapping_negated: bool,
            },
            SyntheticComparison {
                column: SqlColumnRef,
                op: SqlComparisonOp,
                literal: SqlScalarValue,
            },
        }

        try_expand_and_collapse::<
            SqlPredicateFrame<PartiallyApplied>,
            PredicateUnfoldSeed<'_>,
            SqlPredicateAst,
            String,
        >(
            PredicateUnfoldSeed::Expr {
                expr,
                unwrapping_negated: false,
            },
            |seed| match seed {
                PredicateUnfoldSeed::SyntheticComparison {
                    column,
                    op,
                    literal,
                } => Ok(SqlPredicateFrame::Comparison {
                    column,
                    op,
                    literal,
                }),
                PredicateUnfoldSeed::Expr {
                    expr: mut cur,
                    unwrapping_negated,
                } => {
                    if !unwrapping_negated {
                        while let sqlparser::ast::Expr::Nested(inner) = cur {
                            cur = inner.as_ref();
                        }
                    }
                    match cur {
                        sqlparser::ast::Expr::BinaryOp { left, op, right } => match op {
                            sqlparser::ast::BinaryOperator::And => Ok(SqlPredicateFrame::And(
                                PredicateUnfoldSeed::Expr {
                                    expr: left.as_ref(),
                                    unwrapping_negated: false,
                                },
                                PredicateUnfoldSeed::Expr {
                                    expr: right.as_ref(),
                                    unwrapping_negated: false,
                                },
                            )),
                            sqlparser::ast::BinaryOperator::Or => Ok(SqlPredicateFrame::Or(
                                PredicateUnfoldSeed::Expr {
                                    expr: left.as_ref(),
                                    unwrapping_negated: false,
                                },
                                PredicateUnfoldSeed::Expr {
                                    expr: right.as_ref(),
                                    unwrapping_negated: false,
                                },
                            )),
                            sqlparser::ast::BinaryOperator::Eq
                            | sqlparser::ast::BinaryOperator::NotEq
                            | sqlparser::ast::BinaryOperator::Lt
                            | sqlparser::ast::BinaryOperator::Gt
                            | sqlparser::ast::BinaryOperator::LtEq
                            | sqlparser::ast::BinaryOperator::GtEq => {
                                let forward_op = match op {
                                    sqlparser::ast::BinaryOperator::Eq => SqlComparisonOp::Eq,
                                    sqlparser::ast::BinaryOperator::NotEq => SqlComparisonOp::Ne,
                                    sqlparser::ast::BinaryOperator::Lt => SqlComparisonOp::Lt,
                                    sqlparser::ast::BinaryOperator::Gt => SqlComparisonOp::Gt,
                                    sqlparser::ast::BinaryOperator::LtEq => SqlComparisonOp::Le,
                                    sqlparser::ast::BinaryOperator::GtEq => SqlComparisonOp::Ge,
                                    _ => unreachable!(),
                                };
                                let reversed_op = match op {
                                    sqlparser::ast::BinaryOperator::Eq => SqlComparisonOp::Eq,
                                    sqlparser::ast::BinaryOperator::NotEq => SqlComparisonOp::Ne,
                                    sqlparser::ast::BinaryOperator::Lt => SqlComparisonOp::Gt,
                                    sqlparser::ast::BinaryOperator::Gt => SqlComparisonOp::Lt,
                                    sqlparser::ast::BinaryOperator::LtEq => SqlComparisonOp::Ge,
                                    sqlparser::ast::BinaryOperator::GtEq => SqlComparisonOp::Le,
                                    _ => unreachable!(),
                                };
                                if let (Ok(column), Ok(literal)) = (
                                    extract_column_ident(left),
                                    SqlScalarValue::from_sqlparser_expr(right),
                                ) {
                                    Ok(SqlPredicateFrame::Comparison {
                                        column,
                                        op: forward_op,
                                        literal,
                                    })
                                } else if let (Ok(literal), Ok(column)) = (
                                    SqlScalarValue::from_sqlparser_expr(left),
                                    extract_column_ident(right),
                                ) {
                                    Ok(SqlPredicateFrame::Comparison {
                                        column,
                                        op: reversed_op,
                                        literal,
                                    })
                                } else {
                                    let column = extract_column_ident(left)?;
                                    let literal = SqlScalarValue::from_sqlparser_expr(right)?;
                                    Ok(SqlPredicateFrame::Comparison {
                                        column,
                                        op: forward_op,
                                        literal,
                                    })
                                }
                            }
                            other => {
                                Err(format!("unsupported binary operator in predicate: {other}"))
                            }
                        },
                        sqlparser::ast::Expr::UnaryOp {
                            op: sqlparser::ast::UnaryOperator::Not,
                            expr: inner,
                        } => Ok(SqlPredicateFrame::Not(PredicateUnfoldSeed::Expr {
                            expr: inner.as_ref(),
                            unwrapping_negated: false,
                        })),
                        sqlparser::ast::Expr::IsNull(inner) => Ok(SqlPredicateFrame::IsNull {
                            column: extract_column_ident(inner)?,
                            negated: false,
                        }),
                        sqlparser::ast::Expr::IsNotNull(inner) => Ok(SqlPredicateFrame::IsNull {
                            column: extract_column_ident(inner)?,
                            negated: true,
                        }),
                        sqlparser::ast::Expr::InList {
                            expr: col_expr,
                            list,
                            negated,
                        } => {
                            if *negated && !unwrapping_negated {
                                Ok(SqlPredicateFrame::Not(PredicateUnfoldSeed::Expr {
                                    expr: cur,
                                    unwrapping_negated: true,
                                }))
                            } else {
                                let literals = list
                                    .iter()
                                    .map(SqlScalarValue::from_sqlparser_expr)
                                    .collect::<Result<Vec<_>, _>>()?;
                                Ok(SqlPredicateFrame::InList {
                                    column: extract_column_ident(col_expr)?,
                                    literals,
                                })
                            }
                        }
                        sqlparser::ast::Expr::Between {
                            expr: col_expr,
                            negated,
                            low,
                            high,
                        } => {
                            let column = extract_column_ident(col_expr)?;
                            let low_lit = SqlScalarValue::from_sqlparser_expr(low)?;
                            let high_lit = SqlScalarValue::from_sqlparser_expr(high)?;
                            if *negated && !unwrapping_negated {
                                Ok(SqlPredicateFrame::Not(PredicateUnfoldSeed::Expr {
                                    expr: cur,
                                    unwrapping_negated: true,
                                }))
                            } else {
                                Ok(SqlPredicateFrame::And(
                                    PredicateUnfoldSeed::SyntheticComparison {
                                        column: column.clone(),
                                        op: SqlComparisonOp::Ge,
                                        literal: low_lit,
                                    },
                                    PredicateUnfoldSeed::SyntheticComparison {
                                        column,
                                        op: SqlComparisonOp::Le,
                                        literal: high_lit,
                                    },
                                ))
                            }
                        }
                        other => Err(format!("unsupported SQL predicate expression: {other}")),
                    }
                }
            },
            |frame| Ok(SqlPredicateAst::from_frame(frame)),
        )
    }

    /// Parse a SQL boolean expression (or `WHERE` / `SELECT` / `UPDATE` / `DELETE` filter) into a
    /// [`SqlPredicateAst`], inverse to [`Self::render_sql`].
    pub fn from_sql_str(sql_or_expr: &str) -> Result<Self, String> {
        Self::parse_sql_where(sql_or_expr)
    }

    /// Parse a SQL `SELECT` / `UPDATE` / `DELETE` statement's `WHERE` clause (or a standalone
    /// SQL boolean expression) into a [`SqlPredicateAst`] via `sqlparser::parser::Parser` and
    /// [`Self::from_sqlparser_expr`] (`try_expand_and_collapse`).
    pub fn parse_sql_where(sql_or_expr: &str) -> Result<Self, String> {
        let trimmed = sql_or_expr.trim().trim_end_matches(';').trim();
        if trimmed.is_empty() {
            return Err("cannot parse empty SQL WHERE predicate".to_string());
        }
        let dialect = SQLiteDialect {};
        let leading_unquoted_keyword = sqlparser::tokenizer::Tokenizer::new(&dialect, trimmed)
            .tokenize()
            .ok()
            .and_then(|toks| {
                toks.into_iter().find_map(|tok| match tok {
                    sqlparser::tokenizer::Token::Whitespace(_) => None,
                    sqlparser::tokenizer::Token::Word(w) if w.quote_style.is_none() => {
                        Some(w.value.to_ascii_uppercase())
                    }
                    _ => Some(String::new()),
                })
            });
        if matches!(
            leading_unquoted_keyword.as_deref(),
            Some("SELECT" | "UPDATE" | "DELETE")
        ) {
            let stmts = sqlparser::parser::Parser::parse_sql(&dialect, trimmed)
                .map_err(|e| format!("failed to parse SQL statement `{sql_or_expr}`: {e}"))?;
            if stmts.len() != 1 {
                return Err(format!(
                    "expected a single SQL statement in `{sql_or_expr}`, got {}",
                    stmts.len()
                ));
            }
            match &stmts[0] {
                sqlparser::ast::Statement::Query(q) => {
                    if let sqlparser::ast::SetExpr::Select(sel) = q.body.as_ref() {
                        if let Some(selection) = &sel.selection {
                            return Self::from_sqlparser_expr(selection);
                        }
                    }
                }
                sqlparser::ast::Statement::Update {
                    selection: Some(sel),
                    ..
                } => {
                    return Self::from_sqlparser_expr(sel);
                }
                sqlparser::ast::Statement::Delete(del) => {
                    if let Some(sel) = &del.selection {
                        return Self::from_sqlparser_expr(sel);
                    }
                }
                _ => {}
            }
            return Err(format!(
                "SQL statement `{sql_or_expr}` does not contain a WHERE clause"
            ));
        }
        let expr_body = if leading_unquoted_keyword.as_deref() == Some("WHERE") {
            trimmed.get(5..).map(str::trim).unwrap_or(trimmed)
        } else {
            trimmed
        };
        let mut parser = sqlparser::parser::Parser::new(&dialect)
            .try_with_sql(expr_body)
            .map_err(|e| format!("failed to tokenize SQL predicate `{sql_or_expr}`: {e}"))?;
        let expr = parser
            .parse_expr()
            .map_err(|e| format!("failed to parse SQL predicate `{sql_or_expr}`: {e}"))?;
        while parser.consume_token(&sqlparser::tokenizer::Token::SemiColon) {}
        let next_tok = parser.peek_token();
        if next_tok.token != sqlparser::tokenizer::Token::EOF {
            return Err(format!(
                "unexpected trailing token `{}` after SQL predicate in `{sql_or_expr}`",
                next_tok.token
            ));
        }
        Self::from_sqlparser_expr(&expr)
    }
}
