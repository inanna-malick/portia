//! Tokenizer-based SQL query-shape syntax verification and evaluation receipts.

use portia_nursery::nursery_scenario::SqlQueryShape;
use serde::{Deserialize, Serialize};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::tokenizer::{Token, Tokenizer};

use crate::engine::SqlRow;
use crate::grammar_trie::inspect_top_level_sql_clauses;

/// Evaluation receipt returned by [`super::evaluate_candidate_query_against_reference`] when comparing
/// a candidate SQL query against the ground-truth reference query for a [`SqlQueryShape`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlQueryShapeEvaluationReceipt {
    pub expected_shape: SqlQueryShape,
    pub reference_sql: String,
    pub candidate_sql: String,
    pub parsed_and_executed: bool,
    pub matches_shape_syntax: bool,
    pub matches_ground_truth_rows: bool,
    pub reference_row_count: usize,
    pub candidate_row_count: usize,
    pub reference_rows: Vec<SqlRow>,
    pub candidate_rows: Vec<SqlRow>,
    pub error_message: Option<String>,
}

impl SqlQueryShapeEvaluationReceipt {
    /// Returns `true` when the candidate query parsed and executed cleanly against SQLite,
    /// satisfied the structural [`SqlQueryShape`] constraints (e.g. `JOIN` / `LEFT JOIN`),
    /// and returned the exact ground-truth result set.
    pub const fn is_exact_match(&self) -> bool {
        self.parsed_and_executed && self.matches_shape_syntax && self.matches_ground_truth_rows
    }
}

fn tokenize_sql_non_whitespace(sql: &str) -> Vec<Token> {
    let dialect = SQLiteDialect {};
    Tokenizer::new(&dialect, sql)
        .tokenize()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| !matches!(t, Token::Whitespace(_) | Token::EOF))
        .collect()
}

pub(in crate::roguelite_dungeon) fn sql_tokens_reference_identifier(
    sql: &str,
    target_ident: &str,
) -> bool {
    tokenize_sql_non_whitespace(sql)
        .into_iter()
        .any(|tok| match tok {
            Token::Word(w) if w.quote_style != Some('\'') => {
                w.value.eq_ignore_ascii_case(target_ident)
            }
            _ => false,
        })
}

/// Check whether `candidate_sql` satisfies the syntactic structure required by `expected_shape`
/// using `inspect_top_level_sql_clauses` and `sqlparser::tokenizer::Tokenizer` (`SQLiteDialect`),
/// ensuring string literals (`'...'`) never spoof SQL clause or operator classification.
pub fn candidate_sql_matches_shape(candidate_sql: &str, expected_shape: SqlQueryShape) -> bool {
    let clauses = inspect_top_level_sql_clauses(candidate_sql, &[]);
    if clauses.statement_keyword.as_deref() != Some("SELECT") {
        return false;
    }

    let tokens = tokenize_sql_non_whitespace(candidate_sql);
    let mut before_from = true;
    let mut in_where = false;
    let mut has_count_call = false;
    let mut has_where_eq = false;
    let mut has_where_range_or_like = false;

    for (idx, tok) in tokens.iter().enumerate() {
        match tok {
            Token::Word(w) if w.quote_style.is_none() => {
                if w.value.eq_ignore_ascii_case("FROM") {
                    before_from = false;
                } else if w.value.eq_ignore_ascii_case("WHERE") {
                    before_from = false;
                    in_where = true;
                } else if w.value.eq_ignore_ascii_case("LIMIT")
                    || ((w.value.eq_ignore_ascii_case("GROUP")
                        || w.value.eq_ignore_ascii_case("ORDER"))
                        && tokens.get(idx + 1).is_some_and(|next| {
                            matches!(
                                next,
                                Token::Word(nw)
                                    if nw.quote_style.is_none()
                                        && nw.value.eq_ignore_ascii_case("BY")
                            )
                        }))
                {
                    in_where = false;
                } else if before_from
                    && w.value.eq_ignore_ascii_case("COUNT")
                    && matches!(tokens.get(idx + 1), Some(Token::LParen))
                {
                    has_count_call = true;
                } else if in_where && w.value.eq_ignore_ascii_case("LIKE") {
                    has_where_range_or_like = true;
                }
            }
            Token::Eq if in_where => {
                has_where_eq = true;
            }
            Token::Neq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq if in_where => {
                has_where_range_or_like = true;
            }
            _ => {}
        }
    }

    match expected_shape {
        SqlQueryShape::PointLookupEq => {
            clauses.has_where_keyword
                && clauses.has_where_comparison_or_like
                && has_where_eq
                && !clauses.has_join_keyword
                && !clauses.has_aggregate_call
                && !has_where_range_or_like
                && !clauses.has_order_by_clause
                && !clauses.has_limit_keyword
        }
        SqlQueryShape::ColumnProjectionRange => {
            !clauses.has_select_wildcard
                && !clauses.has_join_keyword
                && !clauses.has_aggregate_call
                && clauses.has_where_keyword
                && (has_where_range_or_like
                    || clauses.has_order_by_clause
                    || clauses.has_limit_keyword)
        }
        SqlQueryShape::AggregateCountWhere => {
            clauses.has_aggregate_call
                && has_count_call
                && clauses.has_where_keyword
                && !clauses.has_join_keyword
        }
        SqlQueryShape::InnerJoinOnForeignKey => {
            clauses.has_join_keyword && !clauses.has_left_join && clauses.has_on_keyword
        }
        SqlQueryShape::LeftJoinWithFilter => {
            clauses.has_join_keyword
                && clauses.has_left_join
                && clauses.has_on_keyword
                && clauses.has_where_keyword
        }
    }
}
