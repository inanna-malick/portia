//! Kleene three-valued row and `CHECK` constraint evaluation for [`SqlPredicateAst`].

use std::collections::BTreeMap;

use portia_core::hylo::CollapsibleExt;

use super::coercion::{
    lookup_row_column_value, try_coerce_sql_cell_to_f64, try_sql_cell_equals_bool_literal,
    try_sql_cell_equals_string_literal, SqlRowDomainError,
};
use super::predicate::{SqlPredicateAst, SqlPredicateFrame, SqlTruthValue};
use super::scalar_ops::{SqlComparisonOp, SqlScalarValue};

impl SqlPredicateAst {
    /// Fallible Kleene three-valued SQL evaluator that enforces the input type/coercion
    /// contract across [`SqlPredicateAst`] row and `CHECK` evaluation:
    /// missing columns or `serde_json::Value::Null` produce `Ok(SqlTruthValue::Unknown)`, whereas
    /// non-null cells incompatible with a clause's domain (e.g. `"OPEN"` or `[1, 2]` on an integer
    /// comparison `x >= 1`) return an explicit `Err(SqlRowDomainError)` rather than being silently
    /// reclassified as database `NULL`.
    pub fn try_evaluate_row_three_valued(
        &self,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> Result<SqlTruthValue, SqlRowDomainError> {
        self.try_collapse_frames(|frame| match frame {
            SqlPredicateFrame::Comparison {
                column,
                op,
                literal,
            } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return Ok(SqlTruthValue::Unknown);
                };
                let col_label = column.qualified_or_column_name();
                try_evaluate_sql_value_comparison_three_valued(&col_label, val, op, &literal)
            }
            SqlPredicateFrame::IsNull { column, negated } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return Ok(SqlTruthValue::Unknown);
                };
                let is_null = val.is_null();
                let res = if negated { !is_null } else { is_null };
                Ok(if res {
                    SqlTruthValue::True
                } else {
                    SqlTruthValue::False
                })
            }
            SqlPredicateFrame::InList { column, literals } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return Ok(SqlTruthValue::Unknown);
                };
                if val.is_null() {
                    return Ok(SqlTruthValue::Unknown);
                }
                let col_label = column.qualified_or_column_name();
                let mut saw_true = false;
                let mut saw_unknown = false;
                for lit in &literals {
                    match try_evaluate_sql_value_comparison_three_valued(
                        &col_label,
                        val,
                        SqlComparisonOp::Eq,
                        lit,
                    )? {
                        SqlTruthValue::True => saw_true = true,
                        SqlTruthValue::Unknown => saw_unknown = true,
                        SqlTruthValue::False => {}
                    }
                }
                if saw_true {
                    Ok(SqlTruthValue::True)
                } else if saw_unknown {
                    Ok(SqlTruthValue::Unknown)
                } else {
                    Ok(SqlTruthValue::False)
                }
            }
            SqlPredicateFrame::And(left, right) => Ok(left.and(right)),
            SqlPredicateFrame::Or(left, right) => Ok(left.or(right)),
            SqlPredicateFrame::Not(inner) => Ok(inner.negate()),
        })
    }

    /// Evaluate this SQL predicate AST under Kleene three-valued SQL logic (`True`, `False`,
    /// `Unknown`) against a row map via [`CollapsibleExt::collapse_frames`].
    ///
    /// - Missing columns or `NULL` operands in `=`, `!=`, `<`, `>`, `<=`, `>=` yield [`SqlTruthValue::Unknown`].
    /// - `NOT (Unknown)` remains [`SqlTruthValue::Unknown`].
    /// - `IS NULL` / `IS NOT NULL` evaluate to `True`/`False` when the column is present in `row`,
    ///   and [`SqlTruthValue::Unknown`] if the column is missing from `row`.
    pub fn evaluate_row_three_valued(
        &self,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> SqlTruthValue {
        self.collapse_frames(|frame| match frame {
            SqlPredicateFrame::Comparison {
                column,
                op,
                literal,
            } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return SqlTruthValue::Unknown;
                };
                evaluate_sql_value_comparison_three_valued(val, op, &literal)
            }
            SqlPredicateFrame::IsNull { column, negated } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return SqlTruthValue::Unknown;
                };
                let is_null = val.is_null();
                let res = if negated { !is_null } else { is_null };
                if res {
                    SqlTruthValue::True
                } else {
                    SqlTruthValue::False
                }
            }
            SqlPredicateFrame::InList { column, literals } => {
                let Some(val) = lookup_row_column_value(row, &column) else {
                    return SqlTruthValue::Unknown;
                };
                if val.is_null() {
                    return SqlTruthValue::Unknown;
                }
                let mut saw_unknown = false;
                for lit in &literals {
                    match evaluate_sql_value_comparison_three_valued(val, SqlComparisonOp::Eq, lit)
                    {
                        SqlTruthValue::True => return SqlTruthValue::True,
                        SqlTruthValue::Unknown => saw_unknown = true,
                        SqlTruthValue::False => {}
                    }
                }
                if saw_unknown {
                    SqlTruthValue::Unknown
                } else {
                    SqlTruthValue::False
                }
            }
            SqlPredicateFrame::And(left, right) => left.and(right),
            SqlPredicateFrame::Or(left, right) => left.or(right),
            SqlPredicateFrame::Not(inner) => inner.negate(),
        })
    }

    /// Evaluate this SQL predicate AST for `WHERE` row selection (`self.evaluate_row_three_valued(row) == SqlTruthValue::True`,
    /// where `Unknown` / `NULL` does **not** select the row).
    pub fn evaluate_row(&self, row: &BTreeMap<String, serde_json::Value>) -> bool {
        self.evaluate_row_three_valued(row) == SqlTruthValue::True
    }

    /// Fallibly evaluate this SQL predicate AST for SQLite `CHECK (...)` constraint satisfaction:
    /// returns `Ok(truth != SqlTruthValue::False)` when all non-null cells satisfy their column
    /// domain contracts (so both `True` and SQL `NULL` / `Unknown` satisfy `CHECK`), and returns
    /// `Err(SqlRowDomainError)` on domain violations (such as `"OPEN"` or `[1, 2]` in numeric
    /// `CHECK (balance >= 0)`).
    pub fn try_evaluate_check_constraint(
        &self,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> Result<bool, SqlRowDomainError> {
        self.try_evaluate_row_three_valued(row)
            .map(|truth| truth != SqlTruthValue::False)
    }

    /// Ergonomic alias for [`Self::try_evaluate_check_constraint`].
    pub fn try_evaluate_row_for_check(
        &self,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> Result<bool, SqlRowDomainError> {
        self.try_evaluate_check_constraint(row)
    }

    /// Evaluate this SQL predicate AST for SQLite `CHECK (...)` constraint satisfaction, failing
    /// closed (`false`) on incompatible cell domain errors (`Err(SqlRowDomainError)`) while
    /// preserving SQL's `NULL`-pass semantics (`Ok(SqlTruthValue::Unknown)` -> `true`).
    pub fn evaluate_check_constraint(&self, row: &BTreeMap<String, serde_json::Value>) -> bool {
        self.try_evaluate_check_constraint(row).unwrap_or(false)
    }

    /// Evaluate this SQL predicate AST for SQLite `CHECK (...)` constraint satisfaction, failing
    /// closed (`false`) on incompatible cell domain errors (`Err(SqlRowDomainError)`) while
    /// preserving SQL's `NULL`-pass semantics (`Ok(SqlTruthValue::Unknown)` -> `true`).
    pub fn evaluate_row_for_check(&self, row: &BTreeMap<String, serde_json::Value>) -> bool {
        self.evaluate_check_constraint(row)
    }
}

fn try_evaluate_sql_value_comparison_three_valued(
    column: &str,
    val: &serde_json::Value,
    op: SqlComparisonOp,
    literal: &SqlScalarValue,
) -> Result<SqlTruthValue, SqlRowDomainError> {
    if val.is_null() {
        return Ok(SqlTruthValue::Unknown);
    }
    match literal {
        SqlScalarValue::Null => Ok(SqlTruthValue::Unknown),
        SqlScalarValue::Int(rhs_i) => {
            if let serde_json::Value::Number(num) = val {
                if let Some(lhs_i) = num.as_i64() {
                    return Ok(if compare_ord(lhs_i, *rhs_i, op) {
                        SqlTruthValue::True
                    } else {
                        SqlTruthValue::False
                    });
                }
            }
            let lhs_f = try_coerce_sql_cell_to_f64(column, val)?;
            Ok(if compare_f64(lhs_f, *rhs_i as f64, op) {
                SqlTruthValue::True
            } else {
                SqlTruthValue::False
            })
        }
        SqlScalarValue::Float(rhs_f) => {
            let lhs_f = try_coerce_sql_cell_to_f64(column, val)?;
            Ok(if compare_f64(lhs_f, *rhs_f, op) {
                SqlTruthValue::True
            } else {
                SqlTruthValue::False
            })
        }
        SqlScalarValue::Bool(expected_bool) => {
            let eq = try_sql_cell_equals_bool_literal(column, val, *expected_bool)?;
            let matched = match op {
                SqlComparisonOp::Eq => eq,
                SqlComparisonOp::Ne => !eq,
                _ => {
                    return Err(SqlRowDomainError {
                        column: column.to_string(),
                        expected_domain: "equality operator (= or !=) on boolean literal"
                            .to_string(),
                        actual_value: op.as_sql_str().to_string(),
                    })
                }
            };
            Ok(if matched {
                SqlTruthValue::True
            } else {
                SqlTruthValue::False
            })
        }
        SqlScalarValue::Text(lit_str) => match op {
            SqlComparisonOp::Eq => {
                if try_sql_cell_equals_string_literal(column, val, lit_str)? {
                    Ok(SqlTruthValue::True)
                } else {
                    Ok(SqlTruthValue::False)
                }
            }
            SqlComparisonOp::Ne => {
                if !try_sql_cell_equals_string_literal(column, val, lit_str)? {
                    Ok(SqlTruthValue::True)
                } else {
                    Ok(SqlTruthValue::False)
                }
            }
            _ => {
                let matched = match val {
                    serde_json::Value::String(s) => compare_ord(s.as_str(), lit_str.as_str(), op),
                    serde_json::Value::Number(num) => {
                        if let (Some(lhs_f), Ok(rhs_f)) = (num.as_f64(), lit_str.parse::<f64>()) {
                            compare_f64(lhs_f, rhs_f, op)
                        } else {
                            return Err(SqlRowDomainError {
                                column: column.to_string(),
                                expected_domain: "comparable string or numeric literal".to_string(),
                                actual_value: val.to_string(),
                            });
                        }
                    }
                    _ => {
                        return Err(SqlRowDomainError {
                            column: column.to_string(),
                            expected_domain: "comparable string or numeric cell".to_string(),
                            actual_value: val.to_string(),
                        });
                    }
                };
                Ok(if matched {
                    SqlTruthValue::True
                } else {
                    SqlTruthValue::False
                })
            }
        },
    }
}

fn evaluate_sql_value_comparison_three_valued(
    val: &serde_json::Value,
    op: SqlComparisonOp,
    literal: &SqlScalarValue,
) -> SqlTruthValue {
    try_evaluate_sql_value_comparison_three_valued("", val, op, literal)
        .unwrap_or(SqlTruthValue::Unknown)
}

fn compare_ord<T: Ord>(lhs: T, rhs: T, op: SqlComparisonOp) -> bool {
    match op {
        SqlComparisonOp::Eq => lhs == rhs,
        SqlComparisonOp::Ne => lhs != rhs,
        SqlComparisonOp::Lt => lhs < rhs,
        SqlComparisonOp::Gt => lhs > rhs,
        SqlComparisonOp::Le => lhs <= rhs,
        SqlComparisonOp::Ge => lhs >= rhs,
    }
}

fn compare_f64(lhs: f64, rhs: f64, op: SqlComparisonOp) -> bool {
    match op {
        SqlComparisonOp::Eq => lhs == rhs,
        SqlComparisonOp::Ne => lhs != rhs,
        SqlComparisonOp::Lt => lhs < rhs,
        SqlComparisonOp::Gt => lhs > rhs,
        SqlComparisonOp::Le => lhs <= rhs,
        SqlComparisonOp::Ge => lhs >= rhs,
    }
}
