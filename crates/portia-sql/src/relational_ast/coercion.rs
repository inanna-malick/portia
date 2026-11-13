//! Row-cell lookup and scalar domain coercion helpers for Kleene three-valued evaluation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ident::SqlColumnRef;

pub(super) fn lookup_row_column_value<'a>(
    row: &'a BTreeMap<String, serde_json::Value>,
    col_ref: &SqlColumnRef,
) -> Option<&'a serde_json::Value> {
    if let Some(qual_key) = col_ref.qualified_key() {
        if let Some(v) = row.get(&qual_key) {
            return Some(v);
        }
        if let Some((_, v)) = row.iter().find(|(k, _)| k.eq_ignore_ascii_case(&qual_key)) {
            return Some(v);
        }
    }
    if let Some(v) = row.get(&col_ref.column.name) {
        return Some(v);
    }
    row.iter()
        .find(|(k, _)| col_ref.column.matches_ascii_case_insensitive(k))
        .map(|(_, v)| v)
}

/// Explicit row-cell domain/coercion error returned when a non-null JSON cell in a named row is
/// incompatible with the operator/literal domain of a SQL predicate clause (for example, a
/// non-numeric string `"OPEN"` or JSON array `[1, 2]` compared against an integer threshold
/// `x >= 1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlRowDomainError {
    pub column: String,
    pub expected_domain: String,
    pub actual_value: String,
}

impl std::fmt::Display for SqlRowDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SQL row domain error on column `{}`: expected {}, got {}",
            self.column, self.expected_domain, self.actual_value
        )
    }
}

impl std::error::Error for SqlRowDomainError {}

/// Canonical numeric cell coercion for [`super::SqlPredicateAst`]: extracts a finite `f64` from
/// `serde_json::Value::Number`, numeric `serde_json::Value::String`, or `serde_json::Value::Bool`.
fn coerce_sql_cell_to_f64(val: &serde_json::Value) -> Option<f64> {
    match val {
        serde_json::Value::Number(num) => num.as_f64().filter(|v| v.is_finite()),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok().filter(|v| v.is_finite()),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Fallible numeric cell coercion returning an explicit [`SqlRowDomainError`] when a non-null cell
/// cannot be coerced to a finite number.
pub(super) fn try_coerce_sql_cell_to_f64(
    column: &str,
    val: &serde_json::Value,
) -> Result<f64, SqlRowDomainError> {
    coerce_sql_cell_to_f64(val).ok_or_else(|| SqlRowDomainError {
        column: column.to_string(),
        expected_domain: "finite numeric value (Number, numeric String, or Bool)".to_string(),
        actual_value: val.to_string(),
    })
}

/// Canonical string-equality coercion for [`super::SqlPredicateAst`] scalar comparisons.
fn sql_cell_equals_string_literal(val: &serde_json::Value, expected: &str) -> bool {
    match val {
        serde_json::Value::String(s) => s == expected,
        serde_json::Value::Number(num) => {
            if let (Some(lhs_f), Ok(rhs_f)) = (num.as_f64(), expected.trim().parse::<f64>()) {
                lhs_f == rhs_f
            } else {
                num.to_string() == expected
            }
        }
        serde_json::Value::Bool(b) => {
            (*b && (expected.eq_ignore_ascii_case("true") || expected == "1"))
                || (!*b && (expected.eq_ignore_ascii_case("false") || expected == "0"))
        }
        _ => false,
    }
}

/// Fallible string-literal equality coercion returning an explicit [`SqlRowDomainError`] on
/// non-scalar JSON values (`Array` / `Object`).
pub(super) fn try_sql_cell_equals_string_literal(
    column: &str,
    val: &serde_json::Value,
    expected: &str,
) -> Result<bool, SqlRowDomainError> {
    match val {
        serde_json::Value::String(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::Bool(_) => Ok(sql_cell_equals_string_literal(val, expected)),
        other => Err(SqlRowDomainError {
            column: column.to_string(),
            expected_domain: "scalar string/number/bool cell".to_string(),
            actual_value: other.to_string(),
        }),
    }
}

/// Fallible boolean-literal equality coercion returning an explicit [`SqlRowDomainError`] when a
/// non-null cell is not a valid boolean representation (`Bool`, `0`/`1`, or `"true"`/`"false"`/`"1"`/`"0"`).
pub(super) fn try_sql_cell_equals_bool_literal(
    column: &str,
    val: &serde_json::Value,
    expected: bool,
) -> Result<bool, SqlRowDomainError> {
    match val {
        serde_json::Value::Bool(b) => Ok(*b == expected),
        serde_json::Value::Number(num) => {
            if let Some(lhs_f) = num.as_f64() {
                if (lhs_f - 1.0).abs() <= f64::EPSILON {
                    return Ok(expected);
                } else if lhs_f.abs() <= f64::EPSILON {
                    return Ok(!expected);
                }
            }
            Err(SqlRowDomainError {
                column: column.to_string(),
                expected_domain: "boolean cell (true/false/1/0)".to_string(),
                actual_value: val.to_string(),
            })
        }
        serde_json::Value::String(s) => {
            if s.eq_ignore_ascii_case("true") || s == "1" {
                Ok(expected)
            } else if s.eq_ignore_ascii_case("false") || s == "0" {
                Ok(!expected)
            } else {
                Err(SqlRowDomainError {
                    column: column.to_string(),
                    expected_domain: "boolean cell (true/false/1/0)".to_string(),
                    actual_value: val.to_string(),
                })
            }
        }
        other => Err(SqlRowDomainError {
            column: column.to_string(),
            expected_domain: "boolean cell (true/false/1/0)".to_string(),
            actual_value: other.to_string(),
        }),
    }
}
