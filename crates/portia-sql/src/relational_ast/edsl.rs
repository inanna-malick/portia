//! Compositional SQL `WHERE` predicate eDSL (`col`, `qualified_col`, `&`, `|`, `!`).

use serde::{Deserialize, Serialize};

use super::ident::{SqlColumnRef, SqlIdent};
use super::predicate::SqlPredicateAst;
use super::scalar_ops::{SqlComparisonOp, SqlScalarValue};

/// Typed column handle for the compositional SQL `WHERE` eDSL (`col("delay_minutes").ge(30)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlColumnExpr {
    pub column: SqlColumnRef,
}

impl From<SqlColumnRef> for SqlColumnExpr {
    fn from(column: SqlColumnRef) -> Self {
        Self { column }
    }
}

/// Construct a compositional SQL column expression handle (`col("delay_minutes")`).
pub fn col(column: impl Into<SqlColumnRef>) -> SqlColumnExpr {
    SqlColumnExpr {
        column: column.into(),
    }
}

/// Construct a table-qualified compositional SQL column expression handle
/// (`qualified_col("t", "a b")` -> `t."a b"`).
pub fn qualified_col(table: impl Into<SqlIdent>, column: impl Into<SqlIdent>) -> SqlColumnExpr {
    SqlColumnExpr {
        column: SqlColumnRef::qualified(table, column),
    }
}

impl SqlColumnExpr {
    pub fn eq_int(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Eq, SqlScalarValue::Int(val))
    }

    pub fn ne_int(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Ne, SqlScalarValue::Int(val))
    }

    pub fn lt(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Lt, SqlScalarValue::Int(val))
    }

    pub fn le(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Le, SqlScalarValue::Int(val))
    }

    pub fn gt(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Gt, SqlScalarValue::Int(val))
    }

    pub fn ge(self, val: i64) -> SqlPredicateAst {
        SqlPredicateAst::comparison(self.column, SqlComparisonOp::Ge, SqlScalarValue::Int(val))
    }

    pub fn eq_str(self, val: &str) -> SqlPredicateAst {
        SqlPredicateAst::comparison(
            self.column,
            SqlComparisonOp::Eq,
            SqlScalarValue::Text(val.to_string()),
        )
    }

    pub fn ne_str(self, val: &str) -> SqlPredicateAst {
        SqlPredicateAst::comparison(
            self.column,
            SqlComparisonOp::Ne,
            SqlScalarValue::Text(val.to_string()),
        )
    }

    pub fn eq_lit(self, literal: impl Into<String>) -> SqlPredicateAst {
        let raw = literal.into();
        SqlPredicateAst::comparison(
            self.column,
            SqlComparisonOp::Eq,
            SqlScalarValue::parse_from_sql_token_or_str(&raw),
        )
    }

    pub fn ne_lit(self, literal: impl Into<String>) -> SqlPredicateAst {
        let raw = literal.into();
        SqlPredicateAst::comparison(
            self.column,
            SqlComparisonOp::Ne,
            SqlScalarValue::parse_from_sql_token_or_str(&raw),
        )
    }

    pub fn is_null(self) -> SqlPredicateAst {
        SqlPredicateAst::is_null(self.column, false)
    }

    pub fn is_not_null(self) -> SqlPredicateAst {
        SqlPredicateAst::is_null(self.column, true)
    }

    pub fn in_strs<I, S>(self, items: I) -> SqlPredicateAst
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        SqlPredicateAst::in_list(
            self.column,
            items
                .into_iter()
                .map(|s| SqlScalarValue::Text(s.as_ref().to_string()))
                .collect(),
        )
    }

    pub fn in_ints<I>(self, items: I) -> SqlPredicateAst
    where
        I: IntoIterator<Item = i64>,
    {
        SqlPredicateAst::in_list(
            self.column,
            items.into_iter().map(SqlScalarValue::Int).collect(),
        )
    }

    pub fn in_literals(self, literals: Vec<String>) -> SqlPredicateAst {
        SqlPredicateAst::in_list(
            self.column,
            literals
                .into_iter()
                .map(|s| SqlScalarValue::parse_from_sql_token_or_str(&s))
                .collect(),
        )
    }
}

impl std::ops::BitAnd for SqlPredicateAst {
    type Output = SqlPredicateAst;

    fn bitand(self, rhs: Self) -> Self::Output {
        SqlPredicateAst::and(self, rhs)
    }
}

impl std::ops::BitOr for SqlPredicateAst {
    type Output = SqlPredicateAst;

    fn bitor(self, rhs: Self) -> Self::Output {
        SqlPredicateAst::or(self, rhs)
    }
}

impl std::ops::Not for SqlPredicateAst {
    type Output = SqlPredicateAst;

    fn not(self) -> Self::Output {
        SqlPredicateAst::not(self)
    }
}
