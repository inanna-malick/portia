//! Strongly-typed SQL comparison operators ([`SqlComparisonOp`]) and scalar literals ([`SqlScalarValue`]).

use serde::{Deserialize, Serialize};
use sqlparser::dialect::SQLiteDialect;

/// Strongly-typed SQL binary comparison operator (`=`, `!=`, `<`, `<=`, `>`, `>=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SqlComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl SqlComparisonOp {
    pub const fn as_sql_str(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    pub const fn negated(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
        }
    }

    pub const fn is_range_op(self) -> bool {
        matches!(self, Self::Lt | Self::Le | Self::Gt | Self::Ge)
    }

    pub fn parse_sql_op(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "=" | "==" => Ok(Self::Eq),
            "!=" | "<>" => Ok(Self::Ne),
            "<" => Ok(Self::Lt),
            "<=" => Ok(Self::Le),
            ">" => Ok(Self::Gt),
            ">=" => Ok(Self::Ge),
            other => Err(format!("unsupported SQL comparison operator `{other}`")),
        }
    }
}

impl std::fmt::Display for SqlComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_sql_str())
    }
}

fn normalize_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// Strongly-typed SQL scalar literal (`Int`, `Float`, `Text`, `Bool`, `Null`) stored at the
/// leaves of [`super::SqlPredicateFrame`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlScalarValue {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Null,
}

impl PartialEq for SqlScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => normalize_f64_bits(*a) == normalize_f64_bits(*b),
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Null, Self::Null) => true,
            _ => false,
        }
    }
}

impl Eq for SqlScalarValue {}

impl From<i64> for SqlScalarValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f64> for SqlScalarValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<bool> for SqlScalarValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<String> for SqlScalarValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}

impl From<&str> for SqlScalarValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}

impl SqlScalarValue {
    /// Render this typed scalar value as a canonical SQL literal.
    pub fn render_sql(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Float(f) => {
                let s = f.to_string();
                if f.is_finite() && !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    format!("{s}.0")
                } else {
                    s
                }
            }
            Self::Text(s) => format!("'{}'", s.replace('\'', "''")),
            Self::Bool(b) => {
                if *b {
                    "TRUE".to_string()
                } else {
                    "FALSE".to_string()
                }
            }
            Self::Null => "NULL".to_string(),
        }
    }

    /// Lower a [`sqlparser::ast::Expr`] literal node into a [`SqlScalarValue`].
    pub fn from_sqlparser_expr(mut expr: &sqlparser::ast::Expr) -> Result<Self, String> {
        while let sqlparser::ast::Expr::Nested(inner) = expr {
            expr = inner.as_ref();
        }
        match expr {
            sqlparser::ast::Expr::Value(sqlparser::ast::Value::SingleQuotedString(s)) => {
                Ok(Self::Text(s.clone()))
            }
            sqlparser::ast::Expr::Value(sqlparser::ast::Value::DoubleQuotedString(s)) => Err(
                format!(
                    "quoted identifier `\"{s}\"` in scalar position is a column identifier rather than a single-quoted string literal (`'{s}'`); column-to-column comparisons are unsupported"
                ),
            ),
            sqlparser::ast::Expr::Value(sqlparser::ast::Value::Boolean(b)) => Ok(Self::Bool(*b)),
            sqlparser::ast::Expr::Value(sqlparser::ast::Value::Null) => Ok(Self::Null),
            sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(n, _)) => {
                if let Ok(i) = n.parse::<i64>() {
                    Ok(Self::Int(i))
                } else if let Ok(f) = n.parse::<f64>() {
                    Ok(Self::Float(f))
                } else {
                    Err(format!("invalid numeric SQL literal `{n}`"))
                }
            }
            sqlparser::ast::Expr::UnaryOp {
                op: sqlparser::ast::UnaryOperator::Minus,
                expr: inner,
            } => {
                if let sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(n, _)) =
                    inner.as_ref()
                {
                    let signed = format!("-{n}");
                    if let Ok(i) = signed.parse::<i64>() {
                        Ok(Self::Int(i))
                    } else if let Ok(f) = signed.parse::<f64>() {
                        Ok(Self::Float(f))
                    } else {
                        Err(format!("invalid negative numeric SQL literal `{signed}`"))
                    }
                } else {
                    Err(format!("unsupported unary minus expression in SQL literal: {expr}"))
                }
            }
            sqlparser::ast::Expr::UnaryOp {
                op: sqlparser::ast::UnaryOperator::Plus,
                expr: inner,
            } => Self::from_sqlparser_expr(inner),
            sqlparser::ast::Expr::Identifier(ident) if ident.quote_style.is_some() => Err(format!(
                "quoted identifier `{ident}` in scalar position is a column identifier rather than a single-quoted string literal (`'{}'`); column-to-column comparisons are unsupported",
                ident.value
            )),
            other => Err(format!(
                "unsupported SQL scalar literal expression in predicate: {other}"
            )),
        }
    }

    /// Parse a SQL literal token or raw string at the parser/builder boundary into a
    /// [`SqlScalarValue`] via `sqlparser`.
    pub fn parse_from_sql_token_or_str(raw: &str) -> Self {
        let trimmed = raw.trim();
        let dialect = SQLiteDialect {};
        if let Ok(mut parser) = sqlparser::parser::Parser::new(&dialect).try_with_sql(trimmed) {
            if let Ok(expr) = parser.parse_expr() {
                if let Ok(val) = Self::from_sqlparser_expr(&expr) {
                    return val;
                }
            }
        }
        Self::Text(trimmed.to_string())
    }
}

impl std::fmt::Display for SqlScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render_sql())
    }
}
