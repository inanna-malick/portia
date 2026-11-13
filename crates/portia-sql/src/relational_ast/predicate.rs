//! Recursive SQL predicate AST ([`SqlPredicateAst`]), functorial frame ([`SqlPredicateFrame`]),
//! Kleene three-valued logic ([`SqlTruthValue`]), and canonical SQL rendering.

use std::collections::BTreeSet;

use portia_core::hylo::{Collapsible, CollapsibleExt, Expandable, MappableFrame, PartiallyApplied};
use serde::{Deserialize, Serialize};

use super::ident::{SqlColumnRef, SqlIdent};
use super::scalar_ops::{SqlComparisonOp, SqlScalarValue};

/// Functorial single-layer frame for a recursive SQL `WHERE` predicate expression tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlPredicateFrame<Child> {
    Comparison {
        column: SqlColumnRef,
        op: SqlComparisonOp,
        literal: SqlScalarValue,
    },
    IsNull {
        column: SqlColumnRef,
        negated: bool,
    },
    InList {
        column: SqlColumnRef,
        literals: Vec<SqlScalarValue>,
    },
    And(Child, Child),
    Or(Child, Child),
    Not(Child),
}

impl MappableFrame for SqlPredicateFrame<PartiallyApplied> {
    type Frame<X> = SqlPredicateFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            SqlPredicateFrame::Comparison {
                column,
                op,
                literal,
            } => SqlPredicateFrame::Comparison {
                column,
                op,
                literal,
            },
            SqlPredicateFrame::IsNull { column, negated } => {
                SqlPredicateFrame::IsNull { column, negated }
            }
            SqlPredicateFrame::InList { column, literals } => {
                SqlPredicateFrame::InList { column, literals }
            }
            SqlPredicateFrame::And(left, right) => {
                let l = f(left);
                let r = f(right);
                SqlPredicateFrame::And(l, r)
            }
            SqlPredicateFrame::Or(left, right) => {
                let l = f(left);
                let r = f(right);
                SqlPredicateFrame::Or(l, r)
            }
            SqlPredicateFrame::Not(inner) => SqlPredicateFrame::Not(f(inner)),
        }
    }
}

/// Recursive SQL `WHERE` predicate expression tree backed by [`SqlPredicateFrame`]
/// and stack-safe [`portia_core::hylo`] schemes (`Collapsible`, `Expandable`, `MappableFrame`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlPredicateAst(pub Box<SqlPredicateFrame<SqlPredicateAst>>);

impl Expandable for SqlPredicateAst {
    type FrameToken = SqlPredicateFrame<PartiallyApplied>;

    fn from_frame(val: SqlPredicateFrame<Self>) -> Self {
        Self(Box::new(val))
    }
}

impl Collapsible for SqlPredicateAst {
    type FrameToken = SqlPredicateFrame<PartiallyApplied>;

    fn into_frame(mut self) -> SqlPredicateFrame<Self> {
        let dummy = SqlPredicateFrame::IsNull {
            column: SqlColumnRef::unqualified(SqlIdent::unquoted(String::new())),
            negated: false,
        };
        std::mem::replace(self.0.as_mut(), dummy)
    }
}

impl Drop for SqlPredicateAst {
    fn drop(&mut self) {
        if matches!(
            self.0.as_ref(),
            SqlPredicateFrame::Comparison { .. }
                | SqlPredicateFrame::IsNull { .. }
                | SqlPredicateFrame::InList { .. }
        ) {
            return;
        }
        let dummy_frame = || SqlPredicateFrame::IsNull {
            column: SqlColumnRef::unqualified(SqlIdent::unquoted(String::new())),
            negated: false,
        };
        let mut stack = vec![std::mem::replace(self.0.as_mut(), dummy_frame())];
        while let Some(frame) = stack.pop() {
            match frame {
                SqlPredicateFrame::Comparison { .. }
                | SqlPredicateFrame::IsNull { .. }
                | SqlPredicateFrame::InList { .. } => {}
                SqlPredicateFrame::Not(mut inner) => {
                    stack.push(std::mem::replace(inner.0.as_mut(), dummy_frame()));
                }
                SqlPredicateFrame::And(mut l, mut r) | SqlPredicateFrame::Or(mut l, mut r) => {
                    stack.push(std::mem::replace(l.0.as_mut(), dummy_frame()));
                    stack.push(std::mem::replace(r.0.as_mut(), dummy_frame()));
                }
            }
        }
    }
}

impl Collapsible for &SqlPredicateAst {
    type FrameToken = SqlPredicateFrame<PartiallyApplied>;

    fn into_frame(self) -> SqlPredicateFrame<Self> {
        match self.0.as_ref() {
            SqlPredicateFrame::Comparison {
                column,
                op,
                literal,
            } => SqlPredicateFrame::Comparison {
                column: column.clone(),
                op: *op,
                literal: literal.clone(),
            },
            SqlPredicateFrame::IsNull { column, negated } => SqlPredicateFrame::IsNull {
                column: column.clone(),
                negated: *negated,
            },
            SqlPredicateFrame::InList { column, literals } => SqlPredicateFrame::InList {
                column: column.clone(),
                literals: literals.clone(),
            },
            SqlPredicateFrame::And(left, right) => SqlPredicateFrame::And(left, right),
            SqlPredicateFrame::Or(left, right) => SqlPredicateFrame::Or(left, right),
            SqlPredicateFrame::Not(inner) => SqlPredicateFrame::Not(inner),
        }
    }
}

/// Kleene three-valued logic truth value (`TRUE`, `FALSE`, `UNKNOWN` / `NULL`) for SQL
/// predicate evaluation, distinguishing `WHERE` filter selection (`== SqlTruthValue::True`)
/// from `CHECK` constraint satisfaction (`!= SqlTruthValue::False`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlTruthValue {
    True,
    False,
    Unknown,
}

impl SqlTruthValue {
    pub fn negate(self) -> Self {
        match self {
            SqlTruthValue::True => SqlTruthValue::False,
            SqlTruthValue::False => SqlTruthValue::True,
            SqlTruthValue::Unknown => SqlTruthValue::Unknown,
        }
    }

    pub fn and(self, rhs: Self) -> Self {
        match (self, rhs) {
            (SqlTruthValue::False, _) | (_, SqlTruthValue::False) => SqlTruthValue::False,
            (SqlTruthValue::True, SqlTruthValue::True) => SqlTruthValue::True,
            _ => SqlTruthValue::Unknown,
        }
    }

    pub fn or(self, rhs: Self) -> Self {
        match (self, rhs) {
            (SqlTruthValue::True, _) | (_, SqlTruthValue::True) => SqlTruthValue::True,
            (SqlTruthValue::False, SqlTruthValue::False) => SqlTruthValue::False,
            _ => SqlTruthValue::Unknown,
        }
    }
}

impl std::ops::Not for SqlTruthValue {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.negate()
    }
}

impl std::ops::BitAnd for SqlTruthValue {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        SqlTruthValue::and(self, rhs)
    }
}

impl std::ops::BitOr for SqlTruthValue {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        SqlTruthValue::or(self, rhs)
    }
}

impl SqlPredicateAst {
    pub fn new(frame: SqlPredicateFrame<Self>) -> Self {
        Self::from_frame(frame)
    }

    pub fn frame(&self) -> &SqlPredicateFrame<Self> {
        &self.0
    }

    pub fn comparison(
        column: impl Into<SqlColumnRef>,
        op: SqlComparisonOp,
        literal: impl Into<SqlScalarValue>,
    ) -> Self {
        Self::from_frame(SqlPredicateFrame::Comparison {
            column: column.into(),
            op,
            literal: literal.into(),
        })
    }

    pub fn is_null(column: impl Into<SqlColumnRef>, negated: bool) -> Self {
        Self::from_frame(SqlPredicateFrame::IsNull {
            column: column.into(),
            negated,
        })
    }

    pub fn in_list(column: impl Into<SqlColumnRef>, literals: Vec<SqlScalarValue>) -> Self {
        Self::from_frame(SqlPredicateFrame::InList {
            column: column.into(),
            literals,
        })
    }

    pub fn and(left: Self, right: Self) -> Self {
        Self::from_frame(SqlPredicateFrame::And(left, right))
    }

    pub fn or(left: Self, right: Self) -> Self {
        Self::from_frame(SqlPredicateFrame::Or(left, right))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(inner: Self) -> Self {
        Self::from_frame(SqlPredicateFrame::Not(inner))
    }

    /// Render this predicate AST as canonical SQLite SQL with automatic precedence parenthesization
    /// and valid double-quoting on structured [`SqlColumnRef`] identifiers.
    pub fn render_sql(&self) -> String {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum RenderPrec {
            Atom,
            And,
            Or,
        }

        let (rendered, _prec) = self.collapse_frames(
            |frame: SqlPredicateFrame<(String, RenderPrec)>| match frame {
                SqlPredicateFrame::Comparison {
                    column,
                    op,
                    literal,
                } => (
                    format!("{} {op} {}", column.render_sql(), literal.render_sql()),
                    RenderPrec::Atom,
                ),
                SqlPredicateFrame::IsNull { column, negated } => {
                    let col_sql = column.render_sql();
                    let s = if negated {
                        format!("{col_sql} IS NOT NULL")
                    } else {
                        format!("{col_sql} IS NULL")
                    };
                    (s, RenderPrec::Atom)
                }
                SqlPredicateFrame::InList { column, literals } => {
                    let rendered_lits: Vec<String> =
                        literals.iter().map(SqlScalarValue::render_sql).collect();
                    (
                        format!("{} IN ({})", column.render_sql(), rendered_lits.join(", ")),
                        RenderPrec::Atom,
                    )
                }
                SqlPredicateFrame::And((left, left_prec), (right, right_prec)) => {
                    let l = if left_prec == RenderPrec::Or {
                        format!("({left})")
                    } else {
                        left
                    };
                    let r = if matches!(right_prec, RenderPrec::Or | RenderPrec::And) {
                        format!("({right})")
                    } else {
                        right
                    };
                    (format!("{l} AND {r}"), RenderPrec::And)
                }
                SqlPredicateFrame::Or((left, _), (right, right_prec)) => {
                    let r = if right_prec == RenderPrec::Or {
                        format!("({right})")
                    } else {
                        right
                    };
                    (format!("{left} OR {r}"), RenderPrec::Or)
                }
                SqlPredicateFrame::Not((inner, _)) => (format!("NOT ({inner})"), RenderPrec::Atom),
            },
        );
        rendered
    }

    /// Collect all structured [`SqlColumnRef`] handles referenced anywhere in this predicate AST.
    pub fn referenced_column_refs(&self) -> BTreeSet<SqlColumnRef> {
        self.collapse_frames(
            |frame: SqlPredicateFrame<BTreeSet<SqlColumnRef>>| match frame {
                SqlPredicateFrame::Comparison { column, .. }
                | SqlPredicateFrame::IsNull { column, .. }
                | SqlPredicateFrame::InList { column, .. } => BTreeSet::from([column]),
                SqlPredicateFrame::And(mut left, right)
                | SqlPredicateFrame::Or(mut left, right) => {
                    left.extend(right);
                    left
                }
                SqlPredicateFrame::Not(inner) => inner,
            },
        )
    }

    /// Collect all column names (`<table>.<column>` when qualified, `<column>` when unqualified)
    /// referenced anywhere in this predicate AST.
    pub fn referenced_columns(&self) -> BTreeSet<String> {
        self.referenced_column_refs()
            .into_iter()
            .map(|c| c.qualified_or_column_name())
            .collect()
    }
}
