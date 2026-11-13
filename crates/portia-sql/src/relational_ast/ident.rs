//! Structured SQL identifiers ([`SqlIdent`]) and qualified column references ([`SqlColumnRef`]).

use serde::{Deserialize, Serialize};
use sqlparser::dialect::SQLiteDialect;

/// Sorted list of additional SQLite 3.25+ window/upsert/returning & boolean literal keywords
/// that must be double-quoted when rendered in identifier position so both SQLite and `sqlparser`
/// parse them unambiguously as identifiers rather than keywords or boolean/temporal literals.
const SQLITE_EXTRA_QUOTED_KEYWORDS: &[&str] = &[
    "always",
    "current",
    "do",
    "exclude",
    "false",
    "filter",
    "first",
    "following",
    "generated",
    "groups",
    "last",
    "materialized",
    "nothing",
    "nulls",
    "others",
    "over",
    "partition",
    "preceding",
    "range",
    "returning",
    "rows",
    "ties",
    "true",
    "unbounded",
    "window",
    "within",
];

/// Structured SQL identifier preserving exact raw spelling (`name` without `.trim()`) and
/// delimiter quote metadata (`quoted`), with canonical SQLite double-quote rendering (`""` escape).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SqlIdent {
    pub name: String,
    pub quoted: bool,
}

impl SqlIdent {
    /// Construct a [`SqlIdent`], automatically marking `quoted = true` when `name` requires
    /// SQLite double-quoting (empty, whitespace, punctuation, leading digits, or SQLite keyword).
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let quoted = Self::needs_quoting(&name);
        Self { name, quoted }
    }

    /// Construct an explicitly double-quoted [`SqlIdent`] (`quoted = true`).
    pub fn quoted(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            quoted: true,
        }
    }

    /// Construct an unquoted [`SqlIdent`] (`quoted = false`). Note that [`Self::render_sql`] still
    /// quotes `name` defensively if [`Self::needs_quoting`] holds so rendered SQL is always valid.
    pub fn unquoted(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            quoted: false,
        }
    }

    /// Convert a [`sqlparser::ast::Ident`] without trimming whitespace (`" x "` stays `" x "`).
    pub fn from_sqlparser_ident(ident: &sqlparser::ast::Ident) -> Self {
        Self {
            name: ident.value.clone(),
            quoted: ident.quote_style.is_some(),
        }
    }

    /// Return `true` if `name` (exact spelling, without `.trim()`) matches a SQLite keyword or
    /// boolean/temporal literal keyword.
    pub fn is_sqlite_keyword(name: &str) -> bool {
        if !name.is_ascii() {
            return false;
        }
        let low = name.to_ascii_lowercase();
        let s = low.as_str();
        crate::grammar_trie::SQLITE_RESERVED_KEYWORDS
            .binary_search(&s)
            .is_ok()
            || crate::grammar_trie::SQLITE_LARK_NONRESERVED_KEYWORDS
                .binary_search(&s)
                .is_ok()
            || SQLITE_EXTRA_QUOTED_KEYWORDS.binary_search(&s).is_ok()
    }

    /// Return `true` when `name` cannot be rendered safely as a bare unquoted SQLite identifier
    /// (empty string, leading digit, whitespace, punctuation, or SQLite keyword).
    pub fn needs_quoting(name: &str) -> bool {
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return true;
        };
        let valid_start = first.is_ascii_alphabetic()
            || first == '_'
            || (!first.is_ascii() && first.is_alphabetic());
        if !valid_start {
            return true;
        }
        let valid_rest = chars.all(|c| {
            c.is_ascii_alphanumeric() || c == '_' || (!c.is_ascii() && c.is_alphanumeric())
        });
        if !valid_rest {
            return true;
        }
        Self::is_sqlite_keyword(name)
    }

    /// Render this identifier as valid SQLite SQL, wrapping in double quotes and escaping internal
    /// `"` as `""` whenever `self.quoted` is true or [`Self::needs_quoting`] holds.
    pub fn render_sql(&self) -> String {
        if self.quoted || Self::needs_quoting(&self.name) {
            format!("\"{}\"", self.name.replace('"', "\"\""))
        } else {
            self.name.clone()
        }
    }

    /// Exact character-for-character match against `candidate` without `.trim()`.
    #[inline]
    pub fn matches_exact(&self, candidate: &str) -> bool {
        self.name == candidate
    }

    /// SQLite ASCII case-insensitive match against `candidate` without `.trim()` (so `" x "`
    /// matches `" X "` and `" x "`, never `"x"`).
    #[inline]
    pub fn matches_ascii_case_insensitive(&self, candidate: &str) -> bool {
        self.name.eq_ignore_ascii_case(candidate)
    }
}

impl From<&str> for SqlIdent {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for SqlIdent {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl std::fmt::Display for SqlIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render_sql())
    }
}

/// Structured SQL column reference distinguishing an optional `table_qualifier` (`t."a b"`)
/// from a single unqualified column identifier that may itself contain dots (`"a.b"`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SqlColumnRef {
    pub table_qualifier: Option<SqlIdent>,
    pub column: SqlIdent,
}

impl SqlColumnRef {
    pub fn new(table_qualifier: Option<SqlIdent>, column: impl Into<SqlIdent>) -> Self {
        Self {
            table_qualifier,
            column: column.into(),
        }
    }

    pub fn unqualified(column: impl Into<SqlIdent>) -> Self {
        Self {
            table_qualifier: None,
            column: column.into(),
        }
    }

    pub fn qualified(table: impl Into<SqlIdent>, column: impl Into<SqlIdent>) -> Self {
        Self {
            table_qualifier: Some(table.into()),
            column: column.into(),
        }
    }

    /// Return `Some("<table>.<column>")` (raw unescaped names, without `.trim()`) when
    /// `self.table_qualifier` is `Some`, or `None` when unqualified.
    pub fn qualified_key(&self) -> Option<String> {
        self.table_qualifier
            .as_ref()
            .map(|tbl| format!("{}.{}", tbl.name, self.column.name))
    }

    /// Return `<table>.<column>` when qualified or `<column>` when unqualified (raw unescaped
    /// names without `.trim()`).
    pub fn qualified_or_column_name(&self) -> String {
        self.qualified_key()
            .unwrap_or_else(|| self.column.name.clone())
    }

    /// Render this column reference as valid SQLite SQL (e.g. `"\" x \""`, `"\"a b\""`,
    /// `"\"a.b\""`, `t."a b"`).
    pub fn render_sql(&self) -> String {
        match &self.table_qualifier {
            Some(tbl) => format!("{}.{}", tbl.render_sql(), self.column.render_sql()),
            None => self.column.render_sql(),
        }
    }

    /// Parse a SQLite column binding specification (`"col_name"`, `"\"a.b\""`, `"table.col"`,
    /// or `"\"my.tbl\".\"a.b\""`) into a structured [`SqlColumnRef`] using `sqlparser` so quoted
    /// identifiers containing dots (`"\"a.b\""`) remain a single unqualified column rather than
    /// splitting naively at `.`.
    pub fn parse_sql_binding(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let dialect = SQLiteDialect {};
        if let Ok(mut parser) = sqlparser::parser::Parser::new(&dialect).try_with_sql(trimmed) {
            if let Ok(expr) = parser.parse_expr() {
                if matches!(parser.next_token().token, sqlparser::tokenizer::Token::EOF) {
                    match expr {
                        sqlparser::ast::Expr::Identifier(ident)
                            if ident.quote_style != Some('\'') =>
                        {
                            return Some(Self::unqualified(SqlIdent::from_sqlparser_ident(&ident)));
                        }
                        sqlparser::ast::Expr::CompoundIdentifier(idents) => match idents.as_slice()
                        {
                            [col_ident] if col_ident.quote_style != Some('\'') => {
                                return Some(Self::unqualified(SqlIdent::from_sqlparser_ident(
                                    col_ident,
                                )));
                            }
                            [tbl_ident, col_ident]
                                if tbl_ident.quote_style != Some('\'')
                                    && col_ident.quote_style != Some('\'') =>
                            {
                                return Some(Self::qualified(
                                    SqlIdent::from_sqlparser_ident(tbl_ident),
                                    SqlIdent::from_sqlparser_ident(col_ident),
                                ));
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
        }
        // Fallback for unquoted SQLite keywords (e.g. `order` or `t.order`) that `parse_expr`
        // rejects as bare expressions: wrap unquoted segments or split once only when unquoted.
        if !trimmed.contains(['"', '`', '[']) {
            if let Some((tbl, col)) = trimmed.split_once('.') {
                let tbl = tbl.trim();
                let col = col.trim();
                if !tbl.is_empty() && !col.is_empty() && !col.contains('.') {
                    return Some(Self::qualified(
                        SqlIdent::unquoted(tbl),
                        SqlIdent::unquoted(col),
                    ));
                }
            } else {
                return Some(Self::unqualified(SqlIdent::unquoted(trimmed)));
            }
        }
        None
    }

    /// Match this column binding case-insensitively against `(table_name, col_name)` (where
    /// `table_name = None` matches when `self` is unqualified or when caller lacks table context).
    #[must_use]
    pub fn matches_table_and_column(&self, table_name: Option<&str>, col_name: &str) -> bool {
        if !self.column.matches_ascii_case_insensitive(col_name) {
            return false;
        }
        match (&self.table_qualifier, table_name) {
            (Some(spec_tbl), Some(actual_tbl)) => {
                spec_tbl.matches_ascii_case_insensitive(actual_tbl)
            }
            (Some(_), None) => true,
            (None, _) => true,
        }
    }

    /// Strict `(table_name, col_name)` match requiring `table_qualifier` (when present) to match
    /// `table_name` case-insensitively.
    #[must_use]
    pub fn matches_strict_table_and_column(&self, table_name: &str, col_name: &str) -> bool {
        self.column.matches_ascii_case_insensitive(col_name)
            && self
                .table_qualifier
                .as_ref()
                .is_none_or(|tbl| tbl.matches_ascii_case_insensitive(table_name))
    }
}

impl From<SqlIdent> for SqlColumnRef {
    fn from(column: SqlIdent) -> Self {
        Self::unqualified(column)
    }
}

impl From<&str> for SqlColumnRef {
    fn from(column: &str) -> Self {
        Self::unqualified(SqlIdent::from(column))
    }
}

impl From<String> for SqlColumnRef {
    fn from(column: String) -> Self {
        Self::unqualified(SqlIdent::from(column))
    }
}

impl std::fmt::Display for SqlColumnRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render_sql())
    }
}
