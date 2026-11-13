//! Upstream `sqlite.lark` specialization, SQLite keyword tables, and [`SqliteSchemaGrammarCartridge`].

use std::collections::BTreeSet;

use portia_core::TableSchema;

/// Upstream SQLite Lark EBNF grammar (`grammars/upstream/sqlite.lark` from `ligurio/lark-grammars`),
/// with full `expr` productions enabled and standard Lark `STRING_LITERAL` / `IDENTIFIER` / `%ignore` rules.
pub const UPSTREAM_SQLITE_LARK_GRAMMAR: &str =
    include_str!("../../../../grammars/upstream/sqlite.lark");

/// Sorted list of the 58 SQLite reserved keywords rejected unquoted in identifier positions by
/// SQLite's C parser (`SELECT order FROM ...`, `PRAGMA table_info(order);`), which MUST be quoted
/// as `"kw"` or `` `kw` `` when used as a table or column name.
pub const SQLITE_RESERVED_KEYWORDS: &[&str] = &[
    "add",
    "all",
    "alter",
    "and",
    "as",
    "autoincrement",
    "between",
    "case",
    "cast",
    "check",
    "collate",
    "commit",
    "constraint",
    "create",
    "default",
    "deferrable",
    "delete",
    "distinct",
    "drop",
    "else",
    "escape",
    "except",
    "exists",
    "foreign",
    "from",
    "group",
    "having",
    "in",
    "index",
    "insert",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "limit",
    "not",
    "notnull",
    "null",
    "on",
    "or",
    "order",
    "primary",
    "raise",
    "references",
    "select",
    "set",
    "table",
    "then",
    "to",
    "transaction",
    "union",
    "unique",
    "update",
    "using",
    "values",
    "when",
    "where",
];

/// Sorted list of the 66 non-reserved `K_*` keyword terminals defined in `grammars/upstream/sqlite.lark`
/// (`key`, `view`, `row`, `plan`, `query`, `action`, `temp`, `trigger`, etc.) that SQLite accepts BOTH
/// unquoted and quoted as identifiers, and which therefore require a rule-level `"kw"i` literal alongside
/// the uppercase DFA terminal so `K_*` lexer priority in `sqlite.lark` does not shadow bare `kw`.
pub(crate) const SQLITE_LARK_NONRESERVED_KEYWORDS: &[&str] = &[
    "abort",
    "action",
    "after",
    "analyze",
    "asc",
    "attach",
    "before",
    "begin",
    "by",
    "cascade",
    "column",
    "conflict",
    "cross",
    "current_date",
    "current_time",
    "current_timestamp",
    "database",
    "deferred",
    "desc",
    "detach",
    "each",
    "end",
    "exclusive",
    "explain",
    "fail",
    "for",
    "full",
    "glob",
    "if",
    "ignore",
    "immediate",
    "indexed",
    "initially",
    "inner",
    "instead",
    "key",
    "left",
    "like",
    "match",
    "natural",
    "no",
    "of",
    "offset",
    "outer",
    "plan",
    "pragma",
    "query",
    "recursive",
    "regexp",
    "reindex",
    "release",
    "rename",
    "replace",
    "restrict",
    "right",
    "rollback",
    "row",
    "savepoint",
    "temp",
    "temporary",
    "trigger",
    "vacuum",
    "view",
    "virtual",
    "with",
    "without",
];

/// Return `true` if `ident` (unquoted) is a SQLite reserved keyword that SQLite rejects in
/// bare identifier positions unless quoted (`"..."` or `` `...` ``).
#[inline]
pub fn is_sqlite_reserved_keyword(ident: &str) -> bool {
    let low = ident.trim().to_ascii_lowercase();
    SQLITE_RESERVED_KEYWORDS
        .binary_search(&low.as_str())
        .is_ok()
}

#[inline]
fn is_sqlite_lark_nonreserved_keyword(ident_lower: &str) -> bool {
    SQLITE_LARK_NONRESERVED_KEYWORDS
        .binary_search(&ident_lower)
        .is_ok()
}

#[inline]
fn is_simple_ascii_sql_ident(ident: &str) -> bool {
    let mut chars = ident.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn escape_lark_regex_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '\\' | '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
            | '/' => {
                out.push('\\');
                out.push(ch);
            }
            '"' => {
                out.push_str("\\\"");
            }
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

/// Compile a deduplicated list of schema identifiers into `(rule_rhs, optional_terminal_def)` where:
/// - All valid identifiers (including reserved keywords and special-character identifiers) are
///   accepted in double-quoted (`"..."` with `""` escaping) and backtick-quoted (`` `...` `` with
///   `` `` `` escaping) forms inside a single quote-factored uppercase Lark DFA terminal `terminal_name`,
///   preventing both `llguidance` 2,000-item Earley row overflow and `ParserTooComplex` NFA blowup
///   even on 60-table × 15-column (`900+` column) schemas.
/// - Simple ASCII non-reserved identifiers are also accepted unquoted inside `terminal_name` (or as
///   rule-level `"kw"i` literals when colliding with one of `sqlite.lark`'s 66 non-reserved `K_*` tokens).
/// - SQLite's 58 reserved keywords (`order`, `group`, `select`, `where`, `table`, `index`, `limit`, `values`,
///   etc.) are permitted ONLY in quoted form (`"order"` / `` `order` ``), matching SQLite's C parser.
pub(super) fn compile_schema_ident_lark_rule_and_terminal<'a, I>(
    idents: I,
    terminal_name: &str,
) -> Option<(String, Option<String>)>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen = BTreeSet::new();
    let mut unquoted_branches = Vec::new();
    let mut double_quoted_branches = Vec::new();
    let mut backtick_quoted_branches = Vec::new();
    let mut nonreserved_kw_literals = Vec::new();

    for raw in idents {
        let ident = raw.trim();
        if ident.is_empty() {
            continue;
        }
        let low = ident.to_ascii_lowercase();
        if !seen.insert(low.clone()) {
            continue;
        }
        let dq_escaped = escape_lark_regex_literal(&ident.replace('"', "\"\""));
        let bq_escaped = escape_lark_regex_literal(&ident.replace('`', "``"));
        double_quoted_branches.push(dq_escaped);
        backtick_quoted_branches.push(bq_escaped);

        if is_simple_ascii_sql_ident(ident) && !is_sqlite_reserved_keyword(ident) {
            if is_sqlite_lark_nonreserved_keyword(&low) {
                let lit_esc = ident.replace('\\', "\\\\").replace('"', "\\\"");
                nonreserved_kw_literals.push(format!("\"{lit_esc}\"i"));
            } else {
                unquoted_branches.push(escape_lark_regex_literal(ident));
            }
        }
    }

    if seen.is_empty() {
        return None;
    }

    let mut top_groups = Vec::with_capacity(3);
    if !unquoted_branches.is_empty() {
        top_groups.push(format!("(?:{})", unquoted_branches.join("|")));
    }
    if !double_quoted_branches.is_empty() {
        top_groups.push(format!("\\\"(?:{})\\\"", double_quoted_branches.join("|")));
    }
    if !backtick_quoted_branches.is_empty() {
        top_groups.push(format!("`(?:{})`", backtick_quoted_branches.join("|")));
    }

    let mut rhs_parts = Vec::with_capacity(1 + nonreserved_kw_literals.len());
    let terminal_def = if !top_groups.is_empty() {
        rhs_parts.push(terminal_name.to_string());
        Some(format!(
            "{terminal_name}: /(?i-u:(?:{}))/",
            top_groups.join("|")
        ))
    } else {
        None
    };
    rhs_parts.extend(nonreserved_kw_literals);

    Some((rhs_parts.join(" | "), terminal_def))
}

pub(super) fn specialize_sqlite_lark_for_schemas(
    schemas: &[TableSchema],
    start_rule: &str,
) -> portia_core::GrammarCartridgeSpec {
    let Some((table_rhs, table_term_def)) = compile_schema_ident_lark_rule_and_terminal(
        schemas.iter().map(|sc| sc.table_name.as_str()),
        "SCHEMA_TABLE_IDENT",
    ) else {
        return portia_core::GrammarCartridgeSpec::Unconstrained;
    };

    let col_compiled = compile_schema_ident_lark_rule_and_terminal(
        schemas
            .iter()
            .flat_map(|sc| sc.columns.iter().map(|c| c.name.as_str())),
        "SCHEMA_COLUMN_IDENT",
    );

    let table_rule = format!("table_name\t\t: {table_rhs}");
    let (column_rule, col_term_def) = match col_compiled {
        Some((col_rhs, term_def)) => (format!("column_name\t\t: {col_rhs}"), term_def),
        None => ("column_name\t\t: any_name".to_string(), None),
    };

    // Intentional left-to-right CFG invariant: in SQL (`SELECT q.col FROM tbl AS q`), the qualifier
    // `q` appears BEFORE `FROM tbl AS q` binds any alias. Because Earley CFGs are context-free,
    // loosening `q` to `IDENTIFIER` would let the model commit to `SELECT hallucinated_tbl.col` at
    // token 1; binding `table_name` strictly to discovered `SCHEMA_TABLE_IDENT` keeps every
    // qualified column reference (`Accounts_Ledger_2.record_id`) causally sound left-to-right.
    const SQLITE_BUILTIN_FUNC_RULE: &str = "function_name\t\t: /(?i-u:(?:abs|avg|coalesce|count|date|datetime|group_concat|hex|ifnull|iif|instr|json_extract|julianday|length|lower|ltrim|max|min|nullif|printf|quote|replace|round|rtrim|sign|strftime|substr|substring|sum|time|total|trim|typeof|unicode|upper|zeroblob))/";

    let mut lark = UPSTREAM_SQLITE_LARK_GRAMMAR
        .replacen("start\t\t: parse", start_rule, 1)
        .replacen(
            "database_name\t\t: any_name",
            "database_name\t\t: \"main\"i | \"temp\"i",
            1,
        )
        .replacen("function_name\t\t: any_name", SQLITE_BUILTIN_FUNC_RULE, 1)
        .replacen(
            "STRING_LITERAL \t: /('([^']|'')*')/",
            "STRING_LITERAL \t: /'(?:[^'\\r\\n]|''){0,120}'/",
            1,
        )
        .replacen("table_name\t\t: any_name", &table_rule, 1)
        .replacen("column_name\t\t: any_name", &column_rule, 1);

    if let Some(t_def) = table_term_def {
        lark.push('\n');
        lark.push_str(&t_def);
    }
    if let Some(c_def) = col_term_def {
        lark.push('\n');
        lark.push_str(&c_def);
    }
    lark.push('\n');

    portia_core::GrammarCartridgeSpec::LarkGrammar(lark)
}

/// Compile a mode- and schema-specialized `llguidance` Lark grammar from [`UPSTREAM_SQLITE_LARK_GRAMMAR`]
/// and the exact `schemas` slice passed by the caller (`0` implicit decoy column filtering; callers
/// that explicitly want decoys stripped may call [`TableSchema::without_quarantined_decoys`]).
pub fn compile_sqlite_lark_grammar_for_schemas(
    schemas: &[TableSchema],
    mode: portia_core::Mode,
) -> portia_core::GrammarCartridgeSpec {
    if schemas.is_empty() {
        return portia_core::GrammarCartridgeSpec::Unconstrained;
    }
    let start_rule = match mode {
        portia_core::Mode::Probe => "start: pragma_stmt \";\"?",
        portia_core::Mode::Read => {
            "start: (select_stmt | compound_select_stmt | simple_select_stmt) \";\"?"
        }
        portia_core::Mode::Write => "start: (update_stmt | insert_stmt | delete_stmt) \";\"?",
        portia_core::Mode::Result | portia_core::Mode::Ask | portia_core::Mode::Abstain => {
            return portia_core::GrammarCartridgeSpec::Unconstrained;
        }
    };
    specialize_sqlite_lark_for_schemas(schemas, start_rule)
}

/// Context-specific SQLite schema grammar cartridge wrapping a slice of [`TableSchema`]s
/// and implementing [`portia_core::relay::PayloadGrammarAutomaton`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqliteSchemaGrammarCartridge<'a>(pub &'a [TableSchema]);

impl portia_core::relay::PayloadGrammarAutomaton for SqliteSchemaGrammarCartridge<'_> {
    fn grammar_cartridge_spec(&self, mode: portia_core::Mode) -> portia_core::GrammarCartridgeSpec {
        compile_sqlite_lark_grammar_for_schemas(self.0, mode)
    }
}
