//! Resilient SQLite tokenizer scanning and top-level SQL clause summary inspection.

use serde::{Deserialize, Serialize};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::tokenizer::{Location, Token, Tokenizer, TokenizerError, Word};

use portia_core::TableSchema;

pub(super) fn strip_ident_quotes(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        return trimmed[1..trimmed.len() - 1].replace("\"\"", "\"");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('`') && trimmed.ends_with('`') {
        return trimmed[1..trimmed.len() - 1].replace("``", "`");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('\'') && trimmed.ends_with('\'') {
        return trimmed[1..trimmed.len() - 1].replace("''", "'");
    }
    trimmed.trim_matches(['`', '"']).to_string()
}

pub(super) fn unescape_doubled_ident_body(body: &str, quote: char) -> String {
    match quote {
        '"' => body.replace("\"\"", "\""),
        '`' => body.replace("``", "`"),
        '\'' => body.replace("''", "'"),
        _ => body.to_string(),
    }
}

#[derive(Debug, Clone)]
pub(super) struct SpannedSqlToken {
    pub(super) token: Token,
    pub(super) start_byte: usize,
    pub(super) end_byte: usize,
}

fn location_to_byte_offset(s: &str, target: Location) -> usize {
    if target.line == 0 || target.column == 0 {
        return 0;
    }
    let mut line = 1u64;
    let mut col = 1u64;
    for (byte_idx, ch) in s.char_indices() {
        if line == target.line && col == target.column {
            return byte_idx;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    s.len()
}

fn tokenize_sqlite_spanned(s: &str) -> Result<Vec<SpannedSqlToken>, TokenizerError> {
    let dialect = SQLiteDialect {};
    let raw_tokens = Tokenizer::new(&dialect, s).tokenize_with_location()?;
    let starts: Vec<usize> = raw_tokens
        .iter()
        .map(|t| location_to_byte_offset(s, t.location))
        .collect();
    let mut spanned = Vec::with_capacity(raw_tokens.len());
    for (idx, t) in raw_tokens.into_iter().enumerate() {
        let start_byte = starts[idx];
        let end_byte = starts.get(idx + 1).copied().unwrap_or(s.len());
        if !matches!(t.token, Token::Whitespace(_) | Token::EOF) {
            spanned.push(SpannedSqlToken {
                token: t.token,
                start_byte,
                end_byte,
            });
        }
    }
    Ok(spanned)
}

pub(super) fn tokenize_sqlite_resilient(s: &str) -> (Vec<SpannedSqlToken>, bool) {
    match tokenize_sqlite_spanned(s) {
        Ok(tokens) => (tokens, false),
        Err(_err) => {
            for quote_ch in ['\'', '"', '`'] {
                let closed = format!("{s}{quote_ch}");
                if let Ok(mut tokens) = tokenize_sqlite_spanned(&closed) {
                    for tok in &mut tokens {
                        tok.start_byte = tok.start_byte.min(s.len());
                        tok.end_byte = tok.end_byte.min(s.len());
                    }
                    return (tokens, true);
                }
            }
            (Vec::new(), true)
        }
    }
}

pub(super) fn split_leading_ident(s: &str) -> (&str, &str, bool) {
    let trimmed = s.trim_start();
    if trimmed.is_empty() {
        return ("", "", false);
    }
    let (tokens, had_unclosed_quote) = tokenize_sqlite_resilient(trimmed);
    let Some(first) = tokens.first() else {
        return ("", trimmed, true);
    };
    if first.start_byte != 0 {
        return ("", trimmed, true);
    }
    match &first.token {
        Token::Word(Word {
            quote_style: Some(q @ ('`' | '"' | '\'')),
            ..
        }) => {
            let q_len = q.len_utf8();
            if (!had_unclosed_quote || first.end_byte < trimmed.len())
                && first.end_byte >= 2 * q_len
                && trimmed[..first.end_byte].ends_with(*q)
            {
                (
                    &trimmed[q_len..first.end_byte - q_len],
                    &trimmed[first.end_byte..],
                    true,
                )
            } else {
                (&trimmed[q_len..], "", false)
            }
        }
        Token::SingleQuotedString(_) => {
            if (!had_unclosed_quote || first.end_byte < trimmed.len())
                && first.end_byte >= 2
                && trimmed[..first.end_byte].ends_with('\'')
            {
                (
                    &trimmed[1..first.end_byte - 1],
                    &trimmed[first.end_byte..],
                    true,
                )
            } else {
                (&trimmed[1..], "", false)
            }
        }
        Token::Word(Word {
            quote_style: None, ..
        }) => {
            let mut end_byte = first.end_byte;
            for next in &tokens[1..] {
                if next.start_byte == end_byte
                    && matches!(
                        &next.token,
                        Token::Word(Word {
                            quote_style: None,
                            ..
                        }) | Token::Number(_, _)
                    )
                {
                    end_byte = next.end_byte;
                } else {
                    break;
                }
            }
            let after = &trimmed[end_byte..];
            (&trimmed[..end_byte], after, !after.is_empty())
        }
        _ => ("", trimmed, true),
    }
}

pub(super) fn strip_pragma_table_info_prefix<'a>(s: &'a str, upper: &str) -> Option<&'a str> {
    if !upper.starts_with("PRAGMA") {
        return None;
    }
    let (tokens, _) = tokenize_sqlite_resilient(s);
    if tokens.len() >= 3
        && matches!(&tokens[0].token, Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("PRAGMA"))
        && matches!(&tokens[1].token, Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("TABLE_INFO"))
        && matches!(tokens[2].token, Token::LParen)
    {
        Some(&s[tokens[2].end_byte..])
    } else {
        None
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopLevelSqlClauseSummary {
    pub statement_keyword: Option<String>,
    pub primary_table: Option<String>,
    pub joined_tables: Vec<String>,
    pub has_join_keyword: bool,
    pub has_left_join: bool,
    pub has_on_keyword: bool,
    pub has_where_keyword: bool,
    pub has_group_by_clause: bool,
    pub has_order_by_clause: bool,
    pub has_limit_keyword: bool,
    pub has_select_wildcard: bool,
    pub has_aggregate_call: bool,
    pub has_where_comparison_or_like: bool,
}

/// Inspect `partial_sql` using `tokenize_sqlite_resilient` (`sqlparser::tokenizer`) so
/// SQL keywords inside string literals (`'...'`) are never confused with structural clauses.
pub fn inspect_top_level_sql_clauses(
    partial_sql: &str,
    schemas: &[TableSchema],
) -> TopLevelSqlClauseSummary {
    let (tokens, _) = tokenize_sqlite_resilient(partial_sql);
    let mut summary = TopLevelSqlClauseSummary::default();
    if let Some(first) = tokens.first() {
        if let Token::Word(w) = &first.token {
            if w.quote_style.is_none() {
                summary.statement_keyword = Some(w.value.to_ascii_uppercase());
            }
        }
    }
    let find_schema_table = |raw_slice: &str, ident: &str| -> Option<String> {
        let quote_ch = raw_slice.trim_start().chars().next();
        let clean = match quote_ch {
            Some(q @ ('"' | '`' | '\'')) => unescape_doubled_ident_body(ident, q),
            _ => strip_ident_quotes(ident),
        };
        schemas
            .iter()
            .find(|sc| sc.table_name.eq_ignore_ascii_case(&clean))
            .map(|sc| sc.table_name.clone())
    };
    let resolve_primary_table = |tok: &SpannedSqlToken, has_after_tok: bool, delim_ch: char| {
        let raw_ident = &partial_sql[tok.start_byte..tok.end_byte];
        let (ident, _, has_delim) = split_leading_ident(raw_ident);
        let tail = &partial_sql[tok.end_byte..];
        let delimited = has_delim
            || has_after_tok
            || tail.starts_with(char::is_whitespace)
            || tail.starts_with(delim_ch);
        let tbl = find_schema_table(raw_ident, ident)?;
        let has_longer_prefix = schemas.iter().any(|sc| {
            sc.table_name.len() > tbl.len()
                && sc
                    .table_name
                    .to_ascii_lowercase()
                    .starts_with(&tbl.to_ascii_lowercase())
        });
        (delimited || !has_longer_prefix).then_some(tbl)
    };

    let mut in_where = false;
    let mut before_from = true;
    for (idx, tok) in tokens.iter().enumerate() {
        match &tok.token {
            Token::Mul if before_from && idx == 1 => {
                summary.has_select_wildcard = true;
            }
            Token::Word(w) if w.quote_style.is_none() => {
                let up = w.value.to_ascii_uppercase();
                match up.as_str() {
                    "FROM" | "UPDATE" | "INTO" => {
                        if up == "FROM" {
                            before_from = false;
                        }
                        if summary.primary_table.is_none() {
                            if let Some(next_tok) = tokens.get(idx + 1) {
                                summary.primary_table = resolve_primary_table(
                                    next_tok,
                                    tokens.get(idx + 2).is_some(),
                                    ';',
                                );
                            }
                        }
                    }
                    "TABLE_INFO" => {
                        if summary.primary_table.is_none() {
                            if let (Some(lp), Some(tbl_tok)) =
                                (tokens.get(idx + 1), tokens.get(idx + 2))
                            {
                                if matches!(lp.token, Token::LParen) {
                                    summary.primary_table = resolve_primary_table(
                                        tbl_tok,
                                        tokens.get(idx + 3).is_some(),
                                        ')',
                                    );
                                }
                            }
                        }
                    }
                    "LEFT" => {
                        if tokens.get(idx + 1).is_some_and(|t| match &t.token {
                            Token::Word(nw) => {
                                nw.quote_style.is_none()
                                    && (nw.value.eq_ignore_ascii_case("JOIN")
                                        || nw.value.eq_ignore_ascii_case("OUTER"))
                            }
                            _ => false,
                        }) {
                            summary.has_left_join = true;
                        }
                    }
                    "JOIN" => {
                        before_from = false;
                        summary.has_join_keyword = true;
                        if let Some(next_tok) = tokens.get(idx + 1) {
                            let raw_ident = &partial_sql[next_tok.start_byte..next_tok.end_byte];
                            let (ident, _, _) = split_leading_ident(raw_ident);
                            if let Some(tbl) = find_schema_table(raw_ident, ident) {
                                summary.joined_tables.push(tbl);
                            }
                        }
                    }
                    "ON" if summary.has_join_keyword => {
                        summary.has_on_keyword = true;
                    }
                    "WHERE" => {
                        before_from = false;
                        summary.has_where_keyword = true;
                        in_where = true;
                    }
                    "GROUP" => {
                        if tokens.get(idx + 1).is_some_and(|t| match &t.token {
                            Token::Word(nw) => {
                                nw.quote_style.is_none() && nw.value.eq_ignore_ascii_case("BY")
                            }
                            _ => false,
                        }) {
                            summary.has_group_by_clause = true;
                            in_where = false;
                        }
                    }
                    "ORDER" => {
                        if tokens.get(idx + 1).is_some_and(|t| match &t.token {
                            Token::Word(nw) => {
                                nw.quote_style.is_none() && nw.value.eq_ignore_ascii_case("BY")
                            }
                            _ => false,
                        }) {
                            summary.has_order_by_clause = true;
                            in_where = false;
                        }
                    }
                    "LIMIT" => {
                        summary.has_limit_keyword = true;
                        in_where = false;
                    }
                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" if before_from => {
                        if tokens
                            .get(idx + 1)
                            .is_some_and(|t| matches!(t.token, Token::LParen))
                        {
                            summary.has_aggregate_call = true;
                        }
                    }
                    "LIKE" | "IS" | "IN" if in_where => {
                        summary.has_where_comparison_or_like = true;
                    }
                    _ => {}
                }
            }
            Token::Eq | Token::Neq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq
                if in_where =>
            {
                summary.has_where_comparison_or_like = true;
            }
            _ => {}
        }
    }
    summary
}
