//! `iocraft` SQL syntax highlighting, XML wire-tag stripping, Steward write-mutation cards, and
//! `<to_user>` prose paragraph extraction.

use crate::DriverObservation;
use iocraft::prelude::{Color, MixedTextContent, Weight};
use portia_core::StewardAuthState;

const WIRE_TAGS: [&str; 6] = [
    "<sql>",
    "</sql>",
    "<to_user>",
    "</to_user>",
    "<to_code>",
    "</to_code>",
];

/// Strip XML wire route tags (`<sql>`, `</sql>`, `<to_user>`, `</to_user>`, `<to_code>`, `</to_code>`)
/// so protocol framing never leaks into the human conversation pane.
#[must_use]
pub fn strip_wire_route_tags(raw: &str) -> String {
    let mut cleaned = raw.trim().to_string();
    loop {
        let mut changed = false;
        for tag in WIRE_TAGS {
            if let Some(rest) = cleaned.strip_prefix(tag) {
                cleaned = rest.trim().to_string();
                changed = true;
            }
            if let Some(rest) = cleaned.strip_suffix(tag) {
                cleaned = rest.trim().to_string();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for tag in WIRE_TAGS {
        if cleaned.contains(tag) {
            cleaned = cleaned.replace(tag, "");
        }
    }
    cleaned.trim().to_string()
}

fn is_sql_keyword(word: &str) -> bool {
    matches!(
        word.to_ascii_uppercase().as_str(),
        "SELECT"
            | "FROM"
            | "WHERE"
            | "JOIN"
            | "LEFT"
            | "RIGHT"
            | "INNER"
            | "OUTER"
            | "CROSS"
            | "ON"
            | "AS"
            | "AND"
            | "OR"
            | "NOT"
            | "IN"
            | "IS"
            | "NULL"
            | "LIKE"
            | "GLOB"
            | "BETWEEN"
            | "EXISTS"
            | "ORDER"
            | "BY"
            | "GROUP"
            | "HAVING"
            | "LIMIT"
            | "OFFSET"
            | "ASC"
            | "DESC"
            | "DISTINCT"
            | "INSERT"
            | "INTO"
            | "VALUES"
            | "UPDATE"
            | "SET"
            | "DELETE"
            | "RETURNING"
            | "CREATE"
            | "ALTER"
            | "DROP"
            | "TABLE"
            | "INDEX"
            | "VIEW"
            | "PRIMARY"
            | "KEY"
            | "FOREIGN"
            | "REFERENCES"
            | "PRAGMA"
            | "WITH"
            | "UNION"
            | "ALL"
            | "CASE"
            | "WHEN"
            | "THEN"
            | "ELSE"
            | "END"
            | "COUNT"
            | "SUM"
            | "AVG"
            | "MIN"
            | "MAX"
            | "COALESCE"
            | "CAST"
    )
}

fn is_mutation_keyword(word: &str) -> bool {
    matches!(
        word.to_ascii_uppercase().as_str(),
        "INSERT" | "UPDATE" | "DELETE" | "DROP" | "ALTER"
    )
}

fn tokenize_sql_line(line: &str, gutter_color: Color) -> Vec<MixedTextContent> {
    let mut spans = vec![MixedTextContent::new("  │ ").color(gutter_color)];
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    while i < n {
        let ch = chars[i];
        if ch == '-' && i + 1 < n && chars[i + 1] == '-' {
            let rest: String = chars[i..].iter().collect();
            spans.push(MixedTextContent::new(rest).color(Color::DarkGrey).italic());
            break;
        }
        if ch == '\'' {
            let start = i;
            i += 1;
            while i < n {
                if chars[i] == '\'' {
                    i += 1;
                    if i < n && chars[i] == '\'' {
                        i += 1;
                        continue;
                    }
                    break;
                }
                i += 1;
            }
            let lit: String = chars[start..i].iter().collect();
            spans.push(MixedTextContent::new(lit).color(Color::Green));
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            i += 1;
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let span = if is_mutation_keyword(&word) {
                MixedTextContent::new(word)
                    .color(Color::Yellow)
                    .weight(Weight::Bold)
            } else if is_sql_keyword(&word) {
                MixedTextContent::new(word)
                    .color(Color::Cyan)
                    .weight(Weight::Bold)
            } else {
                MixedTextContent::new(word).color(Color::White)
            };
            spans.push(span);
            continue;
        }
        if ch.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < n && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let num: String = chars[start..i].iter().collect();
            spans.push(MixedTextContent::new(num).color(Color::Magenta));
            continue;
        }
        let start = i;
        i += 1;
        while i < n
            && !chars[i].is_ascii_alphabetic()
            && !chars[i].is_ascii_digit()
            && chars[i] != '_'
            && chars[i] != '\''
            && !(chars[i] == '-' && i + 1 < n && chars[i + 1] == '-')
        {
            i += 1;
        }
        let punct: String = chars[start..i].iter().collect();
        spans.push(MixedTextContent::new(punct).color(Color::Grey));
    }

    spans
}

/// Syntax-highlight a multi-line SQL or code block into lines of [`MixedTextContent`].
#[must_use]
pub fn highlight_sql_mixed_lines(payload: &str, is_write: bool) -> Vec<Vec<MixedTextContent>> {
    let cleaned = strip_wire_route_tags(payload);
    if cleaned.is_empty() {
        return Vec::new();
    }
    let gutter_color = if is_write { Color::Yellow } else { Color::Cyan };
    cleaned
        .lines()
        .map(|line| tokenize_sql_line(line, gutter_color))
        .collect()
}

/// Build Steward & SQLite write-mutation diff lines for `Mode::Write` observations.
#[must_use]
pub fn build_write_steward_card_lines(obs: &DriverObservation) -> Vec<Vec<MixedTextContent>> {
    let mut out = Vec::new();
    let op_label = obs.operation.map(|op| op.as_str()).unwrap_or("WRITE");
    let row_noun = if obs.rows_affected == 1 {
        "row"
    } else {
        "rows"
    };

    let mut header = vec![
        MixedTextContent::new(format!("  ✓ {op_label} applied"))
            .color(Color::Green)
            .weight(Weight::Bold),
        MixedTextContent::new(format!(
            " · {} {row_noun} affected · db v{}",
            obs.rows_affected, obs.db_version
        ))
        .color(Color::White),
    ];
    if obs.steward_auth == StewardAuthState::Granted {
        header.push(
            MixedTextContent::new(" · STEWARD: GRANTED")
                .color(Color::Yellow)
                .weight(Weight::Bold),
        );
    }
    out.push(header);

    if !obs.table_versions.is_empty() {
        let bumps: Vec<String> = obs
            .table_versions
            .iter()
            .map(|(tbl, ver)| format!("{tbl}@v{ver}"))
            .collect();
        out.push(vec![MixedTextContent::new(format!(
            "    versions: {}",
            bumps.join(", ")
        ))
        .color(Color::Grey)]);
    }

    out
}

/// Convert `<to_user>` prose or single-line JSON `Mode::Result` output into styled paragraphs for
/// native `iocraft` `MixedText(wrap: TextWrap::Wrap)` rendering.
#[must_use]
pub fn format_clean_result_blocks(raw_payload: &str) -> Vec<Vec<MixedTextContent>> {
    let cleaned = strip_wire_route_tags(raw_payload);
    if cleaned.is_empty() {
        return Vec::new();
    }

    if cleaned.starts_with('{') && cleaned.ends_with('}') && !cleaned.contains('\n') {
        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&cleaned)
        {
            if !map.is_empty() {
                return map
                    .into_iter()
                    .map(|(k, v)| {
                        let v_txt = match v {
                            serde_json::Value::String(s) => s,
                            other => other.to_string(),
                        };
                        vec![
                            MixedTextContent::new(format!("  {k}: "))
                                .color(Color::Cyan)
                                .weight(Weight::Bold),
                            MixedTextContent::new(v_txt).color(Color::White),
                        ]
                    })
                    .collect();
            }
        }
    }

    format_user_prose_blocks(&cleaned, false)
}

/// Format multi-line Markdown-light `<to_user>` prose into styled `MixedTextContent` lines.
#[must_use]
pub fn format_user_prose_blocks(cleaned: &str, is_abstain: bool) -> Vec<Vec<MixedTextContent>> {
    let mut out = Vec::new();
    let default_color = if is_abstain {
        Color::Yellow
    } else {
        Color::White
    };

    for raw_line in cleaned.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            out.push(vec![MixedTextContent::new("")]);
            continue;
        }
        if let Some(heading) = trimmed
            .strip_prefix("### ")
            .or_else(|| trimmed.strip_prefix("## "))
            .or_else(|| trimmed.strip_prefix("# "))
        {
            out.push(vec![MixedTextContent::new(format!("  {}", heading.trim()))
                .color(Color::Cyan)
                .weight(Weight::Bold)]);
        } else if let Some(bullet) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("• "))
        {
            out.push(vec![
                MixedTextContent::new("  • ").color(Color::Cyan),
                MixedTextContent::new(bullet.trim()).color(default_color),
            ]);
        } else {
            out.push(vec![
                MixedTextContent::new(format!("  {trimmed}")).color(default_color)
            ]);
        }
    }
    out
}
