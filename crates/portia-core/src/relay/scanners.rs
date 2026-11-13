//! Quote-aware SQL/JSON literal scanners and Pretrain-LOTL wire chunk canonicalization.

use super::wire_spec::is_user_prompt_envelope;
use crate::{Mode, RouteTag};

/// True if `s` has an unclosed single-quoted (`'...'`) or double-quoted (`"..."`) SQL literal.
pub fn sql_has_unclosed_quote(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
        } else if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
        } else if b == b'\'' {
            in_single = true;
        } else if b == b'"' {
            in_double = true;
        }
        i += 1;
    }
    in_single || in_double
}

/// Find the byte index of the first occurrence of `target` outside single- or double-quoted SQL literals.
pub fn find_unquoted_sql_char(s: &str, target: char) -> Option<usize> {
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = s.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if in_single {
            if ch == '\'' {
                if chars.peek().is_some_and(|&(_, next_ch)| next_ch == '\'') {
                    let _ = chars.next();
                    continue;
                }
                in_single = false;
            }
        } else if in_double {
            if ch == '"' {
                if chars.peek().is_some_and(|&(_, next_ch)| next_ch == '"') {
                    let _ = chars.next();
                    continue;
                }
                in_double = false;
            }
        } else if ch == '\'' {
            in_single = true;
        } else if ch == '"' {
            in_double = true;
        } else if ch == target {
            return Some(idx);
        }
    }
    None
}

/// True if `s` contains an unquoted SQL `WHERE` keyword at a word boundary.
pub fn sql_has_unquoted_where(s: &str) -> bool {
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < n {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < n && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
        } else if in_double {
            if b == b'"' {
                if i + 1 < n && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
        } else if b == b'\'' {
            in_single = true;
        } else if b == b'"' {
            in_double = true;
        } else if i + 5 <= n && bytes[i..i + 5].eq_ignore_ascii_case(b"WHERE") {
            let prev_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let next_ok =
                i + 5 >= n || !(bytes[i + 5].is_ascii_alphanumeric() || bytes[i + 5] == b'_');
            if prev_ok && next_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// True if `s` has an unclosed double-quoted JSON string literal.
pub fn json_has_unclosed_string(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_str = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
        }
        i += 1;
    }
    in_str
}

/// True if `buf` has a completed closing route tag (`</to_database>`, `</sql>`, `</to_inquirer>`,
/// `</to_user>`, `</report>`, `</tool_call>`) outside SQL/JSON string literals, with a non-empty body
/// before the tag and only whitespace after the tag.
pub fn has_completed_close_tag_outside_literals(buf: &str, mode: Mode) -> bool {
    if let Some((pos, matched_tag)) =
        find_any_close_tag_outside_literals(buf, mode, mode.route_tag())
    {
        let after_tag = &buf[pos + matched_tag.len()..];
        let prefix = &buf[..pos];
        let body_only = prefix.strip_suffix('\n').unwrap_or(prefix);
        after_tag.trim().is_empty() && !body_only.trim().is_empty()
    } else {
        false
    }
}

fn sort_json_object_keys_recursively(val: serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map
                .into_iter()
                .map(|(k, v)| (k, sort_json_object_keys_recursively(v)))
                .collect();
            entries.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
            let mut sorted_map = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                sorted_map.insert(k, v);
            }
            serde_json::Value::Object(sorted_map)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(sort_json_object_keys_recursively)
                .collect(),
        ),
        other => other,
    }
}

/// Canonicalize an outbound worker wire chunk into the single canonical ChatML dialect:
/// - `<|im_start|>user\n...<|im_end|>\n` user prompt envelopes pass through **verbatim** (never
///   mutating literal user text that mentions tags or framing strings),
/// - deterministic `HarnessPen` boundary shims (`"\n<|im_end|>\n"` -> `"<|im_end|>\n"` and
///   `"\nPlan: "` -> `"\nPlan:"`) are canonicalized,
/// - `<tool_response>\n{...}\n</tool_response>` observation blocks serialize canonical sorted JSON
///   keys without rewriting surrounding text, and
/// - native [`super::RelayMachine`] chunks (`<|im_start|>assistant`, `<sql>...</sql>`, `<to_user>...</to_user>`,
///   `\nPlan:`, `<|im_end|>\n`) pass through untouched.
pub fn normalize_lotl_wire_chunk(text: &str, _selected_mode: Option<Mode>) -> String {
    if is_user_prompt_envelope(text) {
        return text.to_string();
    }
    if text == "\n<|im_end|>\n" {
        return "<|im_end|>\n".to_string();
    }
    if text == "\nPlan: " {
        return "\nPlan:".to_string();
    }
    if let Some((before_tr, after_open)) = text.split_once("<tool_response>\n") {
        if let Some((json_body, after_close)) = after_open.rsplit_once("\n</tool_response>") {
            if let Ok(parsed_val) = serde_json::from_str::<serde_json::Value>(json_body.trim()) {
                let sorted_val = sort_json_object_keys_recursively(parsed_val);
                if let Ok(norm_json) = serde_json::to_string(&sorted_val) {
                    return format!(
                        "{before_tr}<tool_response>\n{norm_json}\n</tool_response>{after_close}"
                    );
                }
            }
        }
    }
    text.to_string()
}

fn find_close_tag_outside_literals(buf: &str, mode: Mode, close_tag: &str) -> Option<usize> {
    let mut search_from = 0;
    while let Some(rel) = buf[search_from..].find(close_tag) {
        let pos = search_from + rel;
        let prefix = &buf[..pos];
        let inside_literal = match mode {
            Mode::Probe | Mode::Read | Mode::Write => sql_has_unclosed_quote(prefix),
            Mode::Result | Mode::Abstain => json_has_unclosed_string(prefix),
            Mode::Ask => false,
        };
        if !inside_literal {
            return Some(pos);
        }
        search_from = pos + close_tag.len();
    }
    None
}

pub(super) fn find_any_close_tag_outside_literals(
    buf: &str,
    mode: Mode,
    route: RouteTag,
) -> Option<(usize, &'static str)> {
    for &cand_tag in route.accepted_close_tags() {
        if let Some(pos) = find_close_tag_outside_literals(buf, mode, cand_tag) {
            return Some((pos, cand_tag));
        }
    }
    None
}
