//! Steward `[MODE: ASK]` prompt classification, interactive GUI affordance parsing,
//! candidate row summarization, and clarification/permission reply resolution.

use super::observation::DriverObservation;
use anyhow::{bail, Context, Result};
use portia_core::Mode;
use portia_sql::SqlColumnRef;
use serde::{Deserialize, Serialize};

/// Binary steward authorization decision for `[MODE: ASK]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "UPPERCASE")]
pub enum AskDecision {
    Allow,
    Deny,
}

impl AskDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "ALLOW",
            Self::Deny => "DENY",
        }
    }

    pub fn status_str(&self) -> &'static str {
        match self {
            Self::Allow => "GRANTED",
            Self::Deny => "DENIED",
        }
    }
}

/// Structured steward response parsed from an interactive or canned `[MODE: ASK]` reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AskUserReply {
    Decision {
        decision: AskDecision,
        raw_reply: String,
    },
    Clarification {
        clarification: String,
    },
}

/// Semantic category of an interactive `[MODE: ASK]` question, determined from the prompt itself
/// rather than guessed from the user's reply string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AskPromptKind {
    Permission,
    Clarification,
}

/// Standard coding-agent TUI escape-hatch option appended to every `[MODE: ASK]` `Select` or `MultiSelect` widget.
pub const NONE_OF_ABOVE_ENTER_TEXT_CHOICE: &str = "None of the above — enter text here...";

/// Ground-truth interaction mode for an `[MODE: ASK]` turn in the TUI/REPL, unifying
/// element-as-key JSON (`"multi"`, `"select"`, `"input"`) with SQLite observation state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AskInteractionMode {
    /// Conversational or open-ended clarification (`{"input": "..."}` or plain conversational text);
    /// accepts any free-text user reply in [`CommandInputBox`].
    ConversationalClarify,
    /// Single-choice selection (`{"select": "...", "options": [...]}` or Steward `ALLOW`/`DENY` gate);
    /// mounts a `tui-realm` [`Select`] chooser in the TUI with [`NONE_OF_ABOVE_ENTER_TEXT_CHOICE`] appended.
    StewardPermissionChoice {
        question: String,
        choices: Vec<String>,
    },
    /// Multi-choice selection (`{"multi": "...", "options": [...]}` or multi-row SQLite disambiguation);
    /// mounts a `tui-realm` [`Checkbox`] multi-select chooser in the TUI with `<Space>` toggle, `<Enter>` submit,
    /// and [`NONE_OF_ABOVE_ENTER_TEXT_CHOICE`] appended.
    MultiSelect {
        question: String,
        choices: Vec<String>,
    },
}

/// Exhaustive coroutine step effect of a single OODA turn (`Free CoroutineStepEffect ()`).
///
/// By Boyd's OODA Law, the coroutine loop **continues by default** (`AutoContinueDb`),
/// suspending only when an interactive GUI choice is required (`SuspendForGuiChoice`)
/// and stopping only at `YieldToPrompt` (`Mode::Result`, `Mode::Abstain`, or conversational `Mode::Ask`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoroutineStepEffect {
    /// Non-blocking database / tool step (`Mode::Probe`, `Mode::Read`, `Mode::Write`):
    /// executes against SQLite, prefills `<tool_response>`, and immediately steps Turn `t + 1`.
    AutoContinueDb,
    /// Coroutine suspension on an interactive `Structured-GUI` widget (`Mode::Ask` with
    /// `StewardPermissionChoice` or `MultiSelect`): renders `lead_in_prose` + the interactive
    /// option-set widget, waits for the user's choice, prefills `<tool_response>`, and resumes Turn `t + 1`.
    SuspendForGuiChoice {
        lead_in_prose: String,
        gui_mode: AskInteractionMode,
    },
    /// The single coroutine stop / yield-to-prompt transition (`Mode::Result`, `Mode::Abstain`,
    /// or pure non-action Chitchat): prints output to `PANE 1` and returns control to `portia-tui>`.
    YieldToPrompt,
}

/// Append [`NONE_OF_ABOVE_ENTER_TEXT_CHOICE`] to `opts` if an `"Other"` / `"None of the above"`
/// escape hatch is not already present.
pub fn ensure_none_of_above_escape_choice(mut opts: Vec<String>) -> Vec<String> {
    let has_escape = opts.iter().any(|o| {
        let low = o.to_ascii_lowercase();
        low.starts_with("none of the above") || low.starts_with("other")
    });
    if !has_escape {
        opts.push(NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string());
    }
    opts
}

/// Parse an element-as-key JSON `[MODE: ASK]` payload (`{"multi": "...", "options": [...]}`,
/// `{"select": "...", "options": [...]}`, `{"input": "..."}`, or `{"elements": [...]}`) into
/// an [`AskInteractionMode`].
pub fn extract_ask_interaction_mode(question: &str) -> Option<AskInteractionMode> {
    let trimmed = question.trim();
    if let Some((q_part, opts_part)) = trimmed.split_once("[Options:") {
        if let Some(inner) = opts_part.strip_suffix(']') {
            let opts: Vec<String> = inner
                .split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if opts.len() >= 2 {
                let clean_q = q_part.trim();
                let q_title = if clean_q.is_empty() {
                    "Select option(s):".to_string()
                } else {
                    clean_q.to_string()
                };
                return Some(AskInteractionMode::MultiSelect {
                    question: q_title,
                    choices: ensure_none_of_above_escape_choice(opts),
                });
            }
        }
    }
    if !trimmed.starts_with('{') {
        return None;
    }
    let val = serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
    let root_obj = val.as_object()?;

    // Support top-level `{"title": "...", "elements": [ ... ]}` wrapper as well as flat element-as-key JSON:
    let target_obj = if let Some(elems) = root_obj.get("elements").and_then(|v| v.as_array()) {
        elems
            .iter()
            .filter_map(|e| e.as_object())
            .find(|o| {
                o.contains_key("multi")
                    || o.contains_key("multiselect")
                    || o.contains_key("select")
                    || o.contains_key("choice")
                    || o.contains_key("input")
                    || o.contains_key("textbox")
            })
            .unwrap_or(root_obj)
    } else {
        root_obj
    };

    if target_obj.contains_key("input") || target_obj.contains_key("textbox") {
        return Some(AskInteractionMode::ConversationalClarify);
    }

    let is_explicit_select = target_obj.contains_key("select") || target_obj.contains_key("choice");
    let is_explicit_multi =
        target_obj.contains_key("multi") || target_obj.contains_key("multiselect");

    for key in ["options", "candidates", "choices"] {
        if let Some(arr) = target_obj
            .get(key)
            .or_else(|| root_obj.get(key))
            .and_then(|v| v.as_array())
        {
            let opts: Vec<String> = arr
                .iter()
                .map(|item| match item {
                    serde_json::Value::String(s) => s.trim().to_string(),
                    other => other.to_string(),
                })
                .filter(|s| !s.is_empty())
                .collect();
            if opts.len() >= 2 {
                let choices = ensure_none_of_above_escape_choice(opts);
                let q_title = target_obj
                    .get("multi")
                    .or_else(|| target_obj.get("multiselect"))
                    .or_else(|| target_obj.get("select"))
                    .or_else(|| target_obj.get("choice"))
                    .or_else(|| target_obj.get("question"))
                    .or_else(|| target_obj.get("clarification"))
                    .or_else(|| target_obj.get("prompt"))
                    .or_else(|| root_obj.get("title"))
                    .or_else(|| root_obj.get("question"))
                    .or_else(|| root_obj.get("clarification"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Select option(s):")
                    .trim()
                    .to_string();
                if is_explicit_select && !is_explicit_multi {
                    return Some(AskInteractionMode::StewardPermissionChoice {
                        question: q_title,
                        choices,
                    });
                }
                return Some(AskInteractionMode::MultiSelect {
                    question: q_title,
                    choices,
                });
            }
        }
    }
    None
}

/// Extract arbitrary `1-of-N` candidate choices from a structured JSON `[MODE: ASK]` payload
/// (`{"question": "...", "candidates": [...]}` / `"options"` / `"choices"`).
pub fn extract_ask_select_choices(question: &str) -> Option<(String, Vec<String>)> {
    match extract_ask_interaction_mode(question)? {
        AskInteractionMode::StewardPermissionChoice { question, choices }
        | AskInteractionMode::MultiSelect { question, choices } => Some((question, choices)),
        AskInteractionMode::ConversationalClarify => None,
    }
}

/// Format an `[MODE: ASK]` payload for human-readable display in the TUI/REPL transcript,
/// rendering element-as-key JSON (`{"multi": "...", "options": [...]}`, `{"select": "...", "options": [...]}`,
/// `{"input": "..."}`) into a clean question + options summary while preserving plain text unchanged.
pub fn format_ask_payload_for_display(payload: &str) -> String {
    let trimmed = payload.trim();
    if let Some(mode) = extract_ask_interaction_mode(trimmed) {
        match mode {
            AskInteractionMode::StewardPermissionChoice { question, choices }
            | AskInteractionMode::MultiSelect { question, choices } => {
                let display_opts: Vec<&str> = choices
                    .iter()
                    .map(|s| s.as_str())
                    .filter(|s| {
                        let low = s.to_ascii_lowercase();
                        !low.starts_with("none of the above") && !low.starts_with("other")
                    })
                    .collect();
                if display_opts.is_empty() {
                    return question;
                }
                return format!("{} [{}]", question, display_opts.join(" | "));
            }
            AskInteractionMode::ConversationalClarify => {}
        }
    }
    if trimmed.starts_with('{') {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(root_obj) = val.as_object() {
                let target_obj =
                    if let Some(elems) = root_obj.get("elements").and_then(|v| v.as_array()) {
                        elems
                            .iter()
                            .filter_map(|e| e.as_object())
                            .next()
                            .unwrap_or(root_obj)
                    } else {
                        root_obj
                    };
                for key in [
                    "input",
                    "textbox",
                    "multi",
                    "multiselect",
                    "select",
                    "choice",
                    "question",
                    "clarification",
                    "prompt",
                    "title",
                ] {
                    if let Some(s) = target_obj
                        .get(key)
                        .or_else(|| root_obj.get(key))
                        .and_then(|v| v.as_str())
                    {
                        let clean = s.trim();
                        if !clean.is_empty() {
                            return clean.to_string();
                        }
                    }
                }
            }
        }
    }
    trimmed.to_string()
}

pub fn summarize_candidate_row_choice(row: &[serde_json::Value]) -> Option<String> {
    let mut parts = Vec::with_capacity(3);
    for cell in row {
        let text = match cell {
            serde_json::Value::Null => continue,
            serde_json::Value::String(s) => {
                let t = s.trim();
                if t.is_empty() {
                    continue;
                }
                t.to_string()
            }
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => continue,
        };
        parts.push(text);
        if parts.len() == 3 {
            break;
        }
    }
    match parts.len() {
        0 => None,
        1 => Some(parts.remove(0)),
        _ => Some(format!("{} ({})", parts[0], parts[1..].join(" · "))),
    }
}

/// Return `true` if `col_name` (optionally on `table_name`) matches any explicitly configured
/// `steward_policy_columns` entry (`"col_name"`, `"\"a.b\""`, or `"table_name.col_name"`, case-insensitively).
pub fn steward_column_matches(
    steward_policy_columns: &[String],
    table_name: Option<&str>,
    col_name: &str,
) -> bool {
    if col_name.is_empty() {
        return false;
    }
    steward_policy_columns
        .iter()
        .filter_map(|spec| SqlColumnRef::parse_sql_binding(spec))
        .any(|col_ref| col_ref.matches_table_and_column(table_name, col_name))
}

/// Return `true` if `row` contains a cell equal to `"ASK"` (case-insensitively) inside one of
/// the explicitly configured `steward_policy_columns` (returns `false` when `steward_policy_columns` is empty).
pub fn row_has_steward_ask_policy_in_columns(
    columns: &[String],
    row: &[serde_json::Value],
    steward_policy_columns: &[String],
) -> bool {
    if steward_policy_columns.is_empty() {
        return false;
    }
    columns.iter().zip(row.iter()).any(|(col_name, cell)| {
        steward_column_matches(steward_policy_columns, None, col_name)
            && cell
                .as_str()
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("ASK"))
    })
}

/// Return `true` if `row` contains a cell equal to `"ASK"` (case-insensitively).
pub fn row_has_steward_ask_policy(row: &[serde_json::Value]) -> bool {
    row.iter().any(|cell| {
        cell.as_str()
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("ASK"))
    })
}

/// Return `true` if `haystack_lower` contains `needle_lower` bounded by non-alphanumeric characters.
fn contains_standalone_token_lower(haystack_lower: &str, needle_lower: &str) -> bool {
    if needle_lower.is_empty() {
        return false;
    }
    for (idx, _) in haystack_lower.match_indices(needle_lower) {
        let before_ok = haystack_lower[..idx]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        let after_idx = idx + needle_lower.len();
        let after_ok = haystack_lower[after_idx..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Determine whether a user's `MultiSelect` / clarification reply selects a candidate row from
/// `candidate_rows` that requires Steward `ALLOW`/`DENY` authorization (`Some(true)`) according to
/// explicitly configured `steward_policy_columns`, selects only non-`ASK` candidate row(s) (`Some(false)`),
/// or does not match any candidate row (`None`). Returns `None` whenever `steward_policy_columns` is empty.
pub fn resolve_clarification_steward_ask_from_rows_with_columns(
    clarification: &str,
    columns: &[String],
    candidate_rows: &[Vec<serde_json::Value>],
    steward_policy_columns: &[String],
) -> Option<bool> {
    if steward_policy_columns.is_empty() {
        return None;
    }
    resolve_clarification_steward_ask_from_rows_inner(
        clarification,
        candidate_rows,
        |row| {
            if columns.is_empty() {
                row_has_steward_ask_policy(row)
            } else {
                row_has_steward_ask_policy_in_columns(columns, row, steward_policy_columns)
            }
        },
        false,
    )
}

/// Determine whether a user's `MultiSelect` / clarification reply selects a candidate row from
/// `candidate_rows` that requires Steward `ALLOW`/`DENY` authorization (`Some(true)`), selects
/// only `"ALLOW"` candidate row(s) (`Some(false)`), or does not match any candidate row (`None`).
pub fn resolve_clarification_steward_ask_from_rows(
    clarification: &str,
    candidate_rows: &[Vec<serde_json::Value>],
) -> Option<bool> {
    resolve_clarification_steward_ask_from_rows_inner(
        clarification,
        candidate_rows,
        row_has_steward_ask_policy,
        true,
    )
}

fn resolve_clarification_steward_ask_from_rows_inner<F>(
    clarification: &str,
    candidate_rows: &[Vec<serde_json::Value>],
    row_is_ask: F,
    allow_raw_token_fallback: bool,
) -> Option<bool>
where
    F: Fn(&[serde_json::Value]) -> bool,
{
    let clar_trimmed = clarification.trim();
    if clar_trimmed.is_empty() {
        return None;
    }
    let clar_low = clar_trimmed.to_ascii_lowercase();

    // 1. Match against `summarize_candidate_row_choice(row)` (exact or substring inside comma-separated MultiSelect reply):
    let mut matched_indices = Vec::new();
    for (idx, row) in candidate_rows.iter().enumerate() {
        if let Some(summary) = summarize_candidate_row_choice(row) {
            let sum_low = summary.to_ascii_lowercase();
            if clar_low.contains(&sum_low) {
                matched_indices.push(idx);
            }
        }
    }

    // 2. If no full summary matched, match by primary key / first cell (`row[0]`) as a standalone token:
    if matched_indices.is_empty() {
        for (idx, row) in candidate_rows.iter().enumerate() {
            if let Some(first_cell) = row.first() {
                let key_str = match first_cell {
                    serde_json::Value::String(s) => s.trim().to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => String::new(),
                };
                if !key_str.is_empty()
                    && contains_standalone_token_lower(&clar_low, &key_str.to_ascii_lowercase())
                {
                    matched_indices.push(idx);
                }
            }
        }
    }

    // 3. If still no match, check if a distinctive cell value (or distinctive hyphen/underscore stem
    //    such as `"chamonix"` / `"rye"` inside `"CHAMONIX-RYE-3"`) unique to a single row in
    //    `candidate_rows` is mentioned:
    if matched_indices.is_empty() && candidate_rows.len() >= 2 {
        for (idx, row) in candidate_rows.iter().enumerate() {
            let has_unique_mentioned_cell = row.iter().skip(1).any(|cell| {
                let Some(s) = cell.as_str().map(str::trim) else {
                    return false;
                };
                if s.is_empty()
                    || s.eq_ignore_ascii_case("ASK")
                    || s.eq_ignore_ascii_case("ALLOW")
                    || s.eq_ignore_ascii_case("DENY")
                    || s.eq_ignore_ascii_case("CLEAR")
                    || s.eq_ignore_ascii_case("LOCKED")
                    || s.eq_ignore_ascii_case("ACTIVE")
                    || s.eq_ignore_ascii_case("ARCHIVED")
                {
                    return false;
                };
                let s_low = s.to_ascii_lowercase();
                if contains_standalone_token_lower(&clar_low, &s_low) {
                    let count_in_rows = candidate_rows
                        .iter()
                        .filter(|r| {
                            r.iter().any(|c| {
                                c.as_str()
                                    .is_some_and(|cs| cs.trim().eq_ignore_ascii_case(s))
                            })
                        })
                        .count();
                    if count_in_rows == 1 {
                        return true;
                    }
                }
                // Also check alphanumeric sub-stems (len >= 3) split on `-` or `_` (e.g. `"chamonix"` / `"rye"` -> `"CHAMONIX-RYE-3"`):
                s_low
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .filter(|stem| stem.len() >= 3 && !stem.chars().all(|c| c.is_ascii_digit()))
                    .any(|stem| {
                        if !contains_standalone_token_lower(&clar_low, stem) {
                            return false;
                        }
                        let stem_row_count = candidate_rows
                            .iter()
                            .filter(|r| {
                                r.iter().any(|c| {
                                    c.as_str().is_some_and(|cs| {
                                        cs.to_ascii_lowercase()
                                            .split(|ch: char| !ch.is_ascii_alphanumeric())
                                            .any(|part| part == stem)
                                    })
                                })
                            })
                            .count();
                        stem_row_count == 1
                    })
            });
            if has_unique_mentioned_cell {
                matched_indices.push(idx);
            }
        }
    }

    if !matched_indices.is_empty() {
        let any_ask = matched_indices
            .into_iter()
            .any(|idx| row_is_ask(&candidate_rows[idx]));
        return Some(any_ask);
    }

    // 4. Fallback if the clarification itself is a raw pipe/JSON row containing standalone `"ASK"` or `"ALLOW"`:
    if allow_raw_token_fallback {
        if contains_standalone_token_lower(&clar_low, "ask")
            && !contains_standalone_token_lower(&clar_low, "allow")
        {
            return Some(true);
        }
        if contains_standalone_token_lower(&clar_low, "allow")
            && !contains_standalone_token_lower(&clar_low, "ask")
        {
            return Some(false);
        }
    }

    None
}

/// Extract the user's clarification text from a completed `Mode::Ask` [`DriverObservation`]
/// (returning `None` if the observation was a binary `ALLOW`/`DENY` [`AskUserReply::Decision`]).
pub fn extract_clarification_from_ask_observation(obs: &DriverObservation) -> Option<String> {
    if !obs.ok || obs.mode != Mode::Ask {
        return None;
    }
    match &obs.ask_reply {
        Some(AskUserReply::Clarification { clarification }) => {
            return Some(clarification.trim().to_string());
        }
        Some(AskUserReply::Decision { .. }) => return None,
        None => {}
    }
    if let Some(first_row) = obs.rows.first() {
        let is_clarified = first_row
            .get(1)
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("CLARIFIED"));
        if is_clarified {
            if let Some(clar_str) = first_row.first().and_then(|v| v.as_str()) {
                return Some(clar_str.trim().to_string());
            }
        }
        let is_decision = first_row.get(2).and_then(|v| v.as_str()).is_some_and(|s| {
            let t = s.trim();
            t.eq_ignore_ascii_case("GRANTED") || t.eq_ignore_ascii_case("DENIED")
        });
        if is_decision {
            return None;
        }
    }
    if let Some(msg) = &obs.message {
        let trimmed = msg.trim();
        if !trimmed.is_empty()
            && !trimmed.eq_ignore_ascii_case("ALLOW")
            && !trimmed.eq_ignore_ascii_case("DENY")
            && !trimmed.eq_ignore_ascii_case("GRANTED")
            && !trimmed.eq_ignore_ascii_case("DENIED")
        {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Classify a `[MODE: ASK]` payload into [`AskPromptKind::Permission`] vs.
/// [`AskPromptKind::Clarification`] from structured JSON wire keys (`"kind"`, `"select"`,
/// `"multi"`, `"input"`, `"clarification"`, `"candidates"`) or explicit bracket tags
/// (`[permission]` / `[clarify]`), with zero English keyword or verb/noun heuristics.
pub fn classify_ask_prompt_kind(question: &str) -> AskPromptKind {
    let trimmed = question.trim();
    if trimmed.starts_with('{') {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(obj) = val.as_object() {
                if let Some(k) = obj.get("kind").and_then(|v| v.as_str()) {
                    if k.trim().eq_ignore_ascii_case("clarification")
                        || k.trim().eq_ignore_ascii_case("clarify")
                    {
                        return AskPromptKind::Clarification;
                    }
                    if k.trim().eq_ignore_ascii_case("permission") {
                        return AskPromptKind::Permission;
                    }
                }
                if obj.contains_key("select") || obj.contains_key("decision") {
                    return AskPromptKind::Permission;
                }
                if obj.contains_key("clarification")
                    || obj.contains_key("candidates")
                    || obj.contains_key("multi")
                    || obj.contains_key("input")
                {
                    return AskPromptKind::Clarification;
                }
            }
        }
    }

    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("[permission]") || lower.contains("[permission]") {
        return AskPromptKind::Permission;
    }

    AskPromptKind::Clarification
}

fn has_structured_permission_header(trimmed: &str, keyword: &str) -> bool {
    if trimmed.len() <= keyword.len()
        || !trimmed
            .get(..keyword.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(keyword))
    {
        return false;
    }
    let after = &trimmed[keyword.len()..];
    let first_ch = after.chars().next();
    if !first_ch.is_some_and(|c| c == ':' || c.is_whitespace()) {
        return false;
    }
    let rest = after.trim_start();
    rest.starts_with(':')
        || rest.starts_with('—')
        || rest.starts_with('–')
        || rest.starts_with("- ")
        || rest == "-"
}

/// Parse and validate a steward/user reply for `[MODE: ASK]` according to the question's [`AskPromptKind`]:
/// - [`AskPromptKind::Permission`]: only accepts an explicit binary permission decision token
///   (`ALLOW`/`ALLOWED`/`YES`/`Y`/`APPROVE`/`APPROVED` vs. `DENY`/`DENIED`/`NO`/`N`/`REJECT`/`REJECTED`),
///   JSON `{"decision": "ALLOW"|"DENY"}`, or an explicit structured `ALLOW:` / `DENY:` / `ALLOW —` /
///   `DENY —` / `ALLOW - ` / `DENY - ` header, rejecting casual sentences (`"yes please tell me more"`,
///   `"no problem"`, `"allow me to explain"`) with an explicit error.
/// - [`AskPromptKind::Clarification`]: any non-empty trimmed reply (or JSON `{"clarification": "..."}`)
///   produces [`AskUserReply::Clarification`] and never becomes [`AskDecision::Allow`].
pub fn parse_ask_user_reply_for_kind(kind: AskPromptKind, raw: &str) -> Result<AskUserReply> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("Empty [MODE: ASK] reply for {:?} prompt", kind);
    }

    match kind {
        AskPromptKind::Clarification => {
            if trimmed.eq_ignore_ascii_case("ALLOW")
                || trimmed.eq_ignore_ascii_case("ALLOWED")
                || trimmed.eq_ignore_ascii_case("APPROVE")
                || trimmed.eq_ignore_ascii_case("APPROVED")
                || trimmed.eq_ignore_ascii_case("YES")
                || has_structured_permission_header(trimmed, "ALLOW")
            {
                return Ok(AskUserReply::Decision {
                    decision: AskDecision::Allow,
                    raw_reply: trimmed.to_string(),
                });
            }
            if trimmed.eq_ignore_ascii_case("DENY")
                || trimmed.eq_ignore_ascii_case("DENIED")
                || trimmed.eq_ignore_ascii_case("REJECT")
                || trimmed.eq_ignore_ascii_case("REJECTED")
                || trimmed.eq_ignore_ascii_case("NO")
                || has_structured_permission_header(trimmed, "DENY")
            {
                return Ok(AskUserReply::Decision {
                    decision: AskDecision::Deny,
                    raw_reply: trimmed.to_string(),
                });
            }
            if trimmed.starts_with('{') {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let Some(obj) = parsed.as_object() {
                        if let Some(clar_str) = obj.get("clarification").and_then(|v| v.as_str()) {
                            let clar_trimmed = clar_str.trim();
                            if clar_trimmed.is_empty() {
                                bail!("Empty `clarification` field in [MODE: ASK] JSON reply");
                            }
                            return Ok(AskUserReply::Clarification {
                                clarification: clar_trimmed.to_string(),
                            });
                        }
                        for arr_key in ["selected", "choices", "options"] {
                            if let Some(arr) = obj.get(arr_key).and_then(|v| v.as_array()) {
                                let joined: Vec<String> = arr
                                    .iter()
                                    .map(|item| match item {
                                        serde_json::Value::String(s) => s.trim().to_string(),
                                        other => other.to_string(),
                                    })
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                if !joined.is_empty() {
                                    return Ok(AskUserReply::Clarification {
                                        clarification: joined.join(", "),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            let body = if trimmed
                .get(..8)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("clarify:"))
            {
                let rest = trimmed[8..].trim();
                if rest.is_empty() {
                    bail!("Empty clarification text after `clarify:` prefix");
                }
                rest.to_string()
            } else {
                trimmed.to_string()
            };
            Ok(AskUserReply::Clarification {
                clarification: body,
            })
        }
        AskPromptKind::Permission => {
            if trimmed.starts_with('{') {
                let parsed: serde_json::Value = serde_json::from_str(trimmed)
                    .with_context(|| format!("Invalid JSON in [MODE: ASK] reply: {trimmed:?}"))?;
                if let Some(obj) = parsed.as_object() {
                    if let Some(dec_str) = obj.get("decision").and_then(|v| v.as_str()) {
                        let dec_norm = dec_str.trim().to_ascii_lowercase();
                        let decision = match dec_norm.as_str() {
                            "allow" | "allowed" | "yes" | "y" | "approve" | "approved" => {
                                AskDecision::Allow
                            }
                            "deny" | "denied" | "no" | "n" | "reject" | "rejected" => {
                                AskDecision::Deny
                            }
                            _ => bail!(
                                "Unrecognized JSON decision {:?} in [MODE: ASK] reply; expected ALLOW or DENY",
                                dec_str
                            ),
                        };
                        let raw_reply = obj
                            .get("reply")
                            .and_then(|v| v.as_str())
                            .unwrap_or(trimmed)
                            .to_string();
                        return Ok(AskUserReply::Decision {
                            decision,
                            raw_reply,
                        });
                    }
                }
                bail!(
                    "Unrecognized JSON [MODE: ASK] permission reply {:?}; expected {{\"decision\":\"ALLOW|DENY\"}}",
                    trimmed
                );
            }

            let lower = trimmed.to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "deny" | "denied" | "no" | "n" | "reject" | "rejected"
            ) || has_structured_permission_header(trimmed, "DENY")
            {
                return Ok(AskUserReply::Decision {
                    decision: AskDecision::Deny,
                    raw_reply: trimmed.to_string(),
                });
            }

            if matches!(
                lower.as_str(),
                "allow" | "allowed" | "yes" | "y" | "approve" | "approved"
            ) || has_structured_permission_header(trimmed, "ALLOW")
            {
                return Ok(AskUserReply::Decision {
                    decision: AskDecision::Allow,
                    raw_reply: trimmed.to_string(),
                });
            }

            bail!(
                "Unrecognized [MODE: ASK] permission reply {:?}; expected ALLOW, DENY, YES, or NO",
                trimmed
            )
        }
    }
}

/// Strict permission parser for `[MODE: ASK]` replies, delegating to
/// `parse_ask_user_reply_for_kind(AskPromptKind::Permission, raw)`.
pub fn parse_ask_user_reply(raw: &str) -> Result<AskUserReply> {
    parse_ask_user_reply_for_kind(AskPromptKind::Permission, raw)
}

/// Canonical single-source-of-truth (`∀prompt`-parametric) classifier for [`AskInteractionMode`]
/// strictly from ground-truth SQLite observation state (`saw_steward_ask_row`,
/// `task_candidate_choices`, `active_read_rows`, `has_task_db_read`) and the emitted `question`.
pub fn resolve_ask_interaction_mode_from_observation_state(
    saw_steward_ask_row: bool,
    task_candidate_choices: &[String],
    active_read_rows: &[Vec<serde_json::Value>],
    has_task_db_read: bool,
    question: &str,
) -> AskInteractionMode {
    if let Some(mode) = extract_ask_interaction_mode(question) {
        return mode;
    }
    if classify_ask_prompt_kind(question) == AskPromptKind::Permission {
        return AskInteractionMode::StewardPermissionChoice {
            question: question.trim().to_string(),
            choices: vec![
                "ALLOW — Authorize database write".to_string(),
                "DENY  — Reject and cancel write".to_string(),
                NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string(),
            ],
        };
    }
    if task_candidate_choices.len() >= 2 {
        return AskInteractionMode::MultiSelect {
            question: question.trim().to_string(),
            choices: ensure_none_of_above_escape_choice(task_candidate_choices.to_vec()),
        };
    }
    if saw_steward_ask_row {
        return AskInteractionMode::StewardPermissionChoice {
            question: question.trim().to_string(),
            choices: vec![
                "ALLOW — Authorize database write".to_string(),
                "DENY  — Reject and cancel write".to_string(),
                NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string(),
            ],
        };
    }
    if has_task_db_read && active_read_rows.len() >= 2 && question.trim().ends_with('?') {
        let summarized: Vec<String> = active_read_rows
            .iter()
            .take(6)
            .filter_map(|row| summarize_candidate_row_choice(row))
            .collect();
        if summarized.len() >= 2 {
            return AskInteractionMode::MultiSelect {
                question: question.trim().to_string(),
                choices: ensure_none_of_above_escape_choice(summarized),
            };
        }
    }
    AskInteractionMode::ConversationalClarify
}
