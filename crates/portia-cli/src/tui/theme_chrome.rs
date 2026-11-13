//! `iocraft` workspace chrome: multi-scale `HeaderStatusBar`, pane titles, layout percentages,
//! and pure natural-language composer bar hints.

use super::{InteractionItem, SidebarMode, TwoPaneTuiState};
use iocraft::prelude::*;
use portia_core::{CartridgeBank, Mode};
use std::path::Path;

/// Default placeholder shown when the natural-language prompt composer is empty.
pub const COMPOSER_PLACEHOLDER: &str =
    "Ask a question or give a task in English, or type /help (/model 2b-base, /x, /export, /undo, /schema)...";

/// Extract a compact human-readable database or module basename for header/title chrome.
#[must_use]
pub fn db_basename(db_path: &str) -> &str {
    let trimmed = db_path.trim();
    if trimmed.is_empty() {
        return ":memory:";
    }
    if trimmed == ":memory:" || trimmed.starts_with("rust_ast::") {
        return trimmed;
    }
    Path::new(trimmed)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(trimmed)
}

/// Return horizontal `(left_pct, right_pct)` Flexbox split percentages for `mode`, or `None` when full-width.
#[must_use]
pub const fn sidebar_split_percentages(mode: SidebarMode) -> Option<(f32, f32)> {
    match mode {
        SidebarMode::Hidden => None,
        SidebarMode::SchemaInspector => Some((70.0, 30.0)),
        SidebarMode::RawTokenTrace => Some((46.0, 54.0)),
    }
}

/// Resolve the active `(turn_count, latest_mode)` indicator for the header bar.
#[must_use]
pub fn active_turn_mode_status(state: &TwoPaneTuiState) -> (usize, Option<Mode>) {
    let mut turn_count = 0usize;
    let mut latest_mode = None;
    for item in &state.interactions {
        if let InteractionItem::AssistantTurn {
            turn_index, mode, ..
        } = item
        {
            turn_count = (*turn_index + 1).max(turn_count + 1);
            latest_mode = Some(*mode);
        }
    }
    (turn_count, latest_mode)
}

/// Resolve the Steward / access governance status label and badge color.
#[must_use]
pub fn steward_status_badge(state: &TwoPaneTuiState) -> (String, Color) {
    if state.read_only {
        ("READ-ONLY".to_string(), Color::Cyan)
    } else if state.has_pending_steward_ask_row() {
        ("STEWARD: ASK PENDING".to_string(), Color::Yellow)
    } else if !state.steward_policy_columns.is_empty() {
        (
            format!("STEWARD: {}", state.steward_policy_columns.join(",")),
            Color::Yellow,
        )
    } else {
        ("READ/WRITE".to_string(), Color::Green)
    }
}

/// Build the dynamic header title for the Left Interaction Pane.
#[must_use]
pub fn build_left_pane_title(state: &TwoPaneTuiState) -> String {
    if state.effective_sidebar_mode() == SidebarMode::RawTokenTrace {
        "PANE 1: INTERACTION & SQLITE ENGINE".to_string()
    } else {
        let base = db_basename(&state.db_path);
        format!("Workspace · {base}")
    }
}

/// Build the dynamic header title and border color for the Right Drawer.
#[must_use]
pub fn build_right_pane_title(state: &TwoPaneTuiState) -> (String, Color) {
    match state.effective_sidebar_mode() {
        SidebarMode::RawTokenTrace => (
            "PANE 2: CONTEXT WINDOW & CARTRIDGE BOUNDARIES".to_string(),
            Color::Magenta,
        ),
        SidebarMode::SchemaInspector | SidebarMode::Hidden => {
            let n = state.tables.len();
            let noun = if n == 1 { "table" } else { "tables" };
            (format!("Schema ({n} {noun})"), Color::Cyan)
        }
    }
}

/// Build the 1-line title / keybinding hint bar on the composer dock.
#[must_use]
pub fn build_prompt_bar_title(state: &TwoPaneTuiState, active_label: Option<&str>) -> String {
    let prefix = active_label
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(if state.is_streaming {
            "[streaming...]"
        } else {
            "portia>"
        });
    let schema_hint = match state.effective_sidebar_mode() {
        SidebarMode::SchemaInspector => "Tab: full-width",
        SidebarMode::Hidden | SidebarMode::RawTokenTrace => "Tab: schema",
    };
    let trace_hint = if state.effective_sidebar_mode() == SidebarMode::RawTokenTrace {
        "Ctrl+T: hide trace"
    } else {
        "Ctrl+T: trace"
    };
    let esc_hint = if state.is_streaming {
        "Esc/Ctrl+C: cancel turn"
    } else {
        "Ctrl+D: exit"
    };
    format!("{prefix} · Enter: send · {schema_hint} · {trace_hint} · {esc_hint}")
}

/// Props for [`HeaderStatusBar`].
#[derive(Default, Props)]
pub struct HeaderStatusBarProps {
    pub state: TwoPaneTuiState,
}

/// Top status bar rendering active DB, multi-scale `Qwen3.5` badge (`0.8B` / `2B` / `4B` / `9B`),
/// Steward governance, `/x` & `/undo` counters, and active drawer telemetry.
#[component]
pub fn HeaderStatusBar(props: &HeaderStatusBarProps) -> impl Into<AnyElement<'static>> {
    let state = &props.state;
    let mode = state.effective_sidebar_mode();
    let base = db_basename(&state.db_path);
    let scale = state.model_scale_spec();
    let tbl_count = state.tables.len();
    let tbl_noun = if tbl_count == 1 { "table" } else { "tables" };
    let (gov_label, gov_color) = steward_status_badge(state);
    let (turn_count, latest_mode) = active_turn_mode_status(state);

    let mode_indicator = if state.is_streaming {
        match latest_mode {
            Some(m) => format!("STREAMING · {}", m.as_str()),
            None => "STREAMING".to_string(),
        }
    } else {
        match latest_mode {
            Some(m) => format!("Turn {turn_count} · {}", m.as_str()),
            None => "READY".to_string(),
        }
    };

    let mut left_spans = vec![
        MixedTextContent::new("[portia]")
            .color(Color::Cyan)
            .weight(Weight::Bold),
        MixedTextContent::new(format!(" {base} "))
            .color(Color::White)
            .weight(Weight::Bold),
        MixedTextContent::new(format!("({tbl_count} {tbl_noun}) │ ")).color(Color::Grey),
        MixedTextContent::new(format!("{} ", scale.scale_badge))
            .color(Color::Magenta)
            .weight(Weight::Bold),
        MixedTextContent::new(format!("(L{}) │ ", scale.router_layer)).color(Color::Grey),
        MixedTextContent::new(format!("{gov_label} "))
            .color(gov_color)
            .weight(Weight::Bold),
    ];

    if state.expanded_display {
        left_spans.push(MixedTextContent::new("│ /x ").color(Color::Cyan));
    }
    if state.undo_depth > 0 {
        left_spans.push(
            MixedTextContent::new(format!("│ UNDO:{} ", state.undo_depth)).color(Color::Yellow),
        );
    }

    let (right_text, right_color) = match mode {
        SidebarMode::RawTokenTrace => {
            let bank_str = match state.last_mounted_bank {
                Some(CartridgeBank::OrientTrunk) => "OrientTrunk",
                Some(CartridgeBank::ModePayload(m)) => m.as_str(),
                Some(CartridgeBank::Base) | None => "Base W_0",
            };
            (
                format!(
                    "[{mode_indicator}] │ TRACE: {bank_str} · {} tok · {} swaps",
                    state.total_context_tokens, state.cartridge_switch_count
                ),
                Color::Magenta,
            )
        }
        SidebarMode::SchemaInspector => (
            format!("[{mode_indicator}] │ [Drawer: Schema]"),
            Color::Cyan,
        ),
        SidebarMode::Hidden => (
            format!("[{mode_indicator}] │ [View: Full-Width]"),
            Color::Grey,
        ),
    };

    element! {
        View(
            width: 100pct,
            height: 1,
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
        ) {
            View(flex_shrink: 1.0_f32, overflow: Overflow::Hidden) {
                MixedText(wrap: TextWrap::NoWrap, contents: left_spans)
            }
            View(flex_shrink: 0.0_f32) {
                Text(
                    content: right_text,
                    color: right_color,
                    wrap: TextWrap::NoWrap,
                    weight: if state.is_streaming { Weight::Bold } else { Weight::Normal },
                )
            }
        }
    }
}
