//! `iocraft` Left Interaction & Transcript Pane (`Workspace` mode ⋈ power-user `RawTokenTrace` OODA mode).

use super::result_table_view::render_observation_result_lines;
use super::sql_highlight::{
    build_write_steward_card_lines, format_clean_result_blocks, format_user_prose_blocks,
    highlight_sql_mixed_lines, strip_wire_route_tags,
};
use super::theme_chrome::{build_left_pane_title, db_basename};
use super::{InteractionItem, SidebarMode, TwoPaneTuiState};
use iocraft::prelude::*;
use portia_core::Mode;

/// Styled line entry for `LeftTranscriptPane`, specifying whether `iocraft` should word-wrap it.
#[derive(Clone, Default)]
pub struct TranscriptLine {
    pub contents: Vec<MixedTextContent>,
    pub wrap: bool,
}

impl TranscriptLine {
    fn wrapped(contents: Vec<MixedTextContent>) -> Self {
        Self {
            contents,
            wrap: true,
        }
    }

    fn no_wrap(contents: Vec<MixedTextContent>) -> Self {
        Self {
            contents,
            wrap: false,
        }
    }

    fn blank() -> Self {
        Self {
            contents: Vec::new(),
            wrap: false,
        }
    }
}

fn mode_pill(mode: Mode) -> (&'static str, Color) {
    match mode {
        Mode::Probe => ("[PROBE]", Color::Cyan),
        Mode::Read => ("[READ]", Color::Cyan),
        Mode::Ask => ("[ASK]", Color::Magenta),
        Mode::Write => ("[WRITE]", Color::Yellow),
        Mode::Result => ("[RESULT]", Color::Green),
        Mode::Abstain => ("[ABSTAIN]", Color::Red),
    }
}

fn normalize_plan_text(plan: &str) -> String {
    let cleaned = strip_wire_route_tags(plan);
    if cleaned.is_empty() {
        return "…".to_string();
    }
    let parts: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        "…".to_string()
    } else {
        parts.join(" ")
    }
}

fn push_workspace_gui_card(
    lead_in: &str,
    question: &str,
    banner: &str,
    choices: &[String],
    out: &mut Vec<TranscriptLine>,
) {
    let has_lead = !lead_in.is_empty()
        && !lead_in.ends_with('?')
        && crate::driver::extract_ask_interaction_mode(lead_in).is_none();
    if has_lead {
        for block in format_user_prose_blocks(lead_in, false) {
            out.push(TranscriptLine::wrapped(block));
        }
    } else {
        let clean_q = strip_wire_route_tags(question);
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            format!("  {clean_q}"),
        )
        .color(Color::White)
        .weight(Weight::Bold)]));
    }
    out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
        banner.to_string(),
    )
    .color(Color::Yellow)
    .weight(Weight::Bold)]));
    for opt in choices {
        let is_escape =
            opt.to_ascii_lowercase().starts_with("none of the above") || opt.starts_with("Other");
        let color = if is_escape { Color::Grey } else { Color::Cyan };
        out.push(TranscriptLine::wrapped(vec![
            MixedTextContent::new("    • ").color(color),
            MixedTextContent::new(opt.clone()).color(color),
        ]));
    }
}

fn build_clean_workspace_lines(state: &TwoPaneTuiState, inner_width: usize) -> Vec<TranscriptLine> {
    let mut out = Vec::new();
    let sep_w = inner_width.saturating_sub(2).max(20);
    let base = db_basename(&state.db_path);
    let scale = state.model_scale_spec();

    if state.interactions.is_empty() {
        out.push(TranscriptLine::wrapped(vec![
            MixedTextContent::new(format!("Connected to {base}"))
                .color(Color::White)
                .weight(Weight::Bold),
            MixedTextContent::new(format!(" · {} table(s) loaded", state.tables.len()))
                .color(Color::Grey),
        ]));
        out.push(TranscriptLine::wrapped(vec![
            MixedTextContent::new("Model: ").color(Color::Grey),
            MixedTextContent::new(format!("{} ", scale.scale_badge))
                .color(Color::Magenta)
                .weight(Weight::Bold),
            MixedTextContent::new(format!(
                "({} · {}) · Cartridge: {}",
                scale.canonical_family,
                scale.format_architecture_summary(),
                state.cartridge_label
            ))
            .color(Color::Grey),
        ]));
        if !state.tables.is_empty() {
            let summary: Vec<String> = state
                .tables
                .iter()
                .take(4)
                .map(|t| format!("{} ({})", t.table_name, t.row_count))
                .collect();
            out.push(TranscriptLine::wrapped(vec![
                MixedTextContent::new("Tables: ").color(Color::Grey),
                MixedTextContent::new(summary.join(", ")).color(Color::Grey),
            ]));
        }
        out.push(TranscriptLine::no_wrap(vec![MixedTextContent::new(
            "─".repeat(sep_w),
        )
        .color(Color::Grey)]));
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            "Ready — ask a question or give a task in English, or type /help (/model 2b-base, /x, /export, /undo, /schema).",
        )
        .color(Color::Grey)]));
        return out;
    }

    for item in &state.interactions {
        match item {
            InteractionItem::UserPrompt { user_text, .. } => {
                let cleaned_user = strip_wire_route_tags(user_text);
                out.push(TranscriptLine::wrapped(vec![
                    MixedTextContent::new("❯ ")
                        .color(Color::Cyan)
                        .weight(Weight::Bold),
                    MixedTextContent::new(cleaned_user)
                        .color(Color::White)
                        .weight(Weight::Bold),
                ]));
                out.push(TranscriptLine::blank());
            }
            InteractionItem::AssistantTurn {
                turn_index,
                mode,
                plan,
                payload,
                ..
            } => {
                let (pill, pill_color) = mode_pill(*mode);
                let normalized_plan = normalize_plan_text(plan);
                out.push(TranscriptLine::wrapped(vec![
                    MixedTextContent::new(format!("  {pill}"))
                        .color(pill_color)
                        .weight(Weight::Bold),
                    MixedTextContent::new(" Plan: ").color(Color::Grey),
                    MixedTextContent::new(normalized_plan)
                        .color(Color::Grey)
                        .italic(),
                ]));

                if !payload.trim().is_empty() {
                    match mode {
                        Mode::Probe | Mode::Read | Mode::Write => {
                            for sql_line in highlight_sql_mixed_lines(payload, *mode == Mode::Write)
                            {
                                out.push(TranscriptLine::wrapped(sql_line));
                            }
                        }
                        Mode::Ask => {
                            let clean_payload = strip_wire_route_tags(payload);
                            match state.classify_coroutine_step_effect(
                                *turn_index,
                                Mode::Ask,
                                &clean_payload,
                            ) {
                                crate::driver::CoroutineStepEffect::SuspendForGuiChoice {
                                    lead_in_prose,
                                    gui_mode,
                                } => {
                                    let clean_lead = strip_wire_route_tags(&lead_in_prose);
                                    match gui_mode {
                                        crate::AskInteractionMode::StewardPermissionChoice {
                                            question,
                                            choices,
                                        } => push_workspace_gui_card(
                                            &clean_lead,
                                            &question,
                                            "  ⛨ Steward Authorization (Select 1):",
                                            &choices,
                                            &mut out,
                                        ),
                                        crate::AskInteractionMode::MultiSelect {
                                            question,
                                            choices,
                                        } => push_workspace_gui_card(
                                            &clean_lead,
                                            &question,
                                            "  ▸ Select Candidate Row(s) (<Space> toggle, <Enter> submit):",
                                            &choices,
                                            &mut out,
                                        ),
                                        crate::AskInteractionMode::ConversationalClarify => {}
                                    }
                                }
                                _ => {
                                    let formatted = strip_wire_route_tags(
                                        &crate::driver::format_ask_payload_for_display(
                                            &clean_payload,
                                        ),
                                    );
                                    for block in format_user_prose_blocks(&formatted, false) {
                                        out.push(TranscriptLine::wrapped(block));
                                    }
                                }
                            }
                        }
                        Mode::Result => {
                            for block in format_clean_result_blocks(payload) {
                                out.push(TranscriptLine::wrapped(block));
                            }
                        }
                        Mode::Abstain => {
                            let clean_abs = strip_wire_route_tags(payload);
                            for block in format_user_prose_blocks(&clean_abs, true) {
                                out.push(TranscriptLine::wrapped(block));
                            }
                        }
                    }
                }
                out.push(TranscriptLine::blank());
            }
            InteractionItem::Observation { observation, .. } => {
                if !observation.ok {
                    let err_msg = observation
                        .error
                        .as_deref()
                        .unwrap_or("SQLite execution failed");
                    out.push(TranscriptLine::wrapped(vec![
                        MixedTextContent::new("  ✗ Error: ")
                            .color(Color::Red)
                            .weight(Weight::Bold),
                        MixedTextContent::new(err_msg).color(Color::White),
                    ]));
                } else if observation.mode == Mode::Ask {
                    if let Some(msg) = &observation.message {
                        out.push(TranscriptLine::wrapped(vec![
                            MixedTextContent::new("  ↳ Reply: ").color(Color::Cyan),
                            MixedTextContent::new(msg)
                                .color(Color::White)
                                .weight(Weight::Bold),
                        ]));
                    }
                } else if !observation.rows.is_empty() {
                    let (tbl_lines, wrap_tbl) = render_observation_result_lines(
                        observation,
                        &state.steward_policy_columns,
                        inner_width,
                        state.expanded_display,
                        state.row_display_limit,
                    );
                    for line in tbl_lines {
                        out.push(TranscriptLine {
                            contents: line,
                            wrap: wrap_tbl,
                        });
                    }
                } else if observation.mode == Mode::Write {
                    for line in build_write_steward_card_lines(observation) {
                        out.push(TranscriptLine::wrapped(line));
                    }
                } else if matches!(observation.mode, Mode::Read | Mode::Probe) {
                    out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                        "  ✓ 0 rows returned",
                    )
                    .color(Color::Grey)]));
                }
                out.push(TranscriptLine::blank());
            }
            InteractionItem::HarnessNote { title, detail } => {
                out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                    format!("  ℹ {title}"),
                )
                .color(Color::Yellow)
                .weight(Weight::Bold)]));
                for dl in detail.lines().filter(|l| !l.trim().is_empty()) {
                    out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                        format!("    {}", dl.trim()),
                    )
                    .color(Color::Grey)]));
                }
                out.push(TranscriptLine::blank());
            }
        }
    }
    out
}

fn push_trace_gui_card(
    lead_in_prose: &str,
    question: &str,
    choices: &[String],
    direct_header: &str,
    inner_banner: &str,
    out: &mut Vec<TranscriptLine>,
) {
    let has_prose_lead = !lead_in_prose.trim().is_empty()
        && !lead_in_prose.trim().ends_with('?')
        && crate::driver::extract_ask_interaction_mode(lead_in_prose).is_none();
    if has_prose_lead {
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            "├─ [OODA: CHAT -> STRUCTURED-GUI] Assistant:",
        )
        .color(Color::Green)
        .weight(Weight::Bold)]));
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            format!("│    {lead_in_prose}"),
        )
        .color(Color::White)]));
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            inner_banner,
        )
        .color(Color::Yellow)
        .weight(Weight::Bold)]));
    } else {
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            direct_header,
        )
        .color(Color::Yellow)
        .weight(Weight::Bold)]));
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            format!("│    {question}"),
        )
        .color(Color::White)
        .weight(Weight::Bold)]));
    }
    for opt in choices {
        let is_escape =
            opt.to_ascii_lowercase().starts_with("none of the above") || opt.starts_with("Other");
        let color = if is_escape { Color::Grey } else { Color::Cyan };
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            format!("│      • {opt}"),
        )
        .color(color)]));
    }
}

fn build_ooda_trace_lines(state: &TwoPaneTuiState, inner_width: usize) -> Vec<TranscriptLine> {
    let scale = state.model_scale_spec();
    let sep_w = inner_width.saturating_sub(2).max(20);
    let mut out = vec![TranscriptLine::wrapped(vec![MixedTextContent::new(
        format!(
            "Database: {} | Model: {} ({})",
            state.db_path,
            scale.scale_badge,
            scale.format_architecture_summary()
        ),
    )])];
    if !state.tables.is_empty() {
        let tbl_summary: Vec<String> = state
            .tables
            .iter()
            .map(|t| {
                format!(
                    "{} ({} cols, {} rows)",
                    t.table_name,
                    t.columns.len(),
                    t.row_count
                )
            })
            .collect();
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            format!("Tables:   {}", tbl_summary.join(", ")),
        )]));
    }
    out.push(TranscriptLine::no_wrap(vec![MixedTextContent::new(
        "─".repeat(sep_w),
    )
    .color(Color::Grey)]));

    if state.interactions.is_empty() {
        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
            "Ready — ask a question in English or type /help (/model, /x, /export, /undo, /schema, /trace).",
        )
        .color(Color::Grey)]));
        return out;
    }

    for item in &state.interactions {
        match item {
            InteractionItem::UserPrompt { user_text, .. } => {
                out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                    "┌─ [OODA: INQUIRE] You:",
                )
                .color(Color::Cyan)
                .weight(Weight::Bold)]));
                out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                    format!("│    {user_text}"),
                )
                .color(Color::White)]));
                out.push(TranscriptLine::blank());
            }
            InteractionItem::AssistantTurn {
                turn_index,
                mode,
                plan,
                payload,
                ..
            } => {
                let route_name = mode
                    .route_tag()
                    .open_tag()
                    .trim_matches(|c| c == '<' || c == '>');
                let step_effect = state.classify_coroutine_step_effect(*turn_index, *mode, payload);
                let (orient_txt, orient_col) = match (*mode, &step_effect) {
                    (Mode::Ask, crate::driver::CoroutineStepEffect::SuspendForGuiChoice { .. }) => {
                        (
                            format!(
                            "├─ [OODA: ORIENT] Mode: ASK · Structured-GUI (route: <{route_name}>)"
                        ),
                            Color::Magenta,
                        )
                    }
                    (Mode::Ask, crate::driver::CoroutineStepEffect::YieldToPrompt) => (
                        format!("├─ [OODA: ORIENT] Mode: ASK · Chitchat (route: <{route_name}>)"),
                        Color::Cyan,
                    ),
                    _ => (
                        format!(
                            "├─ [OODA: ORIENT] Mode: {} (route: <{route_name}>)",
                            mode.as_str()
                        ),
                        Color::Magenta,
                    ),
                };
                out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                    orient_txt,
                )
                .color(orient_col)
                .weight(Weight::Bold)]));

                let clean_plan = normalize_plan_text(plan);
                out.push(TranscriptLine::wrapped(vec![
                    MixedTextContent::new("├─ [OODA: DECIDE] Plan: ")
                        .color(Color::Cyan)
                        .weight(Weight::Bold),
                    MixedTextContent::new(clean_plan)
                        .color(Color::White)
                        .italic(),
                ]));

                if !payload.is_empty() {
                    match mode {
                        Mode::Probe | Mode::Read | Mode::Write => {
                            out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                                format!("├─ [OODA: ACT -> sql] SQL {}:", mode.as_str()),
                            )
                            .color(Color::Yellow)
                            .weight(Weight::Bold)]));
                            for pl_line in strip_wire_route_tags(payload).lines() {
                                out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                                    format!("│    sql> {pl_line}"),
                                )
                                .color(Color::Yellow)]));
                            }
                        }
                        Mode::Ask => match step_effect {
                            crate::driver::CoroutineStepEffect::SuspendForGuiChoice {
                                lead_in_prose,
                                gui_mode,
                            } => match gui_mode {
                                crate::AskInteractionMode::StewardPermissionChoice {
                                    question,
                                    choices,
                                } => push_trace_gui_card(
                                    &lead_in_prose,
                                    &question,
                                    &choices,
                                    "├─ [OODA: STRUCTURED-GUI -> to_user] Steward Authorization (Select 1):",
                                    "│    Steward Authorization (Select 1):",
                                    &mut out,
                                ),
                                crate::AskInteractionMode::MultiSelect { question, choices } => {
                                    push_trace_gui_card(
                                        &lead_in_prose,
                                        &question,
                                        &choices,
                                        "├─ [OODA: STRUCTURED-GUI -> to_user] Candidate Disambiguation (Multi-Select):",
                                        "│    Select Candidate Row(s) (<Space> toggle, <Enter> submit):",
                                        &mut out,
                                    )
                                }
                                crate::AskInteractionMode::ConversationalClarify => {}
                            },
                            _ => {
                                out.push(TranscriptLine::wrapped(vec![
                                    MixedTextContent::new("╰─ [OODA: CHITCHAT -> to_user] Assistant:")
                                        .color(Color::Green)
                                        .weight(Weight::Bold),
                                ]));
                                let formatted_ask = strip_wire_route_tags(
                                    &crate::driver::format_ask_payload_for_display(payload),
                                );
                                for block in format_user_prose_blocks(&formatted_ask, false) {
                                    out.push(TranscriptLine::wrapped(block));
                                }
                            }
                        },
                        Mode::Result => {
                            out.push(TranscriptLine::wrapped(vec![
                                MixedTextContent::new("╰─ [OODA: RESULT -> to_user] Assistant:")
                                    .color(Color::Green)
                                    .weight(Weight::Bold),
                            ]));
                            for block in format_clean_result_blocks(payload) {
                                out.push(TranscriptLine::wrapped(block));
                            }
                        }
                        Mode::Abstain => {
                            out.push(TranscriptLine::wrapped(vec![
                                MixedTextContent::new("╰─ [OODA: ABSTAIN -> to_user] Refusal:")
                                    .color(Color::Red)
                                    .weight(Weight::Bold),
                            ]));
                            let clean_abs = strip_wire_route_tags(payload);
                            for block in format_user_prose_blocks(&clean_abs, true) {
                                out.push(TranscriptLine::wrapped(block));
                            }
                        }
                    }
                }
                out.push(TranscriptLine::blank());
            }
            InteractionItem::Observation { observation, .. } => {
                if !observation.ok {
                    out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                        "├─ [OODA: OBSERVE <- SQLite Error]:",
                    )
                    .color(Color::Red)
                    .weight(Weight::Bold)]));
                    if let Some(err) = &observation.error {
                        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                            format!("│    {err}"),
                        )
                        .color(Color::White)]));
                    }
                } else if observation.mode == Mode::Ask {
                    if let Some(msg) = &observation.message {
                        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                            "├─ [OODA: STEWARD REPLY] You:",
                        )
                        .color(Color::Cyan)
                        .weight(Weight::Bold)]));
                        out.push(TranscriptLine::wrapped(vec![MixedTextContent::new(
                            format!("│    {msg}"),
                        )
                        .color(Color::White)]));
                    }
                } else if !observation.rows.is_empty() {
                    let (tbl_lines, wrap_tbl) = render_observation_result_lines(
                        observation,
                        &state.steward_policy_columns,
                        inner_width,
                        state.expanded_display,
                        state.row_display_limit,
                    );
                    for line in tbl_lines {
                        out.push(TranscriptLine {
                            contents: line,
                            wrap: wrap_tbl,
                        });
                    }
                } else if observation.mode == Mode::Write {
                    for line in build_write_steward_card_lines(observation) {
                        out.push(TranscriptLine::wrapped(line));
                    }
                }
                out.push(TranscriptLine::blank());
            }
            InteractionItem::HarnessNote { title, detail } => {
                out.push(TranscriptLine::wrapped(vec![
                    MixedTextContent::new(format!("├─ [HARNESS: {title}]"))
                        .color(Color::Yellow)
                        .weight(Weight::Bold),
                    MixedTextContent::new(format!(" {detail}")).color(Color::Grey),
                ]));
                out.push(TranscriptLine::blank());
            }
        }
    }
    out
}

/// Build left-pane [`TranscriptLine`]s (`Workspace` mode by default, `OODA Trace` in `RawTokenTrace`).
#[must_use]
pub fn build_left_interaction_lines(
    state: &TwoPaneTuiState,
    inner_width: usize,
) -> Vec<TranscriptLine> {
    if state.effective_sidebar_mode() == SidebarMode::RawTokenTrace {
        build_ooda_trace_lines(state, inner_width)
    } else {
        build_clean_workspace_lines(state, inner_width)
    }
}

/// Props for [`LeftTranscriptPane`].
#[derive(Default, Props)]
pub struct LeftTranscriptPaneProps {
    pub state: TwoPaneTuiState,
    pub inner_width: usize,
    pub scroll_handle: Option<Ref<ScrollViewHandle>>,
}

/// Left-hand `iocraft` conversation, SQL observation, and Steward transcript pane.
#[component]
pub fn LeftTranscriptPane(props: &LeftTranscriptPaneProps) -> impl Into<AnyElement<'static>> {
    let title = build_left_pane_title(&props.state);
    let lines = build_left_interaction_lines(&props.state, props.inner_width);
    let auto_scroll = props.state.left_viewport.follow_tail;

    element! {
        View(
            width: 100pct,
            height: 100pct,
            flex_direction: FlexDirection::Column,
            border_style: BorderStyle::Round,
            border_color: Color::Cyan,
        ) {
            View(width: 100pct, padding_left: 1, padding_right: 1) {
                Text(content: title, color: Color::Cyan, weight: Weight::Bold, wrap: TextWrap::NoWrap)
            }
            View(flex_grow: 1.0_f32, width: 100pct, overflow: Overflow::Hidden, padding_left: 1, padding_right: 1) {
                ScrollView(
                    auto_scroll: auto_scroll,
                    keyboard_scroll: Some(false),
                    scrollbar: Some(true),
                    handle: props.scroll_handle,
                ) {
                    View(width: 100pct, flex_direction: FlexDirection::Column) {
                        #(lines.into_iter().map(|line| {
                            if line.contents.is_empty()
                                || (line.contents.len() == 1 && line.contents[0].text.is_empty())
                            {
                                element! { View(height: 1) }.into_any()
                            } else {
                                let wrap = if line.wrap {
                                    TextWrap::Wrap
                                } else {
                                    TextWrap::NoWrap
                                };
                                element! {
                                    MixedText(wrap: wrap, contents: line.contents)
                                }.into_any()
                            }
                        }))
                    }
                }
            }
        }
    }
}
