use super::{
    render_two_pane_tui_frame, ContextSpanKind, InteractionItem, PortiaTuiApp, SidebarMode,
    TwoPaneTuiState, UiLocalMutation, UiToWorkerMsg,
};
use crate::driver::ObservationOperation;
use crate::{DriverObservation, TableSchemaInfo};
use futures::stream::{self, StreamExt};
use iocraft::prelude::*;
use portia_core::bpe::TOKEN_IM_START;
use portia_core::domain::{ColumnSchema, ForeignKeyDeleteRule, ForeignKeyEdge};
use portia_core::relay::RelayPhase;
use portia_core::{CartridgeBank, Mode, PenOwner, StewardAuthState};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use unicode_width::UnicodeWidthStr;

fn simulate_iocraft_keys_and_render(
    state: &mut TwoPaneTuiState,
    keys: &[(KeyCode, KeyModifiers)],
    width: usize,
    height: usize,
) -> String {
    let (cmd_tx, cmd_rx) = smol::channel::unbounded::<UiToWorkerMsg>();
    let mut events: Vec<TerminalEvent> = keys
        .iter()
        .map(|&(code, modifiers)| {
            let mut ev = KeyEvent::new(KeyEventKind::Press, code);
            ev.modifiers = modifiers;
            TerminalEvent::Key(ev)
        })
        .collect();
    let mut exit_ev = KeyEvent::new(KeyEventKind::Press, KeyCode::Char('q'));
    exit_ev.modifiers = KeyModifiers::CONTROL;
    events.push(TerminalEvent::Key(exit_ev));

    let init = state.clone();
    let canvases: Vec<String> = smol::block_on(async move {
        element! {
            PortiaTuiApp(
                initial_state: init,
                cmd_tx: Some(cmd_tx),
                fixed_size: Some((width as u16, height as u16)),
            )
        }
        .mock_terminal_render_loop(MockTerminalConfig::with_events(stream::iter(events)))
        .map(|c| c.to_string())
        .collect()
        .await
    });

    while let Ok(msg) = cmd_rx.try_recv() {
        if let UiToWorkerMsg::UiMutation(m) = msg {
            match m {
                UiLocalMutation::SetSidebarMode(mode) => state.set_sidebar_mode(mode),
                UiLocalMutation::ToggleSchemaDrawer => state.toggle_schema_drawer(),
                UiLocalMutation::ToggleRawTokenTrace => state.toggle_raw_token_trace(),
                UiLocalMutation::ToggleInlineTokens => {
                    state.show_inline_token_ids = !state.show_inline_token_ids;
                }
            }
        }
    }

    canvases
        .last()
        .cloned()
        .unwrap_or_else(|| render_two_pane_tui_frame(state, width, height, false))
}

fn sample_schema() -> Vec<TableSchemaInfo> {
    let campaigns = TableSchemaInfo::new(
        "campaigns",
        vec![
            ColumnSchema::with_sqlite_metadata(0, "campaign_id", "INTEGER", true, true),
            ColumnSchema::with_sqlite_metadata(1, "slug", "TEXT", true, false),
        ],
    )
    .with_row_count(3);

    let ledger = TableSchemaInfo::new(
        "ledger_entries",
        vec![
            ColumnSchema::with_sqlite_metadata(0, "entry_id", "INTEGER", true, true),
            ColumnSchema::with_sqlite_metadata(1, "campaign_id", "INTEGER", true, false),
            ColumnSchema::with_sqlite_metadata(2, "amount_cents", "INTEGER", true, false),
            ColumnSchema::with_sqlite_metadata(3, "policy_decision", "TEXT", true, false),
            ColumnSchema::with_sqlite_metadata(4, "lock_state", "TEXT", true, false),
        ],
    )
    .with_row_count(42)
    .with_foreign_keys(vec![ForeignKeyEdge {
        child_table: "ledger_entries".to_string(),
        child_fk_column: "campaign_id".to_string(),
        parent_table: "campaigns".to_string(),
        parent_pk_column: "campaign_id".to_string(),
        on_delete: ForeignKeyDeleteRule::Cascade,
    }]);

    vec![campaigns, ledger]
}

fn new_test_state(db: &str, tables: Vec<TableSchemaInfo>) -> TwoPaneTuiState {
    TwoPaneTuiState::new(
        db,
        "Qwen/Qwen3.5-0.8B-Base",
        "portia_v3_unified_cartridge.json",
        tables,
    )
}

fn attach_sample_bpe_spans(state: &mut TwoPaneTuiState) {
    state.record_user_prompt_envelope(
        0,
        "",
        "Inspect ledger_entries",
        "<|im_start|>user\nInspect ledger_entries\n",
        vec![TOKEN_IM_START, 872, 198],
    );
    state.push_context_span(
        ContextSpanKind::RelayPenSpan {
            turn_index: 0,
            span_index: 0,
            phase: RelayPhase::ModeLabel,
        },
        PenOwner::RouterPen,
        Some(CartridgeBank::OrientTrunk),
        CartridgeBank::OrientTrunk,
        "READ".to_string(),
        vec![18104],
    );
}

#[allow(clippy::too_many_arguments)]
fn make_obs(
    mode: Mode,
    db_version: u64,
    table_versions: BTreeMap<String, u64>,
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    rows_affected: usize,
    op: &str,
    steward_auth: StewardAuthState,
) -> DriverObservation {
    DriverObservation {
        event_id: "ev-test".to_string(),
        mode,
        ok: true,
        db_version,
        table_versions,
        columns,
        rows,
        rows_affected,
        message: None,
        error: None,
        operation: ObservationOperation::from_wire_str(op),
        ask_reply: None,
        steward_auth,
        is_chat_continuation: false,
    }
}

#[test]
fn default_workspace_launches_in_schema_inspector_or_hidden_without_bpe_noise() {
    let empty_state = new_test_state("/var/db/empty.sqlite", vec![]);
    assert!(!empty_state.show_debug_sidebar);
    assert_eq!(empty_state.effective_sidebar_mode(), SidebarMode::Hidden);
    let empty_frame = render_two_pane_tui_frame(&empty_state, 120, 28, false);
    assert!(empty_frame.contains("Workspace · empty.sqlite"));
    assert!(empty_frame.contains("Enter: send · Tab: schema · Ctrl+T: trace · Ctrl+D: exit"));
    assert!(!empty_frame.contains("Esc: quit"));
    assert!(!empty_frame.contains("Schema ("));
    assert!(!empty_frame.contains("PANE 2: CONTEXT WINDOW"));

    let mut populated = new_test_state("/srv/data/treasury.sqlite", sample_schema())
        .with_steward_policy_columns(["policy_decision"])
        .with_read_only(true);
    attach_sample_bpe_spans(&mut populated);

    assert!(!populated.show_debug_sidebar);
    assert_eq!(
        populated.effective_sidebar_mode(),
        SidebarMode::SchemaInspector
    );

    let frame = render_two_pane_tui_frame(&populated, 148, 36, false);
    for expected in [
        "treasury.sqlite",
        "2 tables",
        "READ-ONLY",
        "Schema (2 tables)",
        "ledger_entries",
        "42 rows",
        "[PK]",
        "-> campaigns.campaign_id",
        "[STEWARD]",
        "[LOCK]",
    ] {
        assert!(frame.contains(expected), "missing {expected}");
    }
    assert!(!frame.contains("248045"));
    assert!(!frame.contains("PANE 2: CONTEXT WINDOW"));
}

#[test]
fn aligned_box_drawing_sql_result_table_and_steward_badges() {
    let mut state = new_test_state("/srv/data/treasury.sqlite", sample_schema())
        .with_steward_policy_columns(["policy_decision"]);

    state.interactions.push(InteractionItem::UserPrompt {
        turn_index: 0,
        schema_header: String::new(),
        user_text: "List ledger entries requiring review".to_string(),
    });
    state.interactions.push(InteractionItem::AssistantTurn {
        turn_index: 0,
        mode: Mode::Read,
        route: Mode::Read.route_tag(),
        plan: "Query ledger entries with governance status.".to_string(),
        payload:
            "SELECT entry_id, region, amount_cents, policy_decision, notes FROM ledger_entries;"
                .to_string(),
        single_pen_verified: true,
    });
    state.interactions.push(InteractionItem::Observation {
        turn_index: 0,
        observation: make_obs(
            Mode::Read,
            2,
            BTreeMap::new(),
            [
                "entry_id",
                "region",
                "amount_cents",
                "policy_decision",
                "notes",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            vec![
                vec![
                    json!(7),
                    json!("東京-NRT"),
                    json!(450),
                    json!("ASK"),
                    json!("Requires human steward sign-off before settlement window closes"),
                ],
                vec![
                    json!(108),
                    json!("chamonix"),
                    json!(129900),
                    json!("ALLOW"),
                    json!("Cleared"),
                ],
            ],
            0,
            "read_only",
            StewardAuthState::NotRequired,
        ),
    });

    let width = 128usize;
    let frame = render_two_pane_tui_frame(&state, width, 26, false);
    if std::env::var_os("PORTIA_PRINT_TUI_PREVIEW").is_some() {
        println!("{frame}");
    }
    for glyph in [
        "┌",
        "┬",
        "├",
        "┼",
        "└",
        "┴",
        "東京-NRT",
        "129900 │",
        "2 rows",
        "…",
    ] {
        assert!(frame.contains(glyph), "missing {glyph}");
    }
    assert!(frame.contains("       7 │") || frame.contains("  7 │"));
    assert!(!frame.contains("\"chamonix\""));

    for (idx, row) in frame.lines().enumerate() {
        assert_eq!(
            UnicodeWidthStr::width(row),
            width,
            "row {idx} width mismatch: {row:?}"
        );
    }
}

#[test]
fn assistant_turns_render_clean_pills_sql_steward_cards_and_strip_xml_tags() {
    let mut state = new_test_state("/srv/data/treasury.sqlite", vec![]);
    state.interactions.push(InteractionItem::UserPrompt {
        turn_index: 0,
        schema_header: String::new(),
        user_text: "Settle entry 7 after steward approval".to_string(),
    });
    for (idx, mode, plan, payload) in [
        (
            0,
            Mode::Probe,
            "Check invariants.",
            "<sql>PRAGMA foreign_key_check;</sql>",
        ),
        (
            0,
            Mode::Read,
            "Inspect entry 7.",
            "<sql>SELECT entry_id FROM ledger_entries WHERE entry_id = 7;</sql>",
        ),
        (
            0,
            Mode::Ask,
            "Request approval.",
            "<to_user>Approve entry 7?</to_user>",
        ),
        (
            1,
            Mode::Write,
            "Apply settlement update.",
            "<sql>UPDATE ledger_entries SET amount_cents = 0 WHERE entry_id = 7;</sql>",
        ),
    ] {
        state.interactions.push(InteractionItem::AssistantTurn {
            turn_index: idx,
            mode,
            route: mode.route_tag(),
            plan: plan.to_string(),
            payload: payload.to_string(),
            single_pen_verified: true,
        });
    }
    let mut tv = BTreeMap::new();
    tv.insert("ledger_entries".to_string(), 5);
    state.interactions.push(InteractionItem::Observation {
        turn_index: 1,
        observation: make_obs(
            Mode::Write,
            5,
            tv,
            vec![],
            vec![],
            1,
            "UPDATE",
            StewardAuthState::Granted,
        ),
    });
    state.interactions.push(InteractionItem::AssistantTurn {
        turn_index: 1,
        mode: Mode::Result,
        route: Mode::Result.route_tag(),
        plan: "Summarize completed settlement.".to_string(),
        payload: "<to_user>Entry 7 settled cleanly after Steward authorization.</to_user>"
            .to_string(),
        single_pen_verified: true,
    });

    let frame = render_two_pane_tui_frame(&state, 132, 44, false);
    for expected in [
        "[PROBE]",
        "[READ]",
        "[ASK]",
        "[WRITE]",
        "[RESULT]",
        "Plan: Apply settlement update.",
        "UPDATE ledger_entries SET amount_cents = 0",
        "STEWARD: GRANTED",
        "db v5",
        "Entry 7 settled cleanly after Steward authorization.",
    ] {
        assert!(frame.contains(expected), "missing {expected}");
    }
    for raw_tag in ["<sql>", "</sql>", "<to_user>", "</to_user>"] {
        assert!(!frame.contains(raw_tag), "leaked raw wire tag {raw_tag}");
    }
}

#[test]
fn keyboard_shortcuts_tab_ctrl_b_and_ctrl_t_toggle_schema_and_raw_token_trace() {
    let mut state = new_test_state("/srv/data/treasury.sqlite", sample_schema())
        .with_steward_policy_columns(["policy_decision"]);
    attach_sample_bpe_spans(&mut state);

    let initial = simulate_iocraft_keys_and_render(&mut state, &[], 144, 32);
    assert_eq!(state.effective_sidebar_mode(), SidebarMode::SchemaInspector);
    assert!(initial.contains("Schema (2 tables)"));
    assert!(!initial.contains("248045"));

    let after_tab = simulate_iocraft_keys_and_render(
        &mut state,
        &[(KeyCode::Tab, KeyModifiers::NONE)],
        144,
        32,
    );
    assert_eq!(state.effective_sidebar_mode(), SidebarMode::Hidden);
    assert!(!after_tab.contains("Schema (2 tables)") && !after_tab.contains("248045"));

    let after_ctrl_b = simulate_iocraft_keys_and_render(
        &mut state,
        &[(KeyCode::Char('b'), KeyModifiers::CONTROL)],
        144,
        32,
    );
    assert_eq!(state.effective_sidebar_mode(), SidebarMode::SchemaInspector);
    assert!(after_ctrl_b.contains("Schema (2 tables)"));

    let after_ctrl_t = simulate_iocraft_keys_and_render(
        &mut state,
        &[(KeyCode::Char('t'), KeyModifiers::CONTROL)],
        160,
        36,
    );
    assert!(
        state.show_debug_sidebar && state.effective_sidebar_mode() == SidebarMode::RawTokenTrace
    );
    for trace_marker in [
        "PANE 2: CONTEXT WINDOW & CARTRIDGE BOUNDARIES",
        "248045",
        "OrientTrunk",
    ] {
        assert!(
            after_ctrl_t.contains(trace_marker),
            "missing {trace_marker}"
        );
    }

    let back_to_schema = simulate_iocraft_keys_and_render(
        &mut state,
        &[(KeyCode::Char('t'), KeyModifiers::CONTROL)],
        144,
        32,
    );
    assert!(
        !state.show_debug_sidebar && state.effective_sidebar_mode() == SidebarMode::SchemaInspector
    );
    assert!(back_to_schema.contains("Schema (2 tables)") && !back_to_schema.contains("248045"));
}
