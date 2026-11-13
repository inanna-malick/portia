//! Carbon-tier agent harness integration tests for the `iocraft 0.9.5` TUI (`PortiaTuiApp`,
//! `SessionActor`, non-lethal `Esc` hierarchy, `PromptHistory`, mid-stream atomic cancellation,
//! SQLite `/undo` savepoint stacks, `/x` expanded vertical record cards, `/export`, and
//! interactive `StewardDock` `Button` / `Checkbox` flows).

use super::{
    delete_prev_word, delete_to_line_start, export_observation_to_path,
    parse_row_display_limit_arg, render_two_pane_tui_frame, spawn_session_actor, ExportFormat,
    InteractionItem, PortiaTuiApp, PromptHistory, SidebarMode, TwoPaneTuiState, UiToWorkerMsg,
    WorkerToUiEvent,
};
use crate::driver::ObservationOperation;
use crate::{
    parse_repl_command, AskInteractionMode, DriverObservation, LiveSqliteDailyDriver, ReplCommand,
    StdioQwenWorkerClient, TableSchemaInfo,
};
use futures::stream::{self, StreamExt};
use iocraft::prelude::*;
use portia_core::domain::ColumnSchema;
use portia_core::{Mode, StewardAuthState};
use rusqlite::Connection;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn press_key(code: KeyCode, modifiers: KeyModifiers) -> TerminalEvent {
    let mut ev = KeyEvent::new(KeyEventKind::Press, code);
    ev.modifiers = modifiers;
    TerminalEvent::Key(ev)
}

fn type_ascii(text: &str) -> Vec<TerminalEvent> {
    text.chars()
        .map(|ch| press_key(KeyCode::Char(ch), KeyModifiers::NONE))
        .collect()
}

fn temp_test_path(stem: &str, ext: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "portia_iocraft_{stem}_{}_{nanos}.{ext}",
        std::process::id()
    ))
}

fn spawn_python_mock_worker(py_body: &str) -> StdioQwenWorkerClient {
    let script_path = temp_test_path("mock_worker", "py");
    std::fs::write(&script_path, py_body).expect("write mock worker script");
    let cmd = format!("python3 -u {:?}", script_path);
    StdioQwenWorkerClient::spawn_shell(&cmd).expect("spawn python3 mock worker")
}

fn wait_until_idle_snapshot(
    rx: &smol::channel::Receiver<WorkerToUiEvent>,
    predicate: impl Fn(&TwoPaneTuiState) -> bool,
) -> TwoPaneTuiState {
    let deadline = Instant::now() + Duration::from_secs(8);
    let mut last_state: Option<TwoPaneTuiState> = None;
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(WorkerToUiEvent::StateSnapshot(snap)) => {
                let s = *snap;
                if !s.is_streaming && predicate(&s) {
                    return s;
                }
                last_state = Some(s);
            }
            Ok(_) => {}
            Err(smol::channel::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(smol::channel::TryRecvError::Closed) => break,
        }
    }
    panic!(
        "timed out waiting for expected SessionActor snapshot; last_state_notes={:?}",
        last_state.map(|s| s.interactions)
    );
}

#[test]
fn non_lethal_esc_hierarchy_clears_draft_closes_drawer_and_never_exits_tui() {
    let tables = vec![TableSchemaInfo::new(
        "campaigns",
        vec![ColumnSchema::with_sqlite_metadata(
            0, "id", "INTEGER", true, true,
        )],
    )
    .with_row_count(5)];
    let initial = TwoPaneTuiState::new(
        "/srv/data/campaigns.sqlite",
        "Qwen/Qwen3.5-0.8B-Base",
        "portia_v3_unified_cartridge.json",
        tables,
    );
    assert_eq!(
        initial.effective_sidebar_mode(),
        SidebarMode::SchemaInspector
    );

    let (cmd_tx, cmd_rx) = smol::channel::unbounded::<UiToWorkerMsg>();
    let mut events = Vec::new();
    // 1. Type draft text into composer:
    events.extend(type_ascii("draft question"));
    // 2. First Esc clears non-empty draft (drawer stays open!):
    events.push(press_key(KeyCode::Esc, KeyModifiers::NONE));
    // 3. Second Esc on empty input closes right drawer to Hidden (does NOT exit!):
    events.push(press_key(KeyCode::Esc, KeyModifiers::NONE));
    // 4. Third Esc on empty input + Hidden drawer shows non-lethal hint (still alive!):
    events.push(press_key(KeyCode::Esc, KeyModifiers::NONE));
    // 5. Type after 3 Esc presses and submit via Enter to prove the TUI never exited:
    events.extend(type_ascii("Count active campaigns"));
    events.push(press_key(KeyCode::Enter, KeyModifiers::NONE));
    // 6. Clean exit via Ctrl+D on empty composer:
    events.push(press_key(KeyCode::Char('d'), KeyModifiers::CONTROL));

    let frames: Vec<String> = smol::block_on(async move {
        let paced = stream::unfold(events.into_iter(), |mut iter| async move {
            smol::future::yield_now().await;
            iter.next().map(|ev| (ev, iter))
        });
        element! {
            PortiaTuiApp(
                initial_state: initial,
                cmd_tx: Some(cmd_tx),
                fixed_size: Some((136, 30)),
            )
        }
        .mock_terminal_render_loop(MockTerminalConfig::with_events(paced))
        .map(|c| c.to_string())
        .collect()
        .await
    });

    assert!(
        frames
            .iter()
            .any(|f| f.contains("Esc never exits — use /quit or Ctrl+D on empty input")),
        "expected non-lethal Esc hint in frames"
    );

    let mut submitted = Vec::new();
    let mut saw_shutdown = false;
    while let Ok(msg) = cmd_rx.try_recv() {
        match msg {
            UiToWorkerMsg::SubmitInput(line) => submitted.push(line),
            UiToWorkerMsg::Shutdown => saw_shutdown = true,
            UiToWorkerMsg::UiMutation(_) => {}
        }
    }
    assert_eq!(submitted, vec!["Count active campaigns".to_string()]);
    assert!(saw_shutdown, "Ctrl+D on empty input should emit Shutdown");
}

#[test]
fn prompt_history_preserves_in_progress_draft_and_ctrl_u_ctrl_w_edit_cleanly() {
    let mut hist = PromptHistory::new();
    hist.push("Show top 5 suppliers");
    hist.push("/model 2b-base");

    // Up from an in-progress draft saves the draft and walks backward:
    assert_eq!(
        hist.prev("half-written query"),
        Some("/model 2b-base".to_string())
    );
    assert_eq!(
        hist.prev("/model 2b-base"),
        Some("Show top 5 suppliers".to_string())
    );
    assert_eq!(hist.prev("Show top 5 suppliers"), None);

    // Down walks forward and restores the exact unsent bottom draft:
    assert_eq!(hist.next_entry(), Some("/model 2b-base".to_string()));
    assert_eq!(hist.next_entry(), Some("half-written query".to_string()));
    assert_eq!(hist.next_entry(), None);

    // UTF-8 safe Ctrl+W & Ctrl+U helpers:
    let sample_w = "Inspect 東京_ledger   now";
    let cursor_before_now = sample_w.find("now").unwrap();
    let (after_w, cur_w) = delete_prev_word(sample_w, cursor_before_now);
    assert_eq!(after_w, "Inspect now");
    assert_eq!(cur_w, "Inspect ".len());

    let (after_u, cur_u) = delete_to_line_start("prefix keep_tail", 7);
    assert_eq!(after_u, "keep_tail");
    assert_eq!(cur_u, 0);

    // Verify pure natural-language harness command parser (SQL strings stay TaskPrompt, never a local SQL bypass):
    assert_eq!(parse_repl_command("/x"), ReplCommand::ToggleExpandedDisplay);
    assert_eq!(
        parse_repl_command("/rows 25"),
        ReplCommand::SetRowLimit("25".to_string())
    );
    assert_eq!(
        parse_repl_command("/export /tmp/out.csv"),
        ReplCommand::ExportResults(PathBuf::from("/tmp/out.csv"))
    );
    assert_eq!(parse_repl_command("/undo"), ReplCommand::Undo);
    assert_eq!(
        parse_repl_command("SELECT * FROM ledger;"),
        ReplCommand::TaskPrompt("SELECT * FROM ledger;".to_string())
    );
}

#[test]
fn expanded_vertical_record_cards_row_caps_and_csv_json_export_work_end_to_end() {
    assert_eq!(parse_row_display_limit_arg("all").unwrap(), None);
    assert_eq!(parse_row_display_limit_arg("5").unwrap(), Some(5));
    assert!(parse_row_display_limit_arg("0").is_err());

    let mut state = TwoPaneTuiState::new(
        "/srv/data/incidents.sqlite",
        "Qwen/Qwen3.5-2B-Base",
        "portia_v3_unified_cartridge_2b_base.json",
        vec![],
    );
    let long_note =
        "BGP session flap on peer AS64512 with full untruncated diagnostic payload TAIL_MARKER_991";
    let obs = DriverObservation {
        event_id: "ev-1".to_string(),
        mode: Mode::Read,
        ok: true,
        db_version: 1,
        table_versions: BTreeMap::new(),
        columns: vec![
            "incident_id".to_string(),
            "pop_code".to_string(),
            "notes".to_string(),
        ],
        rows: vec![
            vec![json!(101), json!("NRT-TOKYO, \"East\""), json!(long_note)],
            vec![json!(102), json!("FRA-DECIX-02"), json!("Nominal")],
        ],
        rows_affected: 0,
        message: None,
        error: None,
        operation: ObservationOperation::from_wire_str("SELECT"),
        ask_reply: None,
        steward_auth: StewardAuthState::NotRequired,
        is_chat_continuation: false,
    };
    state.interactions.push(InteractionItem::Observation {
        turn_index: 0,
        observation: obs,
    });

    // Enable `/x` expanded vertical record cards and `/rows 1`:
    state.expanded_display = true;
    state.row_display_limit = Some(1);
    let frame = render_two_pane_tui_frame(&state, 120, 32, false);
    assert!(
        frame.contains("RECORD 1"),
        "missing RECORD 1 header:\n{frame}"
    );
    assert!(
        frame.contains("TAIL_MARKER_991"),
        "expanded /x card must show untruncated long field:\n{frame}"
    );
    assert!(
        frame.contains("1 additional row(s) hidden") && frame.contains("/rows all"),
        "missing row-cap hint:\n{frame}"
    );

    // Export to CSV and JSON on disk:
    let csv_path = temp_test_path("incidents_export", "csv");
    let json_path = temp_test_path("incidents_export", "json");
    let latest = state.latest_tabular_observation().expect("latest obs");

    let (fmt_csv, r_csv, c_csv) = export_observation_to_path(latest, &csv_path).unwrap();
    assert_eq!((fmt_csv, r_csv, c_csv), (ExportFormat::Csv, 2, 3));
    let csv_body = std::fs::read_to_string(&csv_path).unwrap();
    assert!(csv_body.starts_with("incident_id,pop_code,notes\n"));
    assert!(csv_body.contains("\"NRT-TOKYO, \"\"East\"\"\""));

    let (fmt_json, r_json, c_json) = export_observation_to_path(latest, &json_path).unwrap();
    assert_eq!((fmt_json, r_json, c_json), (ExportFormat::Json, 2, 3));
    let parsed_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    assert_eq!(parsed_json.as_array().unwrap().len(), 2);
    assert_eq!(parsed_json[0]["incident_id"], json!(101));

    let _ = std::fs::remove_file(csv_path);
    let _ = std::fs::remove_file(json_path);
}

#[test]
fn session_actor_sqlite_undo_stack_rolls_back_last_step_refreshes_schema_and_persists_prior_steps()
{
    let db_path = temp_test_path("undo_stack", "sqlite");
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT NOT NULL);
             INSERT INTO items (id, label) VALUES (1, 'seed');",
        )
        .unwrap();
    }

    // Scripted worker executes two multi-turn WRITE prompts:
    // Prompt 1 inserts id=2; Prompt 2 inserts id=3.
    let worker = spawn_python_mock_worker(
        r#"import json, sys

turns = [
    [
        [45829, " WRITE"],
        [60, "]"],
        [1001, "Insert second item.\n"],
        [2001, "INSERT INTO items (id, label) VALUES (2, 'step_one');\n</sql>"],
    ],
    [
        [34549, " RESULT"],
        [60, "]"],
        [1001, "Confirm step one.\n"],
        [2001, "Inserted step_one.\n</to_user>"],
    ],
    [
        [45829, " WRITE"],
        [60, "]"],
        [1001, "Insert third item.\n"],
        [2001, "INSERT INTO items (id, label) VALUES (3, 'step_two');\n</sql>"],
    ],
    [
        [34549, " RESULT"],
        [60, "]"],
        [1001, "Confirm step two.\n"],
        [2001, "Inserted step_two.\n</to_user>"],
    ],
]
turn_idx = 0
tok_idx = 0

for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    op = req.get("op")
    if op in ("reset_session", "prefill"):
        resp = {"status": "prefilled", "token_ids": [1, 2]}
    elif op == "sample_token":
        tid, piece = turns[turn_idx][tok_idx]
        tok_idx += 1
        if tok_idx >= len(turns[turn_idx]):
            turn_idx += 1
            tok_idx = 0
        resp = {"status": "sampled_token", "token_id": tid, "token_piece": piece}
    elif op == "consume_token":
        resp = {"status": "consumed_token", "token_ids": [int(req["token_id"])]}
    else:
        resp = {"status": "error", "message": f"unknown op {op}"}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"#,
    );

    let driver = LiveSqliteDailyDriver::open(&db_path, 100).unwrap();
    let initial_schema = driver.inspect_schema().unwrap();
    assert_eq!(initial_schema[0].row_count, 1);

    let tui_state = TwoPaneTuiState::new(
        db_path.display().to_string(),
        "Qwen/Qwen3.5-0.8B-Base",
        "none",
        initial_schema,
    );
    let mut actor = spawn_session_actor(
        driver,
        worker,
        tui_state,
        8,
        None,
        Box::new(|_, _, _, _| unreachable!()),
    );

    // Step 1: insert row 2 -> auto-refreshes schema row_count to 2 and increments undo_depth to 1!
    actor
        .cmd_tx
        .send_blocking(UiToWorkerMsg::SubmitInput("Add step_one".to_string()))
        .unwrap();
    let after_step1 = wait_until_idle_snapshot(&actor.event_rx, |s| s.undo_depth == 1);
    assert_eq!(after_step1.tables[0].row_count, 2);

    // Step 2: insert row 3 -> row_count becomes 3 and undo_depth becomes 2!
    actor
        .cmd_tx
        .send_blocking(UiToWorkerMsg::SubmitInput("Add step_two".to_string()))
        .unwrap();
    let after_step2 = wait_until_idle_snapshot(&actor.event_rx, |s| s.undo_depth == 2);
    assert_eq!(after_step2.tables[0].row_count, 3);

    // Execute `/undo` -> rolls back ONLY step 2, refreshes schema row_count to 2, undo_depth == 1!
    actor
        .cmd_tx
        .send_blocking(UiToWorkerMsg::SubmitInput("/undo".to_string()))
        .unwrap();
    let after_undo = wait_until_idle_snapshot(&actor.event_rx, |s| s.undo_depth == 1);
    assert_eq!(after_undo.tables[0].row_count, 2);
    assert!(after_undo.interactions.iter().any(|item| matches!(
        item,
        InteractionItem::HarnessNote { title, .. } if title.starts_with("Undo Applied")
    )));

    // Cleanly shut down actor (flushes open savepoint for Step 1 to SQLite disk!):
    actor.shutdown();

    let verify_conn = Connection::open(&db_path).unwrap();
    let labels: Vec<String> = verify_conn
        .prepare("SELECT label FROM items ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(labels, vec!["seed".to_string(), "step_one".to_string()]);
    let _ = std::fs::remove_file(db_path);
}

#[test]
fn session_actor_mid_stream_cancellation_interrupts_slow_worker_without_corrupting_db() {
    let db_path = temp_test_path("cancel_stream", "sqlite");
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL);
             INSERT INTO notes (id, body) VALUES (1, 'intact');",
        )
        .unwrap();
    }

    // Slow streaming worker yields tokens every 25ms in Payload phase so UI thread can cancel mid-flight:
    let worker = spawn_python_mock_worker(
        r#"import json, sys, time

step = 0
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    op = req.get("op")
    if op == "reset_session":
        step = 0
        resp = {"status": "prefilled", "token_ids": [1]}
    elif op == "prefill":
        resp = {"status": "prefilled", "token_ids": [1, 2]}
    elif op == "consume_token":
        resp = {"status": "consumed_token", "token_ids": [int(req["token_id"])]}
    elif op == "sample_token":
        step += 1
        if step == 1:
            resp = {"status": "sampled_token", "token_id": 34549, "token_piece": " RESULT"}
        elif step == 2:
            resp = {"status": "sampled_token", "token_id": 60, "token_piece": "]"}
        elif step == 3:
            resp = {"status": "sampled_token", "token_id": 1001, "token_piece": "Long answer.\n"}
        else:
            time.sleep(0.025)
            resp = {"status": "sampled_token", "token_id": 2001, "token_piece": "word "}
    else:
        resp = {"status": "error", "message": f"unknown op {op}"}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"#,
    );

    let driver = LiveSqliteDailyDriver::open(&db_path, 100).unwrap();
    let schema = driver.inspect_schema().unwrap();
    let tui_state = TwoPaneTuiState::new(
        db_path.display().to_string(),
        "Qwen/Qwen3.5-0.8B-Base",
        "none",
        schema,
    );
    let mut actor = spawn_session_actor(
        driver,
        worker,
        tui_state,
        4,
        None,
        Box::new(|_, _, _, _| unreachable!()),
    );

    actor
        .cmd_tx
        .send_blocking(UiToWorkerMsg::SubmitInput(
            "Stream a very long explanation".to_string(),
        ))
        .unwrap();

    // Wait until at least one streaming snapshot arrives (`is_streaming == true`), then cancel!
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_streaming = false;
    while Instant::now() < deadline {
        if let Ok(WorkerToUiEvent::StateSnapshot(snap)) = actor.event_rx.try_recv() {
            if snap.is_streaming && !snap.interactions.is_empty() {
                saw_streaming = true;
                break;
            }
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(
        saw_streaming,
        "expected live streaming snapshot before cancel"
    );

    actor.cancel_in_flight_turn();
    let cancelled_snap = wait_until_idle_snapshot(&actor.event_rx, |s| {
        s.interactions.iter().any(|it| {
            matches!(
                it,
                InteractionItem::HarnessNote { title, .. } if title == "Turn Cancelled"
            )
        })
    });
    assert!(!cancelled_snap.is_streaming);
    assert_eq!(cancelled_snap.undo_depth, 0);

    actor.shutdown();
    let _ = std::fs::remove_file(db_path);
}

#[test]
fn interactive_steward_dock_keyboard_navigation_resolves_permission_and_multiselect() {
    let initial = TwoPaneTuiState::new(
        "/srv/data/treasury.sqlite",
        "Qwen/Qwen3.5-0.8B-Base",
        "none",
        vec![],
    );
    let (event_tx, event_rx) = smol::channel::unbounded::<WorkerToUiEvent>();
    let (perm_reply_tx, perm_reply_rx) = smol::channel::bounded::<Option<String>>(1);

    event_tx
        .send_blocking(WorkerToUiEvent::AskRequested {
            question: "Approve settlement on row 7?".to_string(),
            ask_mode: AskInteractionMode::StewardPermissionChoice {
                question: "Approve settlement on row 7?".to_string(),
                choices: vec![
                    "ALLOW — Authorize database write".to_string(),
                    "DENY — Block database write".to_string(),
                ],
            },
            context_hint: Some("policy_cols=[policy_decision]".to_string()),
            reply_tx: perm_reply_tx,
        })
        .unwrap();

    // Move Down from ALLOW (0) to DENY (1), press Enter to submit DENY, then Ctrl+Q to exit:
    let events = vec![
        press_key(KeyCode::Down, KeyModifiers::NONE),
        press_key(KeyCode::Enter, KeyModifiers::NONE),
        press_key(KeyCode::Char('q'), KeyModifiers::CONTROL),
    ];

    let frames: Vec<String> = smol::block_on(async move {
        element! {
            PortiaTuiApp(
                initial_state: initial,
                event_rx: Some(event_rx),
                fixed_size: Some((132, 30)),
            )
        }
        .mock_terminal_render_loop(MockTerminalConfig::with_events(stream::iter(events)))
        .map(|c| c.to_string())
        .collect()
        .await
    });

    assert!(
        frames
            .iter()
            .any(|f| f.contains("Steward Authorization Required")),
        "StewardDock did not render in frames"
    );
    assert_eq!(perm_reply_rx.try_recv().unwrap(), Some("DENY".to_string()));
}
