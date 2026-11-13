//! Dedicated background worker & SQLite session actor decoupling the `iocraft` UI event loop
//! from blocking stdio inference and SQLite transactions, with atomic token-boundary cancellation,
//! auto-refreshed schemas after `Mode::Write`, `/x`, `/rows`, `/export`, and `/undo` savepoints.

use super::export_data::{export_observation_to_path, parse_row_display_limit_arg};
use super::snapshot::format_surfaced_error_note;
use super::{InteractionItem, SidebarMode, TwoPaneTuiState};
use crate::model_identity::normalize_model_identifier;
use crate::worker::{
    resolve_default_cartridge_label_for_model,
    run_live_worker_turn_sequence_streaming_cancellable_into, SessionRunError,
    StdioQwenWorkerClient,
};
use crate::{
    format_human_readable_session_events, parse_ask_user_reply_for_kind, parse_repl_command,
    AskInteractionMode, AskPromptKind, LiveSqliteDailyDriver, ReplCommand,
};
use anyhow::{bail, Result};
use smol::channel::{self, Receiver as SmolReceiver, Sender as SmolSender, TryRecvError};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Callback invoked on the worker thread when `/model <scale>` spawns a replacement worker.
pub type WorkerSpawner =
    Box<dyn FnMut(&str, Option<&Path>, &str, bool) -> Result<StdioQwenWorkerClient> + Send>;

/// Instant UI-side toggle synchronized to the actor's canonical [`TwoPaneTuiState`].
#[derive(Debug, Clone, Copy)]
pub enum UiLocalMutation {
    SetSidebarMode(SidebarMode),
    ToggleSchemaDrawer,
    ToggleRawTokenTrace,
    ToggleInlineTokens,
}

/// Commands sent non-blockingly from the `iocraft` UI thread to the background worker thread.
pub enum UiToWorkerMsg {
    SubmitInput(String),
    UiMutation(UiLocalMutation),
    Shutdown,
}

/// Events streamed asynchronously from the background worker thread to the `iocraft` UI.
pub enum WorkerToUiEvent {
    StateSnapshot(Box<TwoPaneTuiState>),
    AskRequested {
        question: String,
        ask_mode: AskInteractionMode,
        context_hint: Option<String>,
        reply_tx: SmolSender<Option<String>>,
    },
    InlineTurnCompleted(String),
    ExitRequested,
}

/// Handle held by `PortiaTuiApp` (and unit tests) to drive and observe the background session actor.
pub struct SessionActorHandle {
    pub cmd_tx: SmolSender<UiToWorkerMsg>,
    pub event_rx: SmolReceiver<WorkerToUiEvent>,
    pub cancel_requested: Arc<AtomicBool>,
    pub initial_state: TwoPaneTuiState,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl SessionActorHandle {
    /// Request immediate token-boundary cancellation of any in-flight generation (`Esc` / `Ctrl+C`).
    pub fn cancel_in_flight_turn(&self) {
        self.cancel_requested.store(true, Ordering::SeqCst);
    }

    /// Shut down the background worker thread cleanly, flushing open SQLite `/undo` savepoints.
    pub fn shutdown(&mut self) {
        self.cancel_requested.store(true, Ordering::SeqCst);
        let _ = self.cmd_tx.send_blocking(UiToWorkerMsg::Shutdown);
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

impl Drop for SessionActorHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn summarize_prompt_label(prompt: &str) -> String {
    let single = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.chars().count() <= 48 {
        single
    } else {
        let head: String = single.chars().take(47).collect();
        format!("{head}…")
    }
}

fn format_status_detail(state: &TwoPaneTuiState) -> String {
    let spec = state.model_scale_spec();
    let row_cap = match state.row_display_limit {
        Some(n) => format!("{n}"),
        None => "all".to_string(),
    };
    format!(
        "Model: {} ({} · {}) | Cartridge: {} | DB: {} ({} tables, read_only={}, undo={}) | Display: expanded={}, rows={} | Context: {} tok, {} swaps",
        spec.canonical_family,
        spec.scale_badge,
        spec.format_architecture_summary(),
        state.cartridge_label,
        state.db_path,
        state.tables.len(),
        state.read_only,
        state.undo_depth,
        state.expanded_display,
        row_cap,
        state.total_context_tokens,
        state.cartridge_switch_count
    )
}

fn build_ask_context_hint(
    driver: &LiveSqliteDailyDriver,
    state: &TwoPaneTuiState,
) -> Option<String> {
    let mut parts = Vec::new();
    if !state.steward_policy_columns.is_empty() {
        parts.push(format!(
            "policy_cols=[{}]",
            state.steward_policy_columns.join(", ")
        ));
    }
    let rows = driver.session_last_read_rows();
    if !rows.is_empty() {
        let previews: Vec<String> = rows
            .iter()
            .take(2)
            .filter_map(|r| crate::driver::summarize_candidate_row_choice(r))
            .collect();
        if !previews.is_empty() {
            parts.push(format!("candidate_rows={}", previews.join(" | ")));
        }
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// Spawn the dedicated background SQLite + Qwen worker session actor thread.
pub fn spawn_session_actor(
    mut driver: LiveSqliteDailyDriver,
    mut worker: StdioQwenWorkerClient,
    mut tui_state: TwoPaneTuiState,
    max_turns: usize,
    canned_ask_reply: Option<String>,
    mut worker_spawner: WorkerSpawner,
) -> SessionActorHandle {
    tui_state.undo_depth = driver.undo_depth();
    tui_state.is_streaming = false;
    let initial_state = tui_state.clone();

    let (cmd_tx, cmd_rx) = channel::unbounded::<UiToWorkerMsg>();
    let (event_tx, event_rx) = channel::unbounded::<WorkerToUiEvent>();
    let cancel_requested = Arc::new(AtomicBool::new(false));
    let cancel_worker = Arc::clone(&cancel_requested);

    let join_handle = std::thread::Builder::new()
        .name("portia-tui-worker".to_string())
        .spawn(move || {
            let mut needs_session_reset = true;
            let push_snap = |state: &TwoPaneTuiState| {
                let _ = event_tx.send_blocking(WorkerToUiEvent::StateSnapshot(Box::new(
                    state.clone(),
                )));
            };

            while let Ok(msg) = cmd_rx.recv_blocking() {
                match msg {
                    UiToWorkerMsg::Shutdown => {
                        let _ = driver.flush_open_undo_savepoints();
                        let _ = event_tx.send_blocking(WorkerToUiEvent::ExitRequested);
                        break;
                    }
                    UiToWorkerMsg::UiMutation(m) => {
                        match m {
                            UiLocalMutation::SetSidebarMode(mode) => {
                                tui_state.set_sidebar_mode(mode)
                            }
                            UiLocalMutation::ToggleSchemaDrawer => tui_state.toggle_schema_drawer(),
                            UiLocalMutation::ToggleRawTokenTrace => {
                                tui_state.toggle_raw_token_trace()
                            }
                            UiLocalMutation::ToggleInlineTokens => {
                                tui_state.show_inline_token_ids = !tui_state.show_inline_token_ids;
                            }
                        }
                        push_snap(&tui_state);
                    }
                    UiToWorkerMsg::SubmitInput(raw_line) => {
                        let cmd = parse_repl_command(&raw_line);
                        match cmd {
                            ReplCommand::Empty => {}
                            ReplCommand::Quit => {
                                let _ = driver.flush_open_undo_savepoints();
                                let _ = event_tx.send_blocking(WorkerToUiEvent::ExitRequested);
                                break;
                            }
                            ReplCommand::Help => {
                                tui_state.interactions.push(InteractionItem::HarnessNote {
                                    title: "Agent Harness Commands".to_string(),
                                    detail: concat!(
                                        "/model <0.8b-base|2b-base|4b-base|9b-base> [--cartridge <path|none>] · hot-swap Qwen3.5 scale\n",
                                        "/x · toggle expanded vertical record cards (─[ RECORD 1 ]─) for wide tables\n",
                                        "/rows <N|all> · set visible result row cap (default 12)\n",
                                        "/export <path.csv|path.json> · export latest SQL result rows to disk\n",
                                        "/undo · roll back most recent database write step & refresh schema/KV\n",
                                        "/schema (Tab / Ctrl+B) · refresh & toggle Schema Inspector drawer\n",
                                        "/trace (/debug, Ctrl+T) · toggle Raw BPE & CartridgeBank Inspector\n",
                                        "/status · show active model/DB/undo telemetry | /clear (/reset) · clear chat | /quit (Ctrl+D) · exit"
                                    )
                                    .to_string(),
                                });
                                push_snap(&tui_state);
                            }
                            ReplCommand::Status => {
                                tui_state.interactions.push(InteractionItem::HarnessNote {
                                    title: format!(
                                        "Session Status · {}",
                                        tui_state.model_scale_spec().scale_badge
                                    ),
                                    detail: format_status_detail(&tui_state),
                                });
                                push_snap(&tui_state);
                            }
                            ReplCommand::ToggleExpandedDisplay => {
                                tui_state.expanded_display = !tui_state.expanded_display;
                                let mode_lbl = if tui_state.expanded_display {
                                    "ON (psql vertical ─[ RECORD k ]─ cards; untruncated values)"
                                } else {
                                    "OFF (compact aligned Unicode box grid)"
                                };
                                tui_state.interactions.push(InteractionItem::HarnessNote {
                                    title: format!("Expanded Display (/x): {mode_lbl}"),
                                    detail: "Applies immediately to all SQL result observations."
                                        .to_string(),
                                });
                                push_snap(&tui_state);
                            }
                            ReplCommand::SetRowLimit(raw_arg) => {
                                match parse_row_display_limit_arg(&raw_arg) {
                                    Ok(limit) => {
                                        tui_state.row_display_limit = limit;
                                        let desc = match limit {
                                            Some(n) => format!("{n} rows per table"),
                                            None => "all rows (unlimited)".to_string(),
                                        };
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: format!("Row Display Limit -> {desc}"),
                                            detail: "Updated result table row cap.".to_string(),
                                        });
                                    }
                                    Err(err) => {
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: "Invalid /rows Argument".to_string(),
                                            detail: err.to_string(),
                                        });
                                    }
                                }
                                push_snap(&tui_state);
                            }
                            ReplCommand::ExportResults(path) => {
                                let note = match tui_state.latest_tabular_observation() {
                                    Some(obs) => match export_observation_to_path(obs, &path) {
                                        Ok((fmt, n_rows, n_cols)) => InteractionItem::HarnessNote {
                                            title: format!(
                                                "Exported {} ({n_rows} rows × {n_cols} cols)",
                                                fmt.label()
                                            ),
                                            detail: format!("Wrote {}", path.display()),
                                        },
                                        Err(err) => InteractionItem::HarnessNote {
                                            title: "Export Failed".to_string(),
                                            detail: err.to_string(),
                                        },
                                    },
                                    None => InteractionItem::HarnessNote {
                                        title: "Nothing to Export".to_string(),
                                        detail: "No SQL result rows available in the session yet."
                                            .to_string(),
                                    },
                                };
                                tui_state.interactions.push(note);
                                push_snap(&tui_state);
                            }
                            ReplCommand::Undo => {
                                match driver.undo_last_step() {
                                    Ok(Some(receipt)) => {
                                        let _ = worker.reset_session();
                                        needs_session_reset = true;
                                        tui_state.reset_kv_window_tracking();
                                        if let Ok(fresh_schema) = driver.inspect_schema() {
                                            tui_state.tables = fresh_schema;
                                        }
                                        tui_state.undo_depth = receipt.remaining_undo_depth;
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: format!(
                                                "Undo Applied ({})",
                                                receipt.savepoint_name
                                            ),
                                            detail: format!(
                                                "Rolled back {:?} -> db v{} ({} undo step(s) remaining; schema & model KV refreshed).",
                                                receipt.prompt_summary,
                                                receipt.restored_db_version,
                                                receipt.remaining_undo_depth
                                            ),
                                        });
                                    }
                                    Ok(None) => {
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: "Nothing to Undo".to_string(),
                                            detail: "No committed database writes in the current session undo stack."
                                                .to_string(),
                                        });
                                    }
                                    Err(err) => {
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: "Undo Error".to_string(),
                                            detail: err.to_string(),
                                        });
                                    }
                                }
                                push_snap(&tui_state);
                            }
                            ReplCommand::DebugSidebar => {
                                tui_state.toggle_raw_token_trace();
                                push_snap(&tui_state);
                            }
                            ReplCommand::Tokens => {
                                tui_state.show_inline_token_ids = !tui_state.show_inline_token_ids;
                                push_snap(&tui_state);
                            }
                            ReplCommand::TuiSnapshot => {
                                push_snap(&tui_state);
                            }
                            ReplCommand::Reset => {
                                let _ = worker.reset_session();
                                tui_state.reset_conversation_context();
                                needs_session_reset = true;
                                push_snap(&tui_state);
                            }
                            ReplCommand::Schema => {
                                if let Ok(schema) = driver.inspect_schema() {
                                    tui_state.tables = schema.clone();
                                    tui_state.set_sidebar_mode(SidebarMode::SchemaInspector);
                                    let summary = if schema.is_empty() {
                                        "No user tables found in database.".to_string()
                                    } else {
                                        schema
                                            .iter()
                                            .map(|t| {
                                                format!(
                                                    "{} ({} cols, {} rows)",
                                                    t.table_name,
                                                    t.columns.len(),
                                                    t.row_count
                                                )
                                            })
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    };
                                    tui_state.interactions.push(InteractionItem::HarnessNote {
                                        title: format!(
                                            "Schema Refreshed ({} tables)",
                                            schema.len()
                                        ),
                                        detail: summary,
                                    });
                                }
                                push_snap(&tui_state);
                            }
                            ReplCommand::SwitchModel { target, cartridge } => {
                                let Some(raw_target) = target
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                else {
                                    tui_state.interactions.push(InteractionItem::HarnessNote {
                                        title: format!(
                                            "Active Model · {}",
                                            tui_state.model_scale_spec().scale_badge
                                        ),
                                        detail: format!(
                                            "{} — Usage: /model <0.8b-base|2b-base|4b-base|9b-base> [--cartridge <path|none>]",
                                            format_status_detail(&tui_state)
                                        ),
                                    });
                                    push_snap(&tui_state);
                                    continue;
                                };
                                let normalized_target = normalize_model_identifier(&raw_target);
                                let new_cart_label = resolve_default_cartridge_label_for_model(
                                    &normalized_target,
                                    cartridge.as_deref(),
                                );
                                match worker_spawner(
                                    &normalized_target,
                                    cartridge.as_deref(),
                                    &new_cart_label,
                                    tui_state.read_only,
                                ) {
                                    Ok(new_worker) => {
                                        worker = new_worker;
                                        let spec = tui_state.apply_hot_swapped_model(
                                            &normalized_target,
                                            &new_cart_label,
                                        );
                                        needs_session_reset = true;
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: format!("Model Switched -> {}", spec.scale_badge),
                                            detail: format!(
                                                "Active model: {} ({} · OrientTrunk {} · ModePayload {}) · Cartridge: {} (KV context reset)",
                                                spec.canonical_family,
                                                spec.format_architecture_summary(),
                                                spec.format_orient_slots(),
                                                spec.format_payload_slots(),
                                                tui_state.cartridge_label
                                            ),
                                        });
                                    }
                                    Err(err) => {
                                        let cur_badge = tui_state.model_scale_spec().scale_badge;
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: "Model Switch Error".to_string(),
                                            detail: format!(
                                                "{err} (staying on {cur_badge} · {})",
                                                tui_state.cartridge_label
                                            ),
                                        });
                                    }
                                }
                                push_snap(&tui_state);
                            }
                            ReplCommand::TaskPrompt(task_prompt) => {
                                cancel_worker.store(false, Ordering::SeqCst);
                                tui_state.is_streaming = true;
                                push_snap(&tui_state);

                                let db_ver_before = driver.db_version();
                                let undo_cp = match driver
                                    .open_turn_undo_savepoint(summarize_prompt_label(&task_prompt))
                                {
                                    Ok(cp) => cp,
                                    Err(err) => {
                                        tui_state.is_streaming = false;
                                        tui_state.interactions.push(InteractionItem::HarnessNote {
                                            title: "Savepoint Error".to_string(),
                                            detail: err.to_string(),
                                        });
                                        push_snap(&tui_state);
                                        continue;
                                    }
                                };

                                let latest_snap = std::cell::RefCell::new(tui_state.clone());
                                let ctx_hint = build_ask_context_hint(&driver, &tui_state);
                                let mut events = Vec::new();

                                let res = run_live_worker_turn_sequence_streaming_cancellable_into(
                                    &mut driver,
                                    &mut worker,
                                    &task_prompt,
                                    max_turns,
                                    needs_session_reset,
                                    Some(cancel_worker.as_ref()),
                                    Some(&mut tui_state),
                                    &mut events,
                                    |q| {
                                        let snap = latest_snap.borrow().clone();
                                        let ask_mode = snap.ask_interaction_mode(q);
                                        let kind = match ask_mode {
                                            AskInteractionMode::StewardPermissionChoice {
                                                ..
                                            } => AskPromptKind::Permission,
                                            _ => AskPromptKind::Clarification,
                                        };
                                        if let Some(ref ans) = canned_ask_reply {
                                            let _ = parse_ask_user_reply_for_kind(kind, ans)
                                                .or_else(|_| {
                                                    parse_ask_user_reply_for_kind(
                                                        AskPromptKind::Clarification,
                                                        ans,
                                                    )
                                                })?;
                                            return Ok(ans.trim().to_string());
                                        }
                                        let (reply_tx, reply_rx) = channel::bounded::<Option<String>>(1);
                                        let _ = event_tx.send_blocking(
                                            WorkerToUiEvent::AskRequested {
                                                question: q.to_string(),
                                                ask_mode,
                                                context_hint: ctx_hint.clone(),
                                                reply_tx,
                                            },
                                        );
                                        loop {
                                            if cancel_worker.load(Ordering::SeqCst) {
                                                return Err(SessionRunError::Cancelled {
                                                    tokens_generated: 0,
                                                    phase_at_cancel: "StewardAsk".to_string(),
                                                }
                                                .into());
                                            }
                                            match reply_rx.try_recv() {
                                                Ok(Some(ans)) => {
                                                    let trimmed = ans.trim();
                                                    if trimmed.is_empty() {
                                                        bail!("Empty reply to [MODE: ASK]");
                                                    }
                                                    let _ = parse_ask_user_reply_for_kind(
                                                        kind, trimmed,
                                                    )
                                                    .or_else(|_| {
                                                        parse_ask_user_reply_for_kind(
                                                            AskPromptKind::Clarification,
                                                            trimmed,
                                                        )
                                                    })?;
                                                    return Ok(trimmed.to_string());
                                                }
                                                Ok(None) | Err(TryRecvError::Closed) => {
                                                    cancel_worker.store(true, Ordering::SeqCst);
                                                    return Err(SessionRunError::Cancelled {
                                                        tokens_generated: 0,
                                                        phase_at_cancel: "StewardAskCancelled"
                                                            .to_string(),
                                                    }
                                                    .into());
                                                }
                                                Err(TryRecvError::Empty) => {
                                                    std::thread::sleep(Duration::from_millis(15));
                                                }
                                            }
                                        }
                                    },
                                    |state| {
                                        let mut streaming_clone = state.clone();
                                        streaming_clone.is_streaming = true;
                                        *latest_snap.borrow_mut() = streaming_clone.clone();
                                        push_snap(&streaming_clone);
                                    },
                                );

                                tui_state.is_streaming = false;
                                let was_cancelled = res
                                    .as_ref()
                                    .err()
                                    .and_then(|e| e.downcast_ref::<SessionRunError>())
                                    .is_some_and(|se| matches!(se, SessionRunError::Cancelled { .. }))
                                    || cancel_worker.load(Ordering::SeqCst);

                                if was_cancelled {
                                    if let Some(cp) = undo_cp {
                                        let _ = driver.abort_turn_undo_savepoint(cp);
                                    }
                                    let _ = worker.reset_session();
                                    needs_session_reset = true;
                                    tui_state.prune_cancelled_streaming_turn();
                                    if let Ok(fresh_schema) = driver.inspect_schema() {
                                        tui_state.tables = fresh_schema;
                                    }
                                    tui_state.undo_depth = driver.undo_depth();
                                    tui_state.interactions.push(InteractionItem::HarnessNote {
                                        title: "Turn Cancelled".to_string(),
                                        detail: "Interrupted active turn; uncommitted SQLite changes rolled back and worker KV reset."
                                            .to_string(),
                                    });
                                } else {
                                    if let Some(cp) = undo_cp {
                                        if let Ok(depth) = driver.finalize_turn_undo_savepoint(cp) {
                                            tui_state.undo_depth = depth;
                                        }
                                    }
                                    if driver.db_version() != db_ver_before {
                                        if let Ok(fresh_schema) = driver.inspect_schema() {
                                            tui_state.tables = fresh_schema;
                                        }
                                    }
                                    needs_session_reset = false;
                                    if let Err(ref err) = res {
                                        needs_session_reset = true;
                                        let (title, detail) = format_surfaced_error_note(err);
                                        tui_state
                                            .interactions
                                            .push(InteractionItem::HarnessNote { title, detail });
                                    }
                                    if !events.is_empty() {
                                        let formatted = format!(
                                            "❯ {}\n{}",
                                            task_prompt.trim(),
                                            format_human_readable_session_events(&events)
                                        );
                                        let _ = event_tx.send_blocking(
                                            WorkerToUiEvent::InlineTurnCompleted(formatted),
                                        );
                                    }
                                }

                                if let Ok(log_path) = std::env::var("PORTIA_SESSION_LOG") {
                                    let trimmed = log_path.trim();
                                    if !trimmed.is_empty() {
                                        let mut snap_copy = tui_state.clone();
                                        snap_copy.show_debug_sidebar = true;
                                        let _ = std::fs::write(
                                            trimmed,
                                            super::render_two_pane_tui_frame(
                                                &snap_copy, 180, 60, false,
                                            ),
                                        );
                                    }
                                }
                                push_snap(&tui_state);
                            }
                        }
                    }
                }
            }
        })
        .expect("spawn portia-tui-worker thread");

    SessionActorHandle {
        cmd_tx,
        event_rx,
        cancel_requested,
        initial_state,
        join_handle: Some(join_handle),
    }
}
