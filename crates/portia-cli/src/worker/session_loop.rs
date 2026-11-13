//! Prompt-blind (`∀prompt`) OODA coroutine loops driving [`EpistemicDomainEnv`] and
//! [`LiveSqliteDailyDriver`] over OS pipes against [`StdioQwenWorkerClient`].

use super::protocol::format_observation_wire_chunk;
use super::spawn::StdioQwenWorkerClient;
use super::summary::{
    attach_session_error, emit_session_outcome_stderr, format_lotl_user_prompt_envelope,
    summarize_session_outcome, SessionEvent, SessionRunError, TurnCycleDetector,
};
use crate::driver::{
    AskInteractionMode, CoroutineStepEffect, DriverObservation, EpistemicDomainEnv,
    LiveSqliteDailyDriver,
};
use crate::tui::{InteractionItem, TwoPaneTuiState};
use anyhow::{bail, Result};
use portia_core::{CartridgeBank, KvPrefillDisposition, LotlNormalizedWireChunk, Mode, PenOwner};
use std::collections::BTreeMap;

/// Run a complete multi-turn session connecting [`StdioQwenWorkerClient`] and [`LiveSqliteDailyDriver`].
pub fn run_live_worker_session(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<Vec<SessionEvent>> {
    run_live_worker_session_with_tui(driver, worker, user_prompt, max_turns, None, ask_callback)
}

/// Run a complete multi-turn session while optionally populating a [`TwoPaneTuiState`]
/// with every prefill envelope, `PenSpanRecord`, cartridge switch, and `<tool_response>`.
pub fn run_live_worker_session_with_tui(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    tui_state: Option<&mut TwoPaneTuiState>,
    ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<Vec<SessionEvent>> {
    let mut transcript_events = Vec::new();
    run_live_worker_session_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        tui_state,
        &mut transcript_events,
        ask_callback,
    )?;
    Ok(transcript_events)
}

/// Run a multi-turn session appending each completed turn to `transcript_events` immediately,
/// resetting the worker's `past_key_values` session at the start of the prompt.
pub fn run_live_worker_session_into(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<()> {
    run_live_worker_turn_sequence_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        true,
        tui_state,
        transcript_events,
        ask_callback,
    )
}

/// Run a multi-turn sequence for `user_prompt` returning the collected [`SessionEvent`] vector.
pub fn run_live_worker_turn_sequence(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    tui_state: Option<&mut TwoPaneTuiState>,
    ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<Vec<SessionEvent>> {
    let mut transcript_events = Vec::new();
    run_live_worker_turn_sequence_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        reset_model_session,
        tui_state,
        &mut transcript_events,
        ask_callback,
    )?;
    Ok(transcript_events)
}

/// Run a multi-turn sequence for `user_prompt` against [`StdioQwenWorkerClient`] and [`LiveSqliteDailyDriver`].
///
/// - When `reset_model_session == true` (single-shot `run`, first prompt of `repl` / `tui`, or
///   immediately after `reset` / `/reset`), calls `worker.reset_session()` and prefills
///   `build_lotl_initial_prompt(driver, user_prompt)` (including `[Table ...]`).
/// - When `reset_model_session == false` (subsequent prompts in an interactive `repl` or `tui`
///   session), preserves the worker's live `past_key_values` cache and prefills the follow-up
///   user turn `<|im_start|>user\n{effective_prompt}<|im_end|>\n`.
#[allow(clippy::too_many_arguments)]
pub fn run_live_worker_turn_sequence_into(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<()> {
    run_live_worker_turn_sequence_streaming_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        reset_model_session,
        tui_state,
        transcript_events,
        ask_callback,
        |_| {},
    )
}

struct CachedTurnGrammar {
    legal_modes: Vec<Mode>,
    specs_by_mode: BTreeMap<Mode, portia_core::GrammarCartridgeSpec>,
}

impl CachedTurnGrammar {
    /// Pre-compute and validate all legal mode payload grammars upfront (`∀prompt`-parametric,
    /// fail-closed) before a single token is generated so an environment error can never fall
    /// back to `GrammarCartridgeSpec::Unconstrained`.
    fn new<E: EpistemicDomainEnv + ?Sized>(env: &E) -> Result<Self> {
        let legal_modes = env.legal_modes()?;
        let mut specs_by_mode = BTreeMap::new();
        for &mode in &legal_modes {
            let spec = env.payload_grammar_spec(mode)?;
            specs_by_mode.insert(mode, spec);
        }
        Ok(Self {
            legal_modes,
            specs_by_mode,
        })
    }
}

impl portia_core::PayloadGrammarAutomaton for CachedTurnGrammar {
    fn legal_modes(&self) -> Vec<Mode> {
        self.legal_modes.clone()
    }

    fn grammar_cartridge_spec(&self, mode: Mode) -> portia_core::GrammarCartridgeSpec {
        self.specs_by_mode.get(&mode).cloned().unwrap_or_else(|| {
            portia_core::GrammarCartridgeSpec::LiteralSet(vec!["\n</to_user>".to_string()])
        })
    }
}

/// Generic prompt-blind (`∀prompt`) OODA coroutine loop driving any [`EpistemicDomainEnv`]
/// ([`LiveSqliteDailyDriver`] or [`crate::rust_driver::LiveRustAstDailyDriver`]) over OS pipes
/// against [`StdioQwenWorkerClient`], invoking `on_tui_frame` on every prefill, stepped token,
/// and domain observation.
#[allow(clippy::too_many_arguments)]
pub fn run_epistemic_env_turn_sequence_streaming_into<E: EpistemicDomainEnv + ?Sized>(
    env: &mut E,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    ask_callback: impl FnMut(&str) -> Result<String>,
    on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<()> {
    run_epistemic_env_turn_sequence_streaming_cancellable_into(
        env,
        worker,
        user_prompt,
        max_turns,
        reset_model_session,
        None,
        tui_state,
        transcript_events,
        ask_callback,
        on_tui_frame,
    )
}

/// Generic prompt-blind (`∀prompt`) OODA coroutine loop like
/// [`run_epistemic_env_turn_sequence_streaming_into`], honoring an optional `cancel_requested`
/// atomic flag at token and turn boundaries so interactive UIs can cancel in-flight streaming.
#[allow(clippy::too_many_arguments)]
pub fn run_epistemic_env_turn_sequence_streaming_cancellable_into<
    E: EpistemicDomainEnv + ?Sized,
>(
    env: &mut E,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    cancel_requested: Option<&std::sync::atomic::AtomicBool>,
    mut tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    mut ask_callback: impl FnMut(&str) -> Result<String>,
    mut on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<()> {
    use std::sync::atomic::Ordering;

    let state_version_before = env.state_version();
    let start_event_idx = transcript_events.len();
    env.begin_new_task();
    worker.trace_task_id += 1;
    worker.trace_record(serde_json::json!({
        "event": "task_start", "prompt": user_prompt, "state_version": state_version_before,
        "max_turns": max_turns, "reset_model_session": reset_model_session,
    }))?;
    let mut cycles = TurnCycleDetector::default();

    let inner_res = (|| -> Result<()> {
        if cancel_requested.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            return Err(SessionRunError::Cancelled {
                tokens_generated: 0,
                phase_at_cancel: "TaskPrefill".to_string(),
            }
            .into());
        }
        env.sync_worker_context(worker)?;

        if reset_model_session {
            env.reset_epistemic_session();
            worker.reset_session()?;
        }
        let schema_header = String::new();
        let prompt_envelope = format_lotl_user_prompt_envelope(user_prompt);
        let kv_disp = if reset_model_session {
            KvPrefillDisposition::ResetSessionKv
        } else {
            KvPrefillDisposition::PreserveStatefulSessionKv
        };
        let normalized_prompt =
            LotlNormalizedWireChunk::from_user_prompt_envelope(&prompt_envelope);
        let prompt_tids = worker.prefill_chunk_typed(
            CartridgeBank::Base,
            PenOwner::HarnessPen,
            normalized_prompt,
            kv_disp,
        )?;
        let base_turn_idx = tui_state.as_deref().map_or(0, |tui| {
            tui.interactions
                .iter()
                .filter_map(|item| match item {
                    InteractionItem::UserPrompt { turn_index, .. }
                    | InteractionItem::AssistantTurn { turn_index, .. }
                    | InteractionItem::Observation { turn_index, .. } => Some(*turn_index + 1),
                    InteractionItem::HarnessNote { .. } => None,
                })
                .max()
                .unwrap_or(0)
        });
        if let Some(tui) = tui_state.as_deref_mut() {
            env.on_tui_session_open(tui);
            tui.record_user_prompt_envelope(
                base_turn_idx,
                &schema_header,
                user_prompt.trim(),
                &prompt_envelope,
                prompt_tids,
            );
            on_tui_frame(tui);
        }

        for turn_idx in 0..max_turns {
            if cancel_requested.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                return Err(SessionRunError::Cancelled {
                    tokens_generated: 0,
                    phase_at_cancel: format!("TurnOpen({turn_idx})"),
                }
                .into());
            }
            let tui_turn_idx = base_turn_idx + turn_idx;
            if let Some(tui) = tui_state.as_deref_mut() {
                tui.begin_streaming_assistant_turn(tui_turn_idx);
            }
            let turn_grammar = CachedTurnGrammar::new(env)?;
            let mut partial_turn = Vec::new();
            let generated = worker.generate_turn_streaming_with_grammar_cancellable(
                512,
                &turn_grammar,
                cancel_requested,
                |ev| {
                    partial_turn.push(ev.clone());
                    if let Some(tui) = tui_state.as_deref_mut() {
                        tui.apply_relay_stream_event(tui_turn_idx, ev);
                        on_tui_frame(tui);
                    }
                },
            );
            let completed = match generated {
                Ok(completed) => completed,
                Err(error) => {
                    let trace_result = worker.trace_record(serde_json::json!({
                        "event": "diagnostic", "turn_index": turn_idx,
                        "error": format!("{error:#}"), "partial_turn": partial_turn,
                        "termination_reason": error.downcast_ref::<SessionRunError>(),
                        "legal_modes": turn_grammar.legal_modes,
                        "state_version": env.state_version(),
                    }));
                    return Err(match trace_result {
                        Ok(()) => error,
                        Err(trace_error) => error.context(format!(
                            "also failed writing generation trace: {trace_error:#}"
                        )),
                    });
                }
            };
            if tui_state.is_none() {
                eprintln!(
                    "[portia-cli] turn {}/{}: mode={:?} route={:?} plan={}",
                    turn_idx + 1,
                    max_turns,
                    completed.mode,
                    completed.route,
                    completed.plan.trim()
                );
            }
            if let Some(tui) = tui_state.as_deref_mut() {
                tui.finalize_streaming_assistant_turn(tui_turn_idx, &completed);
                on_tui_frame(tui);
            }
            let turn_state_version = env.state_version();
            let turn_steward_auth = env.steward_auth();
            if let Err(error) = cycles.observe(
                completed.mode,
                &completed.payload,
                turn_state_version,
                turn_steward_auth,
                turn_idx,
            ) {
                let trace_result = worker.trace_record(serde_json::json!({
                    "event": "diagnostic", "turn_index": turn_idx, "error": error,
                    "legal_modes": turn_grammar.legal_modes,
                    "generated_turn": { "mode": completed.mode, "plan": completed.plan,
                        "payload": completed.payload, "action": completed.action,
                        "full_wire_text": completed.full_wire_text, "spans": completed.spans },
                }));
                let error = anyhow::Error::new(error);
                return Err(match trace_result {
                    Ok(()) => error,
                    Err(trace_error) => {
                        error.context(format!("also failed writing cycle trace: {trace_error:#}"))
                    }
                });
            }
            let env_effect = env.classify_step_effect(completed.mode, &completed.payload);
            let step_effect = match (&env_effect, tui_state.as_deref()) {
                (CoroutineStepEffect::YieldToPrompt, Some(tui)) if completed.mode == Mode::Ask => {
                    tui.classify_coroutine_step_effect(
                        tui_turn_idx,
                        completed.mode,
                        &completed.payload,
                    )
                }
                _ => env_effect,
            };
            let obs = match &step_effect {
                CoroutineStepEffect::AutoContinueDb
                | CoroutineStepEffect::SuspendForGuiChoice { .. } => {
                    env.execute_turn_action(&completed.action, &mut ask_callback)
                }
                CoroutineStepEffect::YieldToPrompt => {
                    if completed.mode == Mode::Ask {
                        DriverObservation {
                            event_id: format!("chitchat-{tui_turn_idx:04}"),
                            mode: Mode::Ask,
                            ok: true,
                            db_version: env.state_version(),
                            table_versions: BTreeMap::new(),
                            columns: Vec::new(),
                            rows: Vec::new(),
                            rows_affected: 0,
                            message: None,
                            error: None,
                            operation: None,
                            ask_reply: None,
                            steward_auth: env.steward_auth(),
                            is_chat_continuation: true,
                        }
                    } else {
                        env.execute_turn_action(&completed.action, &mut ask_callback)
                    }
                }
            };
            if let Some(tui) = tui_state.as_deref_mut() {
                env.on_tui_turn_completed(completed.mode, &obs, tui);
            }
            let recorded_ask_mode = match &step_effect {
                CoroutineStepEffect::SuspendForGuiChoice { gui_mode, .. } => Some(gui_mode.clone()),
                CoroutineStepEffect::YieldToPrompt if completed.mode == Mode::Ask => {
                    Some(AskInteractionMode::ConversationalClarify)
                }
                _ => None,
            };
            transcript_events.push(SessionEvent {
                mode: completed.mode,
                route: completed.route,
                plan: completed.plan.clone(),
                payload: completed.payload.clone(),
                action: completed.action.clone(),
                observation: obs.clone(),
                ask_interaction_mode: recorded_ask_mode,
            });
            let observation_wire = match &step_effect {
                CoroutineStepEffect::AutoContinueDb
                | CoroutineStepEffect::SuspendForGuiChoice { .. } => {
                    Some(format_observation_wire_chunk(&obs)?)
                }
                CoroutineStepEffect::YieldToPrompt => None,
            };
            worker.trace_record(serde_json::json!({
                "event": "turn", "turn_index": turn_idx, "legal_modes": turn_grammar.legal_modes,
                "state_version_before": turn_state_version,
                "session_event": transcript_events.last(),
                "full_wire_text": completed.full_wire_text, "spans": completed.spans,
                "observation_wire": observation_wire,
            }))?;
            if obs.ok && obs.ask_reply.is_some() {
                cycles.new_input();
            }
            if completed.mode == Mode::Write && obs.ok && obs.rows_affected > 0 {
                cycles.allow_post_write_verification_reads();
            }
            if completed.mode == Mode::Ask && !obs.ok {
                bail!(
                    "{}",
                    obs.error
                        .as_deref()
                        .unwrap_or("Interactive [MODE: ASK] failed without steward reply")
                );
            }
            if completed.mode == Mode::Result && !obs.ok {
                bail!(
                    "{}",
                    obs.error
                        .as_deref()
                        .unwrap_or("[MODE: RESULT] verification failed")
                );
            }
            match step_effect {
                CoroutineStepEffect::AutoContinueDb
                | CoroutineStepEffect::SuspendForGuiChoice { .. } => {
                    let obs_chunk = format_observation_wire_chunk(&obs)?;
                    let obs_tids = worker.prefill_observation(completed.route, &obs)?;
                    if let Some(tui) = tui_state.as_deref_mut() {
                        tui.record_observation(
                            tui_turn_idx,
                            completed.route,
                            &obs,
                            &obs_chunk,
                            obs_tids,
                        );
                        on_tui_frame(tui);
                    }
                }
                CoroutineStepEffect::YieldToPrompt => {
                    if let Some(tui) = tui_state.as_deref_mut() {
                        tui.interactions.push(InteractionItem::Observation {
                            turn_index: tui_turn_idx,
                            observation: obs,
                        });
                        on_tui_frame(tui);
                    }
                    break;
                }
            }
        }
        Ok(())
    })();

    let state_version_after = env.state_version();
    let seq_events = &transcript_events[start_event_idx..];
    let mut receipt = summarize_session_outcome(
        seq_events,
        inner_res.is_err(),
        state_version_before,
        state_version_after,
    );
    attach_session_error(&mut receipt, inner_res.as_ref().err());
    receipt.provenance = worker.provenance.clone();
    let trace_result = worker.trace_record(serde_json::json!({
        "event": "task_end", "receipt": receipt,
        "error": inner_res.as_ref().err().map(|e| format!("{e:#}")),
    }));
    if tui_state.is_none() {
        emit_session_outcome_stderr(&receipt);
    }
    match (inner_res, trace_result) {
        (Err(error), Err(trace_error)) => Err(error.context(format!(
            "also failed writing task-end trace: {trace_error:#}"
        ))),
        (Ok(()), Err(trace_error)) => Err(trace_error),
        (result, Ok(())) => result,
    }
}

/// Run a multi-turn sequence for `user_prompt` while invoking `on_tui_frame` on every prefill,
/// stepped token, and SQLite observation so live TUI frontends redraw token-by-token.
#[allow(clippy::too_many_arguments)]
pub fn run_live_worker_turn_sequence_streaming_into(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    ask_callback: impl FnMut(&str) -> Result<String>,
    on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<()> {
    run_epistemic_env_turn_sequence_streaming_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        reset_model_session,
        tui_state,
        transcript_events,
        ask_callback,
        on_tui_frame,
    )
}

/// Cancel-aware variant of [`run_live_worker_turn_sequence_streaming_into`] driven by the
/// background TUI session actor (`Esc` / `Ctrl+C` sets `cancel_requested`).
#[allow(clippy::too_many_arguments)]
pub fn run_live_worker_turn_sequence_streaming_cancellable_into(
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    user_prompt: &str,
    max_turns: usize,
    reset_model_session: bool,
    cancel_requested: Option<&std::sync::atomic::AtomicBool>,
    tui_state: Option<&mut TwoPaneTuiState>,
    transcript_events: &mut Vec<SessionEvent>,
    ask_callback: impl FnMut(&str) -> Result<String>,
    on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<()> {
    run_epistemic_env_turn_sequence_streaming_cancellable_into(
        driver,
        worker,
        user_prompt,
        max_turns,
        reset_model_session,
        cancel_requested,
        tui_state,
        transcript_events,
        ask_callback,
        on_tui_frame,
    )
}
