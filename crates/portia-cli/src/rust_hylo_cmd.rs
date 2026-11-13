//! Interactive & Batch `AppType::RustAstHylo` Session Runner (`--app rust-hylo`)
//! driving [`LiveRustAstDailyDriver`] (`Scaffold -> Ask -> Fork × N -> hylo_octo_merge_rust_holes -> rustc --test`)
//! with [`TwoPaneTuiState`] visualization.

use crate::driver::{
    AstOperation, ColumnSchemaInfo, CoroutineStepEffect, DriverObservation, ObservationOperation,
    TableSchemaInfo,
};
use crate::rust_driver::{HyloFanoutExecutionReceipt, LiveRustAstDailyDriver};
use crate::tui::{detect_terminal_size, render_two_pane_tui_frame, TwoPaneTuiState};
use crate::worker::{format_observation_wire_chunk, StdioQwenWorkerClient};
use anyhow::{bail, Result};
use portia_core::relay::{CompletedRelayTurn, PenSpanRecord, RelayPhase, TypedTurnAction};
use portia_core::{CartridgeBank, Mode, PenOwner, RouteTag, StewardAuthState};
use portia_nursery::nursery_scenario::{AppType, DomainSkin, ScenarioCoordinate, ScenarioLanguage};
use portia_nursery::roguelite_lattice::RogueliteStressVector;
use portia_nursery::rust_ast_lens::ExtractedRustTodoHole;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Summary of an interactive or scripted `--app rust-hylo` session run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustHyloAppRunSummary {
    pub receipt: HyloFanoutExecutionReceipt,
    pub rendered_tui_frame: Option<String>,
}

/// Interactive command grammar for stepping a [`LiveRustAstDailyDriver`] in [`run_interactive_rust_hylo_command_loop`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RustHyloInteractiveCommand {
    /// Blank or no-op line.
    Empty,
    /// Run `Mode::Probe` (`format_ast_outline`).
    Probe,
    /// Run `Mode::Ask` with a question or preamble.
    Ask(String),
    /// Run `Mode::Read` on a target function (`fn <fn_name>`).
    Read(String),
    /// Run `Mode::Write` splicing `<fn_name> => <body>` or `fn <fn_name> { <body> }`.
    Write(String),
    /// Execute the full $2 \times 2$ (`Scaffold -> Ask -> Fork × N -> OctoMerge -> Result`) fanout.
    Fanout,
    /// Run `Mode::Result` (including `rustc --test` verification if enabled).
    Result(String),
    /// Reset task-local read/write counters on the driver.
    Reset,
    /// Exit the interactive loop.
    Quit,
}

/// Record of a single step executed inside [`run_interactive_rust_hylo_command_loop`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustHyloInteractiveStepRecord {
    pub turn_index: usize,
    pub command: RustHyloInteractiveCommand,
    pub mode: Option<Mode>,
    pub step_effect: Option<CoroutineStepEffect>,
    pub observation: Option<DriverObservation>,
}

/// Parse a scripted or interactive line into a [`RustHyloInteractiveCommand`].
pub fn parse_rust_hylo_interactive_command(line: &str) -> RustHyloInteractiveCommand {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return RustHyloInteractiveCommand::Empty;
    }
    let lower = trimmed.to_ascii_lowercase();
    if matches!(lower.as_str(), "quit" | "exit" | "\\q" | "/quit") {
        return RustHyloInteractiveCommand::Quit;
    }
    if matches!(lower.as_str(), "reset" | "/reset") {
        return RustHyloInteractiveCommand::Reset;
    }
    if matches!(lower.as_str(), "probe" | "/probe" | "outline" | "/outline") {
        return RustHyloInteractiveCommand::Probe;
    }
    if matches!(lower.as_str(), "fanout" | "/fanout" | "hylo" | "/hylo") {
        return RustHyloInteractiveCommand::Fanout;
    }
    if let Some(rest) = trimmed
        .strip_prefix("ask ")
        .or_else(|| trimmed.strip_prefix("ASK "))
    {
        return RustHyloInteractiveCommand::Ask(rest.trim().to_string());
    }
    if let Some(rest) = trimmed
        .strip_prefix("read ")
        .or_else(|| trimmed.strip_prefix("READ "))
    {
        return RustHyloInteractiveCommand::Read(rest.trim().to_string());
    }
    if let Some(rest) = trimmed
        .strip_prefix("write ")
        .or_else(|| trimmed.strip_prefix("WRITE "))
    {
        return RustHyloInteractiveCommand::Write(rest.trim().to_string());
    }
    if let Some(rest) = trimmed
        .strip_prefix("result ")
        .or_else(|| trimmed.strip_prefix("RESULT "))
    {
        return RustHyloInteractiveCommand::Result(rest.trim().to_string());
    }
    if lower == "result" || lower == "/result" {
        return RustHyloInteractiveCommand::Result(
            "{\"status\": \"RUST_AST_HYLO_COMPLETED\"}".to_string(),
        );
    }
    if trimmed.contains("=>") || trimmed.starts_with("fn ") {
        return RustHyloInteractiveCommand::Write(trimmed.to_string());
    }
    RustHyloInteractiveCommand::Ask(trimmed.to_string())
}

/// Inspect the module name of a [`LiveRustAstDailyDriver`] while optionally referencing [`TwoPaneTuiState`].
pub fn inspect_hylo_driver_module_name(
    drv: &LiveRustAstDailyDriver,
    _tui: Option<&TwoPaneTuiState>,
) -> Result<String> {
    Ok(drv.session().spec.module_name.clone())
}

/// Build a canonical single-pen [`CompletedRelayTurn`] for a `RustAstHylo` turn so [`TwoPaneTuiState`]
/// records both the left-pane OODA tree and the right-pane `HarnessPen -> RouterPen -> PayloadPen`
/// cartridge boundaries (`verify_single_pen_schedule` verified).
fn build_hylo_completed_relay_turn(
    mode: Mode,
    plan: &str,
    payload: &str,
    payload_bank: CartridgeBank,
) -> CompletedRelayTurn {
    build_hylo_completed_relay_turn_with_tokens(mode, plan, payload, payload_bank, None, None)
}

fn build_hylo_completed_relay_turn_with_tokens(
    mode: Mode,
    plan: &str,
    payload: &str,
    payload_bank: CartridgeBank,
    plan_token_ids: Option<Vec<u32>>,
    payload_token_ids: Option<Vec<u32>>,
) -> CompletedRelayTurn {
    let route: RouteTag = mode.route_tag();
    let mode_tok = mode.qwen35_first_token_id();
    let action = TypedTurnAction::parse_for_mode(mode, payload).unwrap_or_else(|_| {
        TypedTurnAction::SubmitResult {
            report_json: serde_json::json!({ "payload": payload }),
        }
    });
    let plan_text = format!("{plan}\n");
    let payload_text = format!("{payload}\n{}", route.close_tag());
    let resolved_plan_tids = plan_token_ids.unwrap_or_default();
    let resolved_payload_tids = payload_token_ids.unwrap_or_default();
    let spans = vec![
        PenSpanRecord {
            step_index: 0,
            phase: RelayPhase::TurnOpenPrefill,
            pen: PenOwner::HarnessPen,
            producer_bank: None,
            consumer_bank: CartridgeBank::OrientTrunk,
            text: "<|im_start|>assistant\n<think>\n[MODE:".to_string(),
            token_ids: vec![151644, 77091, 198, 151667, 198],
        },
        PenSpanRecord {
            step_index: 1,
            phase: RelayPhase::ModeLabel,
            pen: PenOwner::RouterPen,
            producer_bank: Some(CartridgeBank::OrientTrunk),
            consumer_bank: CartridgeBank::OrientTrunk,
            text: format!(" {}]", mode.as_str()),
            token_ids: vec![mode_tok, 60],
        },
        PenSpanRecord {
            step_index: 2,
            phase: RelayPhase::PostModePrefill,
            pen: PenOwner::HarnessPen,
            producer_bank: None,
            consumer_bank: CartridgeBank::Base,
            text: "\nPlan:".to_string(),
            token_ids: vec![198],
        },
        PenSpanRecord {
            step_index: 3,
            phase: RelayPhase::PlanBody,
            pen: PenOwner::PayloadPen,
            producer_bank: Some(CartridgeBank::Base),
            consumer_bank: CartridgeBank::Base,
            text: plan_text,
            token_ids: resolved_plan_tids,
        },
        PenSpanRecord {
            step_index: 4,
            phase: RelayPhase::PostPlanPrefill,
            pen: PenOwner::HarnessPen,
            producer_bank: None,
            consumer_bank: payload_bank,
            text: format!("</think>\n{}\n", route.open_tag()),
            token_ids: vec![151668, 198],
        },
        PenSpanRecord {
            step_index: 5,
            phase: RelayPhase::PayloadBody,
            pen: PenOwner::PayloadPen,
            producer_bank: Some(payload_bank),
            consumer_bank: payload_bank,
            text: payload_text,
            token_ids: resolved_payload_tids,
        },
        PenSpanRecord {
            step_index: 6,
            phase: RelayPhase::TurnComplete,
            pen: PenOwner::HarnessPen,
            producer_bank: None,
            consumer_bank: CartridgeBank::Base,
            text: "<|im_end|>\n".to_string(),
            token_ids: vec![151645, 198],
        },
    ];
    CompletedRelayTurn {
        mode,
        route,
        plan: plan.to_string(),
        payload: payload.to_string(),
        action,
        full_wire_text: format!(
            "<|im_start|>assistant\n<think>\n[MODE: {}]\nPlan:{}\n</think>\n{}\n{}\n{}<|im_end|>\n",
            mode.as_str(),
            plan,
            route.open_tag(),
            payload,
            route.close_tag()
        ),
        spans,
    }
}

fn record_hylo_turn_into_tui(
    tui_state: &mut TwoPaneTuiState,
    turn_index: usize,
    mode: Mode,
    plan: &str,
    payload: &str,
    obs: &DriverObservation,
) {
    let payload_bank = CartridgeBank::ModePayload(mode);
    let completed = build_hylo_completed_relay_turn(mode, plan, payload, payload_bank);
    tui_state.record_completed_relay_turn(turn_index, &completed);
    let route = mode.route_tag();
    if mode != Mode::Result && mode != Mode::Abstain {
        let wire_chunk = format_observation_wire_chunk(obs).unwrap_or_default();
        tui_state.record_observation(turn_index, route, obs, &wire_chunk, Vec::new());
    }
}

/// Construct a [`LiveRustAstDailyDriver`] from `(ScenarioLanguage, DomainSkin, hole_count)` and an
/// optional `.rs` file path, then execute [`run_rust_hylo_session_with_tui`].
pub fn run_rust_hylo_coordinate_session_with_tui(
    language: ScenarioLanguage,
    domain_skin: DomainSkin,
    hole_count: usize,
    rust_file: Option<PathBuf>,
    leaf_solver: impl FnMut(&ExtractedRustTodoHole, &str, &str) -> Result<String>,
    render_tui: bool,
) -> Result<RustHyloAppRunSummary> {
    let clamped_holes = hole_count.clamp(3, 16);
    let mut stress = RogueliteStressVector::sweet_spot_baseline();
    stress.ast_symbol_haystack = clamped_holes;
    stress.schema_graph_topology = clamped_holes;
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        language,
        domain_skin,
        stress,
        0x4000_0001,
    );
    let drv = if let Some(ref path) = rust_file {
        if path.exists() {
            LiveRustAstDailyDriver::open_rust_file(path)?.with_rustc_verification(true)
        } else {
            let spec = coord.build_cleanroom_rust_hylo_spec();
            std::fs::write(path, &spec.ast_valid_token_zero_skeleton)?;
            LiveRustAstDailyDriver::open_rust_file(path)?.with_rustc_verification(true)
        }
    } else {
        let spec = coord.build_cleanroom_rust_hylo_spec();
        LiveRustAstDailyDriver::from_spec(spec)?.with_rustc_verification(true)
    };
    run_rust_hylo_session_with_tui(drv, leaf_solver, render_tui)
}

/// Drive the full $2 \times 2$ (`Scaffold::TypeCart` $\to$ `Scaffold::ValueCart` $\to$ `Mode::Ask`
/// Dialogic Affordances $\to$ `Fork × N` isolated $O(1)$ leaf windows $\to$ `Impl::TypeCart` $\to$
/// `Impl::ValueCart` $\to$ `hylo_octo_merge_rust_holes` $\to$ `rustc --test` binary execution)
/// pipeline on `drv`, recording every turn into [`TwoPaneTuiState`] and returning [`RustHyloAppRunSummary`].
pub fn run_rust_hylo_session_with_tui(
    mut drv: LiveRustAstDailyDriver,
    leaf_solver: impl FnMut(&ExtractedRustTodoHole, &str, &str) -> Result<String>,
    render_tui: bool,
) -> Result<RustHyloAppRunSummary> {
    let receipt = drv.execute_hylo_2x2_fanout_and_octomerge(leaf_solver)?;
    let tui_state = build_tui_state_from_hylo_receipt(&receipt);

    let rendered_tui_frame = if render_tui {
        let (w, h) = detect_terminal_size();
        let frame_w = w.max(120);
        let frame_h = h.max(36);
        Some(render_two_pane_tui_frame(
            &tui_state, frame_w, frame_h, true,
        ))
    } else {
        None
    };

    Ok(RustHyloAppRunSummary {
        receipt,
        rendered_tui_frame,
    })
}

/// Execute the `--app rust-hylo` workflow (`execute_rust_hylo_cli`), driving a live
/// [`StdioQwenWorkerClient`] for isolated $O(1)$ leaf hole synthesis (`<to_code>` or
/// `fn_name => body` payload from `worker.generate_turn`). Requires a live worker.
pub fn execute_rust_hylo_cli(
    language: ScenarioLanguage,
    domain_skin: DomainSkin,
    hole_count: usize,
    rust_file: Option<PathBuf>,
    worker: Option<&mut StdioQwenWorkerClient>,
    max_tokens_per_turn: usize,
    render_tui: bool,
) -> Result<RustHyloAppRunSummary> {
    let Some(w) = worker else {
        bail!("execute_rust_hylo_cli requires a live StdioQwenWorkerClient (no hardcoded leaf solver stub)");
    };
    let max_toks = max_tokens_per_turn.max(32);
    run_rust_hylo_coordinate_session_with_tui(
        language,
        domain_skin,
        hole_count,
        rust_file,
        |hole, outline, fn_window| {
            w.reset_session()?;
            let leaf_prompt = format!(
                "<|im_start|>user\n{outline}\n\n{fn_window}\n\
                 Fill the `todo!(...)` hole in `{}`.<|im_end|>\n",
                hole.fn_name
            );
            let _ = w.prefill_chunk(CartridgeBank::Base, PenOwner::HarnessPen, &leaf_prompt)?;
            let turn = w.generate_turn(max_toks)?;
            let raw_payload = turn.payload.trim();
            if let Some((_, rhs)) =
                portia_nursery::rust_ast_lens::UnifiedRustAstScenarioSession::parse_write_payload(
                    raw_payload,
                )
            {
                Ok(rhs)
            } else if !raw_payload.is_empty() {
                Ok(raw_payload.to_string())
            } else {
                bail!(
                    "worker returned empty payload for Rust AST hole `{}`",
                    hole.fn_name
                )
            }
        },
        render_tui,
    )
}

/// Populate a complete [`TwoPaneTuiState`] from a [`HyloFanoutExecutionReceipt`], recording the
/// initial prompt envelope, `Mode::Probe` (`Scaffold::TypeCart -> Scaffold::ValueCart`),
/// `Mode::Ask` (dialogic clarification + crate approval), each leaf's `Mode::Read` + `Mode::Write`
/// splice (`Fork × N -> Impl::TypeCart -> Impl::ValueCart`), and `Mode::Result` (`OctoMerge` + `rustc --test`).
pub fn build_tui_state_from_hylo_receipt(receipt: &HyloFanoutExecutionReceipt) -> TwoPaneTuiState {
    let mut tui_state = TwoPaneTuiState::new(
        format!("rust_ast::{}", receipt.module_name),
        "Qwen/Qwen3.5-0.8B-Base",
        "RustAstHylo-2x2-Matrix",
        vec![TableSchemaInfo::new(
            receipt.module_name.clone(),
            receipt
                .leaf_contexts
                .iter()
                .enumerate()
                .map(|(i, leaf)| {
                    ColumnSchemaInfo::with_sqlite_metadata(
                        i as i64,
                        leaf.fn_name.clone(),
                        leaf.signature.clone(),
                        true,
                        i == 0,
                    )
                })
                .collect(),
        )
        .with_row_count(receipt.leaf_contexts.len())],
    );

    let prompt_txt = format!(
        "Scaffold, fork x {}, and octo-merge cleanroom Rust module `{}`",
        receipt.leaf_contexts.len(),
        receipt.module_name
    );
    let full_env = format!(
        "<|im_start|>user\n{}\n{}<|im_end|>\n",
        receipt.ast_outline_before, prompt_txt
    );
    tui_state.record_user_prompt_envelope(
        1,
        &receipt.ast_outline_before,
        &prompt_txt,
        &full_env,
        vec![151644, 872, 198],
    );

    let mut turn_idx = 1usize;
    let mut fn_versions = BTreeMap::new();
    for leaf in &receipt.leaf_contexts {
        fn_versions.insert(leaf.fn_name.clone(), 0u64);
    }

    // Turn 1: Mode::Probe (`Scaffold::TypeCart -> Scaffold::ValueCart`)
    let probe_obs = DriverObservation {
        event_id: format!("ast-obs-{turn_idx:04}"),
        mode: Mode::Probe,
        ok: true,
        db_version: 0,
        table_versions: fn_versions.clone(),
        columns: vec!["module".to_string(), "ast_outline".to_string()],
        rows: vec![vec![
            serde_json::Value::String(receipt.module_name.clone()),
            serde_json::Value::String(receipt.ast_outline_before.clone()),
        ]],
        rows_affected: 1,
        message: Some(receipt.ast_outline_before.clone()),
        error: None,
        operation: Some(ObservationOperation::Ast(AstOperation::Probe)),
        ask_reply: None,
        steward_auth: StewardAuthState::NotRequired,
        is_chat_continuation: false,
    };
    record_hylo_turn_into_tui(
        &mut tui_state,
        turn_idx,
        Mode::Probe,
        "Inspect cleanroom Rust AST outline at Scaffold::TypeCart -> Scaffold::ValueCart.",
        "inspect rust ast outline",
        &probe_obs,
    );
    turn_idx += 1;

    // Turn 2: Mode::Ask (Dialogic Affordances: clarify underflow floor + request approved ecosystem crate)
    let (approved_crate, crate_purpose) = receipt.coordinate.domain_skin.approved_ecosystem_crate();
    let ask_question = format!(
        "Please clarify the ambiguous underflow invariant, approve crate {approved_crate}, and review scaffold before fanout."
    );
    let ask_reply_text = format!(
        "[CLARIFIED]: Use saturating floor at 0. [CRATE_APPROVED `{approved_crate}`]: Approved for {crate_purpose}. [SCAFFOLD_REVIEWED]: Approved Fork x {} fanout.",
        receipt.leaf_contexts.len()
    );
    let ask_obs = DriverObservation {
        event_id: format!("ast-obs-{turn_idx:04}"),
        mode: Mode::Ask,
        ok: true,
        db_version: 0,
        table_versions: fn_versions.clone(),
        columns: vec!["clarification".to_string(), "status".to_string()],
        rows: vec![vec![
            serde_json::Value::String(ask_reply_text.clone()),
            serde_json::Value::String("CLARIFIED".to_string()),
        ]],
        rows_affected: 1,
        message: Some(ask_reply_text),
        error: None,
        operation: Some(ObservationOperation::Ast(AstOperation::Ask)),
        ask_reply: None,
        steward_auth: StewardAuthState::NotRequired,
        is_chat_continuation: false,
    };
    record_hylo_turn_into_tui(
        &mut tui_state,
        turn_idx,
        Mode::Ask,
        "Clarify underspecified invariant, request ecosystem crate, and review scaffold before Fork x N.",
        &ask_question,
        &ask_obs,
    );
    turn_idx += 1;

    // Turns 3..: Fork × N isolated leaf `Mode::Read` + `Mode::Write` (`Impl::TypeCart -> Impl::ValueCart`)
    let total_holes = receipt.leaf_contexts.len();
    for (idx, leaf) in receipt.leaf_contexts.iter().enumerate() {
        let read_obs = DriverObservation {
            event_id: format!("ast-obs-{turn_idx:04}"),
            mode: Mode::Read,
            ok: true,
            db_version: idx as u64,
            table_versions: fn_versions.clone(),
            columns: vec!["fn_name".to_string(), "fn_window".to_string()],
            rows: vec![vec![
                serde_json::Value::String(leaf.fn_name.clone()),
                serde_json::Value::String(leaf.leaf_prompt_window.clone()),
            ]],
            rows_affected: 1,
            message: Some(leaf.leaf_prompt_window.clone()),
            error: None,
            operation: Some(ObservationOperation::Ast(AstOperation::ReadFn)),
            ask_reply: None,
            steward_auth: StewardAuthState::NotRequired,
            is_chat_continuation: false,
        };
        record_hylo_turn_into_tui(
            &mut tui_state,
            turn_idx,
            Mode::Read,
            &format!(
                "Slice O(1) surgical AST window for leaf hole #{} `{}` ({} BPE tokens).",
                leaf.hole_index, leaf.fn_name, leaf.leaf_window_bpe_tokens
            ),
            &leaf.fn_name,
            &read_obs,
        );
        turn_idx += 1;

        *fn_versions.entry(leaf.fn_name.clone()).or_insert(0) += 1;
        let remaining = total_holes.saturating_sub(idx + 1);
        let write_payload = format!("{} => {}", leaf.fn_name, leaf.replacement_body);
        let write_obs = DriverObservation {
            event_id: format!("ast-obs-{turn_idx:04}"),
            mode: Mode::Write,
            ok: leaf.leaf_ast_verified,
            db_version: (idx + 1) as u64,
            table_versions: fn_versions.clone(),
            columns: vec!["status".to_string(), "remaining_todo_holes".to_string()],
            rows: vec![vec![
                serde_json::Value::String("SPLICED".to_string()),
                serde_json::Value::from(remaining as u64),
            ]],
            rows_affected: 1,
            message: Some(format!(
                "{{\"status\": \"SPLICED\", \"fn_name\": \"{}\", \"remaining_todo_holes\": {}}}",
                leaf.fn_name, remaining
            )),
            error: None,
            operation: Some(ObservationOperation::Ast(AstOperation::SpliceHole)),
            ask_reply: None,
            steward_auth: StewardAuthState::NotRequired,
            is_chat_continuation: false,
        };
        record_hylo_turn_into_tui(
            &mut tui_state,
            turn_idx,
            Mode::Write,
            &format!(
                "Splice verified leaf body into `{}` at Impl::ValueCart (remaining holes: {}).",
                leaf.fn_name, remaining
            ),
            &write_payload,
            &write_obs,
        );
        turn_idx += 1;
    }

    // Final Turn: Mode::Result (`OctoMerge` + `rustc --test` binary execution)
    let result_payload = format!(
        "{{\"status\": \"HYLO_2X2_OCTO_MERGED_AND_RUSTC_VERIFIED\", \"module\": \"{}\", \"merged_holes\": {}, \"rustc_verified\": {}}}",
        receipt.module_name,
        receipt.leaf_contexts.len(),
        receipt.rustc_verified
    );
    let result_obs = DriverObservation {
        event_id: format!("ast-obs-{turn_idx:04}"),
        mode: Mode::Result,
        ok: receipt.rustc_verified && receipt.episode_summary.cleared,
        db_version: total_holes as u64,
        table_versions: fn_versions,
        columns: vec![],
        rows: vec![],
        rows_affected: 0,
        message: Some(result_payload.clone()),
        error: None,
        operation: Some(ObservationOperation::Ast(AstOperation::Result)),
        ask_reply: None,
        steward_auth: StewardAuthState::NotRequired,
        is_chat_continuation: false,
    };
    record_hylo_turn_into_tui(
        &mut tui_state,
        turn_idx,
        Mode::Result,
        "Octo-merge all N leaf patches via try_expand_and_collapse and execute rustc --test binary.",
        &result_payload,
        &result_obs,
    );

    tui_state
}

/// Step a [`LiveRustAstDailyDriver`] across scripted or interactive commands (`probe`, `ask <question>`,
/// `read <fn_name>`, `write <fn_name> => <body>`, `fanout`, `result`, `reset`, `quit`) using
/// [`CoroutineStepEffect`] (`AutoContinueDb`, `SuspendForGuiChoice`, `YieldToPrompt`)
/// and recording each step into `tui_state`.
pub fn run_interactive_rust_hylo_command_loop<I, S>(
    drv: &mut LiveRustAstDailyDriver,
    tui_state: &mut TwoPaneTuiState,
    commands: I,
    mut ask_callback: impl FnMut(&str) -> Result<String>,
) -> Result<Vec<RustHyloInteractiveStepRecord>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut records = Vec::new();
    let mut turn_idx = 1usize;

    for raw_line in commands {
        let cmd = parse_rust_hylo_interactive_command(raw_line.as_ref());
        match cmd.clone() {
            RustHyloInteractiveCommand::Empty => continue,
            RustHyloInteractiveCommand::Quit => {
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Quit,
                    mode: None,
                    step_effect: Some(CoroutineStepEffect::YieldToPrompt),
                    observation: None,
                });
                break;
            }
            RustHyloInteractiveCommand::Reset => {
                drv.begin_new_task();
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Reset,
                    mode: None,
                    step_effect: None,
                    observation: None,
                });
            }
            RustHyloInteractiveCommand::Probe => {
                let payload = "inspect rust ast outline";
                let plan = "Inspect cleanroom Rust AST outline.";
                let effect = drv.classify_coroutine_step_effect(Mode::Probe, payload);
                let obs = drv.execute_turn(Mode::Probe, payload, &mut ask_callback);
                record_hylo_turn_into_tui(tui_state, turn_idx, Mode::Probe, plan, payload, &obs);
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: cmd,
                    mode: Some(Mode::Probe),
                    step_effect: Some(effect),
                    observation: Some(obs),
                });
                turn_idx += 1;
            }
            RustHyloInteractiveCommand::Ask(question) => {
                let plan = "Clarify invariant, request approved crate, or disambiguate todo hole.";
                let effect = drv.classify_coroutine_step_effect(Mode::Ask, &question);
                let obs = drv.execute_turn(Mode::Ask, &question, &mut ask_callback);
                record_hylo_turn_into_tui(tui_state, turn_idx, Mode::Ask, plan, &question, &obs);
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Ask(question),
                    mode: Some(Mode::Ask),
                    step_effect: Some(effect),
                    observation: Some(obs),
                });
                turn_idx += 1;
            }
            RustHyloInteractiveCommand::Read(fn_name) => {
                let plan = format!("Slice O(1) AST window for `{fn_name}`.");
                let effect = drv.classify_coroutine_step_effect(Mode::Read, &fn_name);
                let obs = drv.execute_turn(Mode::Read, &fn_name, &mut ask_callback);
                record_hylo_turn_into_tui(tui_state, turn_idx, Mode::Read, &plan, &fn_name, &obs);
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Read(fn_name),
                    mode: Some(Mode::Read),
                    step_effect: Some(effect),
                    observation: Some(obs),
                });
                turn_idx += 1;
            }
            RustHyloInteractiveCommand::Write(write_payload) => {
                let plan = "Splice verified replacement body into todo!() hole.";
                let effect = drv.classify_coroutine_step_effect(Mode::Write, &write_payload);
                let obs = drv.execute_turn(Mode::Write, &write_payload, &mut ask_callback);
                record_hylo_turn_into_tui(
                    tui_state,
                    turn_idx,
                    Mode::Write,
                    plan,
                    &write_payload,
                    &obs,
                );
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Write(write_payload),
                    mode: Some(Mode::Write),
                    step_effect: Some(effect),
                    observation: Some(obs),
                });
                turn_idx += 1;
            }
            RustHyloInteractiveCommand::Fanout => {
                bail!(
                    "interactive `fanout` requires explicit `write <fn_name> => <body>` leaf bodies or a live worker"
                );
            }
            RustHyloInteractiveCommand::Result(report) => {
                let plan = "Finalize cleanroom Rust AST module and run rustc --test verification.";
                let effect = drv.classify_coroutine_step_effect(Mode::Result, &report);
                let obs = drv.execute_turn(Mode::Result, &report, &mut ask_callback);
                if !obs.ok {
                    bail!(
                        "Mode::Result failed in interactive RustHylo loop: {}",
                        obs.error.unwrap_or_default()
                    );
                }
                record_hylo_turn_into_tui(tui_state, turn_idx, Mode::Result, plan, &report, &obs);
                records.push(RustHyloInteractiveStepRecord {
                    turn_index: turn_idx,
                    command: RustHyloInteractiveCommand::Result(report),
                    mode: Some(Mode::Result),
                    step_effect: Some(effect),
                    observation: Some(obs),
                });
                turn_idx += 1;
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_rust_hylo_session_with_tui_verifies_7_holes_and_renders_two_pane_frame() {
        let summary = run_rust_hylo_coordinate_session_with_tui(
            ScenarioLanguage::English,
            DomainSkin::OrbitalConstellation,
            7,
            None,
            |hole, outline, fn_window| {
                assert!(outline.contains("[Module `orbital_delta_v_planner`]"));
                assert!(fn_window.contains(&hole.fn_name));
                Ok(format!(
                    "approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    hole.hole_index
                ))
            },
            true,
        )
        .expect("run_rust_hylo_coordinate_session_with_tui must succeed");

        assert!(summary.receipt.rustc_verified);
        assert!(summary.receipt.episode_summary.cleared);
        assert_eq!(summary.receipt.leaf_contexts.len(), 7);
        let frame = summary
            .rendered_tui_frame
            .expect("rendered_tui_frame must be Some when render_tui = true");
        assert!(!frame.trim().is_empty());
        assert!(frame.contains("orbital_delta_v_planner"));
    }

    #[test]
    fn interactive_rust_hylo_command_loop_steps_all_coroutine_effects_and_verifies_rustc() {
        let mut stress = RogueliteStressVector::sweet_spot_baseline();
        stress.ast_symbol_haystack = 3;
        let coord = ScenarioCoordinate::new(
            AppType::RustAstHylo,
            ScenarioLanguage::English,
            DomainSkin::SolarisVault,
            stress,
            0x4000_0042,
        );
        let spec = coord.build_cleanroom_rust_hylo_spec();
        let holes = spec.todo_holes.clone();
        let mut drv = LiveRustAstDailyDriver::from_spec(spec)
            .expect("driver init")
            .with_rustc_verification(true);
        let mut tui_state = TwoPaneTuiState::new(
            "rust_ast::solaris_seal_verifier",
            "Qwen/Qwen3.5-0.8B-Base",
            "RustAstHylo-2x2-Matrix",
            vec![],
        );

        let mut scripted_cmds = vec![
            // 1. Conversational `Mode::Ask` before `Probe` -> `YieldToPrompt`
            "ask Sure, I will inspect the Rust AST outline and fill the todo!() holes.".to_string(),
            // 2. `Probe` -> `AutoContinueDb` (sets `probed_ast_outline = true`)
            "probe".to_string(),
            // 3. Multi-hole disambiguation question after `Probe` -> `SuspendForGuiChoice(MultiSelect)`
            "ask Which todo!() function hole should we implement first?".to_string(),
            // 4. Dialogic clarification + crate approval -> `SuspendForGuiChoice`
            "ask Please clarify the ambiguous underflow invariant, approve crate tree-sitter, and review scaffold before fanout.".to_string(),
        ];
        for h in &holes {
            scripted_cmds.push(format!("read {}", h.fn_name));
            scripted_cmds.push(format!(
                "write {} => approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                h.fn_name, h.hole_index
            ));
        }
        scripted_cmds.push("result".to_string());
        scripted_cmds.push("quit".to_string());

        let records =
            run_interactive_rust_hylo_command_loop(&mut drv, &mut tui_state, scripted_cmds, |_| {
                Ok("ALLOW".to_string())
            })
            .expect("interactive loop must succeed");

        assert_eq!(
            records[0].step_effect,
            Some(CoroutineStepEffect::YieldToPrompt)
        );
        assert_eq!(
            records[1].step_effect,
            Some(CoroutineStepEffect::AutoContinueDb)
        );
        assert!(matches!(
            records[2].step_effect,
            Some(CoroutineStepEffect::SuspendForGuiChoice { .. })
        ));
        assert!(records
            .iter()
            .any(|r| r.mode == Some(Mode::Result) && r.observation.as_ref().unwrap().ok));
        assert!(!tui_state.interactions.is_empty());
        assert!(!tui_state.context_spans.is_empty());
    }

    #[test]
    fn execute_rust_hylo_cli_drives_live_stdio_worker_when_provided() {
        let mock_worker = tempfile::NamedTempFile::new().unwrap();
        let mock_py_code = r#"import sys, json
hole_idx = 0
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    op = req.get("op")
    if op in ("reset_session", "begin_task", "prefill"):
        sys.stdout.write(json.dumps({"status": "prefilled", "token_ids": [256, 257]}) + "\n")
        sys.stdout.flush()
    elif op in ("sample_token", "step_token"):
        status_key = "sampled_token" if op == "sample_token" else "token"
        phase = str(req.get("phase", "")).lower()
        if phase in ("modelabel", "mode_label"):
            resp = {"status": status_key, "token_id": 64794, "token_piece": " WRITE]", "token_ids": [64794, 60]}
        elif phase in ("planbody", "plan_body"):
            resp = {"status": status_key, "token_id": 198, "token_piece": "Splice hole via approved_crate.\n", "token_ids": [1201, 198]}
        elif phase in ("payloadbody", "payload_body"):
            piece = f"approved_crate::eval_domain_step(input_metric, guard_clause, {hole_idx}usize)\n</sql>"
            hole_idx += 1
            resp = {"status": status_key, "token_id": 151668, "token_piece": piece, "token_ids": [2201, 2202]}
        else:
            resp = {"status": "error", "message": f"unexpected phase {phase}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    elif op == "consume_token":
        sys.stdout.write(json.dumps({"status": "consumed_token", "token_ids": [int(req["token_id"])]}) + "\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps({"status": "error", "message": f"unknown op {op}"}) + "\n")
        sys.stdout.flush()
"#;
        std::fs::write(mock_worker.path(), mock_py_code).unwrap();
        let cmd = format!("python3 -u {:?}", mock_worker.path());
        let mut worker = StdioQwenWorkerClient::spawn_shell(&cmd)
            .expect("spawn mock stdio worker for execute_rust_hylo_cli");
        let summary = execute_rust_hylo_cli(
            ScenarioLanguage::English,
            DomainSkin::SolarisVault,
            3,
            None,
            Some(&mut worker),
            64,
            false,
        )
        .expect("execute_rust_hylo_cli with live worker must succeed");
        assert!(summary.receipt.rustc_verified);
        assert!(summary.receipt.episode_summary.cleared);
        assert_eq!(summary.receipt.leaf_contexts.len(), 3);
    }
}
