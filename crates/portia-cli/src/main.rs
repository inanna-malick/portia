//! `portia-cli` — Real External Database Daily-Driver CLI, Live SQLite Provenance Guard,
//! & JSON-over-stdio PyTorch Qwen Forward-Pass Worker Bridge.
//!
//! Designed for the external Qwen + Portia daily-driver workflows:
//! - Connects to any arbitrary `.sqlite` database file on disk (multi-table schemas, arbitrary
//!   `JOIN` / `GROUP BY` / `COUNT(*)` / `ORDER BY` / `LIMIT` / CTEs in `READ`, `PRAGMA table_info` /
//!   `sqlite_master` in `PROBE`) without requiring benchmark `WaveSpec` or 4-column `Columns`!
//! - Enforces real-database safety guardrails (`LiveSqliteDailyDriver`):
//!   - `PROBE` & `READ` execute under `PRAGMA query_only = ON` so `READ` can never mutate tables.
//!   - `WRITE` parses one DML statement via `sqlparser::ast` (`SQLiteDialect`), enforces read-only
//!     and Steward permissions, runs inside a `SAVEPOINT` with configurable `max_write_rows`
//!     rollback protection, and bumps mutation versions for verification and cycle detection.
//!   - `ASK` routes questions to interactive terminal stdin/stdout (or a configured steward reply).
//! - Drives `portia_core::relay::RelayMachine` (`HarnessPen RouterPen PayloadPen`)
//!   over a clean JSON-lines stdio protocol with any PyTorch Qwen3.5 worker process.

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use portia_core::relay::{verify_single_pen_schedule, PenSpanRecord};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

pub mod batch_cmd;
pub mod compile;
pub mod compile_garak;
pub mod compile_v2;
pub mod driver;
pub mod export_safetensors;
pub mod model_identity;
pub mod nursery_cmd;
pub mod rust_driver;
pub mod rust_hylo_cmd;
pub mod synth_user;
pub mod tui;
pub mod worker;
use model_identity::{normalize_model_identifier, RuntimeProvenance, WorkerLaunchOptions};

pub use compile::{
    compile_cartridge_command, compile_cartridge_for_target,
    compile_cartridge_for_target_with_governance, specialize_regime_and_columns_for_compile_target,
    specialize_regime_and_columns_with_governance, specialize_regime_for_compile_target,
    verify_compiled_cartridge_bisimulation, CartridgeBisimulationSummary, CompileProvenanceReceipt,
    SchemaGovernanceColumns,
};
pub use driver::{
    classify_ask_prompt_kind, ensure_none_of_above_escape_choice, extract_ask_interaction_mode,
    extract_ask_select_choices, parse_ask_user_reply, parse_ask_user_reply_for_kind,
    parse_read_or_probe_tables, parse_single_write_target_table,
    resolve_ask_interaction_mode_from_observation_state, AskDecision, AskInteractionMode,
    AskPromptKind, AskUserReply, ColumnSchemaInfo, CoroutineStepEffect, DriverObservation,
    EpistemicDomainEnv, LiveSqliteDailyDriver, TableSchemaInfo, NONE_OF_ABOVE_ENTER_TEXT_CHOICE,
};
pub use rust_driver::{
    parse_domain_skin_flexible, parse_scenario_language_flexible, HyloFanoutExecutionReceipt,
    HyloLeafExecutionRecord, LiveRustAstDailyDriver,
};
pub use tui::{
    coalesce_cartridge_epochs, detect_terminal_size, export_observation_to_path,
    format_cartridge_bank_badge_for_scale, format_surfaced_error_ansi, format_surfaced_error_note,
    parse_row_display_limit_arg, render_two_pane_tui_frame, run_iocraft_tui_session,
    spawn_session_actor, summarize_loaded_cartridge_rack, AnnotatedContextSpan, CartridgeEpoch,
    ControlTokenBadge, InteractionItem, LoadedCartridgeBankCard, PortiaTuiApp,
    PortiaWorkspaceSnapshot, SessionActorHandle, TuiModelScaleSpec, TwoPaneTuiState,
};
pub use worker::{
    build_lotl_followup_prompt, build_lotl_initial_prompt, emit_session_outcome_stderr,
    format_live_schema_summary, format_lotl_user_prompt_envelope, format_observation_wire_chunk,
    resolve_default_cartridge_label, resolve_default_cartridge_label_for_model,
    resolve_default_worker_argv, resolve_default_worker_command,
    run_epistemic_env_turn_sequence_streaming_into, run_live_worker_session,
    run_live_worker_session_into, run_live_worker_session_with_tui,
    run_live_worker_turn_sequence_into, run_live_worker_turn_sequence_streaming_into,
    summarize_session_outcome, SessionEvent, SessionOutcomeStatus, SessionSummaryReceipt,
    StdioQwenWorkerClient, WorkerRequest, WorkerResponse,
};

#[derive(Debug, Parser)]
#[command(
    name = "portia-cli",
    version,
    about = "Run Qwen3.5 database and code tasks with compiled weight cartridges against SQLite databases",
    after_help = "Application Commands (run, inspect, repl, tui):\n  portia-cli inspect --db examples/bgp_incidents.sqlite\n  portia-cli run --db examples/bgp_incidents.sqlite --read-only --prompt \"Inspect bgp_incidents for pop_code = 'NRT-TOKYO-04'.\"\n  portia-cli run --db examples/bgp_incidents.sqlite --read-only --summary-only --prompt \"Inspect bgp_incidents for pop_code = 'NRT-TOKYO-04'.\"\n  portia-cli repl --db examples/alpine_levain.sqlite\n  portia-cli tui --db examples/alpine_levain.sqlite\n\nBuilder Commands (compile, verify-schedule):\n  portia-cli compile --input-json captures.json --db examples/bgp_incidents.sqlite --out-json /tmp/portia_cartridge.json\n  portia-cli verify-schedule --schedule-json schedule.json"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Shared worker, model, cartridge, and SQLite session controls across `run`, `repl`, and `tui`.
#[derive(Debug, Clone, Args)]
pub struct WorkerRuntimeArgs {
    /// HuggingFace model ID, short scale alias (`0.8b-base`, `2b-base`, `4b-base`, `9b-base`), or
    /// local snapshot path (`Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B-Base`,
    /// `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`, `Qwen/Qwen3.5-27B`).
    #[arg(long, default_value = "Qwen/Qwen3.5-0.8B-Base")]
    pub model: String,
    /// Canonical checkpoint identity for local directories without provenance metadata.
    #[arg(long)]
    pub model_family: Option<String>,
    /// Permit deliberate cross-family transfer; tensor and schema validation remain mandatory.
    #[arg(long)]
    pub allow_cartridge_model_mismatch: bool,
    /// Open the SQLite database in strict read-only mode (`SQLITE_OPEN_READ_ONLY` + `PRAGMA query_only = ON`)
    /// and block all `WRITE` mutations while recording any attempted write in the session transcript.
    #[arg(long)]
    pub read_only: bool,
    /// Maximum turns per task before stopping with a turn-limit error.
    #[arg(long, default_value_t = 64)]
    pub max_turns: usize,
    /// Append complete task/turn/diagnostic receipts as JSON Lines.
    #[arg(long)]
    pub trace_jsonl: Option<PathBuf>,
    /// Maximum rows a single `WRITE` statement may modify before automatic `SAVEPOINT` rollback.
    #[arg(long, default_value_t = 100)]
    pub max_write_rows: usize,
    /// Canned steward reply (`ALLOW` or `DENY`) for `[MODE: ASK]` prompts in non-interactive runs
    /// (if omitted, reads interactively from stdin and errors on EOF).
    #[arg(long, value_enum, ignore_case = true)]
    pub ask_reply: Option<AskDecision>,
    /// Explicit Steward governance column binding (`col_name` or `table.col_name`; repeatable) enabling `ASK` authorization gating on matching rows.
    #[arg(long = "steward-col")]
    pub steward_columns: Vec<String>,
    /// Enable multi-row candidate disambiguation prompts on `[MODE: ASK]` when a read matches multiple rows.
    #[arg(long)]
    pub disambiguate_candidates: bool,
    /// Optional path to a compiled cartridge JSON bundle or `none` for unpatched `W_0` (defaults to `artifacts/portia_v3_unified_cartridge_{08b,2b,4b,9b}_base.json` matching `--model`, or `$PORTIA_CARTRIDGE_JSON`).
    #[arg(long, help_heading = "Builder & Runtime Overrides")]
    pub cartridge: Option<PathBuf>,
    /// Optional custom command line to spawn a JSON-over-stdio worker subprocess via `sh -c`
    /// (if omitted, launches `scripts/portia_qwen_stdio_worker.py` directly via `$PORTIA_PYTHON` or `.venv/bin/python`).
    #[arg(long, help_heading = "Builder & Runtime Overrides")]
    pub worker_cmd: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run a single-prompt multi-turn session against an existing `.sqlite` database and print the JSON turn transcript (or `--summary-only` receipt) to stdout.
    #[command(
        name = "run",
        after_help = "Examples:\n  portia-cli run --db examples/bgp_incidents.sqlite --read-only --prompt \"Inspect bgp_incidents for pop_code = 'NRT-TOKYO-04'.\"\n  portia-cli run --db examples/bgp_incidents.sqlite --read-only --summary-only --prompt \"Inspect bgp_incidents for pop_code = 'NRT-TOKYO-04'.\"\n  portia-cli run --db /tmp/bgp_copy.sqlite --ask-reply ALLOW --prompt \"Update mitigation_rule for FRA-DECIX-02.\""
    )]
    Run {
        /// Path to an existing `.sqlite` database file.
        #[arg(long)]
        db: PathBuf,
        /// Initial user request or question.
        #[arg(long)]
        prompt: String,
        #[command(flatten)]
        runtime: WorkerRuntimeArgs,
        /// Emit only the compact `SessionSummaryReceipt` JSON (`status`, `turns_executed`, `writes_attempted`, `writes_committed`, `final_answer`) to stdout instead of the full multi-turn transcript array.
        #[arg(long)]
        summary_only: bool,
        /// Render the Two-Pane TUI frame to stdout instead of the JSON turn array.
        #[arg(long)]
        tui: bool,
    },
    /// Run a JSONL batch of SQLite prompts against one warm Qwen worker process (`worker.reset_session()` between tasks) and emit one `SessionSummaryReceipt` JSON line per task.
    #[command(
        name = "batch",
        after_help = "Examples:\n  portia-cli batch --tasks-jsonl tasks.jsonl --out-jsonl results.jsonl\n  portia-cli batch --tasks-jsonl tasks.jsonl --model 2b-base --ask-reply ALLOW"
    )]
    Batch {
        /// Path to an input JSONL file of task specs (`{\"case_id\": \"...\", \"db\": \"...\", \"prompt\": \"...\"}`).
        #[arg(long)]
        tasks_jsonl: PathBuf,
        /// Optional output JSONL file path (writes one receipt line per task to stdout when omitted).
        #[arg(long)]
        out_jsonl: Option<PathBuf>,
        #[command(flatten)]
        runtime: WorkerRuntimeArgs,
    },
    /// Start an interactive multi-turn REPL against an existing `.sqlite` database.
    #[command(
        name = "repl",
        after_help = "Example:\n  portia-cli repl --db examples/alpine_levain.sqlite"
    )]
    Repl {
        /// Path to an existing `.sqlite` database file.
        #[arg(long)]
        db: PathBuf,
        #[command(flatten)]
        runtime: WorkerRuntimeArgs,
        /// Render the Two-Pane TUI after every REPL turn.
        #[arg(long)]
        tui: bool,
    },
    /// Launch the interactive Agent Harness Terminal UI (`iocraft` split-drawer fullscreen or `--inline` scrollback).
    #[command(
        name = "tui",
        after_help = "Examples:\n  portia-cli tui --db examples/alpine_levain.sqlite\n  portia-cli tui --db examples/alpine_levain.sqlite --inline\n  portia-cli tui --db examples/alpine_levain.sqlite --read-only --prompt \"Inspect alpine_levain_campaigns.\""
    )]
    Tui {
        /// Path to an existing `.sqlite` database file.
        #[arg(long)]
        db: PathBuf,
        /// Optional initial prompt to execute immediately upon opening the TUI.
        #[arg(long)]
        prompt: Option<String>,
        #[command(flatten)]
        runtime: WorkerRuntimeArgs,
        /// Render in inline terminal scrollback mode instead of alternate-screen fullscreen.
        #[arg(long, default_value_t = false)]
        inline: bool,
        /// Terminal frame width in columns (auto-detected if omitted; forces snapshot mode when set).
        #[arg(long)]
        width: Option<usize>,
        /// Terminal frame height in rows (auto-detected if omitted; forces snapshot mode when set).
        #[arg(long)]
        height: Option<usize>,
    },
    /// Inspect an existing `.sqlite` database schema (`sqlite_master` + `PRAGMA table_info`) in read-only mode and print JSON to stdout.
    #[command(
        name = "inspect",
        after_help = "Example:\n  portia-cli inspect --db examples/bgp_incidents.sqlite"
    )]
    Inspect {
        /// Path to an existing `.sqlite` database file.
        #[arg(long)]
        db: PathBuf,
    },
    /// Compile a weight-cartridge JSON bundle in pure Rust via Clarabel SOCP + `SO(d)` and `RegimeExpr` schema specialization.
    #[command(
        name = "compile",
        after_help = "Examples:\n  portia-cli compile --input-json captures.json --db examples/bgp_incidents.sqlite --out-json /tmp/cartridge.json\n  portia-cli compile --input-json captures.json --out-json /tmp/cartridge.json"
    )]
    Compile {
        /// Path to an input JSON file containing captured model activations (`{"activations": ..., "row_states": [...]}` or `"obligations": [...]`).
        #[arg(long)]
        input_json: PathBuf,
        /// Optional path to a `.sqlite` database file for Futamura-1 `RegimeExpr::specialize_for_schema_invariants`.
        #[arg(long)]
        db: Option<PathBuf>,
        /// Optional path to a serialized `RegimeExpr` JSON file (defaults to `canonical_pocket_regime_dsl()` if omitted).
        #[arg(long)]
        regime_json: Option<PathBuf>,
        /// Specialize `RegimeExpr` for strict read-only execution (`Mode::Write` and `Mode::Ask` pruned).
        #[arg(long)]
        read_only: bool,
        /// Explicit steward governance column binding (`col_name` or `table.col_name`; repeatable).
        #[arg(long = "steward-col")]
        steward_columns: Vec<String>,
        /// Explicit lock governance column binding (`col_name` or `table.col_name`; repeatable).
        #[arg(long = "lock-col")]
        lock_columns: Vec<String>,
        /// Declare that target schema metadata is pre-disclosed in context at Turn 0 (pruning `Mode::Probe`). Defaults to `false` so `--db` alone preserves `Mode::Probe`.
        #[arg(long, default_value_t = false)]
        schema_disclosed: bool,
        /// Allow synthetic coordinate fixture readout targets when `compiler_weight_geometry` and `mode_target_vectors` are absent (testing only; `false` in production).
        #[arg(long, default_value_t = false)]
        allow_synthetic_fixture_targets: bool,
        /// Compiler backend (`v2` Pointer-ISA `faer` + Clarabel SOCP + `ASID`).
        #[arg(long, default_value = "v2")]
        compiler_engine: String,
        /// Output path for the compiled cartridge JSON bundle.
        #[arg(long)]
        out_json: PathBuf,
    },
    /// Compile the 8-slot coupled `Garak` static honesty & anti-hallucination cartridge (`permanent_static`) in pure Rust (`f64` `faer` + `SO(d)` + `StrictZeroCone`).
    #[command(
        name = "compile-garak",
        after_help = "Examples:\n  portia-cli compile-garak --input-json garak_states.json --out-json /tmp/garak_cartridge.json\n  portia-cli compile-garak --input-json garak_states.json --out-json /tmp/garak_comount.json --slot-offset 48 --widened-slots 128 --max-bg-leakage 0.25"
    )]
    CompileGarak {
        /// Path to an input JSON file containing `GarakCompileInputSpec` (or `GarakReadoutSpec` with `--readout-only`).
        #[arg(long)]
        input_json: PathBuf,
        /// Output path for the compiled `permanent_static` `WireCartridgeManifestJson` manifest.
        #[arg(long, required_unless_present = "readout_only")]
        out_json: Option<PathBuf>,
        /// Optional starting slot offset (`base_intermediate_size` or relative `< base_intermediate_size`, e.g. `48` or `120` for co-mounting).
        #[arg(long)]
        slot_offset: Option<usize>,
        /// Optional total widened slot bank capacity (`8` standalone, or `128` when co-mounted alongside dynamic banks).
        #[arg(long)]
        widened_slots: Option<usize>,
        /// Optional strict `StaticPermanentScope` benign-guard leakage ceiling (e.g. `0.25`).
        #[arg(long)]
        max_bg_leakage: Option<f64>,
        /// Emit only the resolved anti-echo-projected `v_no` and `v_act` readout vectors as JSON to stdout.
        #[arg(long, default_value_t = false)]
        readout_only: bool,
        /// Compiler backend (`v2` Pointer-ISA `faer` + `StrictZeroCone`).
        #[arg(long, default_value = "v2")]
        compiler_engine: String,
    },
    /// Stream-patch a single-file or sharded HuggingFace `.safetensors` directory (`< 12 MiB` RAM, zero Python/PyTorch) with a compiled cartridge.
    #[command(
        name = "export-safetensors",
        after_help = "Example:\n  portia-cli export-safetensors --checkpoint-dir /models/Qwen3.5-0.8B-Base --cartridge-json /tmp/garak.json --out-dir /tmp/Qwen3.5-0.8B-Patched"
    )]
    ExportSafetensors {
        /// Input HuggingFace checkpoint directory (`model.safetensors` or `model.safetensors.index.json` + `config.json`).
        #[arg(long)]
        checkpoint_dir: PathBuf,
        /// Path to the compiled `WireCartridgeManifestJson` file.
        #[arg(long)]
        cartridge_json: PathBuf,
        /// Output directory (must be absent or empty) for the widened `.safetensors` checkpoint.
        #[arg(long)]
        out_dir: PathBuf,
        /// Optional dynamic bank key(s) (e.g. `ModePayload:Read`) to bake statically alongside `permanent_static`.
        #[arg(long = "include-bank")]
        include_banks: Vec<String>,
        /// Copy unmodified shards byte-for-byte instead of hardlinking.
        #[arg(long, default_value_t = false)]
        copy_unmodified_shards: bool,
    },
    /// Verify a serialized JSON `PenSpanRecord` schedule against the Single-Pen handoff invariant.
    #[command(
        name = "verify-schedule",
        after_help = "Example:\n  portia-cli verify-schedule --schedule-json schedule.json"
    )]
    VerifySchedule {
        /// Path to a JSON file containing `Vec<PenSpanRecord>`.
        #[arg(long)]
        schedule_json: PathBuf,
    },
    /// Run a Culture Mind Nursery evaluation & counterexample-harvest sweep (`SqlSteward` + `RustAstHylo`) across the 3-tuple lattice.
    #[command(
        name = "nursery",
        after_help = "Examples:\n  portia-cli nursery --episodes 8\n  portia-cli nursery --episodes 2 --synth-user luna --out-json /tmp/luna_sql_tree.json\n  portia-cli nursery --episodes 4 --app sql --synth-user luna --beam-width 2 --dynamic-schemas --out-json /tmp/luna_beam.json"
    )]
    Nursery {
        /// Number of episodes to sweep across the (`AppType × Language × Skin × Stress`) lattice.
        #[arg(long, default_value_t = 8)]
        episodes: usize,
        /// Application domain filter (`sql` / `sql-steward`, `all`, or `rust-ast` / `rust` / `hylo`).
        #[arg(long, default_value = "sql")]
        app: String,
        /// Simulated-user coalgebra mode (`oracle`, `deterministic` / `offline`, `luna` / `gpt-6-luna`, or `openai` / `oai` / `codex`).
        #[arg(long, default_value = "oracle")]
        synth_user: String,
        /// HuggingFace model ID or scale alias (`Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`) for live generator sweeps.
        #[arg(long, default_value = "Qwen/Qwen3.5-0.8B-Base")]
        model: String,
        #[arg(long)]
        model_family: Option<String>,
        #[arg(long)]
        allow_cartridge_model_mismatch: bool,
        /// Append complete task/turn/diagnostic receipts as JSON Lines.
        #[arg(long)]
        trace_jsonl: Option<PathBuf>,
        /// Optional path to a compiled cartridge JSON bundle for live generator sweeps.
        #[arg(long)]
        cartridge: Option<PathBuf>,
        /// Optional custom command line to spawn a JSON-over-stdio worker subprocess via `sh -c`.
        #[arg(long)]
        worker_cmd: Option<String>,
        /// Optional OpenAI / Codex model override (defaults to `gpt-6-luna` on `--synth-user luna`, or `$OPENAI_SYNTH_MODEL` / `"gpt-5-mini"` on `--synth-user openai`).
        #[arg(long)]
        openai_model: Option<String>,
        /// Maximum agent OODA turns per user prompt during live generator sweeps.
        #[arg(long, default_value_t = 8)]
        max_turns: usize,
        /// Number of stateful REPL follow-up waves per completed episode (`reset_model_session = false`).
        #[arg(long, default_value_t = 0)]
        followup_waves: usize,
        /// Number of pragmatic stances (`1..=3`) sampled per parent node in SQL beam-tree sweeps (defaults to `3` on `--synth-user luna`, `1` otherwise).
        #[arg(long)]
        branch_factor: Option<usize>,
        /// Optional surviving beam width per depth (`None` / `0` = unpruned full `3^d` tree; `<K>` caps surviving child branches per depth).
        #[arg(long)]
        beam_width: Option<usize>,
        /// Number of multi-turn user depths in SQL beam-tree sweeps (defaults to `6` on `--synth-user luna`, `1` otherwise).
        #[arg(long)]
        user_turns: Option<usize>,
        /// Hard ceiling on total explored branch nodes per SQL beam-tree episode.
        #[arg(long, default_value_t = 750)]
        max_nodes: usize,
        /// Number of concurrent `gpt-6-luna` sampling workers per tree depth via `std::thread::scope`.
        #[arg(long, default_value_t = 8)]
        luna_concurrency: usize,
        /// Synthesize fresh multi-table SQLite `CREATE TABLE` + `INSERT` schemas via `gpt-6-luna` validated in `rusqlite`.
        #[arg(long)]
        dynamic_schemas: bool,
        /// Run post-episode OpenAI Coroner root-cause & shrink-contrast diagnosis on failed episodes (`--synth-user openai`).
        #[arg(long)]
        coroner: bool,
        /// Optional path to write the serialized nursery / beam-tree sweep summary JSON to.
        #[arg(long)]
        out_json: Option<PathBuf>,
        /// Optional path to harvest genuine on-policy `Base` calibration activations + weight geometry (`CompileCartridgeInputSpec`) for `portia-cli compile`.
        #[arg(long)]
        oracle_capture_out: Option<PathBuf>,
        /// Optional transformer layer index for `--oracle-capture-out` (defaults to `num_layers - 2`, i.e. `22` on `Qwen/Qwen3.5-0.8B-Base`).
        #[arg(long)]
        target_layer: Option<usize>,
    },
}

/// Read a steward reply for `[MODE: ASK]` from either `--ask-reply` or an interactive `BufRead` stream,
/// rejecting unattended EOF (`0` bytes read) or blank input with an actionable error and validating
/// the reply against the provided [`AskPromptKind`] (`Permission` vs `Clarification`).
pub fn read_interactive_ask_reply_from_reader_with_kind(
    reader: &mut dyn BufRead,
    question: &str,
    canned_reply: Option<&str>,
    kind: AskPromptKind,
) -> Result<String> {
    if let Some(ans) = canned_reply {
        let _ = parse_ask_user_reply_for_kind(kind, ans)?;
        return Ok(ans.trim().to_string());
    }
    eprint!("[ASK]: {} -> ", question);
    io::stderr().flush()?;
    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        bail!(
            "Interactive [MODE: ASK] prompt received EOF on stdin for question {:?}; pass --ask-reply <ALLOW|DENY> for non-interactive runs",
            question
        );
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        bail!(
            "Interactive [MODE: ASK] prompt received an empty reply for question {:?}; provide ALLOW/DENY or pass --ask-reply",
            question
        );
    }
    let _ = parse_ask_user_reply_for_kind(kind, trimmed)
        .or_else(|_| parse_ask_user_reply_for_kind(AskPromptKind::Clarification, trimmed))?;
    Ok(trimmed.to_string())
}

/// Read a steward reply for `[MODE: ASK]` from either `--ask-reply` or an interactive `BufRead` stream,
/// rejecting unattended EOF (`0` bytes read) or blank input with an actionable error and validating
/// the reply against the question's [`AskPromptKind`] (`Permission` vs `Clarification`).
pub fn read_interactive_ask_reply_from_reader(
    reader: &mut dyn BufRead,
    question: &str,
    canned_reply: Option<&str>,
) -> Result<String> {
    let kind = classify_ask_prompt_kind(question);
    let reply =
        read_interactive_ask_reply_from_reader_with_kind(reader, question, canned_reply, kind)?;
    let _ = parse_ask_user_reply_for_kind(kind, &reply)?;
    Ok(reply)
}

/// Shared interactive command grammar for `portia-cli repl` and `portia-cli tui`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    /// Blank or whitespace-only line (`continue`).
    Empty,
    /// Exit the interactive session (`quit`, `exit`, `\q`, `/quit`, `/exit`).
    Quit,
    /// Show interactive command help (`help`, `/help`, `?`).
    Help,
    /// Reset the model's conversation context while preserving the SQLite database (`reset`, `/reset`, `clear`, `/clear`).
    Reset,
    /// Print or open the live SQLite schema inspector (`schema`, `/schema`).
    Schema,
    /// Toggle inline BPE token IDs in the TUI context pane (`tokens`, `/tokens`).
    Tokens,
    /// Toggle the right-hand `RawTokenTrace` inspector (`trace`, `/trace`, `debug`, `/debug`, `sidebar`, `/sidebar`).
    DebugSidebar,
    /// Render a TUI snapshot (`tui`, `/tui`).
    TuiSnapshot,
    /// Show active model scale, cartridge, governance, and context telemetry (`status`, `/status`).
    Status,
    /// Toggle `psql`-style expanded vertical record cards (`/x`, `\x`).
    ToggleExpandedDisplay,
    /// Set maximum displayed rows per SQL result table (`/rows <N|all>`).
    SetRowLimit(String),
    /// Export the most recent SQL observation rows to CSV or JSON (`/export <path.csv|path.json>`).
    ExportResults(PathBuf),
    /// Roll back the most recent database write step in the current session (`undo`, `/undo`).
    Undo,
    /// Inspect or hot-switch the active `Qwen3.5` model scale (`/model`, `/model <0.8b-base|2b-base|4b-base|9b-base> [--cartridge <path|none>]`).
    SwitchModel {
        target: Option<String>,
        cartridge: Option<PathBuf>,
    },
    /// Execute a natural-language task prompt through the live Qwen worker.
    TaskPrompt(String),
}

fn parse_switch_model_args(rest: &str) -> ReplCommand {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    if toks.is_empty() {
        return ReplCommand::SwitchModel {
            target: None,
            cartridge: None,
        };
    }
    let target = Some(toks[0].to_string());
    let mut cartridge = None;
    let mut idx = 1usize;
    while idx < toks.len() {
        let tok = toks[idx];
        if tok == "--cartridge" {
            if let Some(p) = toks.get(idx + 1) {
                cartridge = Some(PathBuf::from(p));
                idx += 2;
                continue;
            }
        } else if let Some(p) = tok.strip_prefix("--cartridge=") {
            if !p.is_empty() {
                cartridge = Some(PathBuf::from(p));
            }
        } else if cartridge.is_none() {
            cartridge = Some(PathBuf::from(tok));
        }
        idx += 1;
    }
    ReplCommand::SwitchModel { target, cartridge }
}

/// Parse a raw interactive input line into a structured [`ReplCommand`].
pub fn parse_repl_command(line: &str) -> ReplCommand {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ReplCommand::Empty;
    }
    if trimmed.eq_ignore_ascii_case("/model") || trimmed.eq_ignore_ascii_case("/models") {
        return ReplCommand::SwitchModel {
            target: None,
            cartridge: None,
        };
    }
    if let Some(rest) = trimmed
        .strip_prefix("/model ")
        .or_else(|| trimmed.strip_prefix("/model\t"))
    {
        return parse_switch_model_args(rest);
    }
    if trimmed.eq_ignore_ascii_case("/rows") {
        return ReplCommand::SetRowLimit(String::new());
    }
    if let Some(rest) = trimmed
        .strip_prefix("/rows ")
        .or_else(|| trimmed.strip_prefix("/rows\t"))
    {
        return ReplCommand::SetRowLimit(rest.trim().to_string());
    }
    if trimmed.eq_ignore_ascii_case("/export") {
        return ReplCommand::ExportResults(PathBuf::new());
    }
    if let Some(rest) = trimmed
        .strip_prefix("/export ")
        .or_else(|| trimmed.strip_prefix("/export\t"))
    {
        return ReplCommand::ExportResults(PathBuf::from(rest.trim()));
    }
    match trimmed {
        "quit" | "exit" | "\\q" | "/quit" | "/exit" => ReplCommand::Quit,
        "help" | "/help" | "?" => ReplCommand::Help,
        "reset" | "/reset" | "clear" | "/clear" => ReplCommand::Reset,
        "schema" | "/schema" => ReplCommand::Schema,
        "tokens" | "/tokens" => ReplCommand::Tokens,
        "trace" | "/trace" | "debug" | "/debug" | "sidebar" | "/sidebar" => {
            ReplCommand::DebugSidebar
        }
        "status" | "/status" => ReplCommand::Status,
        "/x" | "\\x" => ReplCommand::ToggleExpandedDisplay,
        "undo" | "/undo" => ReplCommand::Undo,
        "tui" | "/tui" => ReplCommand::TuiSnapshot,
        _ => ReplCommand::TaskPrompt(trimmed.to_string()),
    }
}

/// Unified line-input frontend that owns non-TUI terminal or piped stdin across both top-level
/// `portia> ` command entry and mid-turn `[ASK]: ... -> ` prompts.
pub enum LineInputFrontend<'a> {
    /// Interactive line editor with arrow-key navigation, history, and Ctrl-C/Ctrl-D handling.
    Rustyline(Box<rustyline::DefaultEditor>),
    /// Non-interactive or piped buffered line reader.
    Buffered(Box<dyn BufRead + 'a>),
}

impl<'a> LineInputFrontend<'a> {
    /// Construct a REPL line-input frontend (`Rustyline` when stdin is a TTY, `Buffered` otherwise).
    pub fn new_repl_stdio() -> Self {
        if io::stdin().is_terminal() {
            if let Ok(editor) = rustyline::DefaultEditor::new() {
                return Self::Rustyline(Box::new(editor));
            }
        }
        Self::Buffered(Box::new(io::BufReader::new(io::stdin())))
    }

    /// Read the next top-level interactive command line (`Ok(None)` on clean EOF / Ctrl-D).
    pub fn read_command_line(
        &mut self,
        prompt: &str,
        _tui_state: &mut TwoPaneTuiState,
    ) -> Result<Option<String>> {
        match self {
            Self::Rustyline(rl) => match rl.readline(prompt) {
                Ok(line) => {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        let _ = rl.add_history_entry(trimmed);
                    }
                    Ok(Some(line))
                }
                Err(
                    rustyline::error::ReadlineError::Interrupted
                    | rustyline::error::ReadlineError::Eof,
                ) => Ok(None),
                Err(err) => Err(err.into()),
            },
            Self::Buffered(reader) => {
                let mut line = String::new();
                if reader.read_line(&mut line)? == 0 {
                    Ok(None)
                } else {
                    Ok(Some(line))
                }
            }
        }
    }

    /// Read an interactive steward reply for `[MODE: ASK]` from the active line-input frontend.
    pub fn read_ask_reply(
        &mut self,
        question: &str,
        canned_reply: Option<&str>,
        ask_mode: AskInteractionMode,
        _tui_state: &TwoPaneTuiState,
    ) -> Result<String> {
        let kind = match ask_mode {
            AskInteractionMode::StewardPermissionChoice { .. } => AskPromptKind::Permission,
            AskInteractionMode::MultiSelect { .. } | AskInteractionMode::ConversationalClarify => {
                AskPromptKind::Clarification
            }
        };
        if let Some(ans) = canned_reply {
            let _ = parse_ask_user_reply_for_kind(kind, ans)
                .or_else(|_| parse_ask_user_reply_for_kind(AskPromptKind::Clarification, ans))?;
            return Ok(ans.trim().to_string());
        }
        match self {
            Self::Rustyline(rl) => {
                let prompt = format!("[ASK]: {} -> ", question);
                match rl.readline(&prompt) {
                    Ok(line) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            bail!(
                                "Interactive [MODE: ASK] prompt received an empty reply for question {:?}; provide ALLOW/DENY or pass --ask-reply",
                                question
                            );
                        }
                        let _ = parse_ask_user_reply_for_kind(kind, trimmed).or_else(|_| {
                            parse_ask_user_reply_for_kind(AskPromptKind::Clarification, trimmed)
                        })?;
                        let _ = rl.add_history_entry(trimmed);
                        Ok(trimmed.to_string())
                    }
                    Err(
                        rustyline::error::ReadlineError::Interrupted
                        | rustyline::error::ReadlineError::Eof,
                    ) => {
                        bail!(
                            "Interactive [MODE: ASK] prompt received EOF on stdin for question {:?}; pass --ask-reply <ALLOW|DENY> for non-interactive runs",
                            question
                        )
                    }
                    Err(err) => Err(err.into()),
                }
            }
            Self::Buffered(reader) => read_interactive_ask_reply_from_reader_with_kind(
                reader.as_mut(),
                question,
                None,
                kind,
            ),
        }
    }
}

/// Format completed [`SessionEvent`] turns as clean, human-readable lines for non-TUI REPL output
/// so interactive `repl` sessions never dump raw `serde_json` struct arrays.
#[must_use]
pub fn format_human_readable_session_events(events: &[SessionEvent]) -> String {
    let mut lines = Vec::new();

    for ev in events {
        let route_name = ev
            .mode
            .route_tag()
            .open_tag()
            .trim_matches(|c| c == '<' || c == '>');
        let ask_mode_opt = if ev.mode == portia_core::Mode::Ask {
            let mode = if let Some(ref recorded) = ev.ask_interaction_mode {
                recorded.clone()
            } else {
                crate::driver::resolve_ask_interaction_mode_from_observation_state(
                    false,
                    &[],
                    &[],
                    false,
                    &ev.payload,
                )
            };
            Some(mode)
        } else {
            None
        };

        match &ask_mode_opt {
            Some(AskInteractionMode::ConversationalClarify) => {
                lines.push(format!(
                    "├─ [OODA: ORIENT] Mode: ASK · Chitchat (route: <{}>)",
                    route_name
                ));
            }
            Some(
                AskInteractionMode::StewardPermissionChoice { .. }
                | AskInteractionMode::MultiSelect { .. },
            ) => {
                lines.push(format!(
                    "├─ [OODA: ORIENT] Mode: ASK · Structured-GUI (route: <{}>)",
                    route_name
                ));
            }
            None => {
                lines.push(format!(
                    "├─ [OODA: ORIENT] Mode: {} (route: <{}>)",
                    ev.mode.as_str(),
                    route_name
                ));
            }
        }

        let clean_plan = ev.plan.trim();
        let plan_display = if clean_plan.is_empty() {
            "..."
        } else {
            clean_plan
        };
        lines.push(format!("├─ [OODA: DECIDE] Plan: {plan_display}"));
        match ev.mode {
            portia_core::Mode::Probe | portia_core::Mode::Read | portia_core::Mode::Write => {
                lines.push(format!("├─ [OODA: ACT -> sql] SQL {}:", ev.mode.as_str()));
                for pl_line in ev.payload.lines() {
                    lines.push(format!("│    sql> {pl_line}"));
                }
            }
            portia_core::Mode::Ask => {
                match ask_mode_opt.unwrap_or(AskInteractionMode::ConversationalClarify) {
                    AskInteractionMode::ConversationalClarify => {
                        lines.push("╰─ [OODA: CHITCHAT -> to_user] Assistant:".to_string());
                        let formatted_ask =
                            crate::driver::format_ask_payload_for_display(&ev.payload);
                        for pl_line in formatted_ask.lines() {
                            lines.push(format!("     {pl_line}"));
                        }
                    }
                    AskInteractionMode::StewardPermissionChoice { question, choices } => {
                        let has_prose_lead_in = !question.trim().is_empty()
                            && !question.trim().ends_with('?')
                            && crate::driver::extract_ask_interaction_mode(&question).is_none();
                        if has_prose_lead_in {
                            lines.push("├─ [OODA: CHAT -> STRUCTURED-GUI] Assistant:".to_string());
                            for q_line in question.lines() {
                                lines.push(format!("│    {q_line}"));
                            }
                            lines.push("│    Steward Authorization (Select 1):".to_string());
                        } else {
                            lines.push(
                                "├─ [OODA: STRUCTURED-GUI -> to_user] Steward Authorization (Select 1):"
                                    .to_string(),
                            );
                            for q_line in question.lines() {
                                lines.push(format!("│    {q_line}"));
                            }
                        }
                        for opt in &choices {
                            lines.push(format!("│      • {opt}"));
                        }
                    }
                    AskInteractionMode::MultiSelect { question, choices } => {
                        let has_prose_lead_in = !question.trim().is_empty()
                            && !question.trim().ends_with('?')
                            && crate::driver::extract_ask_interaction_mode(&question).is_none();
                        if has_prose_lead_in {
                            lines.push("├─ [OODA: CHAT -> STRUCTURED-GUI] Assistant:".to_string());
                            for q_line in question.lines() {
                                lines.push(format!("│    {q_line}"));
                            }
                            lines.push(
                                "│    Select Candidate Row(s) (<Space> toggle, <Enter> submit):"
                                    .to_string(),
                            );
                        } else {
                            lines.push(
                                "├─ [OODA: STRUCTURED-GUI -> to_user] Candidate Disambiguation (Multi-Select):"
                                    .to_string(),
                            );
                            for q_line in question.lines() {
                                lines.push(format!("│    {q_line}"));
                            }
                        }
                        for opt in &choices {
                            lines.push(format!("│      • {opt}"));
                        }
                    }
                }
            }
            portia_core::Mode::Result => {
                lines.push("╰─ [OODA: RESULT -> to_user] Assistant:".to_string());
                for pl_line in ev.payload.lines() {
                    lines.push(format!("     {pl_line}"));
                }
            }
            portia_core::Mode::Abstain => {
                lines.push("╰─ [OODA: ABSTAIN -> to_user] Refusal:".to_string());
                for pl_line in ev.payload.lines() {
                    lines.push(format!("     {pl_line}"));
                }
            }
        }
        let obs = &ev.observation;
        if !obs.ok {
            lines.push(format!(
                "├─ [OODA: OBSERVE <- SQLite Error]: {}",
                obs.error.as_deref().unwrap_or("unknown error")
            ));
        } else if obs.mode == portia_core::Mode::Ask {
            if obs.is_chat_continuation {
                lines.push(
                    "├─ [OODA: COROUTINE CONTINUE] Proceeding to next OODA step...".to_string(),
                );
            } else if let Some(msg) = &obs.message {
                lines.push(format!("├─ [OODA: STEWARD REPLY] You: {msg}"));
            }
        } else if !obs.rows.is_empty() {
            lines.push(format!(
                "├─ [OODA: OBSERVE <- SQLite] {} row(s):",
                obs.rows.len()
            ));
            lines.push(format!("│    {}", obs.columns.join(" | ")));
            for row in obs.rows.iter().take(6) {
                let vals: Vec<String> = row.iter().map(ToString::to_string).collect();
                lines.push(format!("│    {}", vals.join(" | ")));
            }
            if obs.rows.len() > 6 {
                lines.push(format!("│    ... (+{} more rows)", obs.rows.len() - 6));
            }
        } else if obs.mode == portia_core::Mode::Write {
            lines.push(format!(
                "├─ [OODA: OBSERVE <- SQLite] Applied ({} row(s) updated)",
                obs.rows_affected
            ));
        } else if matches!(obs.mode, portia_core::Mode::Probe | portia_core::Mode::Read) {
            lines.push("├─ [OODA: OBSERVE <- SQLite] 0 rows".to_string());
        }
    }
    lines.join("\n")
}

/// Output rendering mode for the unified REPL/TUI command controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRenderMode {
    /// Emit JSON receipts and observations to stdout (`portia-cli repl` default).
    JsonStdout,
    /// Emit ANSI two-pane TUI snapshots to stdout after each command (`portia-cli repl --tui` or piped `portia-cli tui`).
    SnapshotTui { width: usize, height: usize },
    /// Mutate [`TwoPaneTuiState`] directly for interactive TUI state-controller execution (`portia-cli tui` / unit tests).
    LiveTui,
}

fn format_session_status_detail(state: &TwoPaneTuiState) -> String {
    let scale = state.model_scale_spec();
    let (gov_label, _) = crate::tui::theme_chrome::steward_status_badge(state);
    format!(
        "Model: {} ({} · {}) | Orient: {} · Payload: {} | Cartridge: {} | DB: {} ({} tables · {} · undo={}) | Context: {} tok, {} swaps (scales: 0.8b-base, 2b-base, 4b-base, 9b-base)",
        scale.canonical_family,
        scale.scale_badge,
        scale.format_architecture_summary(),
        scale.format_orient_slots(),
        scale.format_payload_slots(),
        state.cartridge_label,
        crate::tui::theme_chrome::db_basename(&state.db_path),
        state.tables.len(),
        gov_label,
        state.undo_depth,
        state.total_context_tokens,
        state.cartridge_switch_count
    )
}

/// Execute a single parsed [`ReplCommand`] with a custom worker hot-spawner for `/model <scale>`.
#[allow(clippy::too_many_arguments)]
pub fn execute_session_command_with_spawner<F>(
    cmd: ReplCommand,
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    tui_state: &mut TwoPaneTuiState,
    needs_session_reset: &mut bool,
    max_turns: usize,
    canned_ask_reply: Option<&str>,
    render_mode: SessionRenderMode,
    input_frontend: &mut LineInputFrontend<'_>,
    mut spawn_model_worker: F,
) -> Result<bool>
where
    F: FnMut(&str, Option<&std::path::Path>, &str, bool) -> Result<StdioQwenWorkerClient>,
{
    let emit_snapshot_if_needed = |state: &TwoPaneTuiState| {
        if let SessionRenderMode::SnapshotTui { width, height } = render_mode {
            println!("{}", render_two_pane_tui_frame(state, width, height, true));
        }
    };

    match cmd {
        ReplCommand::Empty => Ok(true),
        ReplCommand::Quit => {
            driver.flush_open_undo_savepoints()?;
            Ok(false)
        }
        ReplCommand::Help => {
            let help_msg = "Commands: /model <0.8b-base|2b-base|4b-base|9b-base>, /x, /rows <N|all>, /export <path.csv|path.json>, /undo, /status, /schema (Tab/Ctrl+B), /trace (/debug, Ctrl+T), /clear (/reset), /tokens, /tui, /quit (Ctrl+D)";
            if render_mode == SessionRenderMode::LiveTui {
                tui_state.interactions.push(InteractionItem::HarnessNote {
                    title: "REPL/TUI Command Help".to_string(),
                    detail: help_msg.to_string(),
                });
            } else {
                eprintln!("[portia-cli] {help_msg}");
                emit_snapshot_if_needed(tui_state);
            }
            Ok(true)
        }
        ReplCommand::Status => {
            let detail = format_session_status_detail(tui_state);
            if render_mode == SessionRenderMode::LiveTui {
                tui_state.interactions.push(InteractionItem::HarnessNote {
                    title: format!(
                        "Session Status · {}",
                        tui_state.model_scale_spec().scale_badge
                    ),
                    detail,
                });
            } else {
                eprintln!("[portia-cli] {detail}");
                emit_snapshot_if_needed(tui_state);
            }
            Ok(true)
        }
        ReplCommand::ToggleExpandedDisplay => {
            tui_state.expanded_display = !tui_state.expanded_display;
            let state_lbl = if tui_state.expanded_display {
                "ON (psql vertical ─[ RECORD k ]─ cards)"
            } else {
                "OFF (compact aligned Unicode box grid)"
            };
            tui_state.interactions.push(InteractionItem::HarnessNote {
                title: format!("Expanded Display (/x): {state_lbl}"),
                detail: "Applies immediately to all SQL result observations.".to_string(),
            });
            if render_mode != SessionRenderMode::LiveTui {
                eprintln!("[portia-cli] expanded_display={state_lbl}");
                emit_snapshot_if_needed(tui_state);
            }
            Ok(true)
        }
        ReplCommand::SetRowLimit(raw_arg) => {
            match parse_row_display_limit_arg(&raw_arg) {
                Ok(limit) => {
                    tui_state.row_display_limit = limit;
                    let desc = match limit {
                        Some(n) => format!("{n} rows"),
                        None => "all rows".to_string(),
                    };
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: format!("Row Display Limit -> {desc}"),
                        detail: "Updated result table row cap.".to_string(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("[portia-cli] row_display_limit={desc}");
                    }
                }
                Err(err) => {
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: "Invalid /rows Argument".to_string(),
                        detail: err.to_string(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("{}", format_surfaced_error_ansi(&err));
                    }
                }
            }
            emit_snapshot_if_needed(tui_state);
            Ok(true)
        }
        ReplCommand::ExportResults(path) => {
            match tui_state.latest_tabular_observation() {
                Some(obs) => match export_observation_to_path(obs, &path) {
                    Ok((fmt, n_rows, n_cols)) => {
                        let detail = format!(
                            "Exported {} ({n_rows} rows × {n_cols} cols) -> {}",
                            fmt.label(),
                            path.display()
                        );
                        tui_state.interactions.push(InteractionItem::HarnessNote {
                            title: format!("Exported {}", fmt.label()),
                            detail: detail.clone(),
                        });
                        if render_mode != SessionRenderMode::LiveTui {
                            eprintln!("[portia-cli] {detail}");
                        }
                    }
                    Err(err) => {
                        tui_state.interactions.push(InteractionItem::HarnessNote {
                            title: "Export Failed".to_string(),
                            detail: err.to_string(),
                        });
                        if render_mode != SessionRenderMode::LiveTui {
                            eprintln!("{}", format_surfaced_error_ansi(&err));
                        }
                    }
                },
                None => {
                    let msg = "No SQL result rows available in the session yet.";
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: "Nothing to Export".to_string(),
                        detail: msg.to_string(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("[portia-cli] {msg}");
                    }
                }
            }
            emit_snapshot_if_needed(tui_state);
            Ok(true)
        }
        ReplCommand::Undo => {
            match driver.undo_last_step()? {
                Some(receipt) => {
                    worker.reset_session()?;
                    *needs_session_reset = true;
                    tui_state.reset_kv_window_tracking();
                    tui_state.tables = driver.inspect_schema()?;
                    tui_state.undo_depth = receipt.remaining_undo_depth;
                    let detail = format!(
                        "Rolled back {:?} ({}) -> db v{} ({} undo step(s) remaining)",
                        receipt.prompt_summary,
                        receipt.savepoint_name,
                        receipt.restored_db_version,
                        receipt.remaining_undo_depth
                    );
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: format!("Undo Applied ({})", receipt.savepoint_name),
                        detail: detail.clone(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("[portia-cli] status=undo_applied {detail}");
                    }
                }
                None => {
                    let msg = "No committed database writes in the current session undo stack.";
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: "Nothing to Undo".to_string(),
                        detail: msg.to_string(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("[portia-cli] {msg}");
                    }
                }
            }
            emit_snapshot_if_needed(tui_state);
            Ok(true)
        }
        ReplCommand::SwitchModel { target, cartridge } => {
            let Some(raw_target) = target
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
            else {
                let detail = format!(
                    "{} — Usage: /model <0.8b-base|2b-base|4b-base|9b-base> [--cartridge <path|none>]",
                    format_session_status_detail(tui_state)
                );
                if render_mode == SessionRenderMode::LiveTui {
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: format!(
                            "Active Model · {}",
                            tui_state.model_scale_spec().scale_badge
                        ),
                        detail,
                    });
                } else {
                    eprintln!("[portia-cli] {detail}");
                    emit_snapshot_if_needed(tui_state);
                }
                return Ok(true);
            };

            let normalized_target = normalize_model_identifier(&raw_target);
            let new_cart_label =
                resolve_default_cartridge_label_for_model(&normalized_target, cartridge.as_deref());
            match spawn_model_worker(
                &normalized_target,
                cartridge.as_deref(),
                &new_cart_label,
                tui_state.read_only,
            ) {
                Ok(new_worker) => {
                    *worker = new_worker;
                    let spec =
                        tui_state.apply_hot_swapped_model(&normalized_target, &new_cart_label);
                    *needs_session_reset = true;
                    let detail = format!(
                        "Active model: {} ({} · OrientTrunk {} · ModePayload {}) · Cartridge: {} (KV context reset)",
                        spec.canonical_family,
                        spec.format_architecture_summary(),
                        spec.format_orient_slots(),
                        spec.format_payload_slots(),
                        tui_state.cartridge_label
                    );
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: format!("Model Switched -> {}", spec.scale_badge),
                        detail: detail.clone(),
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("[portia-cli] status=model_switched {detail}");
                        emit_snapshot_if_needed(tui_state);
                    }
                }
                Err(err) => {
                    let cur_badge = tui_state.model_scale_spec().scale_badge;
                    let detail = format!(
                        "{err} (staying on {cur_badge} · {})",
                        tui_state.cartridge_label
                    );
                    tui_state.interactions.push(InteractionItem::HarnessNote {
                        title: "Model Switch Error".to_string(),
                        detail,
                    });
                    if render_mode != SessionRenderMode::LiveTui {
                        eprintln!("{}", format_surfaced_error_ansi(&err));
                        emit_snapshot_if_needed(tui_state);
                    }
                }
            }
            Ok(true)
        }
        ReplCommand::DebugSidebar => {
            tui_state.toggle_raw_token_trace();
            emit_snapshot_if_needed(tui_state);
            Ok(true)
        }
        ReplCommand::Reset => {
            worker.reset_session()?;
            tui_state.reset_conversation_context();
            *needs_session_reset = true;
            if render_mode != SessionRenderMode::LiveTui {
                eprintln!("[portia-cli] status=session_reset (model conversation context cleared; database state preserved)");
            }
            emit_snapshot_if_needed(tui_state);
            Ok(true)
        }
        ReplCommand::Tokens => {
            tui_state.show_inline_token_ids = !tui_state.show_inline_token_ids;
            match render_mode {
                SessionRenderMode::JsonStdout => {
                    let (w, h) = detect_terminal_size();
                    println!("{}", render_two_pane_tui_frame(tui_state, w, h, true));
                }
                SessionRenderMode::SnapshotTui { width, height } => {
                    println!(
                        "{}",
                        render_two_pane_tui_frame(tui_state, width, height, true)
                    );
                }
                SessionRenderMode::LiveTui => {}
            }
            Ok(true)
        }
        ReplCommand::TuiSnapshot => {
            match render_mode {
                SessionRenderMode::JsonStdout => {
                    let (w, h) = detect_terminal_size();
                    println!("{}", render_two_pane_tui_frame(tui_state, w, h, true));
                }
                SessionRenderMode::SnapshotTui { width, height } => {
                    println!(
                        "{}",
                        render_two_pane_tui_frame(tui_state, width, height, true)
                    );
                }
                SessionRenderMode::LiveTui => {}
            }
            Ok(true)
        }
        ReplCommand::Schema => {
            let schema = driver.inspect_schema()?;
            tui_state.tables = schema.clone();
            if render_mode == SessionRenderMode::LiveTui {
                tui_state.set_sidebar_mode(crate::tui::SidebarMode::SchemaInspector);
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
                    title: format!("Schema Drawer Refreshed ({} tables)", schema.len()),
                    detail: summary,
                });
            } else {
                println!("{}", serde_json::to_string_pretty(&schema)?);
            }
            Ok(true)
        }
        ReplCommand::TaskPrompt(task_prompt) => {
            let db_ver_before = driver.db_version();
            let undo_cp = driver.open_turn_undo_savepoint(task_prompt.clone())?;
            let mut events = Vec::new();
            let latest_tui_state = std::cell::RefCell::new(tui_state.clone());
            let frontend_cell = std::cell::RefCell::new(&mut *input_frontend);
            let res = run_live_worker_turn_sequence_streaming_into(
                driver,
                worker,
                &task_prompt,
                max_turns,
                *needs_session_reset,
                Some(tui_state),
                &mut events,
                |q| {
                    let snap = latest_tui_state.borrow().clone();
                    let ask_mode = snap.ask_interaction_mode(q);
                    frontend_cell
                        .borrow_mut()
                        .read_ask_reply(q, canned_ask_reply, ask_mode, &snap)
                },
                |state| {
                    *latest_tui_state.borrow_mut() = state.clone();
                },
            );
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
            *needs_session_reset = false;
            if let Err(ref err) = res {
                *needs_session_reset = true;
                let (title, detail) = format_surfaced_error_note(err);
                tui_state
                    .interactions
                    .push(InteractionItem::HarnessNote { title, detail });
            }
            if let Ok(log_path) = std::env::var("PORTIA_SESSION_LOG") {
                let trimmed = log_path.trim();
                if !trimmed.is_empty() {
                    let mut snap_copy = tui_state.clone();
                    snap_copy.show_debug_sidebar = true;
                    let _ = std::fs::write(
                        trimmed,
                        render_two_pane_tui_frame(&snap_copy, 180, 60, false),
                    );
                }
            }
            match render_mode {
                SessionRenderMode::JsonStdout => {
                    if !events.is_empty() {
                        println!("{}", format_human_readable_session_events(&events));
                    }
                    if let Err(ref err) = res {
                        eprintln!("{}", format_surfaced_error_ansi(err));
                    }
                }
                SessionRenderMode::SnapshotTui { .. } => {
                    emit_snapshot_if_needed(tui_state);
                    if let Err(ref err) = res {
                        eprintln!("{}", format_surfaced_error_ansi(err));
                    }
                }
                SessionRenderMode::LiveTui => {}
            }
            Ok(true)
        }
    }
}

/// Execute a single parsed [`ReplCommand`] against the shared SQLite driver, live Qwen worker,
/// [`TwoPaneTuiState`], and [`LineInputFrontend`]. Returns `Ok(false)` when the user requests exit (`Quit`).
#[allow(clippy::too_many_arguments)]
pub fn execute_session_command(
    cmd: ReplCommand,
    driver: &mut LiveSqliteDailyDriver,
    worker: &mut StdioQwenWorkerClient,
    tui_state: &mut TwoPaneTuiState,
    needs_session_reset: &mut bool,
    max_turns: usize,
    canned_ask_reply: Option<&str>,
    render_mode: SessionRenderMode,
    input_frontend: &mut LineInputFrontend<'_>,
) -> Result<bool> {
    let allow_mismatch = worker.provenance.model_mismatch_override;
    execute_session_command_with_spawner(
        cmd,
        driver,
        worker,
        tui_state,
        needs_session_reset,
        max_turns,
        canned_ask_reply,
        render_mode,
        input_frontend,
        |target_model, explicit_cart, cart_label, read_only| {
            let launch_opts = WorkerLaunchOptions {
                model_family: None,
                allow_cartridge_model_mismatch: allow_mismatch,
                trace_jsonl: None,
            };
            spawn_configured_worker(
                target_model,
                explicit_cart,
                None,
                cart_label,
                read_only,
                &launch_opts,
            )
        },
    )
}

pub(crate) fn spawn_configured_worker(
    model: &str,
    cartridge: Option<&std::path::Path>,
    worker_cmd: Option<&str>,
    cartridge_label: &str,
    read_only: bool,
    options: &WorkerLaunchOptions,
) -> Result<StdioQwenWorkerClient> {
    eprintln!(
        "[portia-cli] status=starting_worker model={} worker=stdio cartridge={} read_only={}",
        model.trim(),
        cartridge_label,
        read_only
    );
    let startup_provenance = RuntimeProvenance {
        requested_model: model.to_string(),
        model_mismatch_override: options.allow_cartridge_model_mismatch,
        ..Default::default()
    };
    let startup_trace = |event: &str, error: Option<String>| -> Result<()> {
        if let Some(path) = &options.trace_jsonl {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            serde_json::to_writer(
                &mut file,
                &serde_json::json!({
                    "event": event, "provenance": startup_provenance, "error": error,
                }),
            )?;
            file.write_all(b"\n")?;
        }
        Ok(())
    };
    startup_trace("worker_starting", None)?;
    let result = (|| {
        if let Some(custom_cmd) = worker_cmd {
            let mut worker = StdioQwenWorkerClient::spawn_shell(custom_cmd)?;
            worker.provenance = startup_provenance.clone();
            if let Some(path) = &options.trace_jsonl {
                worker.set_trace_jsonl(path)?;
            }
            Ok(worker)
        } else {
            let mut worker =
                StdioQwenWorkerClient::spawn_default_with_options(model, cartridge, options)?;
            worker.reset_session()?;
            Ok(worker)
        }
    })();
    if let Err(error) = &result {
        startup_trace("worker_initialization_failed", Some(format!("{error:#}")))?;
    }
    result
}

struct InitializedSqliteSession {
    driver: LiveSqliteDailyDriver,
    tui_state: TwoPaneTuiState,
    worker: StdioQwenWorkerClient,
    read_only: bool,
    max_turns: usize,
    ask_reply: Option<AskDecision>,
    worker_cmd: Option<String>,
    launch_options: WorkerLaunchOptions,
}

fn init_sqlite_worker_session(
    db: &std::path::Path,
    runtime: WorkerRuntimeArgs,
) -> Result<InitializedSqliteSession> {
    let WorkerRuntimeArgs {
        model,
        model_family,
        allow_cartridge_model_mismatch,
        cartridge,
        worker_cmd,
        read_only,
        max_turns,
        max_write_rows,
        ask_reply,
        steward_columns,
        disambiguate_candidates,
        trace_jsonl,
    } = runtime;
    let normalized_model = normalize_model_identifier(&model);
    let launch_options = WorkerLaunchOptions {
        model_family,
        allow_cartridge_model_mismatch,
        trace_jsonl,
    };
    let mut driver = LiveSqliteDailyDriver::open_with_mode(db, max_write_rows, read_only)?;
    if !steward_columns.is_empty() {
        driver.set_steward_policy_columns(&steward_columns);
    }
    if disambiguate_candidates {
        driver.set_candidate_disambiguation(true);
    }
    let cart_label =
        resolve_default_cartridge_label_for_model(&normalized_model, cartridge.as_deref());
    let tui_state = TwoPaneTuiState::new(
        db.display().to_string(),
        &normalized_model,
        cart_label,
        driver.inspect_schema()?,
    )
    .with_read_only(read_only)
    .with_steward_policy_columns(steward_columns)
    .with_candidate_disambiguation(disambiguate_candidates)
    .with_session_preview_rows(driver.session_last_read_rows().to_vec());
    let worker = spawn_configured_worker(
        &normalized_model,
        cartridge.as_deref(),
        worker_cmd.as_deref(),
        &tui_state.cartridge_label,
        read_only,
        &launch_options,
    )?;
    Ok(InitializedSqliteSession {
        driver,
        tui_state,
        worker,
        read_only,
        max_turns,
        ask_reply,
        worker_cmd,
        launch_options,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { db } => {
            let driver = LiveSqliteDailyDriver::open_read_only(&db)?;
            let schema = driver.inspect_schema()?;
            println!("{}", serde_json::to_string_pretty(&schema)?);
        }
        Commands::Run {
            db,
            prompt,
            runtime,
            summary_only,
            tui,
        } => {
            let InitializedSqliteSession {
                mut driver,
                mut tui_state,
                mut worker,
                max_turns,
                ask_reply,
                ..
            } = init_sqlite_worker_session(&db, runtime)?;
            let ask_reply_str = ask_reply.as_ref().map(AskDecision::as_str);
            let ask_cb = |q: &str| -> Result<String> {
                let stdin = io::stdin();
                let mut lock = stdin.lock();
                read_interactive_ask_reply_from_reader(&mut lock, q, ask_reply_str)
            };

            let db_ver_before = driver.db_version();
            let mut transcript_events = Vec::new();
            let session_res = run_live_worker_session_into(
                &mut driver,
                &mut worker,
                &prompt,
                max_turns,
                Some(&mut tui_state),
                &mut transcript_events,
                ask_cb,
            );
            let mut receipt = summarize_session_outcome(
                &transcript_events,
                session_res.is_err(),
                db_ver_before,
                driver.db_version(),
            );
            worker::attach_session_error(&mut receipt, session_res.as_ref().err());
            receipt.provenance = worker.provenance.clone();
            // Always emit any completed turn events (or summary receipt) to stdout even if a later turn errored:
            if tui {
                let (w, h) = detect_terminal_size();
                println!("{}", render_two_pane_tui_frame(&tui_state, w, h, true));
            } else if summary_only {
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            } else if !transcript_events.is_empty() || session_res.is_ok() {
                println!("{}", serde_json::to_string_pretty(&transcript_events)?);
            }
            session_res?;

            if !matches!(
                receipt.status,
                SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
            ) {
                bail!(
                    "Session exhausted max_turns ({}) without reaching terminal RESULT or ABSTAIN (status={})",
                    max_turns,
                    receipt.status
                );
            }
        }
        Commands::Batch {
            tasks_jsonl,
            out_jsonl,
            runtime,
        } => {
            batch_cmd::run_batch_command(&tasks_jsonl, out_jsonl.as_deref(), runtime)?;
        }
        Commands::Repl { db, runtime, tui } => {
            let InitializedSqliteSession {
                mut driver,
                mut tui_state,
                mut worker,
                read_only,
                max_turns,
                ask_reply,
                worker_cmd,
                launch_options,
            } = init_sqlite_worker_session(&db, runtime)?;
            let ask_reply_str = ask_reply.as_ref().map(AskDecision::as_str);
            let reload_opts = WorkerLaunchOptions {
                model_family: None,
                allow_cartridge_model_mismatch: launch_options.allow_cartridge_model_mismatch,
                trace_jsonl: launch_options.trace_jsonl.clone(),
            };
            if io::stdin().is_terminal() && io::stdout().is_terminal() && !tui {
                tui_state.set_debug_sidebar(false);
                let canned = ask_reply_str.map(str::to_string);
                let actor = spawn_session_actor(
                    driver,
                    worker,
                    tui_state,
                    max_turns,
                    canned,
                    Box::new(move |target_model, explicit_cart, cart_label, ro| {
                        spawn_configured_worker(
                            target_model,
                            explicit_cart,
                            worker_cmd.as_deref(),
                            cart_label,
                            ro,
                            &reload_opts,
                        )
                    }),
                );
                run_iocraft_tui_session(actor, false, None)?;
            } else {
                let mut needs_session_reset = true;
                let (w, h) = detect_terminal_size();
                let render_mode = if tui {
                    println!("{}", render_two_pane_tui_frame(&tui_state, w, h, true));
                    SessionRenderMode::SnapshotTui {
                        width: w,
                        height: h,
                    }
                } else {
                    SessionRenderMode::JsonStdout
                };
                eprintln!(
                    "portia-cli REPL attached to {:?} (model={}, read_only={}, max_turns={}, session_memory=stateful (type '/model <scale>', '/x', '/export', '/undo', '/status', 'reset', 'schema', or 'quit'))",
                    db, tui_state.model_name, read_only, max_turns
                );
                let mut input_frontend = LineInputFrontend::new_repl_stdio();
                while let Some(line) =
                    input_frontend.read_command_line("portia> ", &mut tui_state)?
                {
                    let cmd = parse_repl_command(&line);
                    if !execute_session_command_with_spawner(
                        cmd,
                        &mut driver,
                        &mut worker,
                        &mut tui_state,
                        &mut needs_session_reset,
                        max_turns,
                        ask_reply_str,
                        render_mode,
                        &mut input_frontend,
                        |target_model, explicit_cart, cart_label, ro| {
                            spawn_configured_worker(
                                target_model,
                                explicit_cart,
                                worker_cmd.as_deref(),
                                cart_label,
                                ro,
                                &reload_opts,
                            )
                        },
                    )? {
                        break;
                    }
                }
                driver.flush_open_undo_savepoints()?;
            }
        }
        Commands::Tui {
            db,
            prompt,
            runtime,
            inline,
            width,
            height,
        } => {
            let InitializedSqliteSession {
                mut driver,
                mut tui_state,
                mut worker,
                read_only,
                max_turns,
                ask_reply,
                worker_cmd,
                launch_options,
            } = init_sqlite_worker_session(&db, runtime)?;
            let ask_reply_str = ask_reply.as_ref().map(AskDecision::as_str);
            let reload_opts = WorkerLaunchOptions {
                model_family: None,
                allow_cartridge_model_mismatch: launch_options.allow_cartridge_model_mismatch,
                trace_jsonl: launch_options.trace_jsonl.clone(),
            };
            let (auto_w, auto_h) = detect_terminal_size();
            let frame_w = width.unwrap_or(auto_w);
            let frame_h = height.unwrap_or(auto_h);

            let interactive_tty = io::stdin().is_terminal()
                && io::stdout().is_terminal()
                && width.is_none()
                && height.is_none();

            if interactive_tty {
                tui_state.set_debug_sidebar(false);
                let canned = ask_reply_str.map(str::to_string);
                let actor = spawn_session_actor(
                    driver,
                    worker,
                    tui_state,
                    max_turns,
                    canned,
                    Box::new(move |target_model, explicit_cart, cart_label, ro| {
                        spawn_configured_worker(
                            target_model,
                            explicit_cart,
                            worker_cmd.as_deref(),
                            cart_label,
                            ro,
                            &reload_opts,
                        )
                    }),
                );
                run_iocraft_tui_session(actor, inline, prompt)?;
                return Ok(());
            }

            if let Some(initial_p) = prompt {
                let stdin = io::stdin();
                let mut lock = stdin.lock();
                let db_ver_before = driver.db_version();
                let mut events = Vec::new();
                let res = run_live_worker_session_into(
                    &mut driver,
                    &mut worker,
                    &initial_p,
                    max_turns,
                    Some(&mut tui_state),
                    &mut events,
                    |q| read_interactive_ask_reply_from_reader(&mut lock, q, ask_reply_str),
                );
                let receipt = summarize_session_outcome(
                    &events,
                    res.is_err(),
                    db_ver_before,
                    driver.db_version(),
                );
                println!(
                    "{}",
                    render_two_pane_tui_frame(&tui_state, frame_w, frame_h, true)
                );
                res?;
                if !matches!(
                    receipt.status,
                    SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
                ) {
                    bail!(
                        "Session exhausted max_turns ({}) without reaching terminal RESULT or ABSTAIN",
                        max_turns
                    );
                }
                return Ok(());
            }

            let mut needs_session_reset = true;
            eprintln!(
                "portia-cli TUI attached to {:?} (model={}, read_only={}, max_turns={}, session_memory=stateful (type '/model <scale>', '/x', '/export', '/undo', '/status', 'reset', 'schema', or 'quit'))",
                db, tui_state.model_name, read_only, max_turns
            );
            println!(
                "{}",
                render_two_pane_tui_frame(&tui_state, frame_w, frame_h, true)
            );
            let mut input_frontend = LineInputFrontend::new_repl_stdio();
            let render_mode = SessionRenderMode::SnapshotTui {
                width: frame_w,
                height: frame_h,
            };
            while let Some(line) = input_frontend.read_command_line("portia> ", &mut tui_state)? {
                let cmd = parse_repl_command(&line);
                if !execute_session_command_with_spawner(
                    cmd,
                    &mut driver,
                    &mut worker,
                    &mut tui_state,
                    &mut needs_session_reset,
                    max_turns,
                    ask_reply_str,
                    render_mode,
                    &mut input_frontend,
                    |target_model, explicit_cart, cart_label, ro| {
                        spawn_configured_worker(
                            target_model,
                            explicit_cart,
                            worker_cmd.as_deref(),
                            cart_label,
                            ro,
                            &reload_opts,
                        )
                    },
                )? {
                    break;
                }
            }
            driver.flush_open_undo_savepoints()?;
        }
        Commands::Compile {
            input_json,
            db,
            regime_json,
            read_only,
            steward_columns,
            lock_columns,
            schema_disclosed,
            allow_synthetic_fixture_targets,
            compiler_engine,
            out_json,
        } => {
            let t0 = std::time::Instant::now();
            compile_garak::ensure_v2_compiler_engine_flag(&compiler_engine)?;
            let governance = SchemaGovernanceColumns::new(steward_columns, lock_columns)
                .with_schema_disclosed(schema_disclosed);
            let (cartridge, provenance) = compile_v2::compile_cartridge_v2_with_governance(
                &input_json,
                db.as_deref(),
                regime_json.as_deref(),
                read_only,
                &governance,
                allow_synthetic_fixture_targets,
                &out_json,
            )?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let model_family = cartridge
                .header
                .as_ref()
                .map(|h| h.model_family.as_str())
                .or(cartridge.model_id.as_deref());
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true,
                    "out_json": out_json,
                    "schema_version": "portia-wire-v2",
                    "model_family": model_family,
                    "widened_slots": cartridge.widened_slots,
                    "regime_source": provenance.regime_source,
                    "capture_source": provenance.capture_source,
                    "obligation_source": provenance.obligation_source,
                    "hidden_dim": provenance.hidden_dim,
                    "matched_steward_columns": provenance.matched_steward_columns,
                    "matched_lock_columns": provenance.matched_lock_columns,
                    "read_only": provenance.read_only,
                    "active_modes": provenance.active_modes,
                    "guarded_rules": provenance.guarded_rules,
                    "rule_state_bindings": provenance.rule_state_bindings,
                    "bisimulation_verified": provenance.bisimulation_verified,
                    "min_positive_gate_pre_silu": provenance.min_positive_gate_pre_silu,
                    "max_negative_gate_pre_silu": provenance.max_negative_gate_pre_silu,
                    "min_positive_chordal_cosine": provenance.min_positive_chordal_cosine,
                    "max_negative_swiglu_leak_l2": provenance.max_negative_swiglu_leak_l2,
                    "decoded_mode_matches": provenance.decoded_mode_matches,
                    "residual_source": provenance.residual_source,
                    "banks": cartridge.banks.keys().collect::<Vec<_>>(),
                    "elapsed_ms": elapsed_ms,
                }))?
            );
        }
        Commands::CompileGarak {
            input_json,
            out_json,
            slot_offset,
            widened_slots,
            max_bg_leakage,
            readout_only,
            compiler_engine,
        } => {
            compile_garak::ensure_v2_compiler_engine_flag(&compiler_engine)?;
            compile_garak::run_compile_garak_subcommand(
                &input_json,
                out_json.as_deref(),
                &compile_garak::CompileGarakCliOverrides {
                    slot_offset,
                    widened_slots,
                    max_bg_leakage,
                },
                readout_only,
                std::io::stdout().lock(),
            )?;
        }
        Commands::ExportSafetensors {
            checkpoint_dir,
            cartridge_json,
            out_dir,
            include_banks,
            copy_unmodified_shards,
        } => {
            let receipt = export_safetensors::export_cartridge_to_safetensors_dir(
                &checkpoint_dir,
                &cartridge_json,
                &out_dir,
                &include_banks,
                copy_unmodified_shards,
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Commands::VerifySchedule { schedule_json } => {
            let raw = std::fs::read_to_string(&schedule_json)?;
            let spans: Vec<PenSpanRecord> = serde_json::from_str(&raw)?;
            verify_single_pen_schedule(&spans)?;
            println!("{{\"ok\":true,\"verified_spans\":{}}}", spans.len());
        }
        Commands::Nursery {
            episodes,
            app,
            synth_user,
            model,
            model_family,
            allow_cartridge_model_mismatch,
            trace_jsonl,
            cartridge,
            worker_cmd,
            openai_model,
            max_turns,
            followup_waves,
            branch_factor,
            beam_width,
            user_turns,
            max_nodes,
            luna_concurrency,
            dynamic_schemas,
            coroner,
            out_json,
            oracle_capture_out,
            target_layer,
        } => {
            use crate::nursery_cmd::{
                build_nursery_sweep_coordinates, build_oracle_nursery_policy_for_coordinate,
                run_live_openai_generator_sweep, run_live_sql_beam_tree_sweep,
                run_live_synth_user_generator_sweep, run_nursery_batch_sweep,
                run_oracle_capture_harvest, BranchingBeamConfig, LiveGeneratorSweepConfig,
                NurseryAppFilter, NurserySynthMode, OracleCaptureHarvestConfig,
                DEFAULT_CODEX_LUNA_MODEL,
            };

            let launch_options = WorkerLaunchOptions {
                model_family,
                allow_cartridge_model_mismatch,
                trace_jsonl,
            };
            if let Some(ref capture_out_path) = oracle_capture_out {
                let effective_cartridge = cartridge
                    .as_deref()
                    .or_else(|| Some(std::path::Path::new("none")));
                let cart_label = cartridge.as_deref().map_or_else(
                    || "none (Base)".to_string(),
                    |p| resolve_default_cartridge_label(Some(p)),
                );
                let mut worker = spawn_configured_worker(
                    &model,
                    effective_cartridge,
                    worker_cmd.as_deref(),
                    &cart_label,
                    false,
                    &launch_options,
                )?;
                let harvest_cfg = OracleCaptureHarvestConfig {
                    model_family: worker
                        .provenance
                        .resolved_model_family
                        .clone()
                        .unwrap_or_else(|| normalize_model_identifier(&model)),
                    episodes: episodes.max(1),
                    target_layer,
                    ..OracleCaptureHarvestConfig::default()
                };
                let receipt =
                    run_oracle_capture_harvest(&mut worker, &harvest_cfg, capture_out_path)?;
                let json_txt = receipt.to_json_pretty()?;
                if let Some(ref path) = out_json {
                    std::fs::write(path, &json_txt)?;
                }
                println!("{json_txt}");
                return Ok(());
            }

            let app_filter: NurseryAppFilter = app.parse()?;
            let synth_mode: NurserySynthMode = synth_user.parse()?;
            let sweep_cfg = LiveGeneratorSweepConfig {
                episodes: episodes.max(1),
                app_filter,
                synth_mode,
                openai_model,
                max_turns: max_turns.max(1),
                followup_waves,
                run_coroner: coroner,
            };
            let coords = build_nursery_sweep_coordinates(sweep_cfg.episodes, sweep_cfg.app_filter);

            let resolved_branch_factor = branch_factor
                .unwrap_or(if synth_mode == NurserySynthMode::Luna {
                    3
                } else {
                    1
                })
                .clamp(1, 3);
            let resolved_beam_width = beam_width.filter(|&k| k > 0);
            let resolved_user_turns = user_turns
                .unwrap_or(if synth_mode == NurserySynthMode::Luna {
                    6
                } else {
                    1 + followup_waves
                })
                .max(1);
            let beam_cfg = BranchingBeamConfig {
                branch_factor: resolved_branch_factor,
                beam_width: resolved_beam_width,
                max_user_turns: resolved_user_turns,
                max_agent_steps_per_user_turn: sweep_cfg.max_turns,
                max_total_nodes: max_nodes.max(1),
                luna_concurrency: luna_concurrency.max(1),
            };
            let use_sql_beam_runner = synth_mode == NurserySynthMode::Luna
                || (app_filter == NurseryAppFilter::SqlSteward
                    && (branch_factor.is_some()
                        || beam_width.is_some()
                        || user_turns.is_some()
                        || dynamic_schemas));

            let json_txt = match sweep_cfg.synth_mode {
                NurserySynthMode::Oracle => {
                    let (_receipts, summary) = run_nursery_batch_sweep(&coords, |coord| {
                        build_oracle_nursery_policy_for_coordinate(*coord)
                    })?;
                    summary.to_json_pretty()?
                }
                NurserySynthMode::Deterministic => {
                    let cart_label =
                        resolve_default_cartridge_label_for_model(&model, cartridge.as_deref());
                    let mut worker = spawn_configured_worker(
                        &model,
                        cartridge.as_deref(),
                        worker_cmd.as_deref(),
                        &cart_label,
                        false,
                        &launch_options,
                    )?;
                    let mut det_user = crate::synth_user::DeterministicStewardUserEnv::new();
                    if use_sql_beam_runner {
                        let summary = run_live_sql_beam_tree_sweep(
                            &coords,
                            &mut det_user,
                            &mut worker,
                            beam_cfg,
                            |_ep_idx, _coord, _env| Ok(None),
                        )?;
                        summary.to_json_pretty()?
                    } else {
                        let summary = run_live_synth_user_generator_sweep(
                            &coords,
                            &mut det_user,
                            &mut worker,
                            sweep_cfg.max_turns,
                            sweep_cfg.followup_waves,
                        )?;
                        summary.to_json_pretty()?
                    }
                }
                NurserySynthMode::Luna => {
                    let transport = crate::synth_user::CodexExecChatTransport::from_luna_env(
                        sweep_cfg.openai_model.as_deref(),
                    );
                    let model_label = transport
                        .model_override()
                        .unwrap_or(DEFAULT_CODEX_LUNA_MODEL)
                        .to_string();
                    let mut luna_user =
                        crate::synth_user::OpenAiChatSynthUserEnv::new(transport, model_label);
                    let cart_label =
                        resolve_default_cartridge_label_for_model(&model, cartridge.as_deref());
                    let mut worker = spawn_configured_worker(
                        &model,
                        cartridge.as_deref(),
                        worker_cmd.as_deref(),
                        &cart_label,
                        false,
                        &launch_options,
                    )?;
                    let summary = run_live_sql_beam_tree_sweep(
                        &coords,
                        &mut luna_user,
                        &mut worker,
                        beam_cfg,
                        |ep_idx, coord, env| {
                            if dynamic_schemas {
                                let hint = format!(
                                    "{} multi-table operational SQLite ledger #{}",
                                    coord.domain_skin.slug(),
                                    ep_idx + 1
                                );
                                Ok(Some(
                                    env.synthesize_dynamic_sqlite_scenario(Some(&hint), 3)?,
                                ))
                            } else {
                                Ok(None)
                            }
                        },
                    )?;
                    summary.to_json_pretty()?
                }
                NurserySynthMode::OpenAi => {
                    let transport = crate::synth_user::AutoOpenAiTransport::from_env(
                        sweep_cfg.openai_model.as_deref(),
                    )?;
                    let model_label = transport.effective_model_label().to_string();
                    let mut oai_user =
                        crate::synth_user::OpenAiChatSynthUserEnv::new(transport, model_label);
                    let cart_label =
                        resolve_default_cartridge_label_for_model(&model, cartridge.as_deref());
                    let mut worker = spawn_configured_worker(
                        &model,
                        cartridge.as_deref(),
                        worker_cmd.as_deref(),
                        &cart_label,
                        false,
                        &launch_options,
                    )?;
                    if use_sql_beam_runner {
                        let summary = run_live_sql_beam_tree_sweep(
                            &coords,
                            &mut oai_user,
                            &mut worker,
                            beam_cfg,
                            |ep_idx, coord, env| {
                                if dynamic_schemas {
                                    let hint = format!(
                                        "{} multi-table operational SQLite ledger #{}",
                                        coord.domain_skin.slug(),
                                        ep_idx + 1
                                    );
                                    Ok(Some(
                                        env.synthesize_dynamic_sqlite_scenario(Some(&hint), 3)?,
                                    ))
                                } else {
                                    Ok(None)
                                }
                            },
                        )?;
                        summary.to_json_pretty()?
                    } else {
                        let summary = run_live_openai_generator_sweep(
                            &coords,
                            &mut oai_user,
                            &mut worker,
                            sweep_cfg.max_turns,
                            sweep_cfg.followup_waves,
                            sweep_cfg.run_coroner,
                        )?;
                        summary.to_json_pretty()?
                    }
                }
            };

            if let Some(ref path) = out_json {
                std::fs::write(path, &json_txt)?;
            }
            println!("{json_txt}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests;
