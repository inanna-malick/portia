//! Warm-worker multi-task JSONL batch runner (`portia-cli batch`) that spawns a single
//! [`StdioQwenWorkerClient`], opens a fresh [`LiveSqliteDailyDriver`] per task, resets session
//! KV/DeltaNet state between tasks, and streams one [`BatchTaskReceipt`] JSON line per task.

use crate::driver::{AskDecision, LiveSqliteDailyDriver};
use crate::model_identity::{normalize_model_identifier, WorkerLaunchOptions};
use crate::worker::{
    attach_session_error, resolve_default_cartridge_label_for_model, run_live_worker_session_into,
    summarize_session_outcome, SessionSummaryReceipt,
};
use crate::{read_interactive_ask_reply_from_reader, spawn_configured_worker, WorkerRuntimeArgs};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

/// One task line read from `--tasks-jsonl` in `portia-cli batch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchTaskSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    pub db: PathBuf,
    pub prompt: String,
    #[serde(default)]
    pub read_only: Option<bool>,
    #[serde(default)]
    pub steward_columns: Option<Vec<String>>,
    #[serde(default)]
    pub disambiguate_candidates: Option<bool>,
    #[serde(default)]
    pub ask_reply: Option<String>,
    #[serde(default)]
    pub max_turns: Option<usize>,
    #[serde(default)]
    pub max_write_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_jsonl: Option<PathBuf>,
}

/// One JSONL output line emitted by `portia-cli batch`, flattening [`SessionSummaryReceipt`]
/// alongside optional `case_id` and `error` metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchTaskReceipt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    pub db: PathBuf,
    #[serde(flatten)]
    pub summary: SessionSummaryReceipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Parse non-empty JSONL lines from `tasks_jsonl` into typed [`BatchTaskSpec`] records.
pub fn read_batch_task_specs(tasks_jsonl: &Path) -> Result<Vec<BatchTaskSpec>> {
    let file = std::fs::File::open(tasks_jsonl)
        .with_context(|| format!("failed to open batch tasks JSONL {:?}", tasks_jsonl))?;
    let reader = io::BufReader::new(file);
    let mut specs = Vec::new();
    for (idx, line_res) in reader.lines().enumerate() {
        let line = line_res.with_context(|| {
            format!(
                "failed reading line {} of batch tasks JSONL {:?}",
                idx + 1,
                tasks_jsonl
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let spec: BatchTaskSpec = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "invalid BatchTaskSpec JSON on line {} of {:?}",
                idx + 1,
                tasks_jsonl
            )
        })?;
        specs.push(spec);
    }
    Ok(specs)
}

/// Execute all tasks in `tasks_jsonl` against a single warm [`StdioQwenWorkerClient`], writing
/// one [`BatchTaskReceipt`] JSON line per task to `out_jsonl` (or `stdout` when `None`).
pub fn run_batch_command(
    tasks_jsonl: &Path,
    out_jsonl: Option<&Path>,
    runtime: WorkerRuntimeArgs,
) -> Result<Vec<BatchTaskReceipt>> {
    let specs = read_batch_task_specs(tasks_jsonl)?;
    if specs.is_empty() {
        bail!("batch tasks file {:?} contains 0 task lines", tasks_jsonl);
    }

    let WorkerRuntimeArgs {
        model,
        model_family,
        allow_cartridge_model_mismatch,
        read_only: default_read_only,
        max_turns: default_max_turns,
        trace_jsonl: default_trace_jsonl,
        max_write_rows: default_max_write_rows,
        ask_reply: default_ask_reply,
        steward_columns: default_steward_columns,
        disambiguate_candidates: default_disambiguate_candidates,
        cartridge,
        worker_cmd,
    } = runtime;

    let normalized_model = normalize_model_identifier(&model);
    let launch_options = WorkerLaunchOptions {
        model_family,
        allow_cartridge_model_mismatch,
        trace_jsonl: default_trace_jsonl.clone(),
    };
    let cart_label =
        resolve_default_cartridge_label_for_model(&normalized_model, cartridge.as_deref());

    let mut worker = spawn_configured_worker(
        &normalized_model,
        cartridge.as_deref(),
        worker_cmd.as_deref(),
        &cart_label,
        default_read_only,
        &launch_options,
    )?;

    let mut out_file = match out_jsonl {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Some(
                std::fs::File::create(path)
                    .with_context(|| format!("failed to create batch output {:?}", path))?,
            )
        }
        None => None,
    };

    let mut receipts = Vec::with_capacity(specs.len());
    for spec in specs {
        let effective_trace = spec.trace_jsonl.as_ref().or(default_trace_jsonl.as_ref());
        match effective_trace {
            Some(trace_path) => worker.set_trace_jsonl(trace_path)?,
            None => worker.clear_trace_jsonl(),
        }

        let read_only = spec.read_only.unwrap_or(default_read_only);
        let max_turns = spec.max_turns.unwrap_or(default_max_turns);
        let max_write_rows = spec.max_write_rows.unwrap_or(default_max_write_rows);
        let disambiguate = spec
            .disambiguate_candidates
            .unwrap_or(default_disambiguate_candidates);
        let steward_cols = spec
            .steward_columns
            .clone()
            .unwrap_or_else(|| default_steward_columns.clone());

        let resolved_ask_reply: Option<String> = match spec.ask_reply.as_deref() {
            Some(raw) => Some(raw.trim().to_string()),
            None => default_ask_reply
                .as_ref()
                .map(|d| AskDecision::as_str(d).to_string()),
        };

        let mut driver = LiveSqliteDailyDriver::open_with_mode(&spec.db, max_write_rows, read_only)
            .with_context(|| format!("failed opening SQLite db {:?}", spec.db))?;
        if !steward_cols.is_empty() {
            driver.set_steward_policy_columns(&steward_cols);
        }
        if disambiguate {
            driver.set_candidate_disambiguation(true);
        }

        let db_ver_before = driver.db_version();
        let mut transcript_events = Vec::new();
        let ask_cb = |q: &str| -> Result<String> {
            let stdin = io::stdin();
            let mut lock = stdin.lock();
            read_interactive_ask_reply_from_reader(&mut lock, q, resolved_ask_reply.as_deref())
        };

        // `run_live_worker_session_into` passes `reset_model_session = true`, syncing the new
        // driver's schema context and calling `worker.reset_session()` before prefilling.
        let session_res = run_live_worker_session_into(
            &mut driver,
            &mut worker,
            &spec.prompt,
            max_turns,
            None,
            &mut transcript_events,
            ask_cb,
        );

        let mut summary = summarize_session_outcome(
            &transcript_events,
            session_res.is_err(),
            db_ver_before,
            driver.db_version(),
        );
        attach_session_error(&mut summary, session_res.as_ref().err());
        summary.provenance = worker.provenance.clone();

        let receipt = BatchTaskReceipt {
            case_id: spec.case_id,
            db: spec.db,
            summary,
            error: session_res.err().map(|e| format!("{e:#}")),
        };

        let mut line_json = serde_json::to_string(&receipt)?;
        line_json.push('\n');
        if let Some(ref mut f) = out_file {
            f.write_all(line_json.as_bytes())?;
            f.flush()?;
        } else {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(line_json.as_bytes())?;
            lock.flush()?;
        }

        receipts.push(receipt);
    }

    Ok(receipts)
}
