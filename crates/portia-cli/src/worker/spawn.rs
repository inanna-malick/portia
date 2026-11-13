//! Subprocess lifecycle, Python/venv resolution, cartridge provenance handshake, and
//! token-streaming RPC methods on [`StdioQwenWorkerClient`].

use super::protocol::{
    format_observation_wire_chunk, is_unpatched_base_cartridge_sentinel,
    validate_explicit_cartridge_path, OnPolicyCaptureWithBackground, PayloadActivationRow,
    WorkerRequest, WorkerResponse, WorkerWeightGeometryPayload,
};
use super::summary::SessionRunError;
use crate::driver::DriverObservation;
use crate::model_identity::{
    default_cartridge_filename, inspect_runtime_provenance, normalize_model_identifier,
    resolve_model_family, ModelIdentityError, RuntimeProvenance, WorkerLaunchOptions,
};
use anyhow::{bail, Context, Result};
use portia_core::relay::{CompletedRelayTurn, RelayMachine, RelayPhase, RelayStreamEvent};
use portia_core::{
    CartridgeBank, ColumnSchema, KvPrefillDisposition, LotlNormalizedWireChunk, Mode, PenOwner,
    RouteTag, TableSchema,
};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFAULT_WORKER_RPC_TIMEOUT_SECS: u64 = 300;

fn default_worker_rpc_timeout() -> Duration {
    if let Ok(raw) = std::env::var("PORTIA_WORKER_TIMEOUT_SECS") {
        if let Ok(secs) = raw.trim().parse::<u64>() {
            if secs > 0 {
                return Duration::from_secs(secs);
            }
        }
    }
    Duration::from_secs(DEFAULT_WORKER_RPC_TIMEOUT_SECS)
}

fn spawn_stdout_line_reader(stdout: ChildStdout) -> Receiver<std::io::Result<String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if tx.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    });
    rx
}

/// Drain subprocess `stderr` into an in-memory ring buffer so Python/PyTorch/MPS warnings never
/// write raw bytes onto the live TUI alternate screen while preserving crash diagnostics.
fn spawn_stderr_drain(stderr: ChildStderr) -> Arc<Mutex<Vec<String>>> {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let buf_clone = Arc::clone(&buf);
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let trimmed = line.trim_end().to_string();
                    if !trimmed.is_empty() {
                        if let Ok(mut guard) = buf_clone.lock() {
                            if guard.len() >= 32 {
                                guard.remove(0);
                            }
                            guard.push(trimmed);
                        }
                    }
                }
            }
        }
    });
    buf
}

/// Live OS-pipe client that spawns `scripts/portia_qwen_stdio_worker.py` (or any compatible
/// JSON-over-stdio worker) and drives [`RelayMachine`] token-by-token over `stdin`/`stdout`.
pub struct StdioQwenWorkerClient {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<std::io::Result<String>>,
    stderr_log: Arc<Mutex<Vec<String>>>,
    rpc_timeout: Duration,
    pub(super) active_table_name: Option<String>,
    pub(super) active_columns: Vec<ColumnSchema>,
    pub(super) active_tables: Vec<TableSchema>,
    schema_context_synced: bool,
    capture_layer: Option<usize>,
    pub(super) captured_pre_mlp_trace: Vec<Vec<f32>>,
    manifest_mode_grammars: BTreeMap<Mode, portia_core::GrammarCartridgeSpec>,
    trace_file: Option<std::fs::File>,
    pub(super) trace_task_id: u64,
    pub provenance: RuntimeProvenance,
}

fn parse_manifest_mode_key(raw_key: &str) -> Option<Mode> {
    let tail = raw_key
        .trim()
        .trim_end_matches(')')
        .rsplit([':', '('])
        .next()
        .unwrap_or(raw_key)
        .trim()
        .to_ascii_uppercase();
    Mode::parse_flexible(&tail)
}

fn load_manifest_mode_grammars(
    cartridge_path: &Path,
) -> BTreeMap<Mode, portia_core::GrammarCartridgeSpec> {
    let Ok(raw) = std::fs::read_to_string(cartridge_path) else {
        return BTreeMap::new();
    };
    let Ok(bundle) = portia_compiler::wire::WireCartridgeManifestJson::from_json_str(&raw) else {
        return BTreeMap::new();
    };
    let mut grammars = BTreeMap::new();
    for (raw_mode, spec) in bundle.mode_grammars {
        if let Some(mode) = parse_manifest_mode_key(&raw_mode) {
            grammars.insert(mode, spec);
        }
    }
    grammars
}

fn apply_runtime_ld_library_path(cmd: &mut Command) {
    let mut dirs: Vec<String> = Vec::new();
    for env_key in ["PORTIA_EXTRA_LD_LIBRARY_PATH", "NIX_LD_LIBRARY_PATH"] {
        if let Ok(raw) = std::env::var(env_key) {
            for entry in raw.split(':').map(str::trim).filter(|s| !s.is_empty()) {
                if Path::new(entry).is_dir() && !dirs.iter().any(|d| d == entry) {
                    dirs.push(entry.to_string());
                }
            }
        }
    }
    if Path::new("/run/opengl-driver/lib").is_dir()
        && !dirs.iter().any(|d| d == "/run/opengl-driver/lib")
    {
        dirs.push("/run/opengl-driver/lib".to_string());
    }
    if !dirs.is_empty() {
        let prefix = dirs.join(":");
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let combined = if existing.is_empty() {
            prefix
        } else {
            format!("{prefix}:{existing}")
        };
        cmd.env("LD_LIBRARY_PATH", combined);
    }
}

fn ensure_pinned_worker_venv_if_needed() {
    if std::env::var("PORTIA_PYTHON")
        .ok()
        .is_some_and(|s| !s.trim().is_empty())
    {
        return;
    }
    let roots = candidate_asset_roots();
    let Some(root) = roots
        .into_iter()
        .find(|r| r.join("pyproject.toml").is_file() && r.join("uv.lock").is_file())
    else {
        return;
    };
    let venv_py = root.join(".venv/bin/python");
    let has_transformers_v5 = venv_py.is_file() && {
        let mut cmd = Command::new(&venv_py);
        cmd.arg("-c")
            .arg("import transformers; assert int(transformers.__version__.split('.')[0]) >= 5")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        apply_runtime_ld_library_path(&mut cmd);
        cmd.status().is_ok_and(|st| st.success())
    };
    if !has_transformers_v5 {
        eprintln!(
            "[portia-cli] syncing pinned Python environment (.venv) from {:?}/uv.lock via `uv sync --locked`...",
            root
        );
        let status = Command::new("uv")
            .arg("sync")
            .arg("--locked")
            .current_dir(&root)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status();
        if status.is_err() || !status.is_ok_and(|s| s.success()) {
            let _ = Command::new("uv")
                .arg("sync")
                .current_dir(&root)
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status();
        }
    }
}

impl StdioQwenWorkerClient {
    /// Spawn the default Python worker (`scripts/portia_qwen_stdio_worker.py`) directly via
    /// structured OS process arguments (`Command::new(python_bin).args(...)`) without `sh -c`.
    pub fn spawn_default(model: &str, cartridge: Option<&Path>) -> Result<Self> {
        Self::spawn_default_with_options(model, cartridge, &WorkerLaunchOptions::default())
    }

    pub fn spawn_default_with_options(
        model: &str,
        cartridge: Option<&Path>,
        options: &WorkerLaunchOptions,
    ) -> Result<Self> {
        let (python_bin, args) = resolve_worker_argv_with_options(model, cartridge, options)?;
        let cart_index = args
            .iter()
            .position(|a| a == "--cartridge-json")
            .context("missing resolved cartridge argument")?;
        let provenance =
            inspect_runtime_provenance(model, Path::new(&args[cart_index + 1]), options)?;
        ensure_pinned_worker_venv_if_needed();
        let mut manifest_mode_grammars = BTreeMap::new();
        if let Some(idx) = args.iter().position(|a| a == "--cartridge-json") {
            if let Some(cart_arg) = args.get(idx + 1) {
                let cart_path = Path::new(cart_arg);
                if !is_unpatched_base_cartridge_sentinel(cart_path) && cart_path.is_file() {
                    manifest_mode_grammars = load_manifest_mode_grammars(cart_path);
                }
            }
        }
        let mut cmd = Command::new(&python_bin);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_runtime_ld_library_path(&mut cmd);
        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn default stdio worker {:?} with args {:?}",
                python_bin, args
            )
        })?;
        let stdin = child.stdin.take().context("missing child stdin")?;
        let stdout = child.stdout.take().context("missing child stdout")?;
        let stderr = child.stderr.take().context("missing child stderr")?;
        let stdout_rx = spawn_stdout_line_reader(stdout);
        let stderr_log = spawn_stderr_drain(stderr);
        let mut worker = Self {
            child,
            stdin,
            stdout_rx,
            stderr_log,
            rpc_timeout: default_worker_rpc_timeout(),
            active_table_name: None,
            active_columns: Vec::new(),
            active_tables: Vec::new(),
            schema_context_synced: false,
            capture_layer: None,
            captured_pre_mlp_trace: Vec::new(),
            manifest_mode_grammars,
            trace_file: None,
            trace_task_id: 0,
            provenance,
        };
        if let Some(path) = &options.trace_jsonl {
            worker.set_trace_jsonl(path)?;
        }
        match worker.query_stdio_worker(&WorkerRequest::RuntimeMetadata)? {
            WorkerResponse::RuntimeMetadata { mut provenance } => {
                if provenance.cartridge_sha256 != worker.provenance.cartridge_sha256 {
                    bail!("cartridge changed between preflight validation and worker mounting");
                }
                provenance.requested_model = model.to_string();
                worker.provenance = provenance;
            }
            other => bail!("expected runtime_metadata handshake, got {other:?}"),
        }
        worker.trace_record(serde_json::json!({"event":"worker_ready"}))?;
        Ok(worker)
    }

    /// Spawn a custom worker subprocess via `sh -c <cmd_line>` with piped `stdin`/`stdout`/`stderr`.
    pub fn spawn_shell(cmd_line: &str) -> Result<Self> {
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(cmd_line)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_runtime_ld_library_path(&mut cmd);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn stdio worker {:?}", cmd_line))?;
        let stdin = child.stdin.take().context("missing child stdin")?;
        let stdout = child.stdout.take().context("missing child stdout")?;
        let stderr = child.stderr.take().context("missing child stderr")?;
        let stdout_rx = spawn_stdout_line_reader(stdout);
        let stderr_log = spawn_stderr_drain(stderr);
        Ok(Self {
            child,
            stdin,
            stdout_rx,
            stderr_log,
            rpc_timeout: default_worker_rpc_timeout(),
            active_table_name: None,
            active_columns: Vec::new(),
            active_tables: Vec::new(),
            schema_context_synced: false,
            capture_layer: None,
            captured_pre_mlp_trace: Vec::new(),
            manifest_mode_grammars: BTreeMap::new(),
            trace_file: None,
            trace_task_id: 0,
            provenance: RuntimeProvenance::default(),
        })
    }

    /// Append machine-readable task and turn evidence independently of stdout/TUI rendering.
    pub fn set_trace_jsonl(&mut self, path: &Path) -> Result<()> {
        self.trace_file = Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("opening session trace {}", path.display()))?,
        );
        Ok(())
    }

    /// Clear any currently attached JSONL trace output sink.
    pub fn clear_trace_jsonl(&mut self) {
        self.trace_file = None;
    }

    pub(super) fn trace_record(&mut self, mut record: serde_json::Value) -> Result<()> {
        if let Some(file) = &mut self.trace_file {
            record["task_id"] = self.trace_task_id.into();
            record["provenance"] = serde_json::to_value(&self.provenance)?;
            serde_json::to_writer(&mut *file, &record)?;
            file.write_all(b"\n")?;
            file.flush()?;
        }
        Ok(())
    }

    /// Load `mode_grammars` from a cartridge JSON manifest on disk.
    pub fn load_manifest_from_cartridge_path(&mut self, cartridge_path: &Path) -> Result<()> {
        validate_explicit_cartridge_path(cartridge_path)?;
        if is_unpatched_base_cartridge_sentinel(cartridge_path) {
            self.manifest_mode_grammars.clear();
            return Ok(());
        }
        self.manifest_mode_grammars = load_manifest_mode_grammars(cartridge_path);
        Ok(())
    }

    /// Per-mode [`portia_core::GrammarCartridgeSpec`] entries loaded from the cartridge manifest.
    pub fn manifest_mode_grammars(&self) -> &BTreeMap<Mode, portia_core::GrammarCartridgeSpec> {
        &self.manifest_mode_grammars
    }

    /// Configure the per-request RPC timeout for reading responses from the worker subprocess.
    pub fn set_rpc_timeout(&mut self, timeout: Duration) {
        self.rpc_timeout = timeout;
    }

    /// Enable or disable per-prefill/per-step pre-MLP activation capture at `layer`.
    pub fn set_capture_layer(&mut self, layer: Option<usize>) {
        self.capture_layer = layer;
    }

    /// Drain and return all pre-MLP activation vectors captured from `Prefilled` and `Token`
    /// worker responses since the last drain or `reset_session`.
    pub fn take_captured_pre_mlp_trace(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.captured_pre_mlp_trace)
    }

    /// Attach live `PRAGMA table_info` schema metadata (`table_name`, `columns`, and `tables`)
    /// so the worker's `SqlSchemaBpeTrieMasker` receives the finite schema set on `reset_session`
    /// or the first subsequent request without re-sending it on every token step.
    pub fn set_schema_context(&mut self, tables: Vec<TableSchema>, primary_table: Option<&str>) {
        let selected = primary_table
            .and_then(|pt| {
                tables
                    .iter()
                    .find(|t| t.table_name.eq_ignore_ascii_case(pt))
            })
            .or_else(|| {
                self.active_table_name.as_deref().and_then(|prev| {
                    tables
                        .iter()
                        .find(|t| t.table_name.eq_ignore_ascii_case(prev))
                })
            })
            .or_else(|| tables.first());
        let next_table_name = selected.map(|sc| sc.table_name.clone());
        let next_columns = selected.map(|sc| sc.columns.clone()).unwrap_or_default();
        if self.active_table_name != next_table_name
            || self.active_columns != next_columns
            || self.active_tables != tables
        {
            self.active_table_name = next_table_name;
            self.active_columns = next_columns;
            self.active_tables = tables;
            self.schema_context_synced = false;
        }
    }

    fn take_unsynced_schema_payload(
        &mut self,
    ) -> (Option<String>, Vec<ColumnSchema>, Vec<TableSchema>) {
        if self.schema_context_synced {
            (None, Vec::new(), Vec::new())
        } else {
            self.schema_context_synced = true;
            (
                self.active_table_name.clone(),
                self.active_columns.clone(),
                self.active_tables.clone(),
            )
        }
    }

    fn send_raw_rpc(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
        let mut line = serde_json::to_string(req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;

        let resp_line = match self.stdout_rx.recv_timeout(self.rpc_timeout) {
            Ok(Ok(s)) => s,
            Ok(Err(io_err)) => {
                bail!("Qwen stdio worker stdout read error: {}", io_err);
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                bail!(
                    "Qwen stdio worker timed out after {:?} waiting for RPC response",
                    self.rpc_timeout
                );
            }
            Err(RecvTimeoutError::Disconnected) => {
                let tail = self
                    .stderr_log
                    .lock()
                    .ok()
                    .map(|g| g.join("\n"))
                    .unwrap_or_default();
                if tail.is_empty() {
                    bail!("Qwen stdio worker closed stdout unexpectedly");
                } else {
                    bail!("Qwen stdio worker closed stdout unexpectedly:\n{tail}");
                }
            }
        };
        serde_json::from_str(resp_line.trim())
            .with_context(|| format!("invalid JSON from worker: {:?}", resp_line))
    }

    pub fn query_stdio_worker(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
        let resp = self.send_raw_rpc(req)?;
        if let WorkerResponse::Error { message } = &resp {
            bail!("Qwen stdio worker error: {}", message);
        }
        match &resp {
            WorkerResponse::Prefilled {
                captured_pre_mlp: Some(vec),
                ..
            }
            | WorkerResponse::ConsumedToken {
                captured_pre_mlp: Some(vec),
                ..
            } => {
                self.captured_pre_mlp_trace.push(vec.clone());
            }
            _ => {}
        }
        Ok(resp)
    }

    pub fn send_request(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
        self.query_stdio_worker(req)
    }

    pub fn reset_session(&mut self) -> Result<()> {
        let _ = self.query_stdio_worker(&WorkerRequest::ResetSession {
            table_name: self.active_table_name.clone(),
            columns: self.active_columns.clone(),
            tables: self.active_tables.clone(),
        })?;
        self.schema_context_synced = true;
        self.captured_pre_mlp_trace.clear();
        Ok(())
    }

    /// Reset per-task execution state (`db_observations_seen`, `last_tool_response_json`,
    /// `last_observed_row`, `write_applied_in_session`) at the start of a new user task in a
    /// multi-task REPL or TUI session while preserving `past_key_values`.
    pub fn begin_task(&mut self) -> Result<()> {
        let (table_name, columns, tables) = self.take_unsynced_schema_payload();
        let req = WorkerRequest::BeginTask {
            table_name,
            columns,
            tables,
        };
        let resp = self.send_raw_rpc(&req)?;
        if let WorkerResponse::Error { message } = &resp {
            if !message.to_ascii_lowercase().contains("unknown") {
                bail!("Qwen stdio worker error: {}", message);
            }
        }
        Ok(())
    }

    pub fn prefill_chunk_typed(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        chunk: LotlNormalizedWireChunk,
        kv_disposition: KvPrefillDisposition,
    ) -> Result<Vec<u32>> {
        let (table_name, columns, tables) = self.take_unsynced_schema_payload();
        let capture_layer = self.capture_layer;
        match self.query_stdio_worker(&WorkerRequest::Prefill {
            bank,
            pen,
            text: chunk,
            table_name,
            columns,
            tables,
            reset_kv: kv_disposition.should_reset_kv(),
            capture_layer,
        })? {
            WorkerResponse::Prefilled { token_ids, .. } => Ok(token_ids),
            other => bail!("expected Prefilled response, got {:?}", other),
        }
    }

    pub fn prefill_chunk(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        text: &str,
    ) -> Result<Vec<u32>> {
        let inferred_mode = match bank {
            CartridgeBank::ModePayload(mode) => Some(mode),
            CartridgeBank::Base | CartridgeBank::OrientTrunk => None,
        };
        let chunk = LotlNormalizedWireChunk::normalize(text, inferred_mode);
        self.prefill_chunk_typed(bank, pen, chunk, KvPrefillDisposition::AppendIntraTurnKv)
    }

    pub fn step_one_token(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
    ) -> Result<(u32, String)> {
        self.step_one_token_masked(bank, pen, phase, None)
    }

    pub fn step_one_token_masked(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
        allowed_token_ids: Option<Vec<u32>>,
    ) -> Result<(u32, String)> {
        let (token_id, token_piece, _) =
            self.step_one_token_full(bank, pen, phase, allowed_token_ids)?;
        Ok((token_id, token_piece))
    }

    pub fn step_one_token_full(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
        allowed_token_ids: Option<Vec<u32>>,
    ) -> Result<(u32, String, Vec<u32>)> {
        self.step_one_token_with_grammar(bank, pen, phase, allowed_token_ids, None)
    }

    /// Phase 1 of two-phase `Sample -> Plan -> Consume` handoff: sample 1 token from `last_logits`
    /// under `producer_bank` and `grammar_spec` without advancing `past_key_values` yet.
    pub fn sample_one_token_with_grammar(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
        allowed_token_ids: Option<Vec<u32>>,
        grammar_spec: Option<portia_core::GrammarCartridgeSpec>,
    ) -> Result<(u32, String, Vec<u32>)> {
        let (table_name, columns, tables) = self.take_unsynced_schema_payload();
        let req = WorkerRequest::SampleToken {
            bank,
            pen,
            phase,
            table_name,
            columns,
            tables,
            allowed_token_ids,
            grammar_spec,
        };
        let resp = self.send_raw_rpc(&req)?;
        match resp {
            WorkerResponse::SampledToken {
                token_id,
                token_piece,
                token_ids,
            } => {
                let resolved_ids = if token_ids.is_empty() {
                    vec![token_id]
                } else {
                    token_ids
                };
                Ok((token_id, token_piece, resolved_ids))
            }
            WorkerResponse::Error { message } => {
                bail!("Qwen stdio worker error: {}", message);
            }
            other => bail!("expected SampledToken response, got {:?}", other),
        }
    }

    /// Phase 2 of two-phase `Sample -> Plan -> Consume` handoff: mount `consume_bank`
    /// (`CartridgeBank::Base` on closing `</sql>` / `</to_user>`, `ModePayload(mode)` during body)
    /// and forward `token_id` into `past_key_values`.
    pub fn consume_sampled_token(
        &mut self,
        consume_bank: CartridgeBank,
        pen: PenOwner,
        token_id: u32,
    ) -> Result<()> {
        let capture_layer = self.capture_layer;
        match self.query_stdio_worker(&WorkerRequest::ConsumeToken {
            bank: consume_bank,
            pen,
            token_id,
            capture_layer,
        })? {
            WorkerResponse::ConsumedToken { .. } | WorkerResponse::Prefilled { .. } => Ok(()),
            other => bail!("expected ConsumedToken response, got {:?}", other),
        }
    }

    /// Query the worker for `(resolved_target_layer, base_intermediate_size, CompilerWeightGeometrySpec)`.
    pub fn extract_compiler_weight_geometry_for_optional_layer(
        &mut self,
        target_layer: Option<usize>,
    ) -> Result<(usize, usize, crate::compile::CompilerWeightGeometrySpec)> {
        self.extract_compiler_weight_geometry_with_tokens(target_layer, Vec::new())
    }

    pub fn extract_compiler_weight_geometry_with_tokens(
        &mut self,
        target_layer: Option<usize>,
        token_ids: Vec<u32>,
    ) -> Result<(usize, usize, crate::compile::CompilerWeightGeometrySpec)> {
        match self.query_stdio_worker(&WorkerRequest::ExtractCompilerWeightGeometry {
            target_layer,
            token_ids,
        })? {
            WorkerResponse::WeightGeometry(payload) => {
                let WorkerWeightGeometryPayload {
                    target_layer: resp_layer,
                    num_hidden_layers,
                    base_intermediate_size,
                    vocab_top_k_eigvecs,
                    protected_sql_embeddings,
                    norm_weight,
                    mean_natural_token_norm,
                    qk_layer_vectors,
                    deltanet_layer_weights,
                    named_token_unembeddings,
                    downstream_mlp_anti_echo,
                } = *payload;
                let resolved_layer = target_layer
                    .or(resp_layer)
                    .or_else(|| num_hidden_layers.map(|n| n.saturating_sub(2)))
                    .unwrap_or(22);
                Ok((
                    resolved_layer,
                    base_intermediate_size,
                    crate::compile::CompilerWeightGeometrySpec {
                        vocab_top_k_eigvecs,
                        protected_sql_embeddings,
                        norm_weight,
                        mean_natural_token_norm,
                        qk_layer_vectors,
                        deltanet_layer_weights,
                        named_token_unembeddings,
                        downstream_mlp_anti_echo,
                    },
                ))
            }
            other => bail!("expected WeightGeometry response, got {:?}", other),
        }
    }

    pub fn capture_payload_activations(
        &mut self,
        wire_prefix: &str,
        payload: &str,
        closing_tag: &str,
        target_layer: usize,
    ) -> Result<Vec<PayloadActivationRow>> {
        match self.query_stdio_worker(&WorkerRequest::CapturePayloadActivations {
            wire_prefix: wire_prefix.to_string(),
            payload: payload.to_string(),
            closing_tag: closing_tag.to_string(),
            target_layer,
        })? {
            WorkerResponse::PayloadActivations { rows } => Ok(rows),
            other => bail!("expected PayloadActivations response, got {:?}", other),
        }
    }

    /// Query the worker for `(base_intermediate_size, CompilerWeightGeometrySpec)` at `target_layer`.
    pub fn extract_compiler_weight_geometry(
        &mut self,
        target_layer: usize,
    ) -> Result<(usize, crate::compile::CompilerWeightGeometrySpec)> {
        let (_resolved_layer, base_intermediate_size, geom) =
            self.extract_compiler_weight_geometry_for_optional_layer(Some(target_layer))?;
        Ok((base_intermediate_size, geom))
    }

    /// Capture `(pre_mlp, residual)` at `<|im_start|>assistant\n<think>\n[MODE:` on `target_layer`,
    /// force `" <ORACLE_MODE>]\nPlan:"`, and generate genuine on-policy `Plan:` tokens under `Base`.
    pub fn capture_and_generate_on_policy_plan(
        &mut self,
        wire_prefix: &str,
        oracle_mode: Mode,
        target_layer: usize,
        max_plan_tokens: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, String)> {
        let (pre_mlp, residual, plan, _) = self
            .capture_and_generate_on_policy_plan_with_background(
                wire_prefix,
                oracle_mode,
                target_layer,
                max_plan_tokens,
            )?;
        Ok((pre_mlp, residual, plan))
    }

    pub fn capture_and_generate_on_policy_plan_with_background(
        &mut self,
        wire_prefix: &str,
        oracle_mode: Mode,
        target_layer: usize,
        max_plan_tokens: usize,
    ) -> Result<OnPolicyCaptureWithBackground> {
        match self.query_stdio_worker(&WorkerRequest::CaptureAndGenerateOnPolicyPlan {
            wire_prefix: wire_prefix.to_string(),
            oracle_mode: oracle_mode.as_str().to_string(),
            target_layer: Some(target_layer),
            max_plan_tokens: max_plan_tokens.max(1),
        })? {
            WorkerResponse::OnPolicyPlanCaptured {
                pre_mlp,
                residual,
                generated_plan_line,
                router_background,
                ..
            } => Ok((pre_mlp, residual, generated_plan_line, router_background)),
            other => bail!("expected OnPolicyPlanCaptured response, got {:?}", other),
        }
    }

    pub fn step_one_token_with_grammar(
        &mut self,
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
        allowed_token_ids: Option<Vec<u32>>,
        grammar_spec: Option<portia_core::GrammarCartridgeSpec>,
    ) -> Result<(u32, String, Vec<u32>)> {
        let (token_id, token_piece, resolved_ids) =
            self.sample_one_token_with_grammar(bank, pen, phase, allowed_token_ids, grammar_spec)?;
        for &tid in &resolved_ids {
            self.consume_sampled_token(bank, pen, tid)?;
        }
        Ok((token_id, token_piece, resolved_ids))
    }

    /// Return the active [`portia_core::SpanCartridge`] (`(weight_bank, grammar_spec)`) at the
    /// current [`RelayMachine`] phase using `self.active_tables` (falling back to manifest
    /// `mode_grammars` when the caller grammar is `Unconstrained` in `PayloadBody`).
    pub fn active_span_cartridge(&self, relay: &RelayMachine) -> portia_core::SpanCartridge {
        let mut span_cart = relay.active_span_cartridge(&portia_sql::SqliteSchemaGrammarCartridge(
            &self.active_tables,
        ));
        if relay.phase() == RelayPhase::PayloadBody
            && matches!(
                span_cart.grammar,
                portia_core::GrammarCartridgeSpec::Unconstrained
            )
        {
            if let Some(mode) = relay.selected_mode() {
                if let Some(manifest_spec) = self.manifest_mode_grammars.get(&mode) {
                    span_cart.grammar = manifest_spec.clone();
                }
            }
        }
        span_cart
    }

    /// Drive a full researcher turn (`TurnOpenPrefill` -> `ModeLabel` -> `PostModePrefill` ->
    /// `PlanBody` -> `PostPlanPrefill` -> `PayloadBody` -> `TurnComplete`) against the live
    /// stdio worker subprocess, enforcing `|ActivePen(t)| == 1` at every token step!
    pub fn generate_turn(&mut self, max_tokens_per_turn: usize) -> Result<CompletedRelayTurn> {
        self.generate_turn_streaming(max_tokens_per_turn, |_| {})
    }

    /// Drive a full researcher turn while emitting a [`RelayStreamEvent`] on every prefill span
    /// and stepped trunk token for real-time TUI and CLI rendering.
    pub fn generate_turn_streaming(
        &mut self,
        max_tokens_per_turn: usize,
        on_event: impl FnMut(&RelayStreamEvent),
    ) -> Result<CompletedRelayTurn> {
        let tables = self.active_tables.clone();
        self.generate_turn_streaming_with_grammar(
            max_tokens_per_turn,
            &portia_sql::SqliteSchemaGrammarCartridge(&tables),
            on_event,
        )
    }

    /// Drive a full researcher turn under an explicit [`portia_core::PayloadGrammarAutomaton`]
    /// (`SpanCartridge` mounting once per trunk [`RelayPhase`] and resetting across `HarnessPen`
    /// spans) while emitting a [`RelayStreamEvent`] on every prefill span and stepped trunk token.
    pub fn generate_turn_streaming_with_grammar<
        G: portia_core::PayloadGrammarAutomaton + ?Sized,
    >(
        &mut self,
        max_tokens_per_turn: usize,
        grammar: &G,
        on_event: impl FnMut(&RelayStreamEvent),
    ) -> Result<CompletedRelayTurn> {
        self.generate_turn_streaming_with_grammar_cancellable(
            max_tokens_per_turn,
            grammar,
            None,
            on_event,
        )
    }

    /// Drive a full researcher turn like [`Self::generate_turn_streaming_with_grammar`], checking
    /// `cancel_requested` at every prefill and token-sampling boundary so interactive UIs (`Esc` /
    /// `Ctrl+C`) can abort an in-flight turn cleanly without terminating the worker process.
    pub fn generate_turn_streaming_with_grammar_cancellable<
        G: portia_core::PayloadGrammarAutomaton + ?Sized,
    >(
        &mut self,
        max_tokens_per_turn: usize,
        grammar: &G,
        cancel_requested: Option<&std::sync::atomic::AtomicBool>,
        mut on_event: impl FnMut(&RelayStreamEvent),
    ) -> Result<CompletedRelayTurn> {
        use std::sync::atomic::Ordering;

        let mut relay = RelayMachine::new();
        let mut steps = 0usize;
        let mut partial_text = String::new();
        let mut step_token_id_overrides: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut grammar_mounted_for_phase: Option<RelayPhase> = None;

        while relay.phase() != RelayPhase::TurnComplete {
            if cancel_requested.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                return Err(SessionRunError::Cancelled {
                    tokens_generated: steps,
                    phase_at_cancel: format!("{:?}", relay.phase()),
                }
                .into());
            }
            if relay.active_pen() == PenOwner::HarnessPen {
                grammar_mounted_for_phase = None;
                let bank = relay.active_cartridge_bank();
                let selected_mode = relay.selected_mode();
                let mut err_slot: Option<anyhow::Error> = None;
                let maybe_span = relay.flush_harness_prefill(|text| {
                    partial_text.push_str(text);
                    let chunk = LotlNormalizedWireChunk::normalize(text, selected_mode);
                    match self.prefill_chunk_typed(
                        bank,
                        PenOwner::HarnessPen,
                        chunk,
                        KvPrefillDisposition::AppendIntraTurnKv,
                    ) {
                        Ok(tids) => tids,
                        Err(e) => {
                            err_slot = Some(e);
                            vec![]
                        }
                    }
                })?;
                if let Some(e) = err_slot {
                    return Err(e);
                }
                if let Some(span) = maybe_span {
                    on_event(&RelayStreamEvent::HarnessPrefillSpan { span });
                }
            } else {
                if steps >= max_tokens_per_turn {
                    return Err(SessionRunError::TokenBudgetExceeded {
                        tokens_generated: steps,
                        token_budget: max_tokens_per_turn,
                        phase_at_exhaustion: format!("{:?}", relay.phase()),
                        partial_text,
                    }
                    .into());
                }
                steps += 1;
                let producer_bank = relay.active_cartridge_bank();
                let pen = relay.active_pen();
                let phase = relay.phase();
                let mut span_cart = relay.active_span_cartridge(grammar);
                if phase == RelayPhase::PayloadBody
                    && matches!(
                        span_cart.grammar,
                        portia_core::GrammarCartridgeSpec::Unconstrained
                    )
                {
                    if let Some(mode) = relay.selected_mode() {
                        if let Some(manifest_spec) = self.manifest_mode_grammars.get(&mode) {
                            span_cart.grammar = manifest_spec.clone();
                        }
                    }
                }
                let grammar_spec = if grammar_mounted_for_phase != Some(phase) {
                    grammar_mounted_for_phase = Some(phase);
                    Some(span_cart.grammar)
                } else {
                    None
                };
                let (tid, piece, token_ids) = self.sample_one_token_with_grammar(
                    producer_bank,
                    pen,
                    phase,
                    None,
                    grammar_spec,
                )?;
                if cancel_requested.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                    return Err(SessionRunError::Cancelled {
                        tokens_generated: steps,
                        phase_at_cancel: format!("{:?}", relay.phase()),
                    }
                    .into());
                }
                partial_text.push_str(&piece);
                let mut planned = relay.plan_sampled_trunk_token(tid, &piece)?;
                if phase == RelayPhase::PayloadBody
                    && matches!(planned.consume_bank, CartridgeBank::ModePayload(_))
                    && relay.selected_mode().is_some_and(|m| {
                        payload_ends_with_close_tag_prefix(relay.payload_buf(), m.route_tag())
                    })
                {
                    planned.consume_bank = CartridgeBank::Base;
                }
                self.consume_sampled_token(planned.consume_bank, planned.span.pen, tid)?;
                if !token_ids.is_empty() {
                    planned.span.token_ids = token_ids.clone();
                    step_token_id_overrides.push((planned.span.step_index, token_ids));
                }
                on_event(&RelayStreamEvent::TrunkTokenStepped {
                    phase_after: planned.phase_after,
                    selected_mode: relay.selected_mode(),
                    piece: piece.clone(),
                    span: planned.span,
                    plan_so_far: relay.plan_buf().to_string(),
                    payload_so_far: relay.payload_buf().to_string(),
                });
            }
        }

        let selected_mode = relay.selected_mode();
        let mut close_err: Option<anyhow::Error> = None;
        let mut completed = relay.into_completed_turn(|text| {
            let chunk = LotlNormalizedWireChunk::normalize(text, selected_mode);
            match self.prefill_chunk_typed(
                CartridgeBank::Base,
                PenOwner::HarnessPen,
                chunk,
                KvPrefillDisposition::AppendIntraTurnKv,
            ) {
                Ok(tids) => tids,
                Err(e) => {
                    close_err = Some(e);
                    vec![]
                }
            }
        })?;
        if let Some(e) = close_err {
            return Err(e);
        }
        for (step_idx, full_tids) in step_token_id_overrides {
            if let Some(span) = completed.spans.get_mut(step_idx) {
                span.token_ids = full_tids;
            }
        }
        if let Some(close_span) = completed.spans.last().cloned() {
            on_event(&RelayStreamEvent::HarnessPrefillSpan { span: close_span });
        }
        Ok(completed)
    }

    /// Format an environment or inquirer observation envelope and prefill it into the worker's
    /// KV / DeltaNet cache under `CartridgeBank::Base`.
    pub fn prefill_observation(
        &mut self,
        _route: RouteTag,
        obs: &DriverObservation,
    ) -> Result<Vec<u32>> {
        let raw_chunk = format_observation_wire_chunk(obs)?;
        let chunk = LotlNormalizedWireChunk::normalize(&raw_chunk, Some(obs.mode));
        self.prefill_chunk_typed(
            CartridgeBank::Base,
            PenOwner::HarnessPen,
            chunk,
            KvPrefillDisposition::AppendIntraTurnKv,
        )
    }
}

impl Drop for StdioQwenWorkerClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Detect when `payload_buf` ends with an in-progress multi-token close-tag prefix (e.g. `"\n</"`,
/// `"\n</sql"`, `"\n</to_"`) outside string literals so `CartridgeBank::ModePayload` detaches to
/// `CartridgeBank::Base` from `close_0` onward rather than waiting until the final `">"` token.
fn payload_ends_with_close_tag_prefix(payload_buf: &str, route: RouteTag) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_backslash = false;
    let bytes = payload_buf.as_bytes();
    let mut i = 0usize;
    let mut last_unquoted_lt_after_line_start: Option<usize> = None;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if prev_backslash {
                prev_backslash = false;
            } else if b == b'\\' {
                prev_backslash = true;
            } else if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'<' if i == 0 || bytes[i - 1] == b'\n' => {
                last_unquoted_lt_after_line_start = Some(i);
            }
            _ => {}
        }
        i += 1;
    }

    let Some(lt_idx) = last_unquoted_lt_after_line_start else {
        return false;
    };
    let tail = &payload_buf[lt_idx..];
    route
        .accepted_close_tags()
        .iter()
        .any(|close_tag| tail.len() < close_tag.len() && close_tag.starts_with(tail))
}

/// Resolve the default cartridge label for the TUI header for a specific `model`, preferring
/// explicit `--cartridge`, `$PORTIA_CARTRIDGE_JSON`, or the model family's canonical unified
/// `v2` cartridge filename (`portia_v3_unified_cartridge_{08b,2b,4b,9b}_base.json`).
pub fn resolve_default_cartridge_label_for_model(model: &str, cartridge: Option<&Path>) -> String {
    if let Some(p) = cartridge {
        if is_unpatched_base_cartridge_sentinel(p) {
            return "none (Base W_0)".to_string();
        }
        return p.display().to_string();
    }
    if let Ok(env_cart) = std::env::var("PORTIA_CARTRIDGE_JSON") {
        let trimmed = env_cart.trim();
        if !trimmed.is_empty() {
            if is_unpatched_base_cartridge_sentinel(Path::new(trimmed)) {
                return "none (Base W_0, env)".to_string();
            }
            return format!("{trimmed} (env)");
        }
    }
    let normalized = normalize_model_identifier(model);
    let family = resolve_model_family(&normalized, None).ok().flatten();
    if let Some(default_file) = default_cartridge_filename(family.as_deref()) {
        format!("{default_file} (default)")
    } else {
        "unconfigured (pass --cartridge)".to_string()
    }
}

/// Resolve the default cartridge label for the TUI header (defaults to `Qwen/Qwen3.5-0.8B-Base`
/// when model identity is unspecified; prefer [`resolve_default_cartridge_label_for_model`]).
pub fn resolve_default_cartridge_label(cartridge: Option<&Path>) -> String {
    resolve_default_cartridge_label_for_model("Qwen/Qwen3.5-0.8B-Base", cartridge)
}

fn candidate_asset_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(env_home) = std::env::var("PORTIA_HOME") {
        if !env_home.trim().is_empty() {
            roots.push(PathBuf::from(env_home.trim()));
        }
    }
    roots.push(PathBuf::from("."));
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1).take(4) {
            roots.push(ancestor.to_path_buf());
        }
    }
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    roots.push(workspace_root);
    roots
}

/// Resolve structured `(python_executable, argv)` for spawning `scripts/portia_qwen_stdio_worker.py`.
/// Honors `PORTIA_PYTHON`, `PORTIA_WORKER_SCRIPT`, and `PORTIA_CARTRIDGE_JSON` environment variables,
/// then searches current working directory, executable-relative ancestors, and the build workspace.
pub fn resolve_default_worker_argv(
    model: &str,
    cartridge: Option<&Path>,
) -> Result<(String, Vec<String>)> {
    resolve_worker_argv_with_options(model, cartridge, &WorkerLaunchOptions::default())
}

fn resolve_worker_argv_with_options(
    model: &str,
    cartridge: Option<&Path>,
    options: &WorkerLaunchOptions,
) -> Result<(String, Vec<String>)> {
    let roots = candidate_asset_roots();
    let normalized_model = normalize_model_identifier(model);
    let family = resolve_model_family(&normalized_model, options.model_family.as_deref())?;

    let python_bin = std::env::var("PORTIA_PYTHON")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            roots
                .iter()
                .map(|r| r.join(".venv/bin/python"))
                .find(|p| p.is_file())
                .map(|p| p.to_string_lossy().into_owned())
        })
        .or_else(|| {
            std::env::var("VIRTUAL_ENV")
                .ok()
                .map(|v| PathBuf::from(v.trim()).join("bin/python"))
                .filter(|p| p.is_file())
                .map(|p| p.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "python3".to_string());

    let worker_script = std::env::var("PORTIA_WORKER_SCRIPT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            roots
                .iter()
                .map(|r| r.join("scripts/portia_qwen_stdio_worker.py"))
                .find(|p| p.is_file())
        })
        .unwrap_or_else(|| PathBuf::from("scripts/portia_qwen_stdio_worker.py"));

    let mut args = vec![
        worker_script.to_string_lossy().into_owned(),
        "--model-path".to_string(),
        normalized_model.clone(),
    ];

    let explicit = cartridge.map(Path::to_path_buf).or_else(|| {
        std::env::var("PORTIA_CARTRIDGE_JSON")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(|s| PathBuf::from(s.trim()))
    });
    let resolved_cart = explicit
        .or_else(|| {
            default_cartridge_filename(family.as_deref()).and_then(|name| {
                roots
                    .iter()
                    .map(|r| r.join("artifacts").join(name))
                    .find(|p| p.is_file())
            })
        })
        .ok_or_else(|| ModelIdentityError::MissingCartridge {
            model: normalized_model.clone(),
        })?;
    if !is_unpatched_base_cartridge_sentinel(&resolved_cart) && !resolved_cart.is_file() {
        return Err(ModelIdentityError::MissingCartridge {
            model: normalized_model,
        }
        .into());
    }
    args.extend([
        "--cartridge-json".to_string(),
        resolved_cart.to_string_lossy().into_owned(),
    ]);
    if let Some(family) = &options.model_family {
        args.extend(["--model-family".into(), normalize_model_identifier(family)]);
    }
    if options.allow_cartridge_model_mismatch {
        args.push("--allow-cartridge-model-mismatch".into());
    }
    Ok((python_bin, args))
}

/// Resolve the turnkey command-line string representation for `scripts/portia_qwen_stdio_worker.py`.
pub fn resolve_default_worker_command(model: &str, cartridge: Option<&Path>) -> Result<String> {
    let (python_bin, args) = resolve_default_worker_argv(model, cartridge)?;
    let mut parts = vec![python_bin];
    for a in args {
        if a.starts_with("--") {
            parts.push(a);
        } else {
            parts.push(format!("{:?}", a));
        }
    }
    Ok(parts.join(" "))
}
