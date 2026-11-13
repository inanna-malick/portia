//! Live Rust AST Daily-Driver (`LiveRustAstDailyDriver`) wrapping [`UnifiedRustAstScenarioSession`],
//! real `tree-sitter` + `tree-sitter-rust` + `syn` AST provenance,
//! optional on-disk `.rs` file synchronization, `rustc --crate-type lib` compilation verification,
//! and Boyd OODA `CoroutineStepEffect` classification.

use crate::driver::{
    classify_ask_prompt_kind, ensure_none_of_above_escape_choice, extract_ask_interaction_mode,
    parse_ask_user_reply_for_kind, AskDecision, AskInteractionMode, AskPromptKind, AskUserReply,
    AstOperation, CoroutineStepEffect, DriverObservation, EpistemicDomainEnv, ObservationOperation,
    NONE_OF_ABOVE_ENTER_TEXT_CHOICE,
};
use anyhow::{bail, Context, Result};
use portia_core::relay::TypedTurnAction;
use portia_core::{Mode, StewardAuthState};
use portia_nursery::nursery_scenario::{
    AppType, CleanroomRustHyloSpec, DialogicCodingAffordance, DomainSkin, HyloCartridgeLevel,
    HyloCartridgePhase, RustHyloPipelineStage, RustTodoHoleSpec, ScenarioCoordinate,
    ScenarioEpisodeSummary, ScenarioLanguage,
};
use portia_nursery::roguelite_lattice::RogueliteStressVector;
use portia_nursery::rust_ast_lens::{
    extract_rust_todo_holes, format_rust_ast_outline, hylo_octo_merge_rust_holes,
    parse_and_validate_rust_cst_and_ast, slice_rust_fn_window, splice_rust_todo_hole_body,
    ExtractedRustTodoHole, RustAstHoleGrammarCartridge, RustAstTurnOutcome,
    UnifiedRustAstScenarioSession,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Parse a [`DomainSkin`] from a flexible CLI token (`bgp`, `alpine`, `orbital`, `solaris`, etc.).
pub fn parse_domain_skin_flexible(raw: &str) -> Result<DomainSkin> {
    let low = raw.trim().to_ascii_lowercase();
    match low.as_str() {
        "bgp" | "bgp_noc" | "bgpnoc" | "noc" => Ok(DomainSkin::BgpNoc),
        "alpine" | "alpine_levain" | "alpinelevain" | "levain" => Ok(DomainSkin::AlpineLevain),
        "orbital" | "orbital_constellation" | "orbitalconstellation" => {
            Ok(DomainSkin::OrbitalConstellation)
        }
        "solaris" | "solaris_vault" | "solarisvault" | "vault" => Ok(DomainSkin::SolarisVault),
        _ => bail!(
            "unrecognized DomainSkin {:?}; expected bgp, alpine, orbital, or solaris",
            raw
        ),
    }
}

/// Parse a [`ScenarioLanguage`] from a flexible CLI token (`en`, `ja`, `ru`, `es`, `de`, `id`, etc.).
pub fn parse_scenario_language_flexible(raw: &str) -> Result<ScenarioLanguage> {
    let low = raw.trim().to_ascii_lowercase();
    match low.as_str() {
        "en" | "eng" | "english" => Ok(ScenarioLanguage::English),
        "ja" | "jp" | "japanese" => Ok(ScenarioLanguage::Japanese),
        "ru" | "rus" | "russian" => Ok(ScenarioLanguage::Russian),
        "es" | "spa" | "spanish" => Ok(ScenarioLanguage::Spanish),
        "de" | "deu" | "ger" | "german" => Ok(ScenarioLanguage::German),
        "id" | "ind" | "indonesian" => Ok(ScenarioLanguage::Indonesian),
        _ => bail!(
            "unrecognized ScenarioLanguage {:?}; expected en, ja, ru, es, de, or id",
            raw
        ),
    }
}

/// Isolated $O(1)$ leaf execution record captured during the `Fork × N -> Impl::TypeCart -> Impl::ValueCart`
/// stage of [`LiveRustAstDailyDriver::execute_hylo_2x2_fanout_and_octomerge`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyloLeafExecutionRecord {
    pub hole_index: usize,
    pub fn_name: String,
    pub signature: String,
    pub intent_message: String,
    pub leaf_prompt_window: String,
    pub replacement_body: String,
    pub leaf_window_bytes: usize,
    pub leaf_window_bpe_tokens: usize,
    pub leaf_ast_verified: bool,
}

/// Complete execution receipt for the $2 \times 2$ (`Scaffold ⊥ Impl` $\times$ `TypeCart ⊥ ValueCart`)
/// hylomorphic fanout and octo-merge runner (`portia-cli hylo`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyloFanoutExecutionReceipt {
    pub coordinate: ScenarioCoordinate,
    pub module_name: String,
    pub ast_outline_before: String,
    pub ast_outline_after: String,
    pub stages_executed: Vec<RustHyloPipelineStage>,
    pub leaf_contexts: Vec<HyloLeafExecutionRecord>,
    pub merged_source: String,
    pub rustc_verified: bool,
    pub episode_summary: ScenarioEpisodeSummary,
}

/// Live Rust AST Daily-Driver environment for multi-step `Probe -> Read -> Ask -> Write -> Result`
/// hylomorphic coding episodes over real `tree-sitter-rust` CSTs and `syn` ASTs.
pub struct LiveRustAstDailyDriver {
    session: UnifiedRustAstScenarioSession,
    file_path: Option<PathBuf>,
    is_live_workspace_file: bool,
    verify_rustc_on_result: bool,
    event_counter: u64,
    ast_version: u64,
    fn_versions: BTreeMap<String, u64>,
    task_reads_executed: u64,
    task_writes_committed: u64,
    steward_auth: StewardAuthState,
    last_turn_outcome: Option<RustAstTurnOutcome>,
}

impl LiveRustAstDailyDriver {
    /// Construct a [`LiveRustAstDailyDriver`] from a [`CleanroomRustHyloSpec`].
    pub fn from_spec(spec: CleanroomRustHyloSpec) -> Result<Self> {
        let holes = extract_rust_todo_holes(&spec.ast_valid_token_zero_skeleton)?;
        let mut fn_versions = BTreeMap::new();
        for h in &holes {
            fn_versions.insert(h.fn_name.clone(), 0);
        }
        let session = UnifiedRustAstScenarioSession::new(spec)?;
        Ok(Self {
            session,
            file_path: None,
            is_live_workspace_file: false,
            verify_rustc_on_result: false,
            event_counter: 0,
            ast_version: 0,
            fn_versions,
            task_reads_executed: 0,
            task_writes_committed: 0,
            steward_auth: StewardAuthState::NotRequired,
            last_turn_outcome: None,
        })
    }

    /// Construct a [`LiveRustAstDailyDriver`] from raw Rust source code (validated immediately via
    /// `tree-sitter-rust` CST + `syn` AST) and an optional on-disk `.rs` file path.
    pub fn from_rust_source(
        module_name: &str,
        rust_source: &str,
        file_path: Option<PathBuf>,
    ) -> Result<Self> {
        parse_and_validate_rust_cst_and_ast(rust_source)?;
        let extracted = extract_rust_todo_holes(rust_source)?;
        let todo_holes: Vec<RustTodoHoleSpec> = extracted
            .iter()
            .map(|h| RustTodoHoleSpec {
                hole_index: h.hole_index,
                fn_name: h.fn_name.clone(),
                signature: h.signature.clone(),
                todo_stub_literal: format!("todo!(\"{}\")", h.intent_message),
                invariant_contract: h.doc_comments.join(" "),
            })
            .collect();
        let hole_names: Vec<String> = todo_holes.iter().map(|h| h.fn_name.clone()).collect();
        let hole_count = todo_holes.len();
        let spec = CleanroomRustHyloSpec {
            coordinate: ScenarioCoordinate::new(
                AppType::RustAstHylo,
                ScenarioLanguage::English,
                DomainSkin::SolarisVault,
                RogueliteStressVector::sweet_spot_baseline(),
                0x3600_0001,
            ),
            module_name: module_name.to_string(),
            trait_name: DomainSkin::SolarisVault.primary_rust_trait().to_string(),
            state_struct_name: format!("{}State", DomainSkin::SolarisVault.primary_rust_trait()),
            ast_valid_token_zero_skeleton: rust_source.to_string(),
            todo_holes,
            pipeline_schedule: vec![
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Scaffold,
                    level: HyloCartridgeLevel::TypeCart,
                },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Scaffold,
                    level: HyloCartridgeLevel::ValueCart,
                },
                RustHyloPipelineStage::ForkFanout { hole_count },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Impl,
                    level: HyloCartridgeLevel::TypeCart,
                },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Impl,
                    level: HyloCartridgeLevel::ValueCart,
                },
                RustHyloPipelineStage::OctoMerge {
                    merged_hole_count: hole_count,
                },
            ],
            dialogic_affordances: vec![DialogicCodingAffordance::ReviewScaffoldBeforeFanout {
                todo_hole_names: hole_names,
            }],
            has_contrived_decoy_functions: false,
        };
        let mut drv = Self::from_spec(spec)?;
        drv.is_live_workspace_file = file_path.is_some();
        drv.file_path = file_path;
        Ok(drv)
    }

    /// Open an existing `.rs` source file from disk into a [`LiveRustAstDailyDriver`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_rust_file(path)
    }

    /// Open an existing `.rs` source file from disk into a [`LiveRustAstDailyDriver`].
    pub fn open_rust_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let source = std::fs::read_to_string(p)
            .with_context(|| format!("failed to read Rust source file {:?}", p))?;
        let module_name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("live_rust_module");
        let mut drv = Self::from_rust_source(module_name, &source, Some(p.to_path_buf()))?;
        drv.is_live_workspace_file = true;
        Ok(drv)
    }

    /// Whether this driver is attached to a live workspace `.rs` file (`true` on `open` /
    /// `open_rust_file`, `false` on cleanroom nursery `from_spec`).
    pub const fn is_live_workspace_file(&self) -> bool {
        self.is_live_workspace_file
    }

    /// Enable or disable running `rustc --crate-type lib --edition 2021` on `Mode::Result`.
    pub fn with_rustc_verification(mut self, enabled: bool) -> Self {
        self.verify_rustc_on_result = enabled;
        self
    }

    /// Reset task-local read/write counters at the start of a new user task in a multi-task session.
    pub fn begin_new_task(&mut self) {
        self.task_reads_executed = 0;
        self.task_writes_committed = 0;
        self.steward_auth = StewardAuthState::NotRequired;
    }

    pub const fn steward_auth(&self) -> StewardAuthState {
        self.steward_auth
    }

    /// Current live Rust source code (including all spliced `todo!("...")` holes so far).
    pub fn current_source(&self) -> &str {
        &self.session.current_source
    }

    /// Underlying [`UnifiedRustAstScenarioSession`].
    pub fn session(&self) -> &UnifiedRustAstScenarioSession {
        &self.session
    }

    /// Most recent [`RustAstTurnOutcome`] from [`UnifiedRustAstScenarioSession::step_turn`].
    pub fn last_turn_outcome(&self) -> Option<&RustAstTurnOutcome> {
        self.last_turn_outcome.as_ref()
    }

    /// Extract all remaining `todo!("...")` holes from the current live Rust source via `tree-sitter-rust`.
    pub fn remaining_todo_holes(&self) -> Result<Vec<ExtractedRustTodoHole>> {
        Ok(extract_rust_todo_holes(&self.session.current_source)?)
    }

    /// Render the compact `[AST ...]` structural outline of the current live Rust source via `tree-sitter-rust`.
    pub fn format_ast_outline(&self) -> Result<String> {
        Ok(format_rust_ast_outline(
            &self.session.spec.module_name,
            &self.session.current_source,
        )?)
    }

    /// Compile the current Rust source with `rustc --test --edition 2021` (passing `--cfg approved_crate`
    /// only when `self.session.requested_ecosystem_crate` is `true`, enforcing Wadler's
    /// Free Theorem via the sealed `VerifiedDomainOutput` witness type) and execute the compiled
    /// test binary on the cleanroom invariant test suite.
    pub fn verify_with_rustc(&self) -> Result<()> {
        let tmp_dir = tempfile::tempdir().context("failed to create tempdir for rustc check")?;
        let src_path = tmp_dir.path().join("lib.rs");
        let bin_path = tmp_dir.path().join("cleanroom_test_bin");
        std::fs::write(&src_path, &self.session.current_source)
            .context("failed to write temporary lib.rs for rustc check")?;
        let mut cmd = Command::new("rustc");
        cmd.arg("--test")
            .arg("--edition")
            .arg("2021")
            .arg("-A")
            .arg("dead_code")
            .arg("-A")
            .arg("unused_variables");
        if self.session.requested_ecosystem_crate {
            cmd.arg("--cfg").arg("approved_crate");
        }
        let output = cmd
            .arg(&src_path)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .context("failed to invoke `rustc --test` for AST/type verification")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("rustc --test type/borrow check failed:\n{}", stderr.trim());
        }
        let test_run = Command::new(&bin_path)
            .output()
            .context("failed to execute compiled `rustc --test` binary")?;
        if !test_run.status.success() {
            let stdout = String::from_utf8_lossy(&test_run.stdout);
            let stderr = String::from_utf8_lossy(&test_run.stderr);
            bail!(
                "cleanroom `rustc --test` suite failed:\n{}\n{}",
                stdout.trim(),
                stderr.trim()
            );
        }
        Ok(())
    }

    fn next_event_id(&mut self) -> String {
        self.event_counter += 1;
        format!("ast-obs-{:04}", self.event_counter)
    }

    /// Human-readable candidate choices for remaining `todo!("...")` holes in the module.
    pub fn remaining_hole_choices(&self) -> Vec<String> {
        self.remaining_todo_holes()
            .unwrap_or_default()
            .into_iter()
            .map(|h| format!("fn {} — todo!(\"{}\")", h.fn_name, h.intent_message))
            .collect()
    }

    /// Return whether any dialogic `[MODE: ASK]` affordance on the session remains pending.
    pub fn has_pending_dialogic_affordances(&self) -> bool {
        !self.session.asked_clarification
            || !self.session.requested_ecosystem_crate
            || !self.session.reviewed_scaffold_before_fanout
    }

    /// Determine the [`AskInteractionMode`] for `question` purely from AST session state and
    /// structured JSON widget syntax (`∀prompt`: zero `user_prompt` or `plan` peeking).
    pub fn classify_ask_interaction_mode(&self, question: &str) -> AskInteractionMode {
        if let Some(mode) = extract_ask_interaction_mode(question) {
            return mode;
        }
        if self.steward_auth.requires_ask()
            || classify_ask_prompt_kind(question) == AskPromptKind::Permission
        {
            return AskInteractionMode::StewardPermissionChoice {
                question: question.trim().to_string(),
                choices: vec![
                    "ALLOW — Approve Rust AST operation / ecosystem crate".to_string(),
                    "DENY  — Reject and cancel operation".to_string(),
                    NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string(),
                ],
            };
        }
        let hole_choices = self.remaining_hole_choices();
        if self.session.probed_ast_outline && hole_choices.len() >= 2 {
            return AskInteractionMode::MultiSelect {
                question: question.trim().to_string(),
                choices: ensure_none_of_above_escape_choice(hole_choices),
            };
        }
        AskInteractionMode::ConversationalClarify
    }

    /// Classify a completed turn into its [`CoroutineStepEffect`] (`AutoContinueDb`,
    /// `SuspendForGuiChoice`, or `YieldToPrompt`) purely from `Mode` and AST observation state.
    pub fn classify_coroutine_step_effect(&self, mode: Mode, payload: &str) -> CoroutineStepEffect {
        match mode {
            Mode::Probe | Mode::Read | Mode::Write => CoroutineStepEffect::AutoContinueDb,
            Mode::Result | Mode::Abstain => CoroutineStepEffect::YieldToPrompt,
            Mode::Ask => {
                let gui_mode = self.classify_ask_interaction_mode(payload);
                match gui_mode {
                    AskInteractionMode::StewardPermissionChoice { .. }
                    | AskInteractionMode::MultiSelect { .. } => {
                        CoroutineStepEffect::SuspendForGuiChoice {
                            lead_in_prose: payload.trim().to_string(),
                            gui_mode,
                        }
                    }
                    AskInteractionMode::ConversationalClarify => CoroutineStepEffect::YieldToPrompt,
                }
            }
        }
    }

    /// Execute a semantic [`TypedTurnAction`] against the live Rust AST module.
    pub fn execute_typed_action(
        &mut self,
        action: &TypedTurnAction,
        ask_callback: impl FnMut(&str) -> Result<String>,
    ) -> DriverObservation {
        match action {
            TypedTurnAction::ProbeSchema { sql } => {
                self.execute_turn(Mode::Probe, sql, ask_callback)
            }
            TypedTurnAction::ReadQuery { sql } => self.execute_turn(Mode::Read, sql, ask_callback),
            TypedTurnAction::WriteMutation { sql } => {
                self.execute_turn(Mode::Write, sql, ask_callback)
            }
            TypedTurnAction::AskSteward { question } => {
                self.execute_turn(Mode::Ask, question, ask_callback)
            }
            TypedTurnAction::SubmitResult { report_json } => {
                self.execute_turn(Mode::Result, &report_json.to_string(), ask_callback)
            }
            TypedTurnAction::AbstainTask { abstain_json } => {
                self.execute_turn(Mode::Abstain, &abstain_json.to_string(), ask_callback)
            }
        }
    }

    /// Execute one OODA turn (`Probe`, `Read`, `Ask`, `Write`, `Result`, `Abstain`) against the
    /// live Rust AST session, updating on-disk source (if configured) and returning a [`DriverObservation`].
    pub fn execute_turn(
        &mut self,
        mode: Mode,
        payload: &str,
        mut ask_callback: impl FnMut(&str) -> Result<String>,
    ) -> DriverObservation {
        let event_id = self.next_event_id();
        match mode {
            Mode::Probe => {
                self.task_reads_executed += 1;
                let outcome = self.session.step_turn(Mode::Probe, payload);
                let ok = outcome.ok;
                let outline = outcome.tool_response.clone().unwrap_or_default();
                let err = outcome.error_message.clone();
                self.last_turn_outcome = Some(outcome);
                DriverObservation {
                    event_id,
                    mode: Mode::Probe,
                    ok,
                    db_version: self.ast_version,
                    table_versions: self.fn_versions.clone(),
                    columns: vec!["module".to_string(), "ast_outline".to_string()],
                    rows: if ok {
                        vec![vec![
                            serde_json::Value::String(self.session.spec.module_name.clone()),
                            serde_json::Value::String(outline.clone()),
                        ]]
                    } else {
                        vec![]
                    },
                    rows_affected: usize::from(ok),
                    message: if ok { Some(outline) } else { None },
                    error: err,
                    operation: Some(ObservationOperation::Ast(AstOperation::Probe)),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Mode::Read => {
                self.task_reads_executed += 1;
                let outcome = self.session.step_turn(Mode::Read, payload);
                let ok = outcome.ok;
                let window = outcome.tool_response.clone().unwrap_or_default();
                let err = outcome.error_message.clone();
                self.last_turn_outcome = Some(outcome);
                DriverObservation {
                    event_id,
                    mode: Mode::Read,
                    ok,
                    db_version: self.ast_version,
                    table_versions: self.fn_versions.clone(),
                    columns: vec!["fn_name".to_string(), "fn_window".to_string()],
                    rows: if ok {
                        vec![vec![
                            serde_json::Value::String(payload.trim().to_string()),
                            serde_json::Value::String(window.clone()),
                        ]]
                    } else {
                        vec![]
                    },
                    rows_affected: usize::from(ok),
                    message: if ok { Some(window) } else { None },
                    error: err,
                    operation: Some(ObservationOperation::Ast(AstOperation::ReadFn)),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Mode::Ask => {
                let question = payload.trim();
                let user_reply = match ask_callback(question) {
                    Ok(r) => r,
                    Err(e) => {
                        return DriverObservation {
                            event_id,
                            mode: Mode::Ask,
                            ok: false,
                            db_version: self.ast_version,
                            table_versions: self.fn_versions.clone(),
                            columns: vec![],
                            rows: vec![],
                            rows_affected: 0,
                            message: None,
                            error: Some(e.to_string()),
                            operation: Some(ObservationOperation::Ast(AstOperation::Ask)),
                            ask_reply: None,
                            steward_auth: self.steward_auth,
                            is_chat_continuation: false,
                        };
                    }
                };
                let parsed_decision =
                    parse_ask_user_reply_for_kind(AskPromptKind::Permission, &user_reply).ok();
                if let Some(AskUserReply::Decision {
                    decision: AskDecision::Deny,
                    raw_reply,
                }) = parsed_decision
                {
                    self.steward_auth = self.steward_auth.record_decision(false);
                    return DriverObservation {
                        event_id,
                        mode: Mode::Ask,
                        ok: true,
                        db_version: self.ast_version,
                        table_versions: self.fn_versions.clone(),
                        columns: vec!["clarification".to_string(), "status".to_string()],
                        rows: vec![vec![
                            serde_json::Value::String(raw_reply.clone()),
                            serde_json::Value::String("DENIED".to_string()),
                        ]],
                        rows_affected: 0,
                        message: Some(raw_reply.clone()),
                        error: None,
                        operation: Some(ObservationOperation::Ast(AstOperation::Ask)),
                        ask_reply: Some(AskUserReply::Decision {
                            decision: AskDecision::Deny,
                            raw_reply,
                        }),
                        steward_auth: self.steward_auth,
                        is_chat_continuation: false,
                    };
                }
                if matches!(
                    parsed_decision,
                    Some(AskUserReply::Decision {
                        decision: AskDecision::Allow,
                        ..
                    })
                ) {
                    self.steward_auth = self.steward_auth.record_decision(true);
                }
                let outcome = if self.is_live_workspace_file {
                    self.session.record_live_user_clarification(question)
                } else {
                    self.session.step_turn(Mode::Ask, question)
                };
                let sim_msg = outcome.tool_response.clone().unwrap_or_default();
                self.last_turn_outcome = Some(outcome);
                let combined_reply = if self.is_live_workspace_file {
                    user_reply.clone()
                } else if user_reply.trim().is_empty() {
                    sim_msg
                } else if sim_msg.is_empty() {
                    user_reply.clone()
                } else {
                    format!("{} ({})", user_reply.trim(), sim_msg)
                };
                let parsed_ask =
                    parse_ask_user_reply_for_kind(AskPromptKind::Permission, &user_reply)
                        .unwrap_or_else(|_| AskUserReply::Clarification {
                            clarification: combined_reply.clone(),
                        });
                DriverObservation {
                    event_id,
                    mode: Mode::Ask,
                    ok: true,
                    db_version: self.ast_version,
                    table_versions: self.fn_versions.clone(),
                    columns: vec!["clarification".to_string(), "status".to_string()],
                    rows: vec![vec![
                        serde_json::Value::String(combined_reply.clone()),
                        serde_json::Value::String("CLARIFIED".to_string()),
                    ]],
                    rows_affected: 1,
                    message: Some(combined_reply),
                    error: None,
                    operation: Some(ObservationOperation::Ast(AstOperation::Ask)),
                    ask_reply: Some(parsed_ask),
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Mode::Write => {
                let prev_source = self.session.current_source.clone();
                let outcome = self.session.step_turn(Mode::Write, payload);
                let ok = outcome.ok;
                let msg = outcome.tool_response.clone();
                let err = outcome.error_message.clone();
                let remaining = outcome.remaining_todo_holes;
                self.last_turn_outcome = Some(outcome);
                if ok {
                    if let Some(path) = &self.file_path {
                        if let Err(io_err) = std::fs::write(path, &self.session.current_source) {
                            self.session.current_source = prev_source;
                            return DriverObservation {
                                event_id,
                                mode: Mode::Write,
                                ok: false,
                                db_version: self.ast_version,
                                table_versions: self.fn_versions.clone(),
                                columns: vec![],
                                rows: vec![],
                                rows_affected: 0,
                                message: None,
                                error: Some(format!(
                                    "failed to sync spliced Rust AST to {:?}: {}",
                                    path, io_err
                                )),
                                operation: Some(ObservationOperation::Ast(
                                    AstOperation::SpliceHole,
                                )),
                                ask_reply: None,
                                steward_auth: self.steward_auth,
                                is_chat_continuation: false,
                            };
                        }
                    }
                    self.ast_version += 1;
                    self.task_writes_committed += 1;
                    if let Some((clean_fn, _)) =
                        UnifiedRustAstScenarioSession::parse_write_payload(payload)
                    {
                        let entry = self.fn_versions.entry(clean_fn).or_insert(0);
                        *entry += 1;
                    }
                }
                DriverObservation {
                    event_id,
                    mode: Mode::Write,
                    ok,
                    db_version: self.ast_version,
                    table_versions: self.fn_versions.clone(),
                    columns: vec!["status".to_string(), "remaining_todo_holes".to_string()],
                    rows: if ok {
                        vec![vec![
                            serde_json::Value::String("SPLICED".to_string()),
                            serde_json::Value::from(remaining as u64),
                        ]]
                    } else {
                        vec![]
                    },
                    rows_affected: usize::from(ok),
                    message: msg,
                    error: err,
                    operation: Some(ObservationOperation::Ast(AstOperation::SpliceHole)),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Mode::Result | Mode::Abstain => {
                let mut outcome = self.session.step_turn(mode, payload);
                if mode == Mode::Result && outcome.ok && self.verify_rustc_on_result {
                    if let Err(rustc_err) = self.verify_with_rustc() {
                        outcome.ok = false;
                        outcome.error_message = Some(rustc_err.to_string());
                    }
                }
                let ok = outcome.ok;
                let msg = outcome.tool_response.clone();
                let err = outcome.error_message.clone();
                self.last_turn_outcome = Some(outcome);
                let ast_op = if mode == Mode::Result {
                    AstOperation::Result
                } else {
                    AstOperation::Abstain
                };
                DriverObservation {
                    event_id,
                    mode,
                    ok,
                    db_version: self.ast_version,
                    table_versions: self.fn_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: msg,
                    error: err,
                    operation: Some(ObservationOperation::Ast(ast_op)),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
        }
    }

    /// Execute the full $2 \times 2$ (`Scaffold ⊥ Impl` $\times$ `TypeCart ⊥ ValueCart`) hylomorphic
    /// `Scaffold -> Ask -> Fork × N -> OctoMerge -> Result` pipeline over all `N` `todo!("...")` holes,
    /// verifying each isolated leaf with `tree-sitter-rust` + `syn`, octo-merging via
    /// [`hylo_octo_merge_rust_holes`] (`try_expand_and_collapse` in `hylo.rs`), and verifying the
    /// merged module with `rustc --crate-type lib --edition 2021 --emit=metadata`.
    pub fn execute_hylo_2x2_fanout_and_octomerge(
        &mut self,
        mut leaf_solver: impl FnMut(&ExtractedRustTodoHole, &str, &str) -> Result<String>,
    ) -> Result<HyloFanoutExecutionReceipt> {
        if !self.session.spec.verify_ast_valid_from_token_zero() {
            bail!(
                "module `{}` failed verify_ast_valid_from_token_zero at Scaffold::TypeCart",
                self.session.spec.module_name
            );
        }

        // Step 1: Scaffold::TypeCart -> Scaffold::ValueCart (`Mode::Probe` AST outline)
        let probe_obs = self.execute_turn(Mode::Probe, "inspect rust ast outline", |_| {
            Ok("ALLOW".to_string())
        });
        if !probe_obs.ok {
            bail!(
                "Scaffold Mode::Probe failed: {}",
                probe_obs.error.unwrap_or_default()
            );
        }
        let ast_outline_before = probe_obs.message.unwrap_or_default();

        // Step 2: Dialogic Ask check across `DialogicCodingAffordance`s before Fork × N fanout
        let (approved_crate, _) = self
            .session
            .spec
            .coordinate
            .domain_skin
            .approved_ecosystem_crate();
        let ask_question = format!(
            "Please clarify the ambiguous underflow invariant, approve crate {approved_crate}, and review scaffold before fanout."
        );
        let ask_obs = self.execute_turn(Mode::Ask, &ask_question, |_| {
            Ok(format!(
                "Use saturating floor at 0, approve crate {approved_crate}, and review scaffold."
            ))
        });
        if !ask_obs.ok {
            bail!(
                "Dialogic Mode::Ask failed: {}",
                ask_obs.error.unwrap_or_default()
            );
        }

        // Step 3: ForkFanout { hole_count } -> Impl::TypeCart -> Impl::ValueCart ($O(1)$ per leaf)
        let holes = self.remaining_todo_holes()?;
        let hole_count = holes.len();
        let skeleton_source = self.session.current_source.clone();
        let mut leaf_contexts = Vec::with_capacity(hole_count);
        let mut patches = Vec::with_capacity(hole_count);

        for hole in &holes {
            let read_obs =
                self.execute_turn(Mode::Read, &hole.fn_name, |_| Ok("ALLOW".to_string()));
            if !read_obs.ok {
                bail!(
                    "Leaf Mode::Read failed for `{}`: {}",
                    hole.fn_name,
                    read_obs.error.unwrap_or_default()
                );
            }
            let fn_window = slice_rust_fn_window(&skeleton_source, &hole.fn_name)?;
            let leaf_prompt_window = format!("{ast_outline_before}\n\n{fn_window}");
            let leaf_window_bytes = leaf_prompt_window.len();
            let leaf_window_bpe_tokens = leaf_window_bytes.div_ceil(4);

            let replacement_body = leaf_solver(hole, &ast_outline_before, &fn_window)?;
            // Verify isolated leaf splice via `tree-sitter-rust` + `syn`:
            let isolated_spliced =
                splice_rust_todo_hole_body(&skeleton_source, &hole.fn_name, &replacement_body)?;
            parse_and_validate_rust_cst_and_ast(&isolated_spliced)?;

            // Commit the leaf write to the session provenance tracker:
            let write_payload = format!("{} => {}", hole.fn_name, replacement_body);
            let write_obs =
                self.execute_turn(Mode::Write, &write_payload, |_| Ok("ALLOW".to_string()));
            if !write_obs.ok {
                bail!(
                    "Leaf Mode::Write failed for `{}`: {}",
                    hole.fn_name,
                    write_obs.error.unwrap_or_default()
                );
            }

            leaf_contexts.push(HyloLeafExecutionRecord {
                hole_index: hole.hole_index,
                fn_name: hole.fn_name.clone(),
                signature: hole.signature.clone(),
                intent_message: hole.intent_message.clone(),
                leaf_prompt_window,
                replacement_body: replacement_body.clone(),
                leaf_window_bytes,
                leaf_window_bpe_tokens,
                leaf_ast_verified: true,
            });
            patches.push((hole.fn_name.clone(), replacement_body));
        }

        // Step 4: Hylomorphic OctoMerge (`try_expand_and_collapse` over `RoseFrame`) + `rustc` verification
        let merged_source = hylo_octo_merge_rust_holes(&skeleton_source, &patches)?;
        self.session.current_source = merged_source.clone();
        if let Some(path) = &self.file_path {
            std::fs::write(path, &merged_source)
                .with_context(|| format!("failed to write octo-merged Rust file {:?}", path))?;
        }
        self.verify_with_rustc()?;
        let ast_outline_after = self.format_ast_outline()?;

        let result_obs = self.execute_turn(
            Mode::Result,
            "{\"status\": \"HYLO_2X2_OCTO_MERGED_AND_RUSTC_VERIFIED\"}",
            |_| Ok("ALLOW".to_string()),
        );
        if !result_obs.ok {
            bail!(
                "OctoMerge Mode::Result failed: {}",
                result_obs.error.unwrap_or_default()
            );
        }

        let episode_summary = self
            .last_turn_outcome
            .as_ref()
            .and_then(|o| o.episode_summary.clone())
            .context("missing ScenarioEpisodeSummary after Mode::Result")?;

        let stages_executed = vec![
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::ForkFanout { hole_count },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::OctoMerge {
                merged_hole_count: hole_count,
            },
        ];

        Ok(HyloFanoutExecutionReceipt {
            coordinate: self.session.spec.coordinate,
            module_name: self.session.spec.module_name.clone(),
            ast_outline_before,
            ast_outline_after,
            stages_executed,
            leaf_contexts,
            merged_source,
            rustc_verified: true,
            episode_summary,
        })
    }
}

impl EpistemicDomainEnv for LiveRustAstDailyDriver {
    fn begin_new_task(&mut self) {
        LiveRustAstDailyDriver::begin_new_task(self);
    }

    fn reset_epistemic_session(&mut self) {
        self.begin_new_task();
        self.session.probed_ast_outline = false;
        self.session.inspected_fn_holes.clear();
        self.session.wipeout_diagnoses.clear();
    }

    fn state_version(&self) -> u64 {
        self.ast_version
    }

    fn steward_auth(&self) -> StewardAuthState {
        self.steward_auth
    }

    fn legal_modes(&self) -> Result<Vec<Mode>> {
        if !self.session.probed_ast_outline {
            return Ok(vec![Mode::Probe, Mode::Ask, Mode::Abstain]);
        }
        let holes_empty = self.remaining_todo_holes()?.is_empty();
        let mut modes = vec![Mode::Probe, Mode::Read, Mode::Ask];
        if !holes_empty {
            modes.push(Mode::Write);
        }
        if holes_empty {
            modes.push(Mode::Result);
        }
        modes.push(Mode::Abstain);
        Ok(modes)
    }

    fn payload_grammar_spec(&self, mode: Mode) -> Result<portia_core::GrammarCartridgeSpec> {
        let mut hole_cart = self.session.hole_grammar_cartridge();
        if matches!(mode, Mode::Probe | Mode::Read) && hole_cart.hole_fn_names.is_empty() {
            hole_cart = RustAstHoleGrammarCartridge::new(
                self.session
                    .spec
                    .todo_holes
                    .iter()
                    .map(|h| h.fn_name.clone())
                    .collect(),
                hole_cart.approved_crate_name,
            );
        }
        Ok(hole_cart.span_grammar_cartridge_spec(mode))
    }

    fn classify_step_effect(&self, mode: Mode, payload: &str) -> CoroutineStepEffect {
        self.classify_coroutine_step_effect(mode, payload)
    }

    fn execute_turn_action(
        &mut self,
        action: &TypedTurnAction,
        ask_callback: &mut dyn FnMut(&str) -> Result<String>,
    ) -> DriverObservation {
        self.execute_typed_action(action, ask_callback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn treesitter_rust_syn_and_live_rust_ast_daily_driver_with_rustc_verification() {
        let coord = ScenarioCoordinate::new(
            AppType::RustAstHylo,
            ScenarioLanguage::English,
            DomainSkin::SolarisVault,
            RogueliteStressVector::sweet_spot_baseline(),
            0x3600_0042,
        );
        let spec = coord.build_cleanroom_rust_hylo_spec();
        let holes = spec.todo_holes.clone();
        let mut drv = LiveRustAstDailyDriver::from_spec(spec)
            .expect("driver init")
            .with_rustc_verification(true);

        // 1. Before `Mode::Probe`, `Mode::Ask` classifies as conversational `YieldToPrompt` (zero English keyword scanning):
        assert_eq!(
            drv.classify_coroutine_step_effect(
                Mode::Ask,
                "Sure, I will inspect the Rust AST outline and fill the todo!() holes.",
            ),
            CoroutineStepEffect::YieldToPrompt
        );

        // 2. Execute `Mode::Probe` (`probed_ast_outline == true`), then verify `Mode::Ask` surfaces `SuspendForGuiChoice(MultiSelect)` over the remaining `todo!("...")` holes:
        let probe_obs = drv.execute_turn(Mode::Probe, "inspect rust ast", |_| Ok("ALLOW".into()));
        assert!(probe_obs.ok);
        assert!(probe_obs
            .message
            .as_deref()
            .unwrap_or_default()
            .contains("[TODO_HOLE #0:"));

        match drv.classify_coroutine_step_effect(
            Mode::Ask,
            "Which todo!() function hole would you like to implement first?",
        ) {
            CoroutineStepEffect::SuspendForGuiChoice {
                gui_mode: AskInteractionMode::MultiSelect { choices, .. },
                ..
            } => {
                assert!(choices.len() > holes.len());
                assert!(choices.iter().any(|c| c.contains(&holes[0].fn_name)));
                assert_eq!(
                    choices.last().map(String::as_str),
                    Some(NONE_OF_ABOVE_ENTER_TEXT_CHOICE)
                );
            }
            other => panic!("expected SuspendForGuiChoice(MultiSelect), got {other:?}"),
        }

        let ask_obs = drv.execute_turn(
            Mode::Ask,
            "Please clarify the ambiguous invariant, approve crateena, and review scaffold before fanout.",
            |_| Ok("Use saturating floor at 0 and approve crate".into()),
        );
        assert!(ask_obs.ok);

        for (idx, h) in holes.iter().enumerate() {
            let read_obs = drv.execute_turn(Mode::Read, &h.fn_name, |_| Ok("ALLOW".into()));
            assert!(read_obs.ok);
            let write_payload = if idx % 2 == 0 {
                format!(
                    "{} => approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    h.fn_name, h.hole_index
                )
            } else {
                format!(
                    "fn {} {{ approved_crate::eval_domain_step(input_metric, guard_clause, {}usize) }}",
                    h.fn_name, h.hole_index
                )
            };
            let write_obs = drv.execute_turn(Mode::Write, &write_payload, |_| Ok("ALLOW".into()));
            assert!(write_obs.ok, "write failed: {:?}", write_obs.error);
            assert_eq!(
                write_obs.table_versions.get(&h.fn_name).copied(),
                Some(1),
                "fn_versions must increment for both `=>` and `fn ... {{ ... }}` formats"
            );
        }

        assert!(drv.remaining_todo_holes().unwrap().is_empty());

        let result_obs = drv.execute_turn(
            Mode::Result,
            "{\"status\": \"ALL_HOLES_SPLICED_AND_RUSTC_VERIFIED\"}",
            |_| Ok("ALLOW".into()),
        );
        assert!(
            result_obs.ok,
            "expected Mode::Result + rustc --crate-type lib verification to succeed, got: {:?}",
            result_obs.error
        );
    }

    #[test]
    fn hylo_2x2_fanout_and_octomerge_across_7_holes_with_rustc_verification() {
        let mut stress = RogueliteStressVector::sweet_spot_baseline();
        stress.ast_symbol_haystack = 7;
        stress.schema_graph_topology = 7;
        let coord = ScenarioCoordinate::new(
            AppType::RustAstHylo,
            ScenarioLanguage::English,
            DomainSkin::OrbitalConstellation,
            stress,
            0x3800_0007,
        );
        let spec = coord.build_cleanroom_rust_hylo_spec();
        assert_eq!(spec.todo_holes.len(), 7);

        let tmp = tempfile::NamedTempFile::new().expect("temp rs file");
        std::fs::write(tmp.path(), &spec.ast_valid_token_zero_skeleton).expect("write skeleton");

        let mut drv = LiveRustAstDailyDriver::from_spec(spec)
            .expect("driver init")
            .with_rustc_verification(true);
        drv.file_path = Some(tmp.path().to_path_buf());

        let receipt = drv
            .execute_hylo_2x2_fanout_and_octomerge(|hole, outline, fn_window| {
                // Verify each leaf receives only the compact `[AST ...]` outline + its own $O(1)$ function window:
                assert!(outline.contains("[Module `orbital_delta_v_planner`]"));
                assert!(fn_window.contains(&hole.fn_name));
                for other_idx in 0..7 {
                    if other_idx != hole.hole_index {
                        let other_fn = format!("orbital_constellation_step_{:02}", other_idx + 1);
                        assert!(
                            !fn_window.contains(&format!("fn {other_fn}(")),
                            "leaf window for `{}` leaked sibling body `{other_fn}`",
                            hole.fn_name
                        );
                    }
                }
                Ok(format!(
                    "approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    hole.hole_index
                ))
            })
            .expect("7-hole hylomorphic fanout and octo-merge must succeed");

        assert!(receipt.rustc_verified);
        assert!(receipt.episode_summary.cleared);
        assert!(receipt.episode_summary.wipeout_diagnoses.is_empty());
        assert_eq!(receipt.leaf_contexts.len(), 7);
        assert!(receipt.ast_outline_before.contains("[TODO_HOLE #6:"));
        assert!(receipt.ast_outline_after.contains("[IMPLEMENTED]"));
        assert!(!receipt.ast_outline_after.contains("TODO_HOLE"));
        assert_eq!(
            receipt.stages_executed,
            vec![
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Scaffold,
                    level: HyloCartridgeLevel::TypeCart,
                },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Scaffold,
                    level: HyloCartridgeLevel::ValueCart,
                },
                RustHyloPipelineStage::ForkFanout { hole_count: 7 },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Impl,
                    level: HyloCartridgeLevel::TypeCart,
                },
                RustHyloPipelineStage::CartridgeCell {
                    phase: HyloCartridgePhase::Impl,
                    level: HyloCartridgeLevel::ValueCart,
                },
                RustHyloPipelineStage::OctoMerge {
                    merged_hole_count: 7,
                },
            ]
        );

        let disk_source = std::fs::read_to_string(tmp.path()).expect("read synced disk file");
        assert_eq!(disk_source, receipt.merged_source);
        assert!(extract_rust_todo_holes(&disk_source).unwrap().is_empty());
    }

    #[test]
    fn sealed_type_and_cfg_approved_crate_rustc_test_reject_unapproved_or_forged_bodies() {
        let coord = ScenarioCoordinate::new(
            AppType::RustAstHylo,
            ScenarioLanguage::English,
            DomainSkin::SolarisVault,
            RogueliteStressVector::sweet_spot_baseline(),
            0x3900_0001,
        );
        let spec = coord.build_cleanroom_rust_hylo_spec();

        // 1. Attempting to call `approved_crate::eval_domain_step` WITHOUT requesting the crate in Mode::Ask
        //    causes `rustc --test` (without `--cfg approved_crate`) to fail compilation:
        let mut unapproved_drv = LiveRustAstDailyDriver::from_spec(spec.clone())
            .unwrap()
            .with_rustc_verification(true);
        let patches: Vec<(String, String)> = spec
            .todo_holes
            .iter()
            .map(|h| {
                (
                    h.fn_name.clone(),
                    format!(
                        "approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                        h.hole_index
                    ),
                )
            })
            .collect();
        unapproved_drv.session.current_source =
            hylo_octo_merge_rust_holes(&spec.ast_valid_token_zero_skeleton, &patches).unwrap();
        let unapproved_err = unapproved_drv.verify_with_rustc().expect_err(
            "rustc --test without --cfg approved_crate must reject unapproved crate call",
        );
        assert!(unapproved_err.to_string().contains("approved_crate"));

        // 2. Attempting to bypass `approved_crate` by forging `VerifiedDomainOutput` directly fails
        //    `rustc --test` due to the private `_sealed_crate_witness` field (Wadler's Free Theorem):
        let mut forged_drv = LiveRustAstDailyDriver::from_spec(spec.clone())
            .unwrap()
            .with_rustc_verification(true);
        forged_drv.session.requested_ecosystem_crate = true;
        let forged_patches: Vec<(String, String)> = spec
            .todo_holes
            .iter()
            .map(|h| {
                (
                    h.fn_name.clone(),
                    "Ok(VerifiedDomainOutput { metric: input_metric, _sealed_crate_witness: () })"
                        .to_string(),
                )
            })
            .collect();
        forged_drv.session.current_source =
            hylo_octo_merge_rust_holes(&spec.ast_valid_token_zero_skeleton, &forged_patches)
                .unwrap();
        let forged_err = forged_drv
            .verify_with_rustc()
            .expect_err("rustc --test must reject private field forgery on VerifiedDomainOutput");
        assert!(forged_err.to_string().contains("private"));
    }

    #[test]
    fn live_workspace_rust_file_returns_user_reply_verbatim_without_canned_simulator_suffix() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            "pub fn clamp_budget(v: i64) -> i64 { todo!(\"clamp budget\") }\n",
        )
        .unwrap();

        let mut live_drv = LiveRustAstDailyDriver::open(tmp.path()).unwrap();
        assert!(live_drv.is_live_workspace_file());

        let exact_reply = "Clamp negative values to -10 and keep std only.";
        let obs = live_drv.execute_turn(
            Mode::Ask,
            "What floor and crate policy should clamp_budget use?",
            |_| Ok(exact_reply.to_string()),
        );
        assert!(obs.ok);
        assert_eq!(obs.message.as_deref(), Some(exact_reply));
        assert!(
            !obs.message
                .as_deref()
                .unwrap_or_default()
                .contains("Use saturating floor at 0"),
            "live workspace .rs file must never append canned simulator answers onto user_reply"
        );
    }
}
