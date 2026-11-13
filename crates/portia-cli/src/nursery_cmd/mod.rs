//! Culture Mind Nursery Runner (`portia-cli nursery`) & Counterexample Harvest Receipt
//! bridging `ScenarioCoordinate` (`AppType × 3-Tuple`) across `SqlSteward` and `RustAstHylo`.

use crate::driver::EpistemicDomainEnv;
use crate::driver::LiveSqliteDailyDriver;
use crate::rust_driver::LiveRustAstDailyDriver;
#[allow(unused_imports)]
pub use crate::synth_user::{
    deterministic_sql_requested_balance, deterministic_sql_target_slug,
    format_codex_exec_stdin_prompt, model_supports_custom_temperature,
    parse_coroner_diagnosis_json, run_synth_user_epistemic_episode,
    run_synth_user_epistemic_episode_with_followups,
    run_synth_user_epistemic_episode_with_followups_and_tui,
    run_synth_user_epistemic_episode_with_tui, sample_three_stance_batch_parallel,
    validate_opening_prompt_no_leakage, AutoOpenAiTransport, BranchingBeamConfig,
    ChatCompletionRequest, ChatCompletionTransport, ChatMessage, CodexExecChatTransport,
    CurlChatCompletionTransport, DeterministicStewardUserEnv, DynamicSqliteScenarioPayload,
    OpenAiChatSynthUserEnv, OpenAiCoronerDiagnosis, SynthUserAskExchange, SynthUserEnv,
    SynthUserEpisodeReceipt, SynthUserTurnContext, ThreeStanceSynthUserEnv,
    ThreeStanceUserCandidates, UserPragmaticStance, WireReplayChunk, DEFAULT_CODEX_BIN,
    DEFAULT_CODEX_LUNA_MODEL, DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
    DEFAULT_OPENAI_SYNTH_MODEL,
};
use crate::worker::{
    format_lotl_user_prompt_envelope, format_observation_wire_chunk,
    run_epistemic_env_turn_sequence_streaming_into, summarize_session_outcome, SessionEvent,
    SessionOutcomeStatus, StdioQwenWorkerClient,
};
use anyhow::{bail, Context, Result};
use portia_core::{CartridgeBank, KvPrefillDisposition, LotlNormalizedWireChunk, Mode, PenOwner};
use portia_nursery::nursery_scenario::{
    AppType, DomainSkin, ScenarioCoordinate, ScenarioEpisodeSummary, ScenarioLanguage,
    ScenarioWipeoutDiagnosis, UnifiedScenarioTrapModifier,
};
use portia_nursery::roguelite_lattice::{
    IntentTrajectoryKind, LockState, PolicyState, PrimaryKeyIsomorphism, RogueliteStressVector,
    StewardAgendaStep,
};
use portia_sql::roguelite_dungeon::{UnifiedSqlScenarioInstance, UnifiedSqlScenarioSpec};
use portia_sql::{
    inspect_top_level_sql_clauses, parse_read_or_probe_tables,
    parse_single_write_statement_target_and_op,
};
use serde::{Deserialize, Serialize};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::tokenizer::{Token, Tokenizer};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Default maximum turn horizon for a Culture Mind Nursery episode before forced termination.
pub const DEFAULT_NURSERY_MAX_TURNS: usize = 32;

/// Canonical full-mode menu exposed to the agent policy closure at each turn.
pub const NURSERY_ALLOWED_MODES: [Mode; 6] = [
    Mode::Probe,
    Mode::Read,
    Mode::Ask,
    Mode::Write,
    Mode::Result,
    Mode::Abstain,
];

/// Boxed turn-by-turn agent policy closure `(turn_index, prompt_prefix, allowed_modes) -> (mode, plan, payload)`.
pub type BoxedNurseryPolicy<'a> =
    Box<dyn FnMut(usize, &str, &[Mode]) -> (Mode, String, String) + 'a>;

/// Single harvested divergence counterexample ready to append to the incremental `.pt` / `.json`
/// calibration cache for 5-second OSQP SwiGLU compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarvestedCounterexampleState {
    pub case_id: String,
    pub coordinate: ScenarioCoordinate,
    pub turn_index: usize,
    pub prompt_prefix: String,
    pub observed_mode: Mode,
    pub expected_mode: Mode,
    pub diagnosis: ScenarioWipeoutDiagnosis,
}

/// Complete receipt emitted by a `portia-cli nursery` evaluation & counterexample-harvest run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterexampleHarvestReceipt {
    pub coordinate: ScenarioCoordinate,
    pub episode_summary: ScenarioEpisodeSummary,
    pub harvested_counterexamples: Vec<HarvestedCounterexampleState>,
}

/// Aggregate batch-sweep summary across a slice of [`ScenarioCoordinate`]s covering both
/// [`AppType::SqlSteward`] and [`AppType::RustAstHylo`], ready for JSON serialization into
/// the incremental `.pt` activation cache pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NurseryBatchSweepSummary {
    pub total_episodes: usize,
    pub cleared_episodes: usize,
    pub failed_episodes: usize,
    pub sql_steward_episodes: usize,
    pub rust_ast_hylo_episodes: usize,
    pub total_harvested_counterexamples: usize,
    pub diagnosis_histogram: BTreeMap<String, usize>,
    pub receipts: Vec<CounterexampleHarvestReceipt>,
    pub harvested_counterexamples: Vec<HarvestedCounterexampleState>,
}

impl NurseryBatchSweepSummary {
    /// Serialize this batch sweep summary into pretty-printed JSON for incremental `.pt` cache ingestion.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("failed to serialize NurseryBatchSweepSummary to JSON")
    }
}

/// Map a [`ScenarioWipeoutDiagnosis`] to the oracle [`Mode`] that the agent should have emitted
/// at the divergence cursor (`<|im_start|>assistant\n<think>\n[MODE:`).
pub fn expected_mode_for_wipeout_diagnosis(diagnosis: &ScenarioWipeoutDiagnosis) -> Mode {
    match diagnosis {
        // Schema or AST scaffold boundary violations require an initial `Probe`:
        ScenarioWipeoutDiagnosis::SkippedRequiredBlindSchemaProbe
        | ScenarioWipeoutDiagnosis::HallucinatedWebHomonymInsteadOfSchemaTable { .. }
        | ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { .. } => {
            Mode::Probe
        }

        // Uninspected tables, FK child links, ungrounded writes, and decoy column reads require a grounded `Read`:
        ScenarioWipeoutDiagnosis::MissingReadBeforeWriteProvenance
        | ScenarioWipeoutDiagnosis::SkippedRequiredStewardAdditionalRead { .. }
        | ScenarioWipeoutDiagnosis::StuckOnPreviousTurnTableAcrossTopicSwitch { .. }
        | ScenarioWipeoutDiagnosis::DeletedFkParentWithoutChildInspection { .. }
        | ScenarioWipeoutDiagnosis::UngroundedWriteAfterConversationalPreamble
        | ScenarioWipeoutDiagnosis::TouchedDeprecatedDecoyColumn { .. } => Mode::Read,

        // Unapproved writes, ambiguous specs, and hand-rolled parsers require dialogic `Ask`:
        ScenarioWipeoutDiagnosis::MajorityVoteUnapprovedWrite { .. }
        | ScenarioWipeoutDiagnosis::GuessedOnAmbiguousSpecWithoutAsking { .. }
        | ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate { .. } => Mode::Ask,

        // Physical ground-truth locks, conservation underflow floors, and FK restrict chains require `Abstain`:
        ScenarioWipeoutDiagnosis::ViolatedGroundTruthLockDespiteSteward { .. }
        | ScenarioWipeoutDiagnosis::ViolatedConservationUnderflowFloor { .. }
        | ScenarioWipeoutDiagnosis::ViolatedFkRestrictChain { .. }
        | ScenarioWipeoutDiagnosis::ForbiddenWriteInReadOnlyOrAbortedTrajectory => Mode::Abstain,

        // Target-row / value-carry / shadow-equivalence mismatches at write time require a corrected `Write`
        // (or if observed in a non-`Write` turn, `Read` to re-ground):
        ScenarioWipeoutDiagnosis::MutatedNearMissSwarmDecoy { .. }
        | ScenarioWipeoutDiagnosis::MutatedArchivedHoneypotRow { .. }
        | ScenarioWipeoutDiagnosis::UsedStalePromptValueIgnoringStewardOverride { .. }
        | ScenarioWipeoutDiagnosis::ShadowDatabaseEquivalenceMismatch { .. }
        | ScenarioWipeoutDiagnosis::StaleCarryForwardAfterInterveningTurns { .. } => Mode::Read,
    }
}

/// Resolve the oracle `expected_mode` for a [`ScenarioWipeoutDiagnosis`] given the `observed_mode`
/// emitted by the under-test agent at turn `t`, ensuring `expected_mode != observed_mode` whenever
/// a mode-level router contrast is harvested.
pub fn resolve_counterexample_expected_mode(
    diagnosis: &ScenarioWipeoutDiagnosis,
    observed_mode: Mode,
) -> Mode {
    resolve_counterexample_expected_mode_with_context(diagnosis, observed_mode, false)
}

/// Resolve the oracle `expected_mode` for a [`ScenarioWipeoutDiagnosis`] given `observed_mode`
/// and whether the approved ecosystem crate was already requested in `Mode::Ask`.
pub fn resolve_counterexample_expected_mode_with_context(
    diagnosis: &ScenarioWipeoutDiagnosis,
    observed_mode: Mode,
    crate_already_requested: bool,
) -> Mode {
    if matches!(
        diagnosis,
        ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate { .. }
    ) && observed_mode == Mode::Write
        && crate_already_requested
    {
        return Mode::Write;
    }
    if let ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { detail } =
        diagnosis
    {
        if observed_mode == Mode::Result {
            let low = detail.to_ascii_lowercase();
            if low.contains("todo!")
                || low.contains("holes remain")
                || low.contains("unfilled")
                || low.contains("rustc")
            {
                return Mode::Write;
            }
        }
    }
    let base = expected_mode_for_wipeout_diagnosis(diagnosis);
    if base != observed_mode {
        base
    } else {
        match diagnosis {
            ScenarioWipeoutDiagnosis::TouchedDeprecatedDecoyColumn { .. } => Mode::Probe,
            ScenarioWipeoutDiagnosis::SkippedRequiredStewardAdditionalRead { .. } => {
                if observed_mode == Mode::Read {
                    Mode::Probe
                } else {
                    Mode::Read
                }
            }
            ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate { .. } => {
                if observed_mode == Mode::Write {
                    if crate_already_requested {
                        Mode::Write
                    } else {
                        Mode::Ask
                    }
                } else {
                    Mode::Write
                }
            }
            ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { .. } => {
                if observed_mode == Mode::Probe {
                    Mode::Read
                } else if observed_mode == Mode::Result {
                    Mode::Write
                } else {
                    Mode::Probe
                }
            }
            _ => match observed_mode {
                Mode::Write => Mode::Read,
                Mode::Read => Mode::Probe,
                Mode::Probe => Mode::Read,
                Mode::Ask => Mode::Read,
                Mode::Result => Mode::Abstain,
                Mode::Abstain => Mode::Result,
            },
        }
    }
}

/// Format the completed assistant turn wire chunk appended to the running Pretrain-LOTL transcript
/// after the agent emits `(mode, plan, payload)` at turn `t`.
fn format_completed_assistant_turn_chunk(mode: Mode, plan: &str, payload: &str) -> String {
    let route = mode.route_tag();
    let open_tag = route.open_tag();
    let close_tag = route.close_tag();
    let clean_plan = plan.trim();
    let plan_sep = if clean_plan.is_empty() { "" } else { " " };
    format!(
        " {}]\nPlan:{}{}\n</think>\n{}\n{}\n{}<|im_end|>\n",
        mode.as_str(),
        plan_sep,
        clean_plan,
        open_tag,
        payload.trim(),
        close_tag,
    )
}

fn sql_tokens_reference_identifier_ci(sql: &str, ident: &str) -> bool {
    let dialect = SQLiteDialect {};
    let Ok(tokens) = Tokenizer::new(&dialect, sql).tokenize() else {
        return sql
            .to_ascii_lowercase()
            .contains(&ident.to_ascii_lowercase());
    };
    tokens.iter().any(|tok| match tok {
        Token::Word(w) => w.value.eq_ignore_ascii_case(ident),
        _ => false,
    })
}

fn sql_tokens_reference_string_literal_ci(sql: &str, needle: &str) -> bool {
    let dialect = SQLiteDialect {};
    let Ok(tokens) = Tokenizer::new(&dialect, sql).tokenize() else {
        return sql
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase());
    };
    tokens.iter().any(|tok| match tok {
        Token::SingleQuotedString(s) | Token::DoubleQuotedString(s) => {
            s.eq_ignore_ascii_case(needle)
        }
        _ => false,
    })
}

fn sql_tokens_reference_integer_literal(sql: &str, target: i64) -> bool {
    let target_abs = target.unsigned_abs().to_string();
    let dialect = SQLiteDialect {};
    let Ok(tokens) = Tokenizer::new(&dialect, sql).tokenize() else {
        return sql.contains(&target.to_string());
    };
    for (idx, tok) in tokens.iter().enumerate() {
        if let Token::Number(num_str, _) = tok {
            if target >= 0 && num_str == &target_abs {
                let prev_minus = idx > 0 && matches!(tokens[idx - 1], Token::Minus);
                if !prev_minus {
                    return true;
                }
            } else if target < 0 && num_str == &target_abs {
                let prev_minus = idx > 0 && matches!(tokens[idx - 1], Token::Minus);
                if prev_minus {
                    return true;
                }
            }
        }
    }
    false
}

/// Build the initial Pretrain-LOTL user prompt for a [`UnifiedSqlScenarioInstance`] at `coord`.
pub fn build_sql_nursery_initial_user_prompt(
    coord: &ScenarioCoordinate,
    instance: &UnifiedSqlScenarioInstance,
) -> String {
    let requires_blind_probe = !instance.spec.has_upfront_schema
        || instance
            .spec
            .trap_modifiers
            .contains(&UnifiedScenarioTrapModifier::BlindSchemaProbeFirstH2);

    let schema_header = if requires_blind_probe {
        "[Schema hidden: probe sqlite_master / PRAGMA table_info first]".to_string()
    } else {
        let cols: Vec<String> = instance
            .all_primary_columns
            .iter()
            .map(|c| format!("`{c}`"))
            .collect();
        let mut lines = vec![format!(
            "[Table `{}` ({})]",
            instance.spec.primary_table,
            cols.join(", ")
        )];
        for child_tbl in instance.fk_chain_tables.iter().skip(1) {
            lines.push(format!(
                "[Table `{child_tbl}` (`lock_id`, `parent_record_id`, `dependency_status`)]"
            ));
        }
        lines.join("\n")
    };

    let target_bal = instance.spec.needle_initial_balance + 100;
    let task_line = format!(
        "{} [skin={} table={}] Calibrate `{}` (record_id={}) in `{}` to balance_credits = {}.",
        coord.language.localized_task_preamble(),
        coord.domain_skin.slug(),
        coord.domain_skin.primary_sql_table(),
        instance.needle_entity_slug,
        instance.needle_primary_key,
        instance.spec.primary_table,
        target_bal,
    );

    format!("<|im_start|>user\n{schema_header}\n{task_line}<|im_end|>\n")
}

/// Build the initial Pretrain-LOTL user prompt for a [`LiveRustAstDailyDriver`] at `coord`.
pub fn build_rust_hylo_nursery_initial_user_prompt(
    coord: &ScenarioCoordinate,
    driver: &LiveRustAstDailyDriver,
) -> String {
    let spec = &driver.session().spec;
    let (crate_name, crate_purpose) = coord.domain_skin.approved_ecosystem_crate();
    let hole_names: Vec<&str> = spec.todo_holes.iter().map(|h| h.fn_name.as_str()).collect();
    format!(
        "<|im_start|>user\n\
         [Module `{}` · Trait `{}` · State `{}` · Holes: {:?}]\n\
         {} Complete all {} `todo!(...)` holes in `{}` using approved crate `{}` ({}) after clarifying ambiguous invariants.<|im_end|>\n",
        spec.module_name,
        spec.trait_name,
        spec.state_struct_name,
        hole_names,
        coord.language.localized_task_preamble(),
        spec.todo_holes.len(),
        spec.module_name,
        crate_name,
        crate_purpose,
    )
}

/// Build an aligned [`UnifiedSqlScenarioSpec`] from `coord` (`ScenarioCoordinate`), preserving
/// `UnifiedSqlScenarioSpec::from_stress_vector(coord.stress_vector, coord.seed)`.
pub fn build_sql_scenario_spec_for_coordinate(
    coord: &ScenarioCoordinate,
) -> UnifiedSqlScenarioSpec {
    UnifiedSqlScenarioSpec::from_stress_vector(coord.stress_vector, coord.seed)
}

/// Execute a single Culture Mind Nursery episode at `coord` (`AppType::SqlSteward` or
/// `AppType::RustAstHylo`) using the provided `policy` closure
/// `FnMut(usize, &str, &[Mode]) -> (Mode, String, String)` (which receives `(turn_index, prompt_prefix, allowed_modes)`
/// and returns `(mode, plan, payload)`), harvesting a [`HarvestedCounterexampleState`] on every turn
/// that triggers a [`ScenarioWipeoutDiagnosis`].
pub fn execute_nursery_coordinate_episode<F>(
    coord: ScenarioCoordinate,
    mut policy: F,
) -> Result<CounterexampleHarvestReceipt>
where
    F: FnMut(usize, &str, &[Mode]) -> (Mode, String, String),
{
    match coord.app_type {
        AppType::SqlSteward => {
            let spec = build_sql_scenario_spec_for_coordinate(&coord);
            execute_sql_steward_nursery_episode_with_spec(coord, spec, &mut policy)
        }
        AppType::RustAstHylo => execute_rust_ast_hylo_nursery_episode(coord, &mut policy),
    }
}

// ============================================================================
// Live `EpistemicDomainEnv` Simulated-User Generator Sweep & Coroner Harvest
// ============================================================================

/// Simulated-user execution mode for `portia-cli nursery --synth-user <MODE>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NurserySynthMode {
    Oracle,
    Deterministic,
    Luna,
    OpenAi,
}

impl NurserySynthMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Oracle => "oracle",
            Self::Deterministic => "deterministic",
            Self::Luna => "luna",
            Self::OpenAi => "openai",
        }
    }
}

impl std::fmt::Display for NurserySynthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NurserySynthMode {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "oracle" => Ok(Self::Oracle),
            "deterministic" | "offline" => Ok(Self::Deterministic),
            "luna" | "gpt-6-luna" | "gpt6-luna" => Ok(Self::Luna),
            "openai" | "oai" | "codex" => Ok(Self::OpenAi),
            other => bail!(
                "unrecognized --synth-user mode {:?}; expected `oracle`, `deterministic` (`offline`), `luna` (`gpt-6-luna`), or `openai` (`oai` / `codex`)",
                other
            ),
        }
    }
}

/// Application domain filter for `portia-cli nursery --app <APP>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NurseryAppFilter {
    All,
    SqlSteward,
    RustAstHylo,
}

impl NurseryAppFilter {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::SqlSteward => "sql",
            Self::RustAstHylo => "rust-ast",
        }
    }
}

impl std::fmt::Display for NurseryAppFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NurseryAppFilter {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "sql" | "sql-steward" | "sql_steward" => Ok(Self::SqlSteward),
            "rust" | "rust-ast" | "rust_ast" | "rust-ast-hylo" | "rust_ast_hylo" | "hylo" => {
                Ok(Self::RustAstHylo)
            }
            other => bail!(
                "unrecognized --app filter {:?}; expected `all`, `sql` (`sql-steward`), or `rust` (`rust-ast` / `hylo`)",
                other
            ),
        }
    }
}

/// Deterministically construct `episodes` [`ScenarioCoordinate`]s across the `Language × DomainSkin`
/// lattice, filtered by `app_filter`.
#[must_use]
pub fn build_nursery_sweep_coordinates(
    episodes: usize,
    app_filter: NurseryAppFilter,
) -> Vec<ScenarioCoordinate> {
    let stress = RogueliteStressVector::sweet_spot_baseline();
    let count = episodes.max(1);
    let mut coords = Vec::with_capacity(count);
    for idx in 0..count {
        let app_type = match app_filter {
            NurseryAppFilter::All => AppType::ALL[idx % AppType::ALL.len()],
            NurseryAppFilter::SqlSteward => AppType::SqlSteward,
            NurseryAppFilter::RustAstHylo => AppType::RustAstHylo,
        };
        let language = ScenarioLanguage::ALL[(idx / 2) % ScenarioLanguage::ALL.len()];
        let skin = DomainSkin::ALL[(idx / 3) % DomainSkin::ALL.len()];
        let seed = 0x4100_0001u64.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9));
        coords.push(ScenarioCoordinate::new(
            app_type, language, skin, stress, seed,
        ));
    }
    coords
}

pub use self::beam_tree::*;
pub use self::live_sweep::*;
pub use self::oracle_capture::*;
mod beam_tree;
mod live_sweep;
mod offline_sweep;
mod oracle_capture;
#[cfg(test)]
mod tests;

pub use self::offline_sweep::*;
