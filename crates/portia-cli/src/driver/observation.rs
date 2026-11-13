//! Typed observation envelopes (`DriverObservation`, `ObservationOperation`, `ReadPurpose`,
//! `AstOperation`) and the prompt-blind `EpistemicDomainEnv` coalgebra trait.

use super::ask_protocol::{AskUserReply, CoroutineStepEffect};
use anyhow::Result;
use portia_core::{Mode, StewardAuthState};
use portia_sql::engine::SqlWriteOp;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema summary for a single table in an arbitrary external SQLite database
/// (canonical alias onto [`portia_core::TableSchema`]).
pub type TableSchemaInfo = portia_core::TableSchema;

/// Column metadata summary in an arbitrary external SQLite database
/// (canonical alias onto [`portia_core::ColumnSchema`]).
pub type ColumnSchemaInfo = portia_core::ColumnSchema;

/// Causal purpose of a `Mode::Read` SQL query in the current task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPurpose {
    ReadOnly,
    Candidate,
    Verification,
}

impl ReadPurpose {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Candidate => "candidate",
            Self::Verification => "verification",
        }
    }
}

/// Typed operation tag for `LiveRustAstDailyDriver` (`rust-hylo`) observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AstOperation {
    #[serde(rename = "AST_PROBE")]
    Probe,
    #[serde(rename = "AST_READ_FN")]
    ReadFn,
    #[serde(rename = "AST_ASK")]
    Ask,
    #[serde(rename = "AST_SPLICE_HOLE")]
    SpliceHole,
    #[serde(rename = "RESULT")]
    Result,
    #[serde(rename = "ABSTAIN")]
    Abstain,
    #[serde(rename = "HYLO_FANOUT_OCTOMERGE")]
    HyloFanoutOctomerge,
}

impl AstOperation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Probe => "AST_PROBE",
            Self::ReadFn => "AST_READ_FN",
            Self::Ask => "AST_ASK",
            Self::SpliceHole => "AST_SPLICE_HOLE",
            Self::Result => "RESULT",
            Self::Abstain => "ABSTAIN",
            Self::HyloFanoutOctomerge => "HYLO_FANOUT_OCTOMERGE",
        }
    }
}

/// Strongly-typed operation descriptor on [`DriverObservation`], serializing directly to the
/// canonical flat wire strings (`"read_only"`, `"candidate"`, `"verification"`, `"INSERT"`,
/// `"UPDATE"`, `"DELETE"`, `"AST_PROBE"`, `"AST_READ_FN"`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObservationOperation {
    SqlRead(ReadPurpose),
    SqlWrite(SqlWriteOp),
    Ast(AstOperation),
}

impl ObservationOperation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SqlRead(purpose) => purpose.as_str(),
            Self::SqlWrite(op) => op.as_str(),
            Self::Ast(op) => op.as_str(),
        }
    }

    pub fn from_wire_str(raw: &str) -> Option<Self> {
        match raw.trim() {
            "read_only" => Some(Self::SqlRead(ReadPurpose::ReadOnly)),
            "candidate" => Some(Self::SqlRead(ReadPurpose::Candidate)),
            "verification" => Some(Self::SqlRead(ReadPurpose::Verification)),
            "INSERT" => Some(Self::SqlWrite(SqlWriteOp::Insert)),
            "UPDATE" => Some(Self::SqlWrite(SqlWriteOp::Update)),
            "DELETE" => Some(Self::SqlWrite(SqlWriteOp::Delete)),
            "AST_PROBE" => Some(Self::Ast(AstOperation::Probe)),
            "AST_READ_FN" => Some(Self::Ast(AstOperation::ReadFn)),
            "AST_ASK" => Some(Self::Ast(AstOperation::Ask)),
            "AST_SPLICE_HOLE" => Some(Self::Ast(AstOperation::SpliceHole)),
            "RESULT" => Some(Self::Ast(AstOperation::Result)),
            "ABSTAIN" => Some(Self::Ast(AstOperation::Abstain)),
            "HYLO_FANOUT_OCTOMERGE" => Some(Self::Ast(AstOperation::HyloFanoutOctomerge)),
            _ => None,
        }
    }
}

impl Serialize for ObservationOperation {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ObservationOperation {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::from_wire_str(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!("unknown ObservationOperation wire tag {raw:?}"))
        })
    }
}

/// Observation envelope returned by [`LiveSqliteDailyDriver`] or [`crate::rust_driver::LiveRustAstDailyDriver`]
/// after executing a turn's action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverObservation {
    pub event_id: String,
    pub mode: Mode,
    pub ok: bool,
    pub db_version: u64,
    pub table_versions: BTreeMap<String, u64>,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub rows_affected: usize,
    pub message: Option<String>,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<ObservationOperation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_reply: Option<AskUserReply>,
    #[serde(default)]
    pub steward_auth: StewardAuthState,
    #[serde(default)]
    pub is_chat_continuation: bool,
}

/// Prompt-blind (`∀prompt` Free Theorem) coalgebra unifying external epistemic environments
/// ([`LiveSqliteDailyDriver`] and [`crate::rust_driver::LiveRustAstDailyDriver`]) so the single
/// OODA coroutine loop in [`crate::worker::run_epistemic_env_turn_sequence_streaming_into`]
/// drives SQLite databases and Rust AST typed-hole modules identically without `user_prompt` access.
pub trait EpistemicDomainEnv {
    fn begin_new_task(&mut self);
    fn reset_epistemic_session(&mut self);
    fn state_version(&self) -> u64;
    fn steward_auth(&self) -> StewardAuthState {
        StewardAuthState::NotRequired
    }
    fn sync_worker_context(
        &self,
        _worker: &mut crate::worker::StdioQwenWorkerClient,
    ) -> Result<()> {
        Ok(())
    }
    fn legal_modes(&self) -> Result<Vec<portia_core::Mode>>;
    fn payload_grammar_spec(
        &self,
        mode: portia_core::Mode,
    ) -> Result<portia_core::GrammarCartridgeSpec>;
    fn classify_step_effect(&self, mode: portia_core::Mode, payload: &str) -> CoroutineStepEffect;
    fn execute_turn_action(
        &mut self,
        action: &portia_core::relay::TypedTurnAction,
        ask_callback: &mut dyn FnMut(&str) -> Result<String>,
    ) -> DriverObservation;
    fn on_tui_session_open(&self, _tui: &mut crate::tui::TwoPaneTuiState) {}
    fn on_tui_turn_completed(
        &self,
        _mode: portia_core::Mode,
        _obs: &DriverObservation,
        _tui: &mut crate::tui::TwoPaneTuiState,
    ) {
    }
}
