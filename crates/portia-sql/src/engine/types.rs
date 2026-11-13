//! Core relational cell values, snapshot differentials, provenance traces, and engine error types.

use std::collections::{BTreeMap, BTreeSet};

use portia_core::Mode;
use rusqlite::types::ValueRef;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical SQL cell value for deterministic relational state comparisons and JSON serialization.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Eq for SqlValue {}

impl SqlValue {
    pub fn from_value_ref(v: ValueRef<'_>) -> Self {
        match v {
            ValueRef::Null => SqlValue::Null,
            ValueRef::Integer(i) => SqlValue::Integer(i),
            ValueRef::Real(f) => SqlValue::Real(if f == -0.0 { 0.0 } else { f }),
            ValueRef::Text(t) => SqlValue::Text(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SqlValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            SqlValue::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// A single row represented as an ordered map of `column_name -> SqlValue`.
pub type SqlRow = BTreeMap<String, SqlValue>;

/// Whole-database canonical dump: `table_name -> Vec<SqlRow>`.
pub type DatabaseSnapshot = BTreeMap<String, Vec<SqlRow>>;

/// A database row paired with its extracted primary key value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyedRow {
    pub primary_key: SqlValue,
    pub row: SqlRow,
}

/// A row modification between the expected shadow state and actual database state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowModification {
    pub primary_key: SqlValue,
    pub expected_row: SqlRow,
    pub actual_row: SqlRow,
}

/// Per-table relational differential between actual database state and shadow reference state:
/// $\Delta T = T_{\text{actual}} \ominus T_{\text{expected}}$.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableDiff {
    /// Rows present in `actual` (`agent_conn`) but absent in `expected` (`ref_conn`).
    pub inserted_rows: Vec<KeyedRow>,
    /// Rows present in `expected` (`ref_conn`) but absent in `actual` (`agent_conn`).
    pub deleted_rows: Vec<KeyedRow>,
    /// Rows sharing a primary key whose column values diverge (`expected_row` vs `actual_row`).
    pub modified_rows: Vec<RowModification>,
}

impl TableDiff {
    pub fn is_empty(&self) -> bool {
        self.inserted_rows.is_empty()
            && self.deleted_rows.is_empty()
            && self.modified_rows.is_empty()
    }
}

/// Whole-database algebraic relational differential $\Delta R_{\text{actual}} \ominus \Delta R_{\text{expected}}$
/// returned by [`super::diff_shadow_databases`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalDiff {
    /// Non-empty table differentials keyed by table name.
    pub table_diffs: BTreeMap<String, TableDiff>,
    /// Serialized differential changeset bytes.
    pub session_changeset_bytes: Vec<u8>,
}

impl RelationalDiff {
    pub fn is_equivalent(&self) -> bool {
        self.table_diffs.values().all(TableDiff::is_empty)
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("C-level bytecode authorizer denied `{mode}` query `{sql}`: {reason}")]
    AuthorizerDenied {
        mode: Mode,
        sql: String,
        reason: String,
    },
    #[error("ReadOnlySessionViolation: session is running with --read-only; blocked attempted write {sql:?}")]
    ReadOnlySessionViolation { sql: String },
    #[error("GrammarTrieValidationError on generated SQL {sql:?}: {detail}")]
    GrammarTrieValidationError { sql: String, detail: String },
    #[error(
        "MaxWriteRowsExceeded: statement modified {rows_affected} rows (> limit {max_write_rows}) on `{sql}`, rolled back SAVEPOINT"
    )]
    MaxWriteRowsExceeded {
        rows_affected: usize,
        max_write_rows: usize,
        sql: String,
    },
    #[error(
        "TransactionFinalizationError: failed to release SAVEPOINT {savepoint}: {error} ({rollback_note})"
    )]
    TransactionFinalizationError {
        savepoint: String,
        error: String,
        rollback_note: String,
    },
    #[error("invalid SQL execution in mode `{mode}`: {detail}")]
    ExecutionError { mode: Mode, detail: String },
}

impl EngineError {
    /// Canonical short error tag for structured telemetry and TUI/REPL badges.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Sqlite(_) => "SqliteError",
            Self::AuthorizerDenied { .. } => "AuthorizerDenied",
            Self::ReadOnlySessionViolation { .. } => "ReadOnlySessionViolation",
            Self::GrammarTrieValidationError { .. } => "GrammarTrieValidationError",
            Self::MaxWriteRowsExceeded { .. } => "MaxWriteRowsExceeded",
            Self::TransactionFinalizationError { .. } => "TransactionFinalizationError",
            Self::ExecutionError { .. } => "ExecutionError",
        }
    }

    /// Format this domain error with bold red ANSI styling for terminal REPL output.
    #[must_use]
    pub fn render_ansi(&self) -> String {
        format!(
            "\x1b[1;31m✗ [{}]:\x1b[0m \x1b[31m{self}\x1b[0m",
            self.code()
        )
    }
}

/// Qualified `(table, column)` reference compiled into a SQLite query plan.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ColumnRef {
    pub table: String,
    pub column: String,
}

impl ColumnRef {
    pub fn new(table: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            column: column.into(),
        }
    }
}

/// C-level bytecode provenance captured during `sqlite3_prepare_v2` via `conn.authorizer`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BytecodeAuthorizerTrace {
    /// Exact `(table_lowercase, column_lowercase)` references accessed by the compiled query plan.
    pub compiled_column_reads: BTreeSet<ColumnRef>,
    /// Tables referenced via `SELECT` / `READ`.
    pub compiled_tables: BTreeSet<String>,
    /// Any `PRAGMA` names invoked.
    pub compiled_pragmas: BTreeSet<String>,
    /// Any forbidden actions blocked by the C-level authorizer.
    pub denied_actions: Vec<String>,
}

/// Row-level mutation captured by C-level `conn.preupdate_hook` or relational snapshot diffing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowMutationRecord {
    pub op: String,
    pub table: String,
    pub old_rowid: Option<i64>,
    pub new_rowid: Option<i64>,
    pub old_values: Vec<SqlValue>,
    pub new_values: Vec<SqlValue>,
}

/// Configuration for Mode A (Raw Tanking) vs. Mode B (Paginated Fog-of-War & `WHERE` Narrowing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryPageConfig {
    /// Maximum number of rows returned in a single `READ` page before truncating (`usize::MAX` for Raw Tanking).
    pub max_page_rows: usize,
    /// Whether to attach structured [`FogOfWarPageMetadata`] when a query returns or truncates rows.
    pub include_fog_metadata: bool,
}

impl Default for QueryPageConfig {
    fn default() -> Self {
        Self {
            max_page_rows: usize::MAX,
            include_fog_metadata: false,
        }
    }
}

/// Structured Fog-of-War pagination metadata attached to [`ReadReceipt`] when [`QueryPageConfig`] is active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FogOfWarPageMetadata {
    pub total_matching_rows: usize,
    pub returned_rows: usize,
    pub has_more: bool,
    pub next_offset: Option<usize>,
    pub hint: Option<String>,
}

/// Structured read observation receipt for SQLite `PROBE` / `READ` queries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadReceipt {
    pub event_id: String,
    pub step_index: usize,
    pub mutation_epoch: u64,
    pub mode: Mode,
    pub sql: String,
    pub columns: Vec<String>,
    pub rows: Vec<SqlRow>,
    pub authorizer_trace: BytecodeAuthorizerTrace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_metadata: Option<FogOfWarPageMetadata>,
}

/// Structured write mutation receipt for SQLite `WRITE` statements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteReceipt {
    pub event_id: String,
    pub step_index: usize,
    pub mutation_epoch_before: u64,
    pub mutation_epoch_after: u64,
    pub sql: String,
    pub rows_affected: usize,
    pub changeset_bytes: Vec<u8>,
    pub row_mutations: Vec<RowMutationRecord>,
    pub touched_tables: BTreeSet<String>,
}
