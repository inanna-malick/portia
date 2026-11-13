//! Live SQLite daily-driver (`LiveSqliteDailyDriver`), `PRAGMA query_only` read guards,
//! Steward `preupdate_hook` row tracking, and session-level `SAVEPOINT` `/undo` stack.

use super::ask_protocol::{
    classify_ask_prompt_kind, parse_ask_user_reply_for_kind,
    resolve_ask_interaction_mode_from_observation_state,
    resolve_clarification_steward_ask_from_rows_with_columns,
    row_has_steward_ask_policy_in_columns, steward_column_matches, summarize_candidate_row_choice,
    AskDecision, AskInteractionMode, AskPromptKind, AskUserReply, CoroutineStepEffect,
};
use super::observation::{
    DriverObservation, EpistemicDomainEnv, ObservationOperation, ReadPurpose, TableSchemaInfo,
};
use anyhow::{bail, Context, Result};
use portia_core::relay::TypedTurnAction;
use portia_core::{Mode, StewardAuthState, TableSchema};
use portia_sql::engine::{
    collect_read_tables_via_sqlite_authorizer, list_sqlite_user_tables, parse_read_or_probe_tables,
    parse_single_write_statement_target_and_op, rollback_and_release_savepoint, EngineError,
};
use portia_sql::grammar_trie::{inspect_all_table_schemas, validate_completed_sql_against_schemas};
use rusqlite::hooks::{Action, AuthAction, AuthContext, Authorization, PreUpdateCase};
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

fn query_sqlite_data_version(conn: &Connection) -> i64 {
    conn.query_row("PRAGMA data_version;", [], |r| r.get(0))
        .unwrap_or(0)
}

/// SQLite supplies undefined row IDs for WITHOUT ROWID tables. Discover their layout
/// from schema metadata before installing the callback; never query from inside a hook.
type SqliteTableLayouts = BTreeMap<(String, String), Option<bool>>;

fn inspect_sqlite_table_layouts(conn: &Connection) -> rusqlite::Result<SqliteTableLayouts> {
    let mut stmt = conn.prepare("PRAGMA table_list;")?;
    let rows = stmt.query_map([], |row| {
        let schema: String = row.get(0)?;
        let table: String = row.get(1)?;
        let kind: String = row.get(2)?;
        let without_rowid: bool = row.get(4)?;
        Ok((
            (schema.to_lowercase(), table.to_lowercase()),
            matches!(kind.as_str(), "table" | "shadow").then_some(!without_rowid),
        ))
    })?;
    rows.collect()
}

/// AUTOINCREMENT metadata can change on INSERT OR IGNORE even without a row callback.
/// Only this schema-sized SQLite metadata is read; user table contents are never scanned.
fn inspect_sqlite_sequences(
    conn: &Connection,
    layouts: &SqliteTableLayouts,
) -> rusqlite::Result<BTreeMap<(String, String), i64>> {
    let mut sequences = BTreeMap::new();
    for (schema, table) in layouts.keys() {
        if table != "sqlite_sequence" {
            continue;
        }
        let mut stmt = conn.prepare(&format!(
            "SELECT name, seq FROM \"{}\".sqlite_sequence;",
            schema.replace('"', "\"\"")
        ))?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (table, sequence) = row?;
            sequences.insert((schema.clone(), table.to_lowercase()), sequence);
        }
    }
    Ok(sequences)
}

fn preupdate_changes_values(case: &PreUpdateCase, has_rowid: Option<bool>) -> bool {
    let PreUpdateCase::Update {
        old_value_accessor: old,
        new_value_accessor: new,
    } = case
    else {
        return true; // Inserts, deletes and unknown events cannot certify an unchanged row.
    };
    let Some(has_rowid) = has_rowid else {
        return true;
    };
    if has_rowid && old.get_old_row_id() != new.get_new_row_id() {
        return true;
    }
    (0..old.get_column_count()).any(|column| {
        match (
            old.get_old_column_value(column),
            new.get_new_column_value(column),
        ) {
            (Ok(ValueRef::Real(a)), Ok(ValueRef::Real(b))) => a.to_bits() != b.to_bits(),
            (Ok(a), Ok(b)) => a != b,
            _ => true, // A value-access failure must not produce a false cycle diagnosis.
        }
    })
}

#[derive(Default)]
struct SqliteMutationTrace {
    changed_tables: BTreeSet<String>,
    violated_steward_ask: bool,
    uncertain: bool,
}

fn preupdate_old_row_touches_steward_ask(
    case: &PreUpdateCase,
    policy_col_indices: &[usize],
) -> Result<bool, ()> {
    if policy_col_indices.is_empty() {
        return Ok(false);
    }
    let old = match case {
        PreUpdateCase::Update {
            old_value_accessor, ..
        } => old_value_accessor,
        PreUpdateCase::Delete(old_value_accessor) => old_value_accessor,
        PreUpdateCase::Insert { .. } | PreUpdateCase::Unknown => return Ok(false),
    };
    let col_count = old.get_column_count();
    for &col_idx in policy_col_indices {
        let Ok(col_i32) = i32::try_from(col_idx) else {
            return Err(());
        };
        if col_i32 < 0 || col_i32 >= col_count {
            return Err(());
        }
        match old.get_old_column_value(col_i32) {
            Ok(ValueRef::Text(bytes)) => {
                if std::str::from_utf8(bytes)
                    .map(|s| s.trim().eq_ignore_ascii_case("ASK"))
                    .unwrap_or(false)
                {
                    return Ok(true);
                }
            }
            Ok(_) => {}
            Err(_) => return Err(()),
        }
    }
    Ok(false)
}

fn sql_row_has_steward_ask_policy(
    table_name: &str,
    row: &portia_sql::engine::SqlRow,
    steward_policy_columns: &[String],
) -> bool {
    if steward_policy_columns.is_empty() {
        return false;
    }
    row.iter().any(|(col_name, cell)| {
        steward_column_matches(steward_policy_columns, Some(table_name), col_name)
            && matches!(
                cell,
                portia_sql::engine::SqlValue::Text(s) if s.trim().eq_ignore_ascii_case("ASK")
            )
    })
}

fn snapshot_mutated_steward_ask_row(
    pre: &portia_sql::engine::DatabaseSnapshot,
    post: &portia_sql::engine::DatabaseSnapshot,
    steward_policy_columns: &[String],
) -> bool {
    if steward_policy_columns.is_empty() {
        return false;
    }
    for (tbl, pre_rows) in pre {
        let post_rows = post.get(tbl).map(Vec::as_slice).unwrap_or(&[]);
        if pre_rows.as_slice() == post_rows {
            continue;
        }
        let mut unmatched_post: Vec<&portia_sql::engine::SqlRow> = post_rows.iter().collect();
        for pre_row in pre_rows {
            if let Some(pos) = unmatched_post.iter().position(|&r| r == pre_row) {
                unmatched_post.swap_remove(pos);
            } else if sql_row_has_steward_ask_policy(tbl, pre_row, steward_policy_columns) {
                return true;
            }
        }
    }
    false
}

/// Schema-agnostic, multi-table SQLite daily-driver environment (`LiveDBGuard`).
///
/// Decoupled completely from benchmark `WaveSpec` / 4-column assumptions (`?e_2`):
/// - Supports any multi-table schema, `JOIN`, `GROUP BY`, `COUNT(*)`, `ORDER BY`, `LIMIT`, CTEs,
///   and table aliases during `READ` under `PRAGMA query_only = ON`.
/// - Constrains generated SQL to discovered schemas without requiring a prior row read.
/// - Wraps every `WRITE` in a SQLite `SAVEPOINT` and automatically rolls back if syntax/execution
///   fails, if `rows_affected > max_write_rows`, or if an unapproved Steward `ASK` row is touched.
pub struct LiveSqliteDailyDriver {
    pub(super) conn: Connection,
    db_version: u64,
    last_sqlite_data_version: i64,
    event_counter: u64,
    table_versions: BTreeMap<String, u64>,
    /// Tables with an unverified committed write (`APPLIED`, `rows_affected > 0`) in the current task.
    pending_verification_tables: BTreeSet<String>,
    pub(super) max_write_rows: usize,
    read_only: bool,
    opened_read_only_handle: bool,
    /// Number of committed writes in the current user task (reset by [`Self::begin_new_task`]).
    task_writes_committed: u64,
    /// Number of `Mode::Probe` / `Mode::Read` turns executed in the current user task (reset by [`Self::begin_new_task`]).
    task_db_reads_executed: u64,
    /// Monotonic task-local Steward authorization state (`NotRequired -> PendingAsk -> {Granted | Denied}`).
    steward_auth: StewardAuthState,
    /// Whether `sqlite_master` catalog discovery has been executed in the current session.
    catalog_discovered: bool,
    /// User tables whose schema/rows have been causally inspected via `PRAGMA table_info(...)` or `Mode::Probe`/`Mode::Read`.
    probed_table_names: BTreeSet<String>,
    /// Compact human-readable candidate row summaries from the most recent multi-row `Mode::Read` in the current task.
    task_candidate_choices: Vec<String>,
    /// Column names from the most recent `Mode::Read` in the current task.
    task_last_read_columns: Vec<String>,
    /// Raw candidate rows from the most recent `Mode::Read` in the current task, used to resolve whether a
    /// `MultiSelect` clarification reply selected an `"ASK"` vs. `"ALLOW"` candidate row.
    task_last_read_rows: Vec<Vec<serde_json::Value>>,
    /// Session-persistent column names from the most recent `Mode::Read` across turns.
    session_last_read_columns: Vec<String>,
    /// Session-persistent candidate rows from the most recent `Mode::Read` across turns (or seeded from
    /// the primary table on initialization), preserved across [`Self::begin_new_task`].
    session_last_read_rows: Vec<Vec<serde_json::Value>>,
    /// Session-persistent human-readable candidate choices (up to 6 rows), preserved across [`Self::begin_new_task`].
    session_candidate_choices: Vec<String>,
    /// Explicitly configured steward policy column bindings (`"col_name"` or `"table.col_name"`).
    /// Empty by default on ordinary databases so arbitrary `"ASK"` strings in user columns never block writes.
    steward_policy_columns: Vec<String>,
    /// Whether automatic multi-row candidate `MultiSelect` disambiguation synthesis is enabled (default `false`).
    candidate_disambiguation_enabled: bool,
    /// Monotonic counter for session-level `/undo` SQLite savepoints (`portia_step_N`).
    undo_seq: u64,
    /// Active stack of committed-turn undo savepoints that can be rolled back via `/undo`.
    undo_stack: Vec<SessionUndoCheckpoint>,
}

/// Snapshot metadata for a session-level SQLite `SAVEPOINT` opened before a mutating prompt turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionUndoCheckpoint {
    pub savepoint_name: String,
    pub prompt_summary: String,
    pub db_version_before: u64,
    pub table_versions_before: BTreeMap<String, u64>,
}

/// Confirmation receipt emitted when `/undo` rolls back the top [`SessionUndoCheckpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoAppliedReceipt {
    pub savepoint_name: String,
    pub prompt_summary: String,
    pub restored_db_version: u64,
    pub remaining_undo_depth: usize,
}

impl Drop for LiveSqliteDailyDriver {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = self.flush_open_undo_savepoints();
        }
    }
}

impl LiveSqliteDailyDriver {
    /// Open an existing SQLite database file (never silently creating a new file when `path` does not exist).
    pub fn open(path: impl AsRef<Path>, max_write_rows: usize) -> Result<Self> {
        Self::open_with_mode(path, max_write_rows, false)
    }

    /// Open an existing SQLite database file in strict read-only mode (`SQLITE_OPEN_READ_ONLY` + `PRAGMA query_only = ON`).
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_mode(path, 0, true)
    }

    /// Open an existing SQLite database file with explicit `read_only` enforcement.
    pub fn open_with_mode(
        path: impl AsRef<Path>,
        max_write_rows: usize,
        read_only: bool,
    ) -> Result<Self> {
        let p = path.as_ref();
        if p == Path::new(":memory:") {
            let conn =
                Connection::open_in_memory().context("failed to open in-memory SQLite database")?;
            let mut drv = Self::from_connection(conn, max_write_rows)?;
            drv.set_read_only(read_only)?;
            return Ok(drv);
        }
        if !p.exists() {
            bail!("SQLite database file not found: {:?}", p);
        }
        if !p.is_file() {
            bail!("SQLite database path is not a regular file: {:?}", p);
        }
        let flags = if read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX
        };
        let conn = Connection::open_with_flags(p, flags)
            .with_context(|| format!("failed to open SQLite DB {:?}", p))?;
        let mut drv = Self::from_connection(conn, max_write_rows)?;
        drv.opened_read_only_handle = read_only;
        if read_only {
            drv.set_read_only(true)?;
        }
        Ok(drv)
    }

    pub fn from_connection(conn: Connection, max_write_rows: usize) -> Result<Self> {
        let user_tables = list_sqlite_user_tables(&conn)?;
        let mut table_versions = BTreeMap::new();
        for tbl in &user_tables {
            table_versions.insert(tbl.to_lowercase(), 0);
        }
        let last_sqlite_data_version = query_sqlite_data_version(&conn);
        Ok(Self {
            conn,
            db_version: 0,
            last_sqlite_data_version,
            event_counter: 0,
            table_versions,
            pending_verification_tables: BTreeSet::new(),
            max_write_rows,
            read_only: false,
            opened_read_only_handle: false,
            task_writes_committed: 0,
            task_db_reads_executed: 0,
            steward_auth: StewardAuthState::NotRequired,
            catalog_discovered: false,
            probed_table_names: BTreeSet::new(),
            task_candidate_choices: Vec::new(),
            task_last_read_columns: Vec::new(),
            task_last_read_rows: Vec::new(),
            session_last_read_columns: Vec::new(),
            session_last_read_rows: Vec::new(),
            session_candidate_choices: Vec::new(),
            steward_policy_columns: Vec::new(),
            candidate_disambiguation_enabled: false,
            undo_seq: 0,
            undo_stack: Vec::new(),
        })
    }

    /// Number of committed prompt steps currently available to roll back via `/undo`.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Inspect the active `/undo` savepoint stack (oldest to newest).
    pub fn undo_stack(&self) -> &[SessionUndoCheckpoint] {
        &self.undo_stack
    }

    /// Open a session-level SQLite `SAVEPOINT portia_step_N` prior to executing a user prompt turn.
    /// Returns `Ok(None)` when the driver is in `read_only` mode.
    pub fn open_turn_undo_savepoint(
        &mut self,
        prompt_summary: impl Into<String>,
    ) -> Result<Option<SessionUndoCheckpoint>> {
        if self.read_only {
            return Ok(None);
        }
        self.sync_external_mutations();
        self.undo_seq += 1;
        let savepoint_name = format!("portia_step_{}", self.undo_seq);
        self.conn
            .execute_batch(&format!("SAVEPOINT {savepoint_name};"))
            .with_context(|| format!("failed to open undo SAVEPOINT {savepoint_name}"))?;
        Ok(Some(SessionUndoCheckpoint {
            savepoint_name,
            prompt_summary: prompt_summary.into(),
            db_version_before: self.db_version,
            table_versions_before: self.table_versions.clone(),
        }))
    }

    /// Release an unchanged turn savepoint (`db_version == checkpoint.db_version_before`) without
    /// pushing an empty entry onto the `/undo` stack, or push it onto the `/undo` stack if the turn
    /// mutated the database. Returns the updated [`Self::undo_depth`].
    pub fn finalize_turn_undo_savepoint(
        &mut self,
        checkpoint: SessionUndoCheckpoint,
    ) -> Result<usize> {
        if self.db_version > checkpoint.db_version_before {
            self.undo_stack.push(checkpoint);
        } else {
            self.conn
                .execute_batch(&format!("RELEASE SAVEPOINT {};", checkpoint.savepoint_name))
                .with_context(|| {
                    format!(
                        "failed to release unchanged SAVEPOINT {}",
                        checkpoint.savepoint_name
                    )
                })?;
        }
        Ok(self.undo_stack.len())
    }

    /// Roll back and release an aborted in-flight turn's savepoint (e.g. when cancelled mid-turn).
    pub fn abort_turn_undo_savepoint(&mut self, checkpoint: SessionUndoCheckpoint) -> Result<()> {
        rollback_and_release_savepoint(&self.conn, &checkpoint.savepoint_name).map_err(|err| {
            anyhow::anyhow!(
                "failed to roll back cancelled SAVEPOINT {}: {err}",
                checkpoint.savepoint_name
            )
        })?;
        self.db_version = checkpoint.db_version_before;
        self.table_versions = checkpoint.table_versions_before;
        self.last_sqlite_data_version = query_sqlite_data_version(&self.conn);
        self.pending_verification_tables.clear();
        self.task_writes_committed = 0;
        Ok(())
    }

    /// Roll back the most recent mutating prompt step (`ROLLBACK TO SAVEPOINT ...; RELEASE ...`),
    /// restoring SQLite state and version counters while keeping earlier savepoints undoable.
    pub fn undo_last_step(&mut self) -> Result<Option<UndoAppliedReceipt>> {
        let Some(top) = self.undo_stack.pop() else {
            return Ok(None);
        };
        rollback_and_release_savepoint(&self.conn, &top.savepoint_name).map_err(|err| {
            anyhow::anyhow!(
                "failed to roll back undo SAVEPOINT {}: {err}",
                top.savepoint_name
            )
        })?;
        self.db_version = top.db_version_before;
        self.table_versions = top.table_versions_before;
        self.last_sqlite_data_version = query_sqlite_data_version(&self.conn);
        self.clear_session_candidate_rows();
        Ok(Some(UndoAppliedReceipt {
            savepoint_name: top.savepoint_name,
            prompt_summary: top.prompt_summary,
            restored_db_version: self.db_version,
            remaining_undo_depth: self.undo_stack.len(),
        }))
    }

    /// Commit all remaining open `/undo` savepoints to disk by releasing the outermost savepoint.
    pub fn flush_open_undo_savepoints(&mut self) -> Result<usize> {
        let count = self.undo_stack.len();
        if let Some(outermost) = self.undo_stack.first() {
            let root_name = outermost.savepoint_name.clone();
            self.conn
                .execute_batch(&format!("RELEASE SAVEPOINT {root_name};"))
                .with_context(|| format!("failed to release outermost SAVEPOINT {root_name}"))?;
            self.undo_stack.clear();
            self.last_sqlite_data_version = query_sqlite_data_version(&self.conn);
        }
        Ok(count)
    }

    /// Configure explicit steward policy column bindings (`"col_name"` or `"table.col_name"`).
    pub fn with_steward_policy_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.set_steward_policy_columns(columns);
        self
    }

    /// Mutably set explicit steward policy column bindings (`"col_name"` or `"table.col_name"`).
    pub fn set_steward_policy_columns<I, S>(&mut self, columns: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.steward_policy_columns = columns
            .into_iter()
            .map(|s| s.into().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    /// Explicitly configured steward policy column bindings.
    pub fn steward_policy_columns(&self) -> &[String] {
        &self.steward_policy_columns
    }

    /// Opt into (or out of) automatic multi-row `MultiSelect` candidate disambiguation menus.
    pub fn with_candidate_disambiguation(mut self, enabled: bool) -> Self {
        self.candidate_disambiguation_enabled = enabled;
        self
    }

    /// Mutably toggle automatic multi-row `MultiSelect` candidate disambiguation menus.
    pub fn set_candidate_disambiguation(&mut self, enabled: bool) {
        self.candidate_disambiguation_enabled = enabled;
        if !enabled {
            self.task_candidate_choices.clear();
            self.session_candidate_choices.clear();
        }
    }

    /// Whether automatic multi-row `MultiSelect` candidate disambiguation menus are enabled.
    pub fn candidate_disambiguation_enabled(&self) -> bool {
        self.candidate_disambiguation_enabled
    }

    /// Synchronize database/table versions and invalidate pending verification if an external
    /// SQLite connection committed one or more mutations to the underlying database file (`PRAGMA data_version`).
    pub fn sync_external_mutations(&mut self) -> bool {
        let current_dv = query_sqlite_data_version(&self.conn);
        if current_dv != self.last_sqlite_data_version {
            let delta = if current_dv > self.last_sqlite_data_version {
                (current_dv - self.last_sqlite_data_version) as u64
            } else {
                1
            };
            self.last_sqlite_data_version = current_dv;
            self.db_version += delta;
            if let Ok(user_tables) = list_sqlite_user_tables(&self.conn) {
                for tbl in user_tables {
                    let low = tbl.to_lowercase();
                    let next_ver = self.table_versions.get(&low).copied().unwrap_or(0) + delta;
                    self.table_versions.insert(low, next_ver);
                }
            } else {
                for ver in self.table_versions.values_mut() {
                    *ver += delta;
                }
            }
            self.pending_verification_tables.clear();
            return true;
        }
        false
    }

    /// Reset task-local write accounting at the start of a new user task in a multi-task
    /// REPL or TUI session while preserving connection-wide `db_version`, `table_versions`,
    /// `catalog_discovered`, `probed_table_names`, `session_last_read_rows`,
    /// and `session_candidate_choices`.
    pub fn begin_new_task(&mut self) {
        self.sync_external_mutations();
        self.task_writes_committed = 0;
        self.pending_verification_tables.clear();
        self.task_db_reads_executed = 0;
        self.steward_auth = StewardAuthState::NotRequired;
        self.task_candidate_choices.clear();
        self.task_last_read_columns.clear();
        self.task_last_read_rows.clear();
    }

    /// Clear session-persistent candidate rows, choices, and causal
    /// schema discovery state (e.g., on explicit session KV reset).
    pub fn clear_session_candidate_rows(&mut self) {
        self.sync_external_mutations();
        self.begin_new_task();
        self.session_last_read_columns.clear();
        self.session_last_read_rows.clear();
        self.session_candidate_choices.clear();
        self.catalog_discovered = false;
        self.probed_table_names.clear();
    }

    /// Whether `sqlite_master` catalog discovery has been executed in the current session.
    pub fn catalog_discovered(&self) -> bool {
        self.catalog_discovered
    }

    /// Set of user table names causally inspected via `PRAGMA table_info(...)` or `Mode::Probe` / `Mode::Read`.
    pub fn probed_table_names(&self) -> &BTreeSet<String> {
        &self.probed_table_names
    }

    /// Monotonic task-local Steward authorization state.
    pub fn steward_auth(&self) -> StewardAuthState {
        self.steward_auth
    }

    /// Whether Steward granted `AskDecision::Allow` during the current user task.
    pub fn task_steward_grant_approved(&self) -> bool {
        self.steward_auth.is_granted()
    }

    /// Explicitly mark all user tables in the database as catalog-discovered and probed
    /// (useful for single-turn unit tests that pre-seed probed schema state).
    pub fn mark_all_tables_probed(&mut self) -> Result<()> {
        let tables = list_sqlite_user_tables(&self.conn)?;
        self.catalog_discovered = true;
        for tbl in tables {
            self.probed_table_names.insert(tbl);
        }
        Ok(())
    }

    /// Return `(all_table_names, probed_schemas)` where `probed_schemas` filters
    /// [`Self::inspect_schema_metadata`] to tables in [`Self::probed_table_names`].
    pub fn inspect_epistemic_schema_state(&self) -> Result<(Vec<String>, Vec<TableSchema>)> {
        let all_schemas = self.inspect_schema_metadata()?;
        let all_table_names: Vec<String> =
            all_schemas.iter().map(|s| s.table_name.clone()).collect();
        let probed_schemas: Vec<TableSchema> = all_schemas
            .into_iter()
            .filter(|s| {
                self.probed_table_names
                    .iter()
                    .any(|p| p.eq_ignore_ascii_case(&s.table_name))
            })
            .collect();
        Ok((all_table_names, probed_schemas))
    }

    /// Session-persistent raw candidate rows from the most recent `Mode::Read` (or initial table preview).
    pub fn session_last_read_rows(&self) -> &[Vec<serde_json::Value>] {
        &self.session_last_read_rows
    }

    /// Session-persistent candidate row choices (up to 6 rows) from the most recent `Mode::Read` (or initial table preview).
    pub fn session_candidate_choices(&self) -> &[String] {
        &self.session_candidate_choices
    }

    /// Number of `Mode::Probe` / `Mode::Read` turns executed in the current user task.
    pub fn task_db_reads_executed(&self) -> u64 {
        self.task_db_reads_executed
    }

    /// Whether the current task requires Steward `ALLOW` / `DENY` authorization before `Mode::Write`
    /// (`PendingAsk` or `Denied` blocks `Mode::Write`; `PendingAsk` activates the permission modal).
    pub fn has_pending_steward_ask_row(&self) -> bool {
        self.steward_auth.requires_ask()
    }

    /// Compact human-readable candidate row summaries from the most recent multi-row `Mode::Read`
    /// in the current user task (up to 6 rows).
    pub fn task_candidate_choices(&self) -> &[String] {
        &self.task_candidate_choices
    }

    /// Determine the [`AskInteractionMode`] for `question` strictly from ground-truth database
    /// observation state and structured JSON widget syntax (`∀prompt` Free Theorem: zero `user_prompt` or `plan` peeking).
    pub fn ask_interaction_mode(&self, question: &str) -> AskInteractionMode {
        let (choices, active_rows): (&[String], &[Vec<serde_json::Value>]) =
            if self.candidate_disambiguation_enabled {
                let rows = if !self.task_last_read_rows.is_empty() {
                    self.task_last_read_rows.as_slice()
                } else {
                    self.session_last_read_rows.as_slice()
                };
                (self.task_candidate_choices.as_slice(), rows)
            } else {
                (&[], &[])
            };
        resolve_ask_interaction_mode_from_observation_state(
            self.steward_auth.requires_ask(),
            choices,
            active_rows,
            self.task_db_reads_executed > 0,
            question,
        )
    }

    /// Classify a completed turn into its [`CoroutineStepEffect`] (`AutoContinueDb`,
    /// `SuspendForGuiChoice`, or `YieldToPrompt`) strictly from `Mode` and ground-truth
    /// SQLite observation state (`∀prompt`: omits `user_prompt` and `plan`).
    pub fn classify_coroutine_step_effect(&self, mode: Mode, payload: &str) -> CoroutineStepEffect {
        match mode {
            Mode::Probe | Mode::Read | Mode::Write => CoroutineStepEffect::AutoContinueDb,
            Mode::Result | Mode::Abstain => CoroutineStepEffect::YieldToPrompt,
            Mode::Ask => {
                let gui_mode = self.ask_interaction_mode(payload);
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

    /// Number of committed writes in the current user task since the last [`Self::begin_new_task`].
    pub fn task_writes_committed(&self) -> u64 {
        self.task_writes_committed
    }

    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        if !read_only && self.opened_read_only_handle {
            bail!("cannot clear read_only on a SQLite handle opened with SQLITE_OPEN_READ_ONLY");
        }
        self.conn
            .execute_batch(if read_only {
                "PRAGMA query_only = ON;"
            } else {
                "PRAGMA query_only = OFF;"
            })
            .context("failed to update SQLite PRAGMA query_only")?;
        self.read_only = read_only;
        Ok(())
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    pub fn db_version(&self) -> u64 {
        let current_dv = query_sqlite_data_version(&self.conn);
        let pending_external = if current_dv > self.last_sqlite_data_version {
            (current_dv - self.last_sqlite_data_version) as u64
        } else if current_dv != self.last_sqlite_data_version {
            1
        } else {
            0
        };
        self.db_version + pending_external
    }

    /// Inspect rows without adding a model observation or changing epistemic state.
    pub(crate) fn database_snapshot(&self) -> Result<portia_sql::engine::DatabaseSnapshot> {
        Ok(portia_sql::engine::dump_all_tables(&self.conn)?)
    }

    /// Inspect table and column metadata (`sqlite_master` + `PRAGMA table_info(...)`) with
    /// `row_count: 0` and **zero** `SELECT COUNT(*)` table scans.
    pub fn inspect_schema_metadata(&self) -> Result<Vec<TableSchemaInfo>> {
        Ok(inspect_all_table_schemas(&self.conn)?)
    }

    /// Inspect table and column metadata along with live `SELECT COUNT(*)` row counts for display views.
    pub fn inspect_schema(&self) -> Result<Vec<TableSchemaInfo>> {
        let mut out = self.inspect_schema_metadata()?;
        for tbl_info in &mut out {
            let count_sql = format!(
                "SELECT COUNT(*) FROM \"{}\"",
                tbl_info.table_name.replace('"', "\"\"")
            );
            let row_count: i64 = self.conn.query_row(&count_sql, [], |r| r.get(0))?;
            tbl_info.row_count = row_count as usize;
        }
        Ok(out)
    }

    /// Backwards-compatible alias for [`Self::inspect_schema_metadata`] returning unified
    /// [`portia_core::TableSchema`] descriptors without `SELECT COUNT(*)` scans.
    pub fn inspect_grammar_trie_schemas(&self) -> Result<Vec<TableSchema>> {
        self.inspect_schema_metadata()
    }

    fn next_event_id(&mut self) -> String {
        self.event_counter += 1;
        format!("obs-{:04}", self.event_counter)
    }

    /// Execute a semantic [`TypedTurnAction`] against the live database or inquirer channel.
    pub fn execute_typed_action(
        &mut self,
        action: &TypedTurnAction,
        ask_callback: impl FnMut(&str) -> Result<String>,
    ) -> DriverObservation {
        match action {
            TypedTurnAction::ProbeSchema { sql } => {
                let event_id = self.next_event_id();
                self.exec_probe_or_read(event_id, Mode::Probe, sql)
            }
            TypedTurnAction::ReadQuery { sql } => {
                let event_id = self.next_event_id();
                self.exec_probe_or_read(event_id, Mode::Read, sql)
            }
            TypedTurnAction::WriteMutation { sql } => {
                let event_id = self.next_event_id();
                self.exec_write(event_id, sql)
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

    /// Execute a completed relay turn against the live database or inquirer channel.
    pub fn execute_turn(
        &mut self,
        mode: Mode,
        payload: &str,
        mut ask_callback: impl FnMut(&str) -> Result<String>,
    ) -> DriverObservation {
        self.sync_external_mutations();
        let event_id = self.next_event_id();
        match mode {
            Mode::Probe => self.exec_probe_or_read(event_id, Mode::Probe, payload),
            Mode::Read => self.exec_probe_or_read(event_id, Mode::Read, payload),
            Mode::Write => self.exec_write(event_id, payload),
            Mode::Ask => {
                let question = payload.trim();
                let prompt_kind = if self.steward_auth.requires_ask() {
                    AskPromptKind::Permission
                } else {
                    classify_ask_prompt_kind(question)
                };
                match ask_callback(question) {
                    Ok(reply) => {
                        let parsed_res = parse_ask_user_reply_for_kind(prompt_kind, &reply)
                            .or_else(|_| {
                                parse_ask_user_reply_for_kind(AskPromptKind::Clarification, &reply)
                            });
                        match parsed_res {
                            Ok(parsed_ask) => {
                                let (cols, ask_rows) = match &parsed_ask {
                                    AskUserReply::Decision {
                                        decision,
                                        raw_reply,
                                    } => {
                                        self.task_candidate_choices.clear();
                                        self.steward_auth = self
                                            .steward_auth
                                            .record_decision(*decision == AskDecision::Allow);
                                        (
                                            vec![
                                                "reply".to_string(),
                                                "decision".to_string(),
                                                "status".to_string(),
                                            ],
                                            vec![vec![
                                                serde_json::Value::String(raw_reply.clone()),
                                                serde_json::Value::String(
                                                    decision.as_str().to_string(),
                                                ),
                                                serde_json::Value::String(
                                                    decision.status_str().to_string(),
                                                ),
                                            ]],
                                        )
                                    }
                                    AskUserReply::Clarification { clarification } => {
                                        self.task_candidate_choices.clear();
                                        let (active_cols, active_rows) =
                                            if !self.task_last_read_rows.is_empty() {
                                                (
                                                    self.task_last_read_columns.as_slice(),
                                                    self.task_last_read_rows.as_slice(),
                                                )
                                            } else {
                                                (
                                                    self.session_last_read_columns.as_slice(),
                                                    self.session_last_read_rows.as_slice(),
                                                )
                                            };
                                        if resolve_clarification_steward_ask_from_rows_with_columns(
                                            clarification,
                                            active_cols,
                                            active_rows,
                                            &self.steward_policy_columns,
                                        ) == Some(true)
                                        {
                                            self.steward_auth =
                                                self.steward_auth.latch_ask_requirement();
                                        }
                                        (
                                            vec!["clarification".to_string(), "status".to_string()],
                                            vec![vec![
                                                serde_json::Value::String(clarification.clone()),
                                                serde_json::Value::String("CLARIFIED".to_string()),
                                            ]],
                                        )
                                    }
                                };
                                DriverObservation {
                                    event_id,
                                    mode: Mode::Ask,
                                    ok: true,
                                    db_version: self.db_version,
                                    table_versions: self.table_versions.clone(),
                                    columns: cols,
                                    rows: ask_rows,
                                    rows_affected: 1,
                                    message: Some(reply),
                                    error: None,
                                    operation: None,
                                    ask_reply: Some(parsed_ask),
                                    steward_auth: self.steward_auth,
                                    is_chat_continuation: false,
                                }
                            }
                            Err(parse_err) => DriverObservation {
                                event_id,
                                mode: Mode::Ask,
                                ok: false,
                                db_version: self.db_version,
                                table_versions: self.table_versions.clone(),
                                columns: vec![],
                                rows: vec![],
                                rows_affected: 0,
                                message: None,
                                error: Some(parse_err.to_string()),
                                operation: None,
                                ask_reply: None,
                                steward_auth: self.steward_auth,
                                is_chat_continuation: false,
                            },
                        }
                    }
                    Err(e) => DriverObservation {
                        event_id,
                        mode: Mode::Ask,
                        ok: false,
                        db_version: self.db_version,
                        table_versions: self.table_versions.clone(),
                        columns: vec![],
                        rows: vec![],
                        rows_affected: 0,
                        message: None,
                        error: Some(e.to_string()),
                        operation: None,
                        ask_reply: None,
                        steward_auth: self.steward_auth,
                        is_chat_continuation: false,
                    },
                }
            }
            Mode::Result | Mode::Abstain => DriverObservation {
                event_id,
                mode,
                ok: true,
                db_version: self.db_version,
                table_versions: self.table_versions.clone(),
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                message: Some(payload.trim().to_string()),
                error: None,
                operation: None,
                ask_reply: None,
                steward_auth: self.steward_auth,
                is_chat_continuation: false,
            },
        }
    }

    fn exec_probe_or_read(
        &mut self,
        event_id: String,
        mode: Mode,
        raw_sql: &str,
    ) -> DriverObservation {
        self.sync_external_mutations();
        let sql_owned = strip_markdown_sql_fence(raw_sql);
        let sql = sql_owned.as_str();
        if let Ok(schemas) = self.inspect_schema_metadata() {
            if let Err(trie_err) = validate_completed_sql_against_schemas(sql, &schemas) {
                return DriverObservation {
                    event_id,
                    mode,
                    ok: false,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: None,
                    error: Some(format!("GrammarTrieValidationError: {trie_err}")),
                    operation: None,
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                };
            }
        }
        let mut referenced_tables = match parse_read_or_probe_tables(sql) {
            Ok(tbls) => tbls,
            Err(err_msg) => {
                return DriverObservation {
                    event_id,
                    mode,
                    ok: false,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: None,
                    error: Some(err_msg),
                    operation: None,
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                };
            }
        };

        let _ = self.conn.execute_batch("PRAGMA query_only = ON;");
        if let Ok(auth_tables) = collect_read_tables_via_sqlite_authorizer(&self.conn, sql) {
            referenced_tables.extend(auth_tables);
        }
        let res = (|| -> Result<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
            let mut stmt = self.conn.prepare(sql)?;
            let col_names: Vec<String> = stmt
                .column_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            let col_count = col_names.len();
            let mut rows_out = Vec::new();
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let mut vals = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let v = match row.get_ref(i)? {
                        ValueRef::Null => serde_json::Value::Null,
                        ValueRef::Integer(n) => serde_json::Value::from(n),
                        ValueRef::Real(f) => serde_json::Value::from(f),
                        ValueRef::Text(bytes) => {
                            serde_json::Value::String(String::from_utf8_lossy(bytes).into_owned())
                        }
                        ValueRef::Blob(b) => {
                            serde_json::Value::String(format!("<blob:{}B>", b.len()))
                        }
                    };
                    vals.push(v);
                }
                rows_out.push(vals);
            }
            Ok((col_names, rows_out))
        })();
        if !self.read_only {
            let _ = self.conn.execute_batch("PRAGMA query_only = OFF;");
        }

        let read_operation = if mode == Mode::Read {
            Some(ObservationOperation::SqlRead(if self.read_only {
                ReadPurpose::ReadOnly
            } else if referenced_tables.iter().any(|table| {
                self.pending_verification_tables
                    .contains(&table.to_lowercase())
            }) {
                ReadPurpose::Verification
            } else {
                ReadPurpose::Candidate
            }))
        } else {
            None
        };
        match res {
            Ok((columns, rows)) => {
                self.task_db_reads_executed += 1;
                if sql.to_ascii_lowercase().contains("sqlite_master") {
                    self.catalog_discovered = true;
                }
                let known_user_tables = list_sqlite_user_tables(&self.conn).unwrap_or_default();
                for tbl in &referenced_tables {
                    if let Some(canonical) = known_user_tables
                        .iter()
                        .find(|u| u.eq_ignore_ascii_case(tbl))
                    {
                        self.probed_table_names.insert(canonical.clone());
                    } else if !tbl.eq_ignore_ascii_case("sqlite_master")
                        && !tbl.to_ascii_lowercase().starts_with("sqlite_")
                    {
                        self.probed_table_names.insert(tbl.clone());
                    }
                }
                if mode == Mode::Read {
                    self.task_last_read_columns = columns.clone();
                    self.task_last_read_rows = rows.clone();
                    if !rows.is_empty() {
                        self.session_last_read_columns = columns.clone();
                        self.session_last_read_rows = rows.clone();
                    }
                    let is_ambiguous_disambiguation_scan =
                        if self.candidate_disambiguation_enabled && rows.len() >= 2 {
                            let summarized: Vec<String> = rows
                                .iter()
                                .take(6)
                                .filter_map(|r| summarize_candidate_row_choice(r))
                                .collect();
                            let ambiguous = summarized.len() >= 2;
                            self.task_candidate_choices = summarized.clone();
                            if ambiguous {
                                self.session_candidate_choices = summarized;
                            }
                            ambiguous
                        } else {
                            self.task_candidate_choices.clear();
                            false
                        };
                    let any_row_requires_ask = rows.iter().any(|row| {
                        row_has_steward_ask_policy_in_columns(
                            &columns,
                            row,
                            &self.steward_policy_columns,
                        )
                    });
                    let all_rows_require_ask = !rows.is_empty()
                        && rows.iter().all(|row| {
                            row_has_steward_ask_policy_in_columns(
                                &columns,
                                row,
                                &self.steward_policy_columns,
                            )
                        });
                    if if is_ambiguous_disambiguation_scan {
                        all_rows_require_ask
                    } else {
                        any_row_requires_ask
                    } {
                        self.steward_auth = self.steward_auth.latch_ask_requirement();
                    }
                    for tbl in referenced_tables {
                        self.pending_verification_tables.remove(&tbl.to_lowercase());
                    }
                }
                let row_len = rows.len();
                DriverObservation {
                    event_id,
                    mode,
                    ok: true,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns,
                    rows,
                    rows_affected: row_len,
                    message: None,
                    error: None,
                    operation: read_operation,
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Err(e) => DriverObservation {
                event_id,
                mode,
                ok: false,
                db_version: self.db_version,
                table_versions: self.table_versions.clone(),
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                message: None,
                error: Some(format!("SQL read error: {}", e)),
                operation: read_operation,
                ask_reply: None,
                steward_auth: self.steward_auth,
                is_chat_continuation: false,
            },
        }
    }

    fn exec_write(&mut self, event_id: String, raw_sql: &str) -> DriverObservation {
        self.sync_external_mutations();
        let sql_owned = strip_markdown_sql_fence(raw_sql);
        let sql = sql_owned.as_str();
        let best_effort_op = parse_single_write_statement_target_and_op(sql)
            .ok()
            .map(|(_, op)| ObservationOperation::SqlWrite(op));
        if self.read_only {
            let dom_err = EngineError::ReadOnlySessionViolation {
                sql: sql.to_string(),
            };
            return DriverObservation {
                event_id,
                mode: Mode::Write,
                ok: false,
                db_version: self.db_version,
                table_versions: self.table_versions.clone(),
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                message: None,
                error: Some(dom_err.to_string()),
                operation: best_effort_op,
                ask_reply: None,
                steward_auth: self.steward_auth,
                is_chat_continuation: false,
            };
        }
        if let Ok(schemas) = self.inspect_schema_metadata() {
            if let Err(trie_err) = validate_completed_sql_against_schemas(sql, &schemas) {
                let dom_err = EngineError::GrammarTrieValidationError {
                    sql: sql.to_string(),
                    detail: trie_err,
                };
                return DriverObservation {
                    event_id,
                    mode: Mode::Write,
                    ok: false,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: None,
                    error: Some(dom_err.to_string()),
                    operation: best_effort_op,
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                };
            }
        }
        let (target_table, sql_write_op) = match parse_single_write_statement_target_and_op(sql) {
            Ok(pair) => pair,
            Err(err_msg) => {
                return DriverObservation {
                    event_id,
                    mode: Mode::Write,
                    ok: false,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: None,
                    error: Some(err_msg),
                    operation: None,
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                };
            }
        };
        let write_op = ObservationOperation::SqlWrite(sql_write_op);

        if self.steward_auth.blocks_write() {
            let reason = if self.steward_auth == StewardAuthState::Denied {
                "StewardAuthorizationViolation: Steward DENY recorded for current task; WRITE blocked"
            } else {
                "StewardAuthorizationViolation: pending ASK-gated row requires Steward ALLOW before WRITE"
            };
            return DriverObservation {
                event_id,
                mode: Mode::Write,
                ok: false,
                db_version: self.db_version,
                table_versions: self.table_versions.clone(),
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                message: None,
                error: Some(reason.to_string()),
                operation: Some(write_op),
                ask_reply: None,
                steward_auth: self.steward_auth,
                is_chat_continuation: false,
            };
        }

        let sp_name = "portia_write_guard";
        if let Err(e) = self.conn.execute_batch(&format!("SAVEPOINT {sp_name};")) {
            return DriverObservation {
                event_id,
                mode: Mode::Write,
                ok: false,
                db_version: self.db_version,
                table_versions: self.table_versions.clone(),
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                message: None,
                error: Some(format!("failed to open SAVEPOINT: {}", e)),
                operation: Some(write_op),
                ask_reply: None,
                steward_auth: self.steward_auth,
                is_chat_continuation: false,
            };
        }

        let layouts_result = inspect_sqlite_table_layouts(&self.conn);
        let layouts_uncertain = layouts_result.is_err();
        let layouts = Arc::new(layouts_result.unwrap_or_default());

        let enforce_steward_guard =
            !self.steward_policy_columns.is_empty() && !self.steward_auth.is_granted();
        let mut steward_col_indices_by_table: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut steward_schema_uncertain = false;
        if enforce_steward_guard {
            match self.inspect_schema_metadata() {
                Ok(schemas) => {
                    for sc in schemas {
                        let idxs: Vec<usize> = sc
                            .columns
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, c)| {
                                steward_column_matches(
                                    &self.steward_policy_columns,
                                    Some(&sc.table_name),
                                    &c.name,
                                )
                                .then_some(idx)
                            })
                            .collect();
                        steward_col_indices_by_table.insert(sc.table_name.to_lowercase(), idxs);
                    }
                }
                Err(_) => {
                    steward_schema_uncertain = true;
                }
            }
        }
        let has_virtual_or_unknown_table = layouts_uncertain
            || steward_schema_uncertain
            || layouts
                .iter()
                .any(|((_schema, tbl), layout)| !tbl.starts_with("sqlite_") && layout.is_none());
        let pre_write_snapshot = if enforce_steward_guard && has_virtual_or_unknown_table {
            match portia_sql::engine::dump_all_tables(&self.conn) {
                Ok(snap) => Some(snap),
                Err(snap_err) => {
                    let _ = rollback_and_release_savepoint(&self.conn, sp_name);
                    return DriverObservation {
                        event_id,
                        mode: Mode::Write,
                        ok: false,
                        db_version: self.db_version,
                        table_versions: self.table_versions.clone(),
                        columns: vec![],
                        rows: vec![],
                        rows_affected: 0,
                        message: None,
                        error: Some(format!("failed to snapshot pre-write state: {snap_err}")),
                        operation: Some(write_op),
                        ask_reply: None,
                        steward_auth: self.steward_auth,
                        is_chat_continuation: false,
                    };
                }
            }
        } else {
            None
        };

        let sequences_before = inspect_sqlite_sequences(&self.conn, &layouts);
        let mutation_trace = Arc::new(Mutex::new(SqliteMutationTrace {
            uncertain: layouts_uncertain || sequences_before.is_err(),
            ..SqliteMutationTrace::default()
        }));
        let row_trace = Arc::clone(&mutation_trace);
        let row_layouts = Arc::clone(&layouts);
        let row_steward_cols = Arc::new(steward_col_indices_by_table);
        self.conn.preupdate_hook(Some(
            move |_: Action, schema: &str, table: &str, case: &PreUpdateCase| {
                let table = table.to_lowercase();
                let has_rowid = row_layouts
                    .get(&(schema.to_lowercase(), table.clone()))
                    .copied()
                    .flatten();
                let changed = preupdate_changes_values(case, has_rowid);
                let steward_check = if enforce_steward_guard {
                    match row_steward_cols.get(&table) {
                        Some(col_indices) => {
                            preupdate_old_row_touches_steward_ask(case, col_indices)
                        }
                        None if !table.starts_with("sqlite_") => Err(()),
                        None => Ok(false),
                    }
                } else {
                    Ok(false)
                };
                if let Ok(mut trace) = row_trace.lock() {
                    if changed {
                        trace.changed_tables.insert(table);
                    }
                    match steward_check {
                        Ok(true) => trace.violated_steward_ask = true,
                        Ok(false) => {}
                        Err(()) => trace.uncertain = true,
                    }
                }
            },
        ));

        let mutated_tables_arc: Arc<Mutex<BTreeSet<String>>> =
            Arc::new(Mutex::new(BTreeSet::new()));
        let mutated_hook = Arc::clone(&mutated_tables_arc);
        let authorizer_trace = Arc::clone(&mutation_trace);
        self.conn
            .authorizer(Some(move |ctx: AuthContext<'_>| -> Authorization {
                if let AuthAction::Insert { table_name }
                | AuthAction::Update { table_name, .. }
                | AuthAction::Delete { table_name } = ctx.action
                {
                    let t_low = table_name.to_lowercase();
                    if !t_low.starts_with("sqlite_") {
                        if let Ok(mut guard) = mutated_hook.lock() {
                            guard.insert(t_low.clone());
                        }
                        let layout = ctx.database_name.and_then(|schema| {
                            layouts
                                .get(&(schema.to_lowercase(), t_low))
                                .copied()
                                .flatten()
                        });
                        // Virtual-table implementations need not fire row callbacks.
                        if layout.is_none() {
                            if let Ok(mut trace) = authorizer_trace.lock() {
                                trace.uncertain = true;
                            }
                        }
                    }
                }
                Authorization::Allow
            }));

        let total_changes_before = self.conn.total_changes();
        let exec_res = self.conn.execute(sql, []);
        self.conn
            .authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
        self.conn
            .preupdate_hook(None::<fn(Action, &str, &str, &PreUpdateCase)>);

        match exec_res {
            Ok(direct_changed) => {
                let total_changes_after = self.conn.total_changes();
                let trigger_inclusive_changed =
                    total_changes_after.saturating_sub(total_changes_before) as usize;
                let affected = direct_changed.max(trigger_inclusive_changed);

                if affected > self.max_write_rows {
                    let rb_status = match rollback_and_release_savepoint(&self.conn, sp_name) {
                        Ok(()) => "rolled back SAVEPOINT".to_string(),
                        Err(rb_err) => format!("ROLLBACK FAILED ({rb_err})"),
                    };
                    let dom_err = EngineError::MaxWriteRowsExceeded {
                        rows_affected: affected,
                        max_write_rows: self.max_write_rows,
                        sql: format!(
                            "{sql} (direct={direct_changed}, total_with_triggers={trigger_inclusive_changed}, {rb_status})"
                        ),
                    };
                    return DriverObservation {
                        event_id,
                        mode: Mode::Write,
                        ok: false,
                        db_version: self.db_version,
                        table_versions: self.table_versions.clone(),
                        columns: vec![],
                        rows: vec![],
                        rows_affected: 0,
                        message: None,
                        error: Some(dom_err.to_string()),
                        operation: Some(write_op),
                        ask_reply: None,
                        steward_auth: self.steward_auth,
                        is_chat_continuation: false,
                    };
                }
                let (mut touched, hook_violated_steward_ask, uncertain) = mutation_trace
                    .lock()
                    .map(|trace| {
                        (
                            trace.changed_tables.clone(),
                            trace.violated_steward_ask,
                            trace.uncertain,
                        )
                    })
                    .unwrap_or_else(|_| (BTreeSet::new(), enforce_steward_guard, true));

                if enforce_steward_guard && affected > 0 {
                    let violated_steward_ask = if hook_violated_steward_ask {
                        true
                    } else if uncertain {
                        match (
                            &pre_write_snapshot,
                            portia_sql::engine::dump_all_tables(&self.conn),
                        ) {
                            (Some(pre_snap), Ok(post_snap)) => snapshot_mutated_steward_ask_row(
                                pre_snap,
                                &post_snap,
                                &self.steward_policy_columns,
                            ),
                            _ => true,
                        }
                    } else {
                        false
                    };
                    if violated_steward_ask {
                        self.steward_auth = self.steward_auth.latch_ask_requirement();
                        let rb_status = match rollback_and_release_savepoint(&self.conn, sp_name) {
                            Ok(()) => "rolled back SAVEPOINT".to_string(),
                            Err(rb_err) => format!("ROLLBACK FAILED ({rb_err})"),
                        };
                        return DriverObservation {
                            event_id,
                            mode: Mode::Write,
                            ok: false,
                            db_version: self.db_version,
                            table_versions: self.table_versions.clone(),
                            columns: vec![],
                            rows: vec![],
                            rows_affected: 0,
                            message: None,
                            error: Some(format!(
                                "StewardAuthorizationViolation: attempted {sql_write_op} on unapproved ASK-gated row without Steward ALLOW ({rb_status})"
                            )),
                            operation: Some(write_op),
                            ask_reply: None,
                            steward_auth: self.steward_auth,
                            is_chat_continuation: false,
                        };
                    }
                }
                if let Err(release_err) = self
                    .conn
                    .execute_batch(&format!("RELEASE SAVEPOINT {sp_name};"))
                {
                    let rb_note = match rollback_and_release_savepoint(&self.conn, sp_name) {
                        Ok(()) => "rolled back SAVEPOINT after release failure".to_string(),
                        Err(rb_err) => {
                            format!("savepoint remains unresolved after rollback error: {rb_err}")
                        }
                    };
                    let dom_err = EngineError::TransactionFinalizationError {
                        savepoint: sp_name.to_string(),
                        error: release_err.to_string(),
                        rollback_note: rb_note,
                    };
                    return DriverObservation {
                        event_id,
                        mode: Mode::Write,
                        ok: false,
                        db_version: self.db_version,
                        table_versions: self.table_versions.clone(),
                        columns: vec![],
                        rows: vec![],
                        rows_affected: 0,
                        message: None,
                        error: Some(dom_err.to_string()),
                        operation: Some(write_op),
                        ask_reply: None,
                        steward_auth: self.steward_auth,
                        is_chat_continuation: false,
                    };
                }
                self.last_sqlite_data_version = query_sqlite_data_version(&self.conn);
                if affected > 0 {
                    self.task_writes_committed += 1;
                    self.pending_verification_tables
                        .insert(target_table.to_lowercase());
                }
                let sequences_after = inspect_sqlite_table_layouts(&self.conn)
                    .and_then(|layouts| inspect_sqlite_sequences(&self.conn, &layouts));
                let metadata_changed = match (&sequences_before, &sequences_after) {
                    (Ok(before), Ok(after)) => before != after,
                    _ => true,
                };
                if uncertain {
                    touched.extend(
                        mutated_tables_arc
                            .lock()
                            .map(|g| g.clone())
                            .unwrap_or_default(),
                    );
                    touched.insert(target_table.to_lowercase());
                }
                if !touched.is_empty() || metadata_changed {
                    self.db_version += 1;
                    for tbl in touched {
                        let next_ver = self.table_versions.get(&tbl).copied().unwrap_or(0) + 1;
                        self.table_versions.insert(tbl.clone(), next_ver);
                        self.pending_verification_tables.insert(tbl);
                    }
                }
                DriverObservation {
                    event_id,
                    mode: Mode::Write,
                    ok: true,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: affected,
                    message: None,
                    error: None,
                    operation: Some(write_op),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
            Err(e) => {
                let err_msg = match rollback_and_release_savepoint(&self.conn, sp_name) {
                    Ok(()) => format!("SQL write error (rolled back): {}", e),
                    Err(rb_err) => {
                        format!("SQL write error: {} (ROLLBACK FAILED: {})", e, rb_err)
                    }
                };
                DriverObservation {
                    event_id,
                    mode: Mode::Write,
                    ok: false,
                    db_version: self.db_version,
                    table_versions: self.table_versions.clone(),
                    columns: vec![],
                    rows: vec![],
                    rows_affected: 0,
                    message: None,
                    error: Some(err_msg),
                    operation: Some(write_op),
                    ask_reply: None,
                    steward_auth: self.steward_auth,
                    is_chat_continuation: false,
                }
            }
        }
    }
}

fn strip_markdown_sql_fence(raw_sql: &str) -> String {
    let mut s = raw_sql.trim();
    if let Some(after_open) = s.strip_prefix("```sql") {
        s = after_open.trim();
    } else if let Some(after_open) = s.strip_prefix("```") {
        s = after_open.trim();
    }
    if let Some(before_close) = s.strip_suffix("```") {
        s = before_close.trim();
    }
    s.to_string()
}

impl EpistemicDomainEnv for LiveSqliteDailyDriver {
    fn begin_new_task(&mut self) {
        LiveSqliteDailyDriver::begin_new_task(self);
    }

    fn reset_epistemic_session(&mut self) {
        self.clear_session_candidate_rows();
    }

    fn state_version(&self) -> u64 {
        self.db_version()
    }

    fn steward_auth(&self) -> StewardAuthState {
        LiveSqliteDailyDriver::steward_auth(self)
    }

    fn sync_worker_context(&self, worker: &mut crate::worker::StdioQwenWorkerClient) -> Result<()> {
        let schemas = self.inspect_schema_metadata()?;
        worker.set_schema_context(schemas, None);
        Ok(())
    }

    fn legal_modes(&self) -> Result<Vec<Mode>> {
        let (all_table_names, probed_schemas) = self.inspect_epistemic_schema_state()?;
        let cart = portia_sql::EpistemicSqlGrammarCartridge {
            all_table_names: &all_table_names,
            catalog_discovered: self.catalog_discovered(),
            probed_schemas: &probed_schemas,
            pending_steward_ask: self.steward_auth.blocks_write(),
            steward_grant_approved: self.task_steward_grant_approved(),
            ambiguous_candidates: self.task_candidate_choices(),
            read_only: self.is_read_only(),
        };
        Ok(portia_core::PayloadGrammarAutomaton::legal_modes(&cart))
    }

    fn payload_grammar_spec(&self, mode: Mode) -> Result<portia_core::GrammarCartridgeSpec> {
        let (all_table_names, probed_schemas) = self.inspect_epistemic_schema_state()?;
        let cart = portia_sql::EpistemicSqlGrammarCartridge {
            all_table_names: &all_table_names,
            catalog_discovered: self.catalog_discovered(),
            probed_schemas: &probed_schemas,
            pending_steward_ask: self.steward_auth.blocks_write(),
            steward_grant_approved: self.task_steward_grant_approved(),
            ambiguous_candidates: self.task_candidate_choices(),
            read_only: self.is_read_only(),
        };
        Ok(portia_core::PayloadGrammarAutomaton::grammar_cartridge_spec(&cart, mode))
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

    fn on_tui_session_open(&self, tui: &mut crate::tui::TwoPaneTuiState) {
        tui.steward_policy_columns = self.steward_policy_columns.clone();
        tui.candidate_disambiguation_enabled = self.candidate_disambiguation_enabled;
        if tui.session_preview_rows.is_empty() && !self.session_last_read_rows().is_empty() {
            tui.session_preview_rows = self.session_last_read_rows().to_vec();
        }
    }

    fn on_tui_turn_completed(
        &self,
        mode: Mode,
        obs: &DriverObservation,
        tui: &mut crate::tui::TwoPaneTuiState,
    ) {
        if mode == Mode::Read && obs.ok && !obs.rows.is_empty() {
            tui.session_preview_rows = self.session_last_read_rows().to_vec();
        }
        if mode == Mode::Write && obs.ok && obs.rows_affected > 0 {
            if let Ok(fresh_tables) = self.inspect_schema() {
                tui.tables = fresh_tables;
            }
        }
    }
}
