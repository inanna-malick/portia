//! Schema-agnostic SQLite Daily-Driver environment (`LiveSqliteDailyDriver`),
//! table/column schema introspection (`TableSchemaInfo`, `ColumnSchemaInfo`),
//! and Steward authorization + `SAVEPOINT` mutation guardrails.

pub mod ask_protocol;
pub mod observation;
pub mod sqlite_env;
#[cfg(test)]
mod tests;

pub use portia_sql::engine::{
    collect_read_tables_via_sqlite_authorizer, collect_tables_from_sql_query_ast,
    list_sqlite_user_tables, parse_read_or_probe_tables,
    parse_single_write_statement_target_and_op, parse_single_write_target_table,
    rollback_and_release_savepoint, EngineError, SqlWriteOp,
};

pub use ask_protocol::{
    classify_ask_prompt_kind, ensure_none_of_above_escape_choice, extract_ask_interaction_mode,
    extract_ask_select_choices, extract_clarification_from_ask_observation,
    format_ask_payload_for_display, parse_ask_user_reply, parse_ask_user_reply_for_kind,
    resolve_ask_interaction_mode_from_observation_state,
    resolve_clarification_steward_ask_from_rows,
    resolve_clarification_steward_ask_from_rows_with_columns, row_has_steward_ask_policy,
    row_has_steward_ask_policy_in_columns, steward_column_matches, summarize_candidate_row_choice,
    AskDecision, AskInteractionMode, AskPromptKind, AskUserReply, CoroutineStepEffect,
    NONE_OF_ABOVE_ENTER_TEXT_CHOICE,
};
pub use observation::{
    AstOperation, ColumnSchemaInfo, DriverObservation, EpistemicDomainEnv, ObservationOperation,
    ReadPurpose, TableSchemaInfo,
};
pub use sqlite_env::{LiveSqliteDailyDriver, SessionUndoCheckpoint, UndoAppliedReceipt};
