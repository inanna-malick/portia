//! SQLite relational provenance, hardened connection configuration, C-level bytecode
//! `authorizer` table extraction, `SAVEPOINT` rollback helpers, and whole-database shadow diffing.

mod authorizer;
mod query_ast;
mod shadow_diff;
mod types;

#[cfg(test)]
mod tests;

pub use authorizer::{
    collect_read_tables_via_sqlite_authorizer, configure_hardened_sqlite, list_sqlite_user_tables,
    rollback_and_release_savepoint,
};
pub use query_ast::{
    collect_tables_from_sql_query_ast, parse_read_or_probe_tables,
    parse_single_write_statement_target_and_op, parse_single_write_target_table, SqlWriteOp,
};
pub use shadow_diff::{
    clone_connection_to_memory, diff_shadow_databases, dump_all_tables, verify_shadow_equivalence,
};
pub use types::{
    BytecodeAuthorizerTrace, ColumnRef, DatabaseSnapshot, EngineError, FogOfWarPageMetadata,
    KeyedRow, QueryPageConfig, ReadReceipt, RelationalDiff, RowModification, RowMutationRecord,
    SqlRow, SqlValue, TableDiff, WriteReceipt,
};
