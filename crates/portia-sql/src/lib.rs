//! `portia-sql` — Cleanroom Relational Lens (`rusqlite::Connection` ⊥ `EpistemicOracle`),
//! C-Level SQLite `authorizer` & `preupdate_hook` Provenance, Shadow Relational Diffing,
//! `sqlparser::ast` Validation, & `proptest` Roguelite World Synthesis.

pub mod engine;
pub mod grammar_trie;
pub mod relational_ast;
pub mod roguelite_dungeon;
pub mod schema_normalizer;
pub mod v2_pointer_bridge;

pub use grammar_trie::{
    compile_epistemic_span_grammar, compile_sqlite_lark_grammar_for_schemas,
    inspect_all_table_schemas, inspect_table_schema, inspect_top_level_sql_clauses,
    is_quarantined_grammar_trie_decoy_column, validate_completed_sql_against_schemas,
    EpistemicSqlGrammarCartridge, SqliteSchemaGrammarCartridge, TopLevelSqlClauseSummary,
    UPSTREAM_SQLITE_LARK_GRAMMAR,
};
pub use portia_core::{ColumnSchema, TableSchema};

pub use engine::{
    collect_read_tables_via_sqlite_authorizer, collect_tables_from_sql_query_ast,
    configure_hardened_sqlite, list_sqlite_user_tables, parse_read_or_probe_tables,
    parse_single_write_statement_target_and_op, parse_single_write_target_table,
    rollback_and_release_savepoint, BytecodeAuthorizerTrace, ColumnRef, EngineError, SqlRow,
    SqlValue, SqlWriteOp,
};
pub use relational_ast::{
    col, qualified_col, ForeignKeyEdge, MutantMatchBasis, MutantMatchDiagnosis,
    RelationalSchemaGraph, SkippedCheckConstraintDiagnostic, SqlColumnExpr, SqlColumnRef,
    SqlComparisonOp, SqlIdent, SqlPredicateAst, SqlPredicateFrame, SqlRowDomainError,
    SqlScalarValue, SqlTruthValue, SqlTypeError,
};
