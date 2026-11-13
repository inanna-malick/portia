//! SQLite `PRAGMA table_info` schema introspection, top-level clause scanning,
//! `llguidance` Lark grammar cartridges, and `sqlparser` AST validation.

mod epistemic_cartridge;
mod introspect;
mod lark_specialize;
mod tokenizer_scan;
mod validate;
mod validate_expr;
mod validate_query;

#[cfg(test)]
mod tests;

pub use portia_core::{ColumnSchema, TableSchema};

pub use epistemic_cartridge::{compile_epistemic_span_grammar, EpistemicSqlGrammarCartridge};
pub use introspect::{
    inspect_all_table_schemas, inspect_table_schema, is_quarantined_grammar_trie_decoy_column,
};
pub(crate) use lark_specialize::SQLITE_LARK_NONRESERVED_KEYWORDS;
pub use lark_specialize::{
    compile_sqlite_lark_grammar_for_schemas, is_sqlite_reserved_keyword,
    SqliteSchemaGrammarCartridge, SQLITE_RESERVED_KEYWORDS, UPSTREAM_SQLITE_LARK_GRAMMAR,
};
pub use tokenizer_scan::{inspect_top_level_sql_clauses, TopLevelSqlClauseSummary};
pub use validate::validate_completed_sql_against_schemas;
