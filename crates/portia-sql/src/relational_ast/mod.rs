//! Relational predicate ASTs, structured SQL identifiers, Kleene three-valued row evaluation,
//! single-bug AST mutant diagnostics, and live SQLite `PRAGMA` / `CHECK` schema graphs.

mod coercion;
mod edsl;
mod ident;
mod mutant_diag;
mod predicate;
mod row_eval;
mod scalar_ops;
mod schema_graph;
mod schema_introspect;
mod sql_lowering;
mod type_check;

#[cfg(test)]
mod tests;

pub use portia_core::{ColumnSchema, TableSchema};

pub use coercion::SqlRowDomainError;
pub use edsl::{col, qualified_col, SqlColumnExpr};
pub use ident::{SqlColumnRef, SqlIdent};
pub use mutant_diag::{MutantMatchBasis, MutantMatchDiagnosis};
pub use predicate::{SqlPredicateAst, SqlPredicateFrame, SqlTruthValue};
pub use scalar_ops::{SqlComparisonOp, SqlScalarValue};
pub use schema_graph::{ForeignKeyEdge, RelationalSchemaGraph, SkippedCheckConstraintDiagnostic};
pub use schema_introspect::validate_check_predicate_against_table_schema;
pub use type_check::SqlTypeError;
