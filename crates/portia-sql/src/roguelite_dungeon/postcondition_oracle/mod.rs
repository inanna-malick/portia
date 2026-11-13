//! SQL Query-Shape Reference Oracle & Post-Episode SQLite State Verifier
//! ([`evaluate_candidate_query_against_reference`] & [`verify_scenario_postcondition_on_connection`]).

mod reference_sql;
mod schema_probe;
mod shape_syntax;
mod state_verifier;

#[cfg(test)]
mod tests;

pub use reference_sql::{
    canonical_reference_sql_for_comparison_op, canonical_reference_sql_for_shape,
    evaluate_candidate_query_against_reference,
};
pub(super) use shape_syntax::sql_tokens_reference_identifier;
pub use shape_syntax::{candidate_sql_matches_shape, SqlQueryShapeEvaluationReceipt};
pub use state_verifier::verify_scenario_postcondition_on_connection;
