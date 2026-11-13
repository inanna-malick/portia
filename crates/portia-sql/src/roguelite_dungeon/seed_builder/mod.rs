//! Unified Tensor-Product SQL Scenario Compiler ([`UnifiedSqlScenarioSpec`] & [`UnifiedSqlScenarioInstance`])
//! seeding `rusqlite::Connection` in-memory or on-disk SQLite files across all stress dimensions.

mod primary_ddl;
mod secondary_ddl;
mod spec;

#[cfg(test)]
mod tests;

pub use portia_nursery::roguelite_lattice::WavePermutation;
pub use spec::{arb_unified_sql_scenario_spec, UnifiedSqlScenarioInstance, UnifiedSqlScenarioSpec};
