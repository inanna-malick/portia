//! Shared Nursery Scenario Coordinate Spine (`AppType` × `(ScenarioLanguage, DomainSkin, RogueliteStressVector)`).
//!
//! Provides the shared scenario coordinate, relational query shapes, failure attribution coroner,
//! steward agenda builder, and `RustAstHylo` specification types for both [`AppType::SqlSteward`]
//! and [`AppType::RustAstHylo`].

pub mod builders;
pub mod coordinate;
pub mod coroner;
pub mod rust_hylo_spec;
pub mod sql_shapes;

#[cfg(test)]
mod tests;

pub use builders::{
    arb_app_type, arb_domain_skin, arb_scenario_coordinate, arb_scenario_language,
    MaterializedScenarioDomain,
};
pub use coordinate::{AppType, DomainSkin, ScenarioCoordinate, ScenarioLanguage};
pub use coroner::{
    build_default_steward_agenda, ScenarioEpisodeSummary, ScenarioWipeoutDiagnosis,
    WaveTransitionReceipt,
};
pub use rust_hylo_spec::{
    evaluate_nursery_candidate_with_rustc, render_sealed_witness_and_approved_crate_block,
    CleanroomRustHyloSpec, DialogicCodingAffordance, HyloCartridgeLevel, HyloCartridgePhase,
    RustHyloPipelineStage, RustTodoHoleSpec, SqlStewardCoordinateVocabulary,
};
pub use sql_shapes::{
    arb_sql_comparison_operator, arb_sql_query_shape, SqlComparisonOperator, SqlQueryShape,
    UnifiedScenarioTrapModifier,
};
