//! Roguelite SQL Dungeon Schema Generators, Unified Tensor-Product Seed Builder,
//! & Postcondition Reference Oracle (`schema_gen`, `seed_builder`, `postcondition_oracle`).

pub mod opaque_gibberish;
mod opaque_gibberish_eval;
pub mod postcondition_oracle;
pub mod schema_gen;
pub mod seed_builder;

pub use portia_core::domain::{
    BiasedScaleBand, CollisionLocus, CombatStrategyClass, DungeonDifficultyBand, LockState,
    NeedleDepth, PolicyState, RowLifecycleStatus, RowPolicyDecision, WriteOp,
};
pub use portia_nursery::nursery_scenario::{
    arb_app_type, arb_domain_skin, arb_scenario_coordinate, arb_scenario_language,
    arb_sql_comparison_operator, arb_sql_query_shape, build_default_steward_agenda, AppType,
    CleanroomRustHyloSpec, DialogicCodingAffordance, DomainSkin, HyloCartridgeLevel,
    HyloCartridgePhase, MaterializedScenarioDomain, RustHyloPipelineStage, RustTodoHoleSpec,
    ScenarioCoordinate, ScenarioEpisodeSummary, ScenarioLanguage, ScenarioWipeoutDiagnosis,
    SqlComparisonOperator, SqlQueryShape, SqlStewardCoordinateVocabulary,
    UnifiedScenarioTrapModifier, WaveTransitionReceipt,
};
pub use portia_nursery::roguelite_lattice::{
    arb_biased_three_band_range, arb_needle_depth, generate_compound_policy_matrix,
    generate_identifier_swarm, roguelite_stress_vector_strategy,
    shrink_stress_vector_to_irreducible_cliff, CompoundPolicyMatrix, IdentifierSwarm,
    IntentTrajectoryKind, IrreducibleStressInteractionReceipt, PolicyLockSurfaceIso,
    PrimaryKeyIsomorphism, RogueliteDimension, RogueliteStressVector, StewardAgendaStep,
    StressContrastiveLowerCover, StressPosetShrinkFrame, WavePermutation,
};

pub use opaque_gibberish::{
    arb_opaque_gibberish_dungeon_spec, ForeignKeyDdlStyle, GibberishDialect,
    OpaqueGibberishDungeonInstance, OpaqueGibberishDungeonSpec, OpaqueGibberishVerdict,
};
pub use postcondition_oracle::{
    candidate_sql_matches_shape, canonical_reference_sql_for_comparison_op,
    canonical_reference_sql_for_shape, evaluate_candidate_query_against_reference,
    verify_scenario_postcondition_on_connection, SqlQueryShapeEvaluationReceipt,
};
pub use schema_gen::{
    arb_multi_table_count, arb_multi_table_fk_graph_spec, arb_row_haystack_count,
    arb_row_haystack_dungeon_spec, arb_wide_column_count, arb_wide_table_dungeon_spec,
    evaluate_raw_tanking_vs_fog_finesse, FkEdgeAction, HaystackStrategyComparison,
    MultiTableFkGraphDungeonInstance, MultiTableFkGraphDungeonSpec, RowHaystackDungeonInstance,
    RowHaystackDungeonSpec, WideTableDungeonInstance, WideTableDungeonSpec,
    WideTableFinesseComparison, DECEPTIVE_COLUMN_STEMS, HAYSTACK_CLIFF_POINTS,
};
pub use seed_builder::{
    arb_unified_sql_scenario_spec, UnifiedSqlScenarioInstance, UnifiedSqlScenarioSpec,
};
