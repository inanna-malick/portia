//! Shared Canonical Domain Primitives (`PolicyState`, `LockState`, `RowLifecycleStatus`,
//! `WriteOp`, `ColumnSchema`, `TableSchema`, `ForeignKeyEdge`, and stratified scale bands).
//!
//! Opt-in synthetic nursery test-oracle decoy classifiers live in [`benchmark_decoys`]
//! (and remain re-exported here for backwards compatibility with downstream nursery oracles).

pub mod benchmark_decoys;
mod governance;
mod policy_matrix;
mod scale_bands;
mod schema;

#[cfg(test)]
mod tests;

pub use benchmark_decoys::*;
pub use governance::{
    DbInterlockState, LockState, PolicyState, RowLifecycleStatus, RowPolicyDecision, SqlWriteOp,
    StewardDecision, WriteOp,
};
pub use policy_matrix::{
    generate_compound_policy_matrix, CompoundPolicyMatrix, PolicyClauseKind, PolicyGuardClause,
};
pub use scale_bands::{
    sample_biased_three_band_u64, AstDungeonFloorBand, BiasedScaleBand, CollisionLocus,
    CombatStrategyClass, DungeonDifficultyBand, HorizonBand, NeedleDepth, NeedleDepthBand,
    NeedleDepthPosition, BOSS_FLOOR_WEIGHT, DEEP_DUNGEON_WEIGHT, SWEET_SPOT_WEIGHT,
};
pub use schema::{ColumnSchema, ForeignKeyDeleteRule, ForeignKeyEdge, TableSchema};
