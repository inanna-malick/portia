//! Multi-dimensional Roguelite stress lattice, biased band samplers, identifier swarms,
//! cliff sweepers, surface isomorphisms, and hylomorphic multi-axis shrinkers.

pub mod cliff_sweep;
pub mod dimensions;
pub mod hylo_shrinker;
pub mod isomorphisms;
pub mod proptest_support;
pub mod stress_vector;
pub mod swarm;
pub mod value_tree;

#[cfg(test)]
mod tests;

pub use portia_core::domain::{
    generate_compound_policy_matrix, sample_biased_three_band_u64, BiasedScaleBand, CollisionLocus,
    CombatStrategyClass, CompoundPolicyMatrix, LockState, NeedleDepth, NeedleDepthBand,
    PolicyClauseKind, PolicyGuardClause, PolicyState, RowLifecycleStatus, RowPolicyDecision,
    WriteOp, BOSS_FLOOR_WEIGHT, DEEP_DUNGEON_WEIGHT, SWEET_SPOT_WEIGHT,
};

pub use cliff_sweep::{
    find_exact_wipeout_cliff, find_multi_axis_ray_wipeout_cliff, interpolate_stress_ray,
    log_spaced_cliff_ladder, MultiAxisRayCliffReceipt, WipeoutCliffReceipt,
};
pub use dimensions::{
    arb_biased_three_band_range, arb_needle_depth, DimensionBandSpec, RogueliteDimension,
    NUM_ROGUELITE_DIMENSIONS,
};
pub use hylo_shrinker::{
    shrink_stress_vector_to_irreducible_cliff, IrreducibleStressInteractionReceipt,
    StressContrastiveLowerCover, StressPosetShrinkFrame,
};
pub use isomorphisms::{
    IntentTrajectoryKind, PolicyLockSurfaceIso, PrimaryKeyIsomorphism, StewardAgendaStep,
    WavePermutation,
};
pub use proptest_support::{
    dimension_biased_strategy, roguelite_stress_vector_strategy, RogueliteStressStrategy,
};
pub use stress_vector::{
    count_empirical_json_bpe_tokens, CombatStrategyAssessment, RogueliteStressVector,
};
pub use swarm::{char_hamming_distance, generate_identifier_swarm, IdentifierSwarm};
pub use value_tree::{sample_biased_roguelite_stress_vector, RogueliteValueTree};
