//! `proptest` `ValueTree` and `Strategy` implementations for [`RogueliteStressVector`].

use proptest::prelude::*;
use proptest::strategy::{NewTree, Strategy, ValueTree};
use proptest::test_runner::TestRunner;

use super::dimensions::{
    arb_biased_three_band_range, RogueliteDimension, NUM_ROGUELITE_DIMENSIONS,
};
use super::stress_vector::RogueliteStressVector;
use super::value_tree::{sample_biased_roguelite_stress_vector, RogueliteValueTree};

impl ValueTree for RogueliteValueTree {
    type Value = RogueliteStressVector;

    fn current(&self) -> Self::Value {
        self.current_vector()
    }

    fn simplify(&mut self) -> bool {
        self.simplify_step()
    }

    fn complicate(&mut self) -> bool {
        self.complicate_step()
    }
}

/// Strategy for [`RogueliteStressVector`] that samples each dimension with the
/// `13 : 5 : 2` (`65%` sweet spot, `25%` deep dungeon, `10%` boss floor) bias
/// and shrinks via [`RogueliteValueTree`] toward [`RogueliteStressVector::sweet_spot_baseline`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RogueliteStressStrategy;

impl Strategy for RogueliteStressStrategy {
    type Tree = RogueliteValueTree;
    type Value = RogueliteStressVector;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let raw = (
            any::<[u8; NUM_ROGUELITE_DIMENSIONS]>(),
            any::<[u64; NUM_ROGUELITE_DIMENSIONS]>(),
            any::<u64>(),
        )
            .new_tree(runner)?;
        let (tickets, offsets, mod_seed) = raw.current();
        let vec = sample_biased_roguelite_stress_vector(tickets, offsets, mod_seed);
        Ok(RogueliteValueTree::new(vec))
    }
}

/// Construct a `proptest` strategy for a single [`RogueliteDimension`] using
/// [`arb_biased_three_band_range`] (`65% / 25% / 10%`).
pub fn dimension_biased_strategy(dim: RogueliteDimension) -> BoxedStrategy<usize> {
    let spec = dim.band_spec();
    arb_biased_three_band_range(
        spec.sweet_spot.0..=spec.sweet_spot.1,
        spec.deep_dungeon.0..=spec.deep_dungeon.1,
        spec.boss_floor.0..=spec.boss_floor.1,
    )
}

/// Canonical `proptest` strategy for [`RogueliteStressVector`].
pub fn roguelite_stress_vector_strategy() -> RogueliteStressStrategy {
    RogueliteStressStrategy
}
