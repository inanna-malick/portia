//! Stateful `ValueTree` binary-search shrinker and biased seed sampler for [`RogueliteStressVector`].

use portia_core::domain::{CollisionLocus, NeedleDepthBand};

use super::dimensions::{RogueliteDimension, NUM_ROGUELITE_DIMENSIONS};
use super::stress_vector::RogueliteStressVector;

/// Stateful `ValueTree` shrinker for [`RogueliteStressVector`] that shrinks any failing
/// coordinate toward [`RogueliteStressVector::sweet_spot_baseline`].
#[derive(Debug, Clone)]
pub struct RogueliteValueTree {
    initial: RogueliteStressVector,
    baseline: RogueliteStressVector,
    current: RogueliteStressVector,
    active_dim_cursor: usize,
    lo_bound: usize,
    hi_fail: usize,
    probing: bool,
}

impl RogueliteValueTree {
    pub const fn new(initial: RogueliteStressVector) -> Self {
        let base = RogueliteStressVector::sweet_spot_baseline();
        let lo0 = base.row_haystack_size;
        let hi0 = if initial.row_haystack_size > lo0 {
            initial.row_haystack_size
        } else {
            lo0
        };
        Self {
            initial,
            baseline: base,
            current: initial,
            active_dim_cursor: 0,
            lo_bound: lo0,
            hi_fail: hi0,
            probing: false,
        }
    }

    pub const fn with_custom_baseline(
        initial: RogueliteStressVector,
        baseline: RogueliteStressVector,
    ) -> Self {
        let lo0 = baseline.row_haystack_size;
        let hi0 = if initial.row_haystack_size > lo0 {
            initial.row_haystack_size
        } else {
            lo0
        };
        Self {
            initial,
            baseline,
            current: initial,
            active_dim_cursor: 0,
            lo_bound: lo0,
            hi_fail: hi0,
            probing: false,
        }
    }

    pub const fn current_vector(&self) -> RogueliteStressVector {
        self.current
    }

    pub const fn initial_vector(&self) -> RogueliteStressVector {
        self.initial
    }

    /// Attempt to simplify `current` one binary-search step closer to `baseline` along the active dimension.
    pub fn simplify_step(&mut self) -> bool {
        if self.active_dim_cursor >= NUM_ROGUELITE_DIMENSIONS {
            return false;
        }

        if self.probing {
            let dim = RogueliteDimension::ALL[self.active_dim_cursor];
            self.hi_fail = self.current.get_dimension(dim);
            self.probing = false;
        }

        while self.active_dim_cursor < NUM_ROGUELITE_DIMENSIONS {
            let dim = RogueliteDimension::ALL[self.active_dim_cursor];
            if self.lo_bound < self.hi_fail {
                let probe = self.lo_bound + (self.hi_fail - 1 - self.lo_bound) / 2;
                self.current = self.current.with_dimension(dim, probe);
                self.probing = true;
                return true;
            }
            self.current = self.current.with_dimension(dim, self.hi_fail);
            self.active_dim_cursor += 1;
            if self.active_dim_cursor < NUM_ROGUELITE_DIMENSIONS {
                let next_dim = RogueliteDimension::ALL[self.active_dim_cursor];
                self.lo_bound = self.baseline.get_dimension(next_dim);
                self.hi_fail = self.current.get_dimension(next_dim).max(self.lo_bound);
                self.probing = false;
            }
        }
        false
    }

    /// Called when the previous `simplify_step()` probe passed the test (`!fails(current)`),
    /// raising `lo_bound = probe + 1` and restoring `current` to `hi_fail`.
    pub fn complicate_step(&mut self) -> bool {
        if !self.probing || self.active_dim_cursor >= NUM_ROGUELITE_DIMENSIONS {
            return false;
        }
        let dim = RogueliteDimension::ALL[self.active_dim_cursor];
        let passed_probe = self.current.get_dimension(dim);
        self.lo_bound = (passed_probe + 1).min(self.hi_fail);
        self.current = self.current.with_dimension(dim, self.hi_fail);
        self.probing = false;
        true
    }
}

/// Deterministic project-from-seed constructor for [`RogueliteStressVector`] honoring the
/// `65%` (`13/20`) sweet-spot, `25%` (`5/20`) deep-dungeon, and `10%` (`2/20`) boss-floor weights.
pub fn sample_biased_roguelite_stress_vector(
    band_tickets: [u8; NUM_ROGUELITE_DIMENSIONS],
    offset_seeds: [u64; NUM_ROGUELITE_DIMENSIONS],
    modifier_seed: u64,
) -> RogueliteStressVector {
    let mut vec = RogueliteStressVector::sweet_spot_baseline();
    for i in 0..NUM_ROGUELITE_DIMENSIONS {
        let dim = RogueliteDimension::ALL[i];
        let ticket = band_tickets[i] % 20;
        let val = dim.band_spec().sample_from_ticket(ticket, offset_seeds[i]);
        vec = vec.with_dimension(dim, val);
    }

    vec.needle_depth = NeedleDepthBand::ALL[(modifier_seed as usize) % 3];
    vec.collision_locus = CollisionLocus::ALL[((modifier_seed >> 8) as usize) % 3];
    let max_decoys = (vec.column_width_arity / 4).min(25);
    vec.deprecated_decoy_columns = if max_decoys == 0 {
        0
    } else {
        ((modifier_seed >> 16) as usize) % (max_decoys + 1)
    };
    let max_fk = vec.schema_graph_topology.clamp(1, 8);
    vec.fk_chain_depth = 1 + (((modifier_seed >> 24) as usize) % max_fk);
    let max_carry = (vec.episode_turn_horizon / 2).min(25);
    vec.chitchat_carry_distance = if max_carry == 0 {
        0
    } else {
        ((modifier_seed >> 32) as usize) % (max_carry + 1)
    };
    vec.steward_negotiation_rounds = 1 + (((modifier_seed >> 40) as usize) % 8);
    vec.homonym_overload_count = 1 + (((modifier_seed >> 48) as usize) % 12);
    vec
}
