//! Hylomorphic multi-axis irreducible interaction shrinker (`shrink_stress_vector_to_irreducible_cliff`).

use portia_core::hylo::{expand_and_collapse, MappableFrame, PartiallyApplied};
use serde::{Deserialize, Serialize};

use super::dimensions::{RogueliteDimension, NUM_ROGUELITE_DIMENSIONS};
use super::stress_vector::RogueliteStressVector;

/// A verified passing Hamming-1 lower cover along a single implicated [`RogueliteDimension`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StressContrastiveLowerCover {
    pub dimension: RogueliteDimension,
    pub passing_value: usize,
    pub failing_value: usize,
    pub passing_lower_cover: RogueliteStressVector,
    pub failing_upper_cover: RogueliteStressVector,
}

/// Complete receipt returned by [`shrink_stress_vector_to_irreducible_cliff`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrreducibleStressInteractionReceipt {
    pub initial_failing: RogueliteStressVector,
    pub irreducible_failing: RogueliteStressVector,
    pub implicated_dimensions: Vec<RogueliteDimension>,
    pub contrastive_lower_covers: Vec<StressContrastiveLowerCover>,
    pub shrink_steps: usize,
    pub total_queries: usize,
}

/// Base functor frame `StressPosetShrinkFrame<Child>` for unfolding and collapsing the
/// multi-axis irreducible cliff descent via [`expand_and_collapse`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StressPosetShrinkFrame<Child> {
    Irreducible {
        irreducible: RogueliteStressVector,
        passing_covers: Vec<StressContrastiveLowerCover>,
        queries_at_leaf: usize,
    },
    DescentStep {
        dimension: RogueliteDimension,
        queries_in_step: usize,
        next: Child,
    },
}

impl MappableFrame for StressPosetShrinkFrame<PartiallyApplied> {
    type Frame<X> = StressPosetShrinkFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            StressPosetShrinkFrame::Irreducible {
                irreducible,
                passing_covers,
                queries_at_leaf,
            } => StressPosetShrinkFrame::Irreducible {
                irreducible,
                passing_covers,
                queries_at_leaf,
            },
            StressPosetShrinkFrame::DescentStep {
                dimension,
                queries_in_step,
                next,
            } => StressPosetShrinkFrame::DescentStep {
                dimension,
                queries_in_step,
                next: f(next),
            },
        }
    }
}

#[derive(Debug, Clone)]
struct StressShrinkSeed {
    current: RogueliteStressVector,
    floor: RogueliteStressVector,
    dim_cursor: usize,
    lo_pass: Option<usize>,
    hi_fail: usize,
}

/// Hylomorphic multi-axis irreducible interaction shrinker powered by [`expand_and_collapse`]:
/// steps down along each active dimension until it isolates a locally irreducible failing
/// coordinate where every active dimension above `floor` has a verified passing Hamming-1 lower cover.
pub fn shrink_stress_vector_to_irreducible_cliff<F>(
    initial: RogueliteStressVector,
    is_failing: F,
) -> IrreducibleStressInteractionReceipt
where
    F: Fn(&RogueliteStressVector) -> bool,
{
    let sweet = RogueliteStressVector::sweet_spot_baseline();
    let abs_min = RogueliteStressVector::absolute_minimum();
    let mut initial_queries = 1usize;
    let floor = if !is_failing(&sweet) && initial.dominates(&sweet) {
        sweet
    } else {
        abs_min
    };

    let seed0 = StressShrinkSeed {
        current: initial,
        floor,
        dim_cursor: 0,
        lo_pass: None,
        hi_fail: initial.get_dimension(RogueliteDimension::ALL[0]),
    };

    let (irreducible, passing_covers, shrink_steps, hylo_queries) = expand_and_collapse::<
        StressPosetShrinkFrame<PartiallyApplied>,
        StressShrinkSeed,
        (
            RogueliteStressVector,
            Vec<StressContrastiveLowerCover>,
            usize,
            usize,
        ),
    >(
        seed0,
        |mut seed| {
            let mut queries = 0usize;
            while seed.dim_cursor < NUM_ROGUELITE_DIMENSIONS {
                let dim = RogueliteDimension::ALL[seed.dim_cursor];
                let floor_val = seed.floor.get_dimension(dim);
                let cur_val = seed.current.get_dimension(dim);

                if cur_val <= floor_val {
                    seed.dim_cursor += 1;
                    if seed.dim_cursor < NUM_ROGUELITE_DIMENSIONS {
                        let next_dim = RogueliteDimension::ALL[seed.dim_cursor];
                        seed.lo_pass = None;
                        seed.hi_fail = seed.current.get_dimension(next_dim);
                    }
                    continue;
                }

                match seed.lo_pass {
                    None => {
                        let floor_probe = seed.current.with_dimension(dim, floor_val);
                        queries += 1;
                        if is_failing(&floor_probe) {
                            seed.current = floor_probe;
                            seed.dim_cursor += 1;
                            if seed.dim_cursor < NUM_ROGUELITE_DIMENSIONS {
                                let next_dim = RogueliteDimension::ALL[seed.dim_cursor];
                                seed.lo_pass = None;
                                seed.hi_fail = seed.current.get_dimension(next_dim);
                            }
                            return StressPosetShrinkFrame::DescentStep {
                                dimension: dim,
                                queries_in_step: queries,
                                next: seed,
                            };
                        } else {
                            seed.lo_pass = Some(floor_val);
                            seed.hi_fail = cur_val;
                        }
                    }
                    Some(lo) => {
                        if lo + 1 < seed.hi_fail {
                            let mid = lo + (seed.hi_fail - lo) / 2;
                            let mid_probe = seed.current.with_dimension(dim, mid);
                            queries += 1;
                            if is_failing(&mid_probe) {
                                seed.current = mid_probe;
                                seed.hi_fail = mid;
                            } else {
                                seed.lo_pass = Some(mid);
                            }
                            return StressPosetShrinkFrame::DescentStep {
                                dimension: dim,
                                queries_in_step: queries,
                                next: seed,
                            };
                        } else {
                            seed.current = seed.current.with_dimension(dim, seed.hi_fail);
                            seed.dim_cursor += 1;
                            if seed.dim_cursor < NUM_ROGUELITE_DIMENSIONS {
                                let next_dim = RogueliteDimension::ALL[seed.dim_cursor];
                                seed.lo_pass = None;
                                seed.hi_fail = seed.current.get_dimension(next_dim);
                            }
                        }
                    }
                }
            }

            let mut covers = Vec::new();
            for dim in RogueliteDimension::ALL {
                let val = seed.current.get_dimension(dim);
                let floor_val = seed.floor.get_dimension(dim);
                if val > floor_val {
                    let lower_vec = seed.current.with_dimension(dim, val - 1);
                    queries += 1;
                    if is_failing(&lower_vec) {
                        seed.current = lower_vec;
                        seed.dim_cursor = 0;
                        let first_dim = RogueliteDimension::ALL[0];
                        seed.lo_pass = None;
                        seed.hi_fail = seed.current.get_dimension(first_dim);
                        return StressPosetShrinkFrame::DescentStep {
                            dimension: dim,
                            queries_in_step: queries,
                            next: seed,
                        };
                    } else {
                        covers.push(StressContrastiveLowerCover {
                            dimension: dim,
                            passing_value: val - 1,
                            failing_value: val,
                            passing_lower_cover: lower_vec,
                            failing_upper_cover: seed.current,
                        });
                    }
                }
            }

            StressPosetShrinkFrame::Irreducible {
                irreducible: seed.current,
                passing_covers: covers,
                queries_at_leaf: queries,
            }
        },
        |frame| match frame {
            StressPosetShrinkFrame::Irreducible {
                irreducible,
                passing_covers,
                queries_at_leaf,
            } => (irreducible, passing_covers, 0, queries_at_leaf),
            StressPosetShrinkFrame::DescentStep {
                queries_in_step,
                next: (irr, covers, steps, q_rest),
                ..
            } => (irr, covers, steps + 1, queries_in_step + q_rest),
        },
    );

    initial_queries += hylo_queries;
    let implicated_dimensions = passing_covers.iter().map(|c| c.dimension).collect();

    IrreducibleStressInteractionReceipt {
        initial_failing: initial,
        irreducible_failing: irreducible,
        implicated_dimensions,
        contrastive_lower_covers: passing_covers,
        shrink_steps,
        total_queries: initial_queries,
    }
}
