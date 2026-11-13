//! Discrete structured `Garak` probe types, deterministic PRNG, and Arm A blind black-box search.

use crate::GarakProbeFamily;
use portia_compiler::concolic::ConcolicDiodeLens;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Discrete structured `Garak` probe genome (`family` + 3 semantic/syntactic mutation coordinates).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DiscreteGarakProbe {
    pub id: String,
    pub family: GarakProbeFamily,
    pub coords: [usize; 3],
    pub prompt: String,
}

impl DiscreteGarakProbe {
    /// Coordinate Hamming distance within the same [`GarakProbeFamily`] (`usize::MAX` across families).
    #[must_use]
    pub fn mutation_distance(&self, other: &Self) -> usize {
        if self.family != other.family {
            return usize::MAX;
        }
        self.coords
            .iter()
            .zip(&other.coords)
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// Evaluation record for a single discrete `Garak` probe against its family [`ConcolicDiodeLens`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcolicProbeEvaluation {
    pub probe: DiscreteGarakProbe,
    pub cliff_margin: f64,
    pub diode_fired: bool,
    pub cell: Vec<i32>,
}

/// Summary report from a fixed-budget (`B`-rollout) boundary-break search arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarakSearchReport {
    pub evaluations_used: usize,
    pub bypassed_probes: Vec<ConcolicProbeEvaluation>,
    pub distinct_bypass_cells: usize,
    pub min_cliff_margin: f64,
    pub archive_cells_populated: usize,
}

pub(super) struct SplitMix64(u64);

impl SplitMix64 {
    pub(super) const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(super) fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() % (bound as u64)) as usize
        }
    }

    pub(super) fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

pub(super) fn pick_unvisited_neighbor_uniform(
    parent: &DiscreteGarakProbe,
    candidates: &[(DiscreteGarakProbe, Vec<f64>)],
    visited: &BTreeSet<usize>,
    rng: &mut SplitMix64,
) -> Option<usize> {
    for dist in [1usize, 2, 3] {
        let mut ring: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(idx, (p, _))| !visited.contains(idx) && parent.mutation_distance(p) == dist)
            .map(|(idx, _)| idx)
            .collect();
        if !ring.is_empty() {
            return Some(ring.swap_remove(rng.next_usize(ring.len())));
        }
    }
    let rem: Vec<usize> = (0..candidates.len())
        .filter(|i| !visited.contains(i))
        .collect();
    (!rem.is_empty()).then(|| rem[rng.next_usize(rem.len())])
}

/// **Arm A (Blind Black-Box Search):** Observes **only** binary `diode_fired` (`w_gate · x > 0`)
/// pass/fail (zero continuous `cliff_margin`, zero `ker(w_gate)` cell coordinates).
pub fn run_blind_blackbox_probe_search(
    lenses: &BTreeMap<String, ConcolicDiodeLens>,
    seeds: &[(DiscreteGarakProbe, Vec<f64>)],
    candidates: &[(DiscreteGarakProbe, Vec<f64>)],
    budget: usize,
    bin_width: f64,
    rng_seed: u64,
) -> GarakSearchReport {
    let mut rng = SplitMix64::new(rng_seed);
    let mut visited = BTreeSet::new();
    let mut known_pool: Vec<(DiscreteGarakProbe, bool)> = seeds
        .iter()
        .map(|(p, x)| {
            (
                p.clone(),
                lenses[p.family.rule_name()].cliff_margin(x) > 0.0,
            )
        })
        .collect();

    let (mut bypassed, mut bypass_cells, mut all_cells) =
        (Vec::new(), BTreeSet::new(), BTreeSet::new());
    let mut min_margin = f64::INFINITY;
    for _ in 0..budget.min(candidates.len()) {
        let parent = known_pool[rng.next_usize(known_pool.len())].0.clone();
        let Some(cand_idx) =
            pick_unvisited_neighbor_uniform(&parent, candidates, &visited, &mut rng)
        else {
            break;
        };
        visited.insert(cand_idx);
        let (probe, x) = &candidates[cand_idx];
        let lens = &lenses[probe.family.rule_name()];
        let margin = lens.cliff_margin(x);
        let fired = margin > 0.0;
        let cell = lens.ridge_cell_coords(x, bin_width);
        let mut tc = vec![probe.family as i32];
        tc.extend_from_slice(&cell);
        min_margin = min_margin.min(margin);
        all_cells.insert(tc.clone());
        known_pool.push((probe.clone(), fired));
        if !fired {
            bypass_cells.insert(tc);
            bypassed.push(ConcolicProbeEvaluation {
                probe: probe.clone(),
                cliff_margin: margin,
                diode_fired: false,
                cell,
            });
        }
    }
    GarakSearchReport {
        evaluations_used: visited.len(),
        bypassed_probes: bypassed,
        distinct_bypass_cells: bypass_cells.len(),
        min_cliff_margin: min_margin,
        archive_cells_populated: all_cells.len(),
    }
}
