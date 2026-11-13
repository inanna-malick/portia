//! Log-spaced cliff sweep and monotone binary-search wipeout boundary finders.

use portia_core::domain::{BiasedScaleBand, CombatStrategyClass};
use serde::{Deserialize, Serialize};

use super::dimensions::RogueliteDimension;
use super::stress_vector::RogueliteStressVector;

/// Return the canonical log-spaced probe ladder for `dim`.
pub fn log_spaced_cliff_ladder(dim: RogueliteDimension) -> Vec<usize> {
    match dim {
        RogueliteDimension::RowHaystackSize => {
            vec![1, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
        }
        RogueliteDimension::IdentifierSwarmCollision => {
            vec![1, 3, 5, 10, 25, 50, 100, 175, 250]
        }
        RogueliteDimension::ColumnWidthArity => {
            vec![4, 6, 10, 20, 35, 50, 75, 100]
        }
        RogueliteDimension::SchemaGraphTopology => {
            vec![1, 2, 4, 6, 10, 15, 22, 30]
        }
        RogueliteDimension::PolicyPredicateArity => {
            vec![1, 2, 3, 5, 8, 10, 13, 16]
        }
        RogueliteDimension::EpisodeTurnHorizon => {
            vec![1, 3, 5, 10, 18, 25, 38, 50]
        }
        RogueliteDimension::AstSymbolHaystack => {
            vec![3, 8, 15, 35, 70, 120, 200, 300]
        }
    }
}

/// Exact wipeout cliff receipt produced by [`find_exact_wipeout_cliff`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WipeoutCliffReceipt {
    pub dimension: RogueliteDimension,
    pub strategy: CombatStrategyClass,
    pub max_passing_value: Option<usize>,
    pub min_failing_value: Option<usize>,
    pub cliff_band: Option<BiasedScaleBand>,
    pub bpe_token_estimate_at_cliff: usize,
    pub ladders_probed: Vec<usize>,
    pub binary_search_steps: usize,
    pub total_oracle_queries: usize,
    pub failure_mode_diagnosis: String,
}

impl WipeoutCliffReceipt {
    /// Returns `true` iff an exact adjacent integer boundary `max_passing_value + 1 == min_failing_value` was isolated.
    pub const fn is_exact_unit_cliff(&self) -> bool {
        match (self.max_passing_value, self.min_failing_value) {
            (Some(pass), Some(fail)) => pass + 1 == fail,
            _ => false,
        }
    }
}

/// Sweep `log_spaced_cliff_ladder(dimension)` starting from `baseline` and binary-search between
/// the highest passing rung and lowest failing rung to isolate `max_passing_value + 1 == min_failing_value`.
pub fn find_exact_wipeout_cliff<F>(
    baseline: &RogueliteStressVector,
    dimension: RogueliteDimension,
    strategy: CombatStrategyClass,
    mut oracle: F,
) -> WipeoutCliffReceipt
where
    F: FnMut(&RogueliteStressVector) -> bool,
{
    let ladder = log_spaced_cliff_ladder(dimension);
    let mut ladders_probed = Vec::new();
    let mut total_queries = 0usize;
    let mut last_pass: Option<usize> = None;
    let mut first_fail: Option<usize> = None;

    for &rung in &ladder {
        let probe_vec = baseline.with_dimension(dimension, rung);
        ladders_probed.push(rung);
        total_queries += 1;
        if oracle(&probe_vec) {
            last_pass = Some(rung);
        } else {
            first_fail = Some(rung);
            break;
        }
    }

    let mut bin_steps = 0usize;
    if let (Some(mut lo_pass), Some(mut hi_fail)) = (last_pass, first_fail) {
        while lo_pass + 1 < hi_fail {
            let mid = lo_pass + (hi_fail - lo_pass) / 2;
            let probe_vec = baseline.with_dimension(dimension, mid);
            bin_steps += 1;
            total_queries += 1;
            if oracle(&probe_vec) {
                lo_pass = mid;
            } else {
                hi_fail = mid;
            }
        }
        last_pass = Some(lo_pass);
        first_fail = Some(hi_fail);
    }

    let cliff_val = first_fail
        .or(last_pass)
        .unwrap_or_else(|| dimension.band_spec().boss_floor.1);
    let cliff_vec = baseline.with_dimension(dimension, cliff_val);
    let bpe_est = cliff_vec.estimate_bpe_tokens(strategy);
    let cliff_band = first_fail.map(|v| dimension.band_spec().classify(v));

    let diagnosis = match (last_pass, first_fail) {
        (Some(p), Some(f)) => {
            let band = dimension.band_spec().classify(f);
            format!(
                "Exact wipeout cliff on {} under {:?}: N*_pass = {} < N*_fail = {} ({:?}, ~{} BPE tokens).",
                dimension.short_name(),
                strategy,
                p,
                f,
                band,
                bpe_est
            )
        }
        (Some(p), None) => format!(
            "Immunity verified on {} under {:?}: passed all rungs up to boss ceiling N = {} (~{} BPE tokens).",
            dimension.short_name(),
            strategy,
            p,
            bpe_est
        ),
        (None, Some(f)) => format!(
            "Immediate baseline failure on {} under {:?} at minimum rung N = {} (~{} BPE tokens).",
            dimension.short_name(),
            strategy,
            f,
            bpe_est
        ),
        (None, None) => "Empty ladder".to_string(),
    };

    WipeoutCliffReceipt {
        dimension,
        strategy,
        max_passing_value: last_pass,
        min_failing_value: first_fail,
        cliff_band,
        bpe_token_estimate_at_cliff: bpe_est,
        ladders_probed,
        binary_search_steps: bin_steps,
        total_oracle_queries: total_queries,
        failure_mode_diagnosis: diagnosis,
    }
}

/// Multi-axis diagonal ray wipeout receipt interpolating from `start` to `target` over `0..=1000`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiAxisRayCliffReceipt {
    pub max_passing_permille: Option<usize>,
    pub min_failing_permille: Option<usize>,
    pub max_passing_vector: Option<RogueliteStressVector>,
    pub min_failing_vector: Option<RogueliteStressVector>,
    pub total_oracle_queries: usize,
}

/// Interpolate linearly between `start` (`permille = 0`) and `target` (`permille = 1000`) across all 7 dimensions.
pub fn interpolate_stress_ray(
    start: &RogueliteStressVector,
    target: &RogueliteStressVector,
    permille: usize,
) -> RogueliteStressVector {
    let t = permille.min(1000);
    let mut out = if t >= 500 { *target } else { *start };
    for dim in RogueliteDimension::ALL {
        let a = start.get_dimension(dim);
        let b = target.get_dimension(dim);
        let v = if b >= a {
            a + ((b - a) * t) / 1000
        } else {
            a - ((a - b) * t) / 1000
        };
        out = out.with_dimension(dim, v);
    }
    out
}

/// Find the exact permille cliff along a simultaneous multi-dimension ray.
pub fn find_multi_axis_ray_wipeout_cliff<F>(
    start: &RogueliteStressVector,
    target: &RogueliteStressVector,
    mut oracle: F,
) -> MultiAxisRayCliffReceipt
where
    F: FnMut(&RogueliteStressVector) -> bool,
{
    let coarse_rungs = [0usize, 50, 100, 200, 350, 500, 750, 1000];
    let mut queries = 0usize;
    let mut last_pass: Option<usize> = None;
    let mut first_fail: Option<usize> = None;

    for &r in &coarse_rungs {
        let v = interpolate_stress_ray(start, target, r);
        queries += 1;
        if oracle(&v) {
            last_pass = Some(r);
        } else {
            first_fail = Some(r);
            break;
        }
    }

    if let (Some(mut lo), Some(mut hi)) = (last_pass, first_fail) {
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            let v = interpolate_stress_ray(start, target, mid);
            queries += 1;
            if oracle(&v) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        last_pass = Some(lo);
        first_fail = Some(hi);
    }

    MultiAxisRayCliffReceipt {
        max_passing_permille: last_pass,
        min_failing_permille: first_fail,
        max_passing_vector: last_pass.map(|t| interpolate_stress_ray(start, target, t)),
        min_failing_vector: first_fail.map(|t| interpolate_stress_ray(start, target, t)),
        total_oracle_queries: queries,
    }
}
