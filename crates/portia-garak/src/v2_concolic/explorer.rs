//! White-Box Concolic `MAP-Elites` Boundary Explorer (`T-MAP` ICLR 2026 / `RISE` NeurIPS 2026).
//!
//! Guides discrete structured `Garak` probe mutations using continuous `cliff_margin(x) = w_gate_bf16 · x`
//! and 4-axis `ker(w_gate) ∩ Ω_dark` `MAP-Elites` ridge cells—never raw continuous-to-BPE `GCG`.

use super::explorer_blind::{pick_unvisited_neighbor_uniform, SplitMix64};
pub use super::explorer_blind::{
    run_blind_blackbox_probe_search, ConcolicProbeEvaluation, DiscreteGarakProbe, GarakSearchReport,
};
use crate::GarakProbeFamily;
use portia_compiler::concolic::{ConcolicDiodeLens, ConcolicMapElitesArchive};
use std::collections::{BTreeMap, BTreeSet};

/// **Arm B (White-Box Concolic `MAP-Elites`):** Guides discrete structured probe mutation strictly
/// from already-evaluated probes via [`ConcolicMapElitesArchive`] (`cliff_margin = w_gate_bf16 · x`
/// descent + 4-axis `ker(w_gate) ∩ Ω_dark` cell novelty + causal per-knob margin sensitivity).
pub fn run_whitebox_concolic_map_elites_search(
    lenses: &BTreeMap<String, ConcolicDiodeLens>,
    seeds: &[(DiscreteGarakProbe, Vec<f64>)],
    candidates: &[(DiscreteGarakProbe, Vec<f64>)],
    budget: usize,
    bin_width: f64,
    rng_seed: u64,
) -> GarakSearchReport {
    let mut rng = SplitMix64::new(rng_seed);
    let mut archives: BTreeMap<GarakProbeFamily, ConcolicMapElitesArchive<DiscreteGarakProbe>> =
        BTreeMap::new();
    let mut spawns_per_cell: BTreeMap<(GarakProbeFamily, Vec<i32>), usize> = BTreeMap::new();
    let mut breaks_per_cell: BTreeMap<(GarakProbeFamily, Vec<i32>), usize> = BTreeMap::new();
    let mut fam_evals: BTreeMap<GarakProbeFamily, usize> = BTreeMap::new();
    let mut knob_stats: BTreeMap<(GarakProbeFamily, usize, usize), (f64, usize)> = BTreeMap::new();
    let mut fam_pot_sums: BTreeMap<GarakProbeFamily, (f64, usize)> = BTreeMap::new();
    let mut frontier: Vec<(DiscreteGarakProbe, f64, Vec<i32>)> = Vec::new();
    let mut fam_scales: BTreeMap<GarakProbeFamily, f64> = BTreeMap::new();
    let cliff_pot = |m: f64, sc: f64| {
        if m > 0.0 {
            -(m / sc)
        } else {
            0.32 + 0.08 * ((-m / sc).min(1.5))
        }
    };

    for (p, x) in seeds {
        let lens = &lenses[p.family.rule_name()];
        let margin = lens.cliff_margin(x);
        let m = margin.abs().max(1e-3);
        fam_scales
            .entry(p.family)
            .and_modify(|s| *s = s.max(m))
            .or_insert(m);
    }
    for (p, x) in seeds {
        let lens = &lenses[p.family.rule_name()];
        let margin = lens.cliff_margin(x);
        let sc = fam_scales[&p.family];
        let fp = fam_pot_sums.entry(p.family).or_insert((0.0, 0));
        fp.0 += cliff_pot(margin, sc);
        fp.1 += 1;
        for (ax, &val) in p.coords.iter().enumerate() {
            let e = knob_stats.entry((p.family, ax, val)).or_insert((0.0, 0));
            e.0 += margin;
            e.1 += 1;
        }
        frontier.push((p.clone(), margin, lens.ridge_cell_coords(x, bin_width)));
        archives
            .entry(p.family)
            .or_insert_with(ConcolicMapElitesArchive::new)
            .observe(lens, bin_width, p.clone(), x);
    }

    let (mut visited, mut bypassed, mut bypass_cells) =
        (BTreeSet::new(), Vec::new(), BTreeSet::new());
    let mut min_margin = f64::INFINITY;

    for _ in 0..budget.min(candidates.len()) {
        let min_dist = |p: &DiscreteGarakProbe| -> Option<usize> {
            candidates
                .iter()
                .enumerate()
                .filter(|(idx, (c, _))| !visited.contains(idx) && c.family == p.family)
                .map(|(_, (c, _))| p.mutation_distance(c))
                .filter(|&d| d >= 1)
                .min()
        };
        let mut scored: Vec<(usize, f64, GarakProbeFamily, Vec<i32>, DiscreteGarakProbe)> =
            Vec::new();
        for (&fam, arch) in &archives {
            let sc = fam_scales.get(&fam).copied().unwrap_or(1.0).max(1e-3);
            let fv = *fam_evals.get(&fam).unwrap_or(&0) as f64;
            for elite in arch.elites_sorted_by_break_proximity() {
                if let Some(d_min) = min_dist(&elite.payload) {
                    let key = (fam, elite.cell.clone());
                    let cs = *spawns_per_cell.get(&key).unwrap_or(&0) as f64;
                    let cb = *breaks_per_cell.get(&key).unwrap_or(&0) as f64;
                    let s = cliff_pot(elite.cliff_margin, sc) - 0.03 * cs - 0.04 * cb - 0.025 * fv
                        + 0.015 * rng.next_f64();
                    scored.push((d_min.clamp(1, 2), s, fam, elite.cell, elite.payload));
                }
            }
        }
        for (p, margin, cell) in &frontier {
            if let Some(d_min) = min_dist(p) {
                if !scored
                    .iter()
                    .any(|(_, _, f, c, _)| *f == p.family && c == cell)
                {
                    let sc = fam_scales.get(&p.family).copied().unwrap_or(1.0).max(1e-3);
                    let fv = *fam_evals.get(&p.family).unwrap_or(&0) as f64;
                    let key = (p.family, cell.clone());
                    let cs = *spawns_per_cell.get(&key).unwrap_or(&0) as f64;
                    let cb = *breaks_per_cell.get(&key).unwrap_or(&0) as f64;
                    let s = cliff_pot(*margin, sc) - 0.04 * cs - 0.04 * cb - 0.025 * fv
                        + 0.015 * rng.next_f64();
                    scored.push((d_min.clamp(1, 2), s, p.family, cell.clone(), p.clone()));
                }
            }
        }
        let Some(chosen) = ({
            scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
            scored.first().cloned()
        }) else {
            break;
        };
        *spawns_per_cell.entry((chosen.2, chosen.3)).or_insert(0) += 1;
        let parent = chosen.4;
        let sc = fam_scales
            .get(&parent.family)
            .copied()
            .unwrap_or(1.0)
            .max(1e-3);

        // Causal first-order ANOVA step `pot(cv) - pot(pv)` with empirical-Bayes unseen prior:
        let prior_pot = fam_pot_sums
            .get(&parent.family)
            .map(|&(s, c)| s / (c.max(1) as f64) + 0.08)
            .unwrap_or(-0.5);
        let pot_of = |fam: GarakProbeFamily, ax: usize, v: usize, s: f64| -> f64 {
            match knob_stats.get(&(fam, ax, v)) {
                Some(&(sum, cnt)) => cliff_pot(sum / (cnt as f64), s),
                None => prior_pot,
            }
        };
        let mut cand_idx = None;
        for dist in [1usize, 2, 3] {
            let mut ring: Vec<(f64, usize)> = candidates
                .iter()
                .enumerate()
                .filter(|(idx, (c, _))| {
                    !visited.contains(idx) && parent.mutation_distance(c) == dist
                })
                .map(|(idx, (c, _))| {
                    let mut est = 0.0;
                    for (ax, (&pv, &cv)) in parent.coords.iter().zip(&c.coords).enumerate() {
                        if pv != cv {
                            est += pot_of(c.family, ax, cv, sc) - pot_of(c.family, ax, pv, sc);
                        }
                    }
                    (est + 0.05 * rng.next_f64(), idx)
                })
                .collect();
            if !ring.is_empty() {
                ring.sort_by(|a, b| b.0.total_cmp(&a.0));
                cand_idx = Some(ring[0].1);
                break;
            }
        }
        let Some(idx) = cand_idx
            .or_else(|| pick_unvisited_neighbor_uniform(&parent, candidates, &visited, &mut rng))
        else {
            break;
        };
        visited.insert(idx);
        let (probe, x) = &candidates[idx];
        *fam_evals.entry(probe.family).or_insert(0) += 1;
        let lens = &lenses[probe.family.rule_name()];
        let margin = lens.cliff_margin(x);
        let cell = lens.ridge_cell_coords(x, bin_width);
        let fp = fam_pot_sums.entry(probe.family).or_insert((0.0, 0));
        fp.0 += cliff_pot(margin, sc);
        fp.1 += 1;
        for (ax, &val) in probe.coords.iter().enumerate() {
            let e = knob_stats
                .entry((probe.family, ax, val))
                .or_insert((0.0, 0));
            e.0 += margin;
            e.1 += 1;
        }
        frontier.push((probe.clone(), margin, cell.clone()));
        archives
            .entry(probe.family)
            .or_insert_with(ConcolicMapElitesArchive::new)
            .observe(lens, bin_width, probe.clone(), x);
        min_margin = min_margin.min(margin);
        if margin <= 0.0 {
            *breaks_per_cell
                .entry((probe.family, cell.clone()))
                .or_insert(0) += 1;
            let mut tc = vec![probe.family as i32];
            tc.extend_from_slice(&cell);
            bypass_cells.insert(tc);
            bypassed.push(ConcolicProbeEvaluation {
                probe: probe.clone(),
                cliff_margin: margin,
                diode_fired: false,
                cell,
            });
        }
    }

    let archive_cells_populated: usize = archives.values().map(ConcolicMapElitesArchive::len).sum();
    GarakSearchReport {
        evaluations_used: visited.len(),
        bypassed_probes: bypassed,
        distinct_bypass_cells: bypass_cells.len(),
        min_cliff_margin: min_margin,
        archive_cells_populated,
    }
}
