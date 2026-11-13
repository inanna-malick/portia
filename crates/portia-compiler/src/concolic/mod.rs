//! White-Box Concolic `MAP-Elites` Boundary Explorer (`concolic`, `T-MAP` / `RISE` 2026).
//!
//! Instead of trying to invert deep Layer-22 activations into discrete BPE tokens (`GCG` gibberish),
//! exposes:
//! 1. **Continuous Cliff-Margin Fitness:** `margin(x) = w_gate_bf16 · x` (turning a flat `98%`
//!    binary `PASS` plateau into a smooth gradient ramp down to `w_gate · x <= 0` failure!), and
//! 2. **Zero-Crossing Ridge Diversity Coordinates:** Projects candidate states onto the top-`k`
//!    orthonormal directions of $\ker(\mathbf{w}_{\text{gate}}) \cap \ker(\mathbf{Q}_{\text{preserve}}) \cap \ker(\mathbf{W}_{U,\text{top}})$
//!    to bin discrete `proptest` / `portia-garak` mutations into distinct `MAP-Elites` behavior cells.

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, StateMatrix};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// White-box concolic lens extracted from a compiled diode's `w_gate` and preserved/vocab subspaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcolicDiodeLens {
    pub rule_name: String,
    pub layer_idx: usize,
    pub w_gate_f64: Vec<f64>,
    pub ridge_basis: OrthonormalSubspace,
}

impl ConcolicDiodeLens {
    /// Construct a `ConcolicDiodeLens` whose `ridge_basis` spans the top `num_grid_axes` variance
    /// directions of `seed_states` inside $\ker(\mathbf{w}_{\text{gate}}) \cap \ker(\mathbf{Q}_{\text{forb}})$.
    pub fn build(
        rule_name: impl Into<String>,
        layer_idx: usize,
        w_gate_bf16: &[bf16],
        q_forbidden_or_wu: Option<&OrthonormalSubspace>,
        seed_states: &StateMatrix,
        num_grid_axes: usize,
    ) -> Result<Self, V2CompilerError> {
        let d = w_gate_bf16.len();
        if seed_states.hidden_dim() != d {
            return Err(V2CompilerError::DimensionMismatch {
                context: "ConcolicDiodeLens::build",
                expected: d,
                actual: seed_states.hidden_dim(),
            });
        }
        let w_gate_f64: Vec<f64> = w_gate_bf16.iter().map(|v| v.to_f64()).collect();
        let wg_mat = StateMatrix::try_new("concolic_w_gate_row", 1, d, w_gate_f64.clone())?;
        let q_wg = OrthonormalSubspace::from_row_space_svd("concolic_wg_basis", &wg_mat, 1e-9, 1)?;
        let q_joint_forb = match q_forbidden_or_wu {
            Some(qf) => OrthonormalSubspace::union_bases(
                "concolic_joint_forb",
                d,
                &[&q_wg, qf],
                (1 + qf.rank()).min(d.saturating_sub(16)),
            )?,
            None => q_wg,
        };

        // Project `seed_states` centered rows into `ker(w_gate) ∩ ker(Q_forb)` to get the live ridge PCA:
        let mu = seed_states.mean_row();
        let n = seed_states.nrows();
        let mut proj_rows = Vec::with_capacity(n * d);
        for r in 0..n {
            let centered: Vec<f64> = seed_states
                .row(r)
                .iter()
                .zip(&mu)
                .map(|(&x, &m)| x - m)
                .collect();
            proj_rows.extend_from_slice(&q_joint_forb.project_out(&centered));
        }
        let proj_mat = StateMatrix::try_new("concolic_ridge_samples", n, d, proj_rows)?;
        let ridge_basis = OrthonormalSubspace::from_row_space_svd(
            "concolic_ridge_pca",
            &proj_mat,
            1e-4,
            num_grid_axes.max(1),
        )?;

        Ok(Self {
            rule_name: rule_name.into(),
            layer_idx,
            w_gate_f64,
            ridge_basis,
        })
    }

    /// Signed pre-SiLU cliff margin `w_gate · x`:
    /// - `+3.2`: deep inside safe diode plateau,
    /// - `-> 0.0`: approaching diode shutoff ridge,
    /// - `< 0.0`: diode bypassed / failed to fire!
    #[inline]
    #[must_use]
    pub fn cliff_margin(&self, x: &[f64]) -> f64 {
        dot_f64(&self.w_gate_f64, x)
    }

    /// Discretize `x`'s projection along `ridge_basis` into quantized `MAP-Elites` cell coordinates.
    #[must_use]
    pub fn ridge_cell_coords(&self, x: &[f64], bin_width: f64) -> Vec<i32> {
        let bw = bin_width.max(1e-3);
        self.ridge_basis
            .coords(x)
            .into_iter()
            .map(|c| (c / bw).floor() as i32)
            .collect()
    }
}

/// Elite entry stored in a `MAP-Elites` cell, minimizing `cliff_margin` (seeking boundary breaks!).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcolicElite<P> {
    pub payload: P,
    pub cliff_margin: f64,
    pub cell: Vec<i32>,
}

/// Quality-Diversity (`MAP-Elites`) archive tracking the lowest-margin (closest-to-break) discrete
/// input `P` discovered in every distinct `ker(w_gate) ∩ Ω_dark` behavioral ridge cell.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcolicMapElitesArchive<P> {
    elites: BTreeMap<Vec<i32>, ConcolicElite<P>>,
}

impl<P: Clone> ConcolicMapElitesArchive<P> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            elites: BTreeMap::new(),
        }
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elites.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elites.is_empty()
    }

    /// Insert `payload` with activation `x` if its cell is empty or `cliff_margin < existing.cliff_margin`.
    /// Returns `true` iff the archive was updated with a new or closer-to-break elite.
    pub fn observe(
        &mut self,
        lens: &ConcolicDiodeLens,
        bin_width: f64,
        payload: P,
        x: &[f64],
    ) -> bool {
        let cliff_margin = lens.cliff_margin(x);
        let cell = lens.ridge_cell_coords(x, bin_width);
        match self.elites.get(&cell) {
            Some(prev) if prev.cliff_margin <= cliff_margin => false,
            _ => {
                self.elites.insert(
                    cell.clone(),
                    ConcolicElite {
                        payload,
                        cliff_margin,
                        cell,
                    },
                );
                true
            }
        }
    }

    /// Return all discovered elites sorted from lowest `cliff_margin` (true breaks `< 0` first!) upward.
    #[must_use]
    pub fn elites_sorted_by_break_proximity(&self) -> Vec<ConcolicElite<P>> {
        let mut v: Vec<_> = self.elites.values().cloned().collect();
        v.sort_by(|a, b| {
            a.cliff_margin
                .partial_cmp(&b.cliff_margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}
