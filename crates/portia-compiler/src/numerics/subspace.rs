//! `faer`-accelerated SIMD/Rayon Orthonormal Subspaces, Nullspace Projections (`AlphaEdit`),
//! and Joint Forbidden-Subspace Unions (`ker(A) ∩ ker(B) = (row(A) + row(B))^⊥`).

use crate::error::V2CompilerError;
use crate::numerics::{dot_f64, l2_norm, StateMatrix};
use faer::Mat;
use serde::{Deserialize, Serialize};

/// Orthonormal column basis $\mathbf{Q} \in \mathbb{R}^{d \times r}$ ($\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}_r$)
/// stored in column-major layout (`cols.len() == r * d`) for cache-local dot-product projections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrthonormalSubspace {
    hidden_dim: usize,
    rank: usize,
    col_major: Vec<f64>,
    singular_values: Vec<f64>,
}

impl OrthonormalSubspace {
    /// Empty zero-rank subspace in $\mathbb{R}^d$ (identity `project_out`).
    #[must_use]
    pub const fn empty(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            rank: 0,
            col_major: Vec::new(),
            singular_values: Vec::new(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    #[inline]
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    #[must_use]
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    #[inline]
    #[must_use]
    pub fn basis_col(&self, col_idx: usize) -> &[f64] {
        let s = col_idx * self.hidden_dim;
        &self.col_major[s..s + self.hidden_dim]
    }

    /// Extract the top-rank row-space basis of an `(m, d)` matrix via `faer` thin SVD:
    /// retains singular directions with $\sigma_i \ge \text{rel\_sv\_tol} \cdot \sigma_0$, capped at `max_rank`.
    pub fn from_row_space_svd(
        context: &'static str,
        rows_mat: &StateMatrix,
        rel_sv_tol: f64,
        max_rank: usize,
    ) -> Result<Self, V2CompilerError> {
        let d = rows_mat.hidden_dim();
        if max_rank == 0 {
            return Ok(Self::empty(d));
        }
        let mat = rows_mat.to_faer();
        let svd = mat
            .thin_svd()
            .map_err(|_| V2CompilerError::SvdConvergence { context })?;
        let s_diag = svd.S().column_vector();
        let v_mat = svd.V();
        let n_sv = s_diag.nrows();
        if n_sv == 0 {
            return Ok(Self::empty(d));
        }
        let s0 = s_diag[0];
        if !s0.is_finite() || s0 <= 1e-12 {
            return Ok(Self::empty(d));
        }
        let cutoff = (rel_sv_tol.max(1e-12)) * s0;
        let mut col_major = Vec::new();
        let mut kept_svs = Vec::new();
        for k in 0..n_sv.min(max_rank) {
            let sv = s_diag[k];
            if sv >= cutoff {
                kept_svs.push(sv);
                for i in 0..d {
                    col_major.push(v_mat[(i, k)]);
                }
            }
        }
        let rank = kept_svs.len();
        Ok(Self {
            hidden_dim: d,
            rank,
            col_major,
            singular_values: kept_svs,
        })
    }

    /// Combine multiple forbidden subspaces into a single orthonormal basis spanning their sum
    /// $\operatorname{span}(\mathbf{Q}_1 \cup \dots \cup \mathbf{Q}_m)$ via `faer` thin SVD,
    /// so projecting out the returned basis enforces $\mathbf{v} \in \bigcap_i \ker(\mathbf{Q}_i^\top)$!
    pub fn union_bases(
        context: &'static str,
        hidden_dim: usize,
        bases: &[&OrthonormalSubspace],
        max_rank: usize,
    ) -> Result<Self, V2CompilerError> {
        let total_cols: usize = bases.iter().map(|b| b.rank).sum();
        if total_cols == 0 || max_rank == 0 {
            return Ok(Self::empty(hidden_dim));
        }
        let mut rows_data = Vec::with_capacity(total_cols * hidden_dim);
        for b in bases {
            if b.hidden_dim != hidden_dim {
                return Err(V2CompilerError::DimensionMismatch {
                    context,
                    expected: hidden_dim,
                    actual: b.hidden_dim,
                });
            }
            for c in 0..b.rank {
                rows_data.extend_from_slice(b.basis_col(c));
            }
        }
        let stacked = StateMatrix::try_new(context, total_cols, hidden_dim, rows_data)?;
        Self::from_row_space_svd(context, &stacked, 1e-6, max_rank)
    }

    /// Coordinate projection $\mathbf{Q}^\top \mathbf{v} \in \mathbb{R}^r$.
    #[must_use]
    pub fn coords(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.hidden_dim);
        (0..self.rank)
            .map(|c| dot_f64(self.basis_col(c), v))
            .collect()
    }

    /// Leakage norm $\|\mathbf{Q}^\top \mathbf{v}\|_2$.
    #[must_use]
    pub fn leakage_norm(&self, v: &[f64]) -> f64 {
        let c = self.coords(v);
        l2_norm(&c)
    }

    /// Fraction of $\|\mathbf{v}\|_2^2$ surviving after projecting out $\mathbf{Q}$.
    #[must_use]
    pub fn retained_energy_fraction(&self, v: &[f64]) -> f64 {
        let full_sq = dot_f64(v, v);
        if full_sq <= 1e-24 {
            return 0.0;
        }
        let proj = self.project_out(v);
        (dot_f64(&proj, &proj) / full_sq).clamp(0.0, 1.0)
    }

    /// Two-pass Classical Gram-Schmidt nullspace projection $(\mathbf{I} - \mathbf{Q}\mathbf{Q}^\top)\mathbf{v}$
    /// achieving machine-precision ($< 10^{-15}$) orthogonality.
    #[must_use]
    pub fn project_out(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.hidden_dim);
        let mut out = v.to_vec();
        if self.rank == 0 {
            return out;
        }
        for _pass in 0..2 {
            for c in 0..self.rank {
                let q = self.basis_col(c);
                let alpha = dot_f64(q, &out);
                for (oi, &qi) in out.iter_mut().zip(q.iter()) {
                    *oi -= alpha * qi;
                }
            }
        }
        out
    }

    /// Project onto $\operatorname{span}(\mathbf{Q})$: $\mathbf{Q}\mathbf{Q}^\top \mathbf{v}$.
    #[must_use]
    pub fn project_onto(&self, v: &[f64]) -> Vec<f64> {
        let perp = self.project_out(v);
        v.iter()
            .zip(perp.iter())
            .map(|(&vi, &pi)| vi - pi)
            .collect()
    }

    /// Convert to a `(d, r)` `faer::Mat<f64>`.
    #[must_use]
    pub fn to_faer_tall(&self) -> Mat<f64> {
        Mat::from_fn(self.hidden_dim, self.rank, |i, j| {
            self.col_major[j * self.hidden_dim + i]
        })
    }
}
