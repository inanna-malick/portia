//! Math-library-agnostic contiguous state matrices (`RowMajorStates`) and `half::bf16` helpers.
//!
//! Zero `nalgebra` or `faer` types appear in public fields—domain callers pass plain `&[f64]`,
//! `&[f32]`, or `&[bf16]` slices, keeping internal numerical backends 100% swappable.

pub mod safetensors_io;
pub mod sod;
pub mod subspace;

use crate::error::V2CompilerError;
use faer::Mat;
use half::bf16;
use serde::{Deserialize, Serialize};

/// Validated non-empty `(n_samples, hidden_dim)` row-major matrix of Transformer activations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMatrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl StateMatrix {
    /// Construct a validated `(nrows, ncols)` row-major activation matrix.
    pub fn try_new(
        context: &'static str,
        nrows: usize,
        ncols: usize,
        data: Vec<f64>,
    ) -> Result<Self, V2CompilerError> {
        if nrows == 0 || ncols == 0 {
            return Err(V2CompilerError::EmptyMatrix {
                context,
                nrows,
                ncols,
            });
        }
        if data.len() != nrows.saturating_mul(ncols) {
            return Err(V2CompilerError::DimensionMismatch {
                context,
                expected: nrows * ncols,
                actual: data.len(),
            });
        }
        if !data.iter().all(|v| v.is_finite()) {
            return Err(V2CompilerError::NonFiniteValue { context });
        }
        Ok(Self { nrows, ncols, data })
    }

    /// Construct from nested row vectors `Vec<Vec<f64>>` (e.g. stdio JSON capture responses).
    pub fn from_rows(context: &'static str, rows: &[Vec<f64>]) -> Result<Self, V2CompilerError> {
        let Some(first) = rows.first() else {
            return Err(V2CompilerError::EmptyMatrix {
                context,
                nrows: 0,
                ncols: 0,
            });
        };
        let ncols = first.len();
        if ncols == 0 {
            return Err(V2CompilerError::EmptyMatrix {
                context,
                nrows: rows.len(),
                ncols: 0,
            });
        }
        let mut data = Vec::with_capacity(rows.len() * ncols);
        for row in rows {
            if row.len() != ncols {
                return Err(V2CompilerError::DimensionMismatch {
                    context,
                    expected: ncols,
                    actual: row.len(),
                });
            }
            data.extend_from_slice(row);
        }
        Self::try_new(context, rows.len(), ncols, data)
    }

    /// Vertically concatenate multiple `StateMatrix` blocks sharing the same `ncols`.
    pub fn vstack(context: &'static str, blocks: &[&StateMatrix]) -> Result<Self, V2CompilerError> {
        let Some(first) = blocks.first() else {
            return Err(V2CompilerError::EmptyMatrix {
                context,
                nrows: 0,
                ncols: 0,
            });
        };
        let d = first.ncols;
        let total_rows: usize = blocks.iter().map(|b| b.nrows).sum();
        let mut data = Vec::with_capacity(total_rows * d);
        for b in blocks {
            if b.ncols != d {
                return Err(V2CompilerError::DimensionMismatch {
                    context,
                    expected: d,
                    actual: b.ncols,
                });
            }
            data.extend_from_slice(&b.data);
        }
        Self::try_new(context, total_rows, d, data)
    }

    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.ncols
    }

    #[inline]
    #[must_use]
    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.ncols;
        &self.data[start..start + self.ncols]
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Convert to a `faer::Mat<f64>` of shape `(nrows, ncols)`.
    #[must_use]
    pub fn to_faer(&self) -> Mat<f64> {
        Mat::from_fn(self.nrows, self.ncols, |i, j| self.data[i * self.ncols + j])
    }

    /// Quantize every coordinate to `half::bf16` and convert back to `f64` (matching GPU `bf16` grid).
    #[must_use]
    pub fn quantize_bf16(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|&v| bf16::from_f64(v).to_f64())
            .collect();
        Self {
            nrows: self.nrows,
            ncols: self.ncols,
            data,
        }
    }

    /// Mean row centroid $\bar{\mathbf{x}} \in \mathbb{R}^d$.
    #[must_use]
    pub fn mean_row(&self) -> Vec<f64> {
        let mut mu = vec![0.0; self.ncols];
        let inv_n = 1.0 / (self.nrows as f64);
        for r in 0..self.nrows {
            let row = self.row(r);
            for j in 0..self.ncols {
                mu[j] += row[j] * inv_n;
            }
        }
        mu
    }

    /// Matrix-vector product $\mathbf{X} \mathbf{w} \in \mathbb{R}^{\text{nrows}}$.
    pub fn matvec(&self, w: &[f64]) -> Result<Vec<f64>, V2CompilerError> {
        if w.len() != self.ncols {
            return Err(V2CompilerError::DimensionMismatch {
                context: "StateMatrix::matvec",
                expected: self.ncols,
                actual: w.len(),
            });
        }
        let mut out = Vec::with_capacity(self.nrows);
        for r in 0..self.nrows {
            out.push(dot_f64(self.row(r), w));
        }
        Ok(out)
    }
}

#[inline]
#[must_use]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
#[must_use]
pub fn l2_norm(v: &[f64]) -> f64 {
    dot_f64(v, v).sqrt()
}

#[must_use]
pub fn normalize_unit_vec(v: &[f64]) -> Option<Vec<f64>> {
    let n = l2_norm(v);
    if !n.is_finite() || n <= 1e-12 {
        None
    } else {
        Some(v.iter().map(|&x| x / n).collect())
    }
}

#[inline]
#[must_use]
pub fn quantize_slice_bf16_f64(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| bf16::from_f64(x).to_f64()).collect()
}

#[inline]
#[must_use]
pub fn slice_to_bf16_vec(v: &[f64]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f64(x)).collect()
}

#[inline]
#[must_use]
pub fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[must_use]
pub fn rms_norm_slice(x: &[f64], scale: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(x.len(), scale.len());
    let d = x.len().max(1) as f64;
    let inv_rms = 1.0 / (dot_f64(x, x) / d + eps).sqrt();
    x.iter()
        .zip(scale.iter())
        .map(|(&xi, &si)| xi * inv_rms * si)
        .collect()
}
