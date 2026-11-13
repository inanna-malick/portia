//! Constant-energy $\operatorname{SO}(d)$ Stiefel 2-frame $(\hat{\mathbf{h}}_{\text{rad}}, \hat{\mathbf{v}}_{\text{tan}}) \in \operatorname{St}(d, 2)$
//! with exact nullspace preservation on the tangential leg (`rotSOd`).

use crate::error::V2CompilerError;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::{dot_f64, l2_norm, normalize_unit_vec, StateMatrix};
use serde::{Deserialize, Serialize};

/// Geodesic rotation angle $\theta \in (-\pi, \pi)$ on the residual sphere $S^{d-1}(R)$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChordalAngleRad(pub f64);

impl ChordalAngleRad {
    #[inline]
    #[must_use]
    pub const fn new(radians: f64) -> Self {
        Self(radians)
    }

    #[inline]
    #[must_use]
    pub const fn radians(self) -> f64 {
        self.0
    }

    /// Even-radial step-back factor $\cos\theta - 1 \le 0$.
    #[inline]
    #[must_use]
    pub fn radial_coeff(self) -> f64 {
        self.0.cos() - 1.0
    }

    /// Odd-tangential step-forward factor $\sin\theta$.
    #[inline]
    #[must_use]
    pub fn tangential_coeff(self) -> f64 {
        self.0.sin()
    }
}

/// Certified 2-frame on $\operatorname{St}(d, 2)$ preserving residual L2 energy $\|\mathbf{h} + \Delta\mathbf{h}\|_2 == \|\mathbf{h}\|_2$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StiefelChordalFrame {
    pub h_radial_hat: Vec<f64>,
    pub v_tangent_hat: Vec<f64>,
    pub residual_radius: f64,
    pub angle: ChordalAngleRad,
    pub orthogonality_err: f64,
    pub forbidden_leak_err: f64,
}

impl StiefelChordalFrame {
    /// Build a certified $(\hat{\mathbf{h}}_{\text{rad}}, \hat{\mathbf{v}}_{\text{tan}})$ frame from positive
    /// un-normalized residual states `pos_residuals`, raw `target_dir`, and optional `forbidden_nullspace`
    /// (e.g. $\operatorname{span}(W_V \cup W_{U,\text{top}} \cup Q_{\text{anti}})$).
    pub fn construct(
        rule_name: &str,
        pos_residuals: &StateMatrix,
        target_dir: &[f64],
        angle: ChordalAngleRad,
        forbidden_nullspace: Option<&OrthonormalSubspace>,
    ) -> Result<Self, V2CompilerError> {
        let d = pos_residuals.hidden_dim();
        if target_dir.len() != d {
            return Err(V2CompilerError::DimensionMismatch {
                context: "StiefelChordalFrame::construct(target_dir)",
                expected: d,
                actual: target_dir.len(),
            });
        }
        if !angle.radians().is_finite() {
            return Err(V2CompilerError::NonFiniteValue {
                context: "StiefelChordalFrame::angle",
            });
        }

        let mean_h = pos_residuals.mean_row();
        let residual_radius = l2_norm(&mean_h);
        let h_radial_hat = normalize_unit_vec(&mean_h).ok_or_else(|| {
            V2CompilerError::DegenerateProjectedTarget {
                rule: rule_name.to_string(),
                norm: residual_radius,
            }
        })?;

        // Orthogonalize `target_dir` against BOTH `h_radial_hat` AND `forbidden_nullspace` (2 passes):
        let mut v_tan = target_dir.to_vec();
        for _pass in 0..2 {
            if let Some(q_forb) = forbidden_nullspace {
                v_tan = q_forb.project_out(&v_tan);
            }
            let alpha_h = dot_f64(&h_radial_hat, &v_tan);
            for (vi, &hi) in v_tan.iter_mut().zip(h_radial_hat.iter()) {
                *vi -= alpha_h * hi;
            }
        }
        if let Some(q_forb) = forbidden_nullspace {
            // Re-project if `h_radial_hat` had non-zero overlap with `q_forb`, via joint basis:
            let h_mat = StateMatrix::try_new("h_rad_row", 1, d, h_radial_hat.clone())?;
            let q_h = OrthonormalSubspace::from_row_space_svd("h_rad_basis", &h_mat, 1e-9, 1)?;
            let joint = OrthonormalSubspace::union_bases(
                "stiefel_joint_null",
                d,
                &[q_forb, &q_h],
                q_forb.rank() + 1,
            )?;
            v_tan = joint.project_out(target_dir);
        }

        let tan_norm = l2_norm(&v_tan);
        let v_tangent_hat = normalize_unit_vec(&v_tan).ok_or_else(|| {
            V2CompilerError::DegenerateProjectedTarget {
                rule: rule_name.to_string(),
                norm: tan_norm,
            }
        })?;

        let orthogonality_err = dot_f64(&h_radial_hat, &v_tangent_hat).abs();
        let forbidden_leak_err =
            forbidden_nullspace.map_or(0.0, |q| q.leakage_norm(&v_tangent_hat));

        Ok(Self {
            h_radial_hat,
            v_tangent_hat,
            residual_radius,
            angle,
            orthogonality_err,
            forbidden_leak_err,
        })
    }

    /// Radial `w_down` write vector scaled for unit post-SwiGLU activation `act_rad`:
    /// $\mathbf{w}_{\text{down,rad}} = \frac{R(\cos\theta - 1)}{\bar{a}_{\text{rad}}}\,\hat{\mathbf{h}}_{\text{rad}}$.
    #[must_use]
    pub fn scaled_radial_down(&self, mean_act_rad: f64) -> Vec<f64> {
        let scale = (self.residual_radius * self.angle.radial_coeff()) / mean_act_rad;
        self.h_radial_hat.iter().map(|&h| h * scale).collect()
    }

    /// Tangential `w_down` write vector scaled for unit signed activation `act_tan`:
    /// $\mathbf{w}_{\text{down,tan}} = \frac{R\sin\theta}{\bar{a}_{\text{tan}}}\,\hat{\mathbf{v}}_{\text{tan}}$.
    #[must_use]
    pub fn scaled_tangential_down(&self, mean_signed_act_tan: f64) -> Vec<f64> {
        let scale = (self.residual_radius * self.angle.tangential_coeff()) / mean_signed_act_tan;
        self.v_tangent_hat.iter().map(|&v| v * scale).collect()
    }
}
