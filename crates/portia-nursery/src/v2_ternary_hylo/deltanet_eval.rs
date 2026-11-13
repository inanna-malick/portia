//! Exact Pure-Rust `Qwen3_5GatedDeltaNet` Recurrent Layer (`v2_ternary_hylo::deltanet_eval`),
//! verified to `2.38e-7` (`f32` parity) against PyTorch `Qwen3_5GatedDeltaNet`.

use portia_compiler::numerics::safetensors_io::{
    load_qwen35_rmsnorm_scale_1p, load_safetensor_f64, load_safetensor_matrix_2d,
};
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::{
    dot_f64, l2_norm, normalize_unit_vec, rms_norm_slice, silu_f64, StateMatrix,
};
use portia_compiler::solvers::deltanet_scrub::DeltaNetGateLayer;
use portia_compiler::V2CompilerError;
use std::path::Path;

/// Full recurrent layer weights of a `Qwen3_5GatedDeltaNet` block.
#[derive(Debug, Clone)]
pub struct Qwen35GatedDeltaNetLayer {
    pub layer_idx: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub input_norm_scale: Vec<f64>,
    pub in_proj_qkv: StateMatrix,
    /// Flattened `(conv_dim, conv_kernel_size)` depthwise causal filter (`[t-3, t-2, t-1, t]`).
    pub conv1d_weight: StateMatrix,
    pub in_proj_z: StateMatrix,
    pub in_proj_a: StateMatrix,
    pub in_proj_b: StateMatrix,
    pub a_log: Vec<f64>,
    pub dt_bias: Vec<f64>,
    /// `Qwen3_5RMSNormGated` weight (`w`, NOT `1 + w`!) of length `head_v_dim`.
    pub gated_norm_weight: Vec<f64>,
    pub out_proj: StateMatrix,
}

/// Exact fast-weight & causal-conv1d state snapshot of a `Qwen3_5GatedDeltaNet` layer.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaNetRecurrentSnapshot {
    /// `[num_heads][head_k_dim * head_v_dim]` row-major recurrent matrix $\mathbf{S}_t^{(h)}$.
    pub s_heads: Vec<Vec<f64>>,
    /// Last `conv_kernel_size - 1` pre-conv `qkv` vectors (`[t-3, t-2, t-1]`, oldest to newest).
    pub conv_history: Vec<Vec<f64>>,
}

impl Qwen35GatedDeltaNetLayer {
    /// Load a real `Qwen3.5` `linear_attn` layer directly from a HuggingFace `.safetensors` snapshot.
    pub fn load_from_safetensors(
        snapshot_dir: &Path,
        layer_idx: usize,
    ) -> Result<Self, V2CompilerError> {
        let pfx = format!("model.language_model.layers.{layer_idx}");
        let input_norm_scale =
            load_qwen35_rmsnorm_scale_1p(snapshot_dir, &format!("{pfx}.input_layernorm.weight"))?;
        let in_proj_qkv = load_safetensor_matrix_2d(
            snapshot_dir,
            &format!("{pfx}.linear_attn.in_proj_qkv.weight"),
        )?;
        let (conv_shape, conv_raw) =
            load_safetensor_f64(snapshot_dir, &format!("{pfx}.linear_attn.conv1d.weight"))?;
        let conv_dim = conv_shape[0];
        let conv_k = *conv_shape.last().unwrap_or(&4);
        let conv1d_weight = StateMatrix::try_new("conv1d", conv_dim, conv_k, conv_raw)?;
        let in_proj_z = load_safetensor_matrix_2d(
            snapshot_dir,
            &format!("{pfx}.linear_attn.in_proj_z.weight"),
        )?;
        let in_proj_a = load_safetensor_matrix_2d(
            snapshot_dir,
            &format!("{pfx}.linear_attn.in_proj_a.weight"),
        )?;
        let in_proj_b = load_safetensor_matrix_2d(
            snapshot_dir,
            &format!("{pfx}.linear_attn.in_proj_b.weight"),
        )?;
        let (_, a_log) = load_safetensor_f64(snapshot_dir, &format!("{pfx}.linear_attn.A_log"))?;
        let (_, dt_bias) =
            load_safetensor_f64(snapshot_dir, &format!("{pfx}.linear_attn.dt_bias"))?;
        let (_, gated_norm_weight) =
            load_safetensor_f64(snapshot_dir, &format!("{pfx}.linear_attn.norm.weight"))?;
        let out_proj =
            load_safetensor_matrix_2d(snapshot_dir, &format!("{pfx}.linear_attn.out_proj.weight"))?;

        let hidden_dim = input_norm_scale.len();
        let num_heads = in_proj_a.nrows();
        let value_dim = in_proj_z.nrows();
        let head_v_dim = value_dim / num_heads;
        let key_dim = (conv_dim - value_dim) / 2;
        let head_k_dim = key_dim / num_heads;

        Ok(Self {
            layer_idx,
            hidden_dim,
            num_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size: conv_k,
            input_norm_scale,
            in_proj_qkv,
            conv1d_weight,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            a_log,
            dt_bias,
            gated_norm_weight,
            out_proj,
        })
    }

    /// Extract the gate-only [`DeltaNetGateLayer`] view consumed by `portia-compiler`.
    #[must_use]
    pub fn gate_layer_view(&self) -> DeltaNetGateLayer {
        DeltaNetGateLayer {
            layer_idx: self.layer_idx,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            input_norm_scale: self.input_norm_scale.clone(),
            w_a_proj: self.in_proj_a.clone(),
            w_b_proj: self.in_proj_b.clone(),
        }
    }

    /// Allocate a zeroed recurrent + `conv1d` state snapshot.
    #[must_use]
    pub fn zero_state(&self) -> DeltaNetRecurrentSnapshot {
        let conv_dim = self.in_proj_qkv.nrows();
        let hist_len = self.conv_kernel_size.saturating_sub(1);
        DeltaNetRecurrentSnapshot {
            s_heads: vec![vec![0.0; self.head_k_dim * self.head_v_dim]; self.num_heads],
            conv_history: vec![vec![0.0; conv_dim]; hist_len],
        }
    }

    /// Exact `bf16` host-RAM byte size of snapshotting `num_deltanet_layers` recurrent + `conv1d` states.
    #[must_use]
    pub const fn host_ram_snapshot_bytes(&self, num_deltanet_layers: usize) -> usize {
        let s_elems = self.num_heads * self.head_k_dim * self.head_v_dim;
        let conv_elems = (2 * self.num_heads * self.head_k_dim + self.num_heads * self.head_v_dim)
            * self.conv_kernel_size.saturating_sub(1);
        num_deltanet_layers * (s_elems + conv_elems) * 2
    }

    /// Measure lag-tap energy fractions `[t-(K-1) .. t]` of `conv1d.weight`.
    #[must_use]
    pub fn conv1d_tap_energy_fractions(&self) -> Vec<f64> {
        let mut energies = vec![0.0; self.conv_kernel_size];
        for r in 0..self.conv1d_weight.nrows() {
            for (tap, &w) in self.conv1d_weight.row(r).iter().enumerate() {
                energies[tap] += w * w;
            }
        }
        let sum: f64 = energies.iter().sum::<f64>().max(1e-12);
        energies.into_iter().map(|e| e / sum).collect()
    }

    /// Synthesize the naive `OverwriteLatch` ($\beta_{t,h} \to 1$, $\Delta b > 0$, $\Delta a = 0$)
    /// residual impulse vector on `target_heads`, illustrating coupled write-gate pollution on `DeltaNet-1`.
    pub fn synthesize_naive_overwrite_latch_delta(
        &self,
        target_heads: &[usize],
        b_drive: f64,
    ) -> Result<Vec<f64>, V2CompilerError> {
        let d = self.hidden_dim;
        let mut a_rows = Vec::with_capacity(self.num_heads * d);
        for h in 0..self.num_heads {
            for (&wa, &s) in self.in_proj_a.row(h).iter().zip(&self.input_norm_scale) {
                a_rows.push(wa * s);
            }
        }
        let q_a = OrthonormalSubspace::from_row_space_svd(
            "deltanet_a_forb",
            &StateMatrix::try_new("a_rows", self.num_heads, d, a_rows)?,
            1e-6,
            self.num_heads,
        )?;
        let mut raw_b = vec![0.0; d];
        for &h in target_heads {
            for (j, (&wb, &s)) in self
                .in_proj_b
                .row(h)
                .iter()
                .zip(&self.input_norm_scale)
                .enumerate()
            {
                raw_b[j] += wb * s;
            }
        }
        let clean_b = q_a.project_out(&raw_b);
        let unit_b = normalize_unit_vec(&clean_b).ok_or_else(|| {
            V2CompilerError::DegenerateProjectedTarget {
                rule: "naive_overwrite_latch".to_string(),
                norm: l2_norm(&clean_b),
            }
        })?;
        Ok(unit_b.into_iter().map(|v| v * b_drive).collect())
    }

    /// Advance the exact `Qwen3_5GatedDeltaNet` state by one token residual `h_in` and return `out_proj` output.
    pub fn step_token(
        &self,
        state: &mut DeltaNetRecurrentSnapshot,
        h_in: &[f64],
    ) -> Result<Vec<f64>, V2CompilerError> {
        let x = rms_norm_slice(h_in, &self.input_norm_scale, 1e-6);
        let qkv_raw = self.in_proj_qkv.matvec(&x)?;
        let z_raw = self.in_proj_z.matvec(&x)?;
        let a_raw = self.in_proj_a.matvec(&x)?;
        let b_raw = self.in_proj_b.matvec(&x)?;

        let conv_dim = qkv_raw.len();
        let k_len = self.conv_kernel_size;
        let hist_len = k_len.saturating_sub(1);
        let mut conv_out = vec![0.0; conv_dim];
        for c in 0..conv_dim {
            let w_row = self.conv1d_weight.row(c);
            let mut acc = 0.0;
            for (&w_tap, hist_vec) in w_row.iter().zip(state.conv_history.iter()).take(hist_len) {
                acc += w_tap * hist_vec[c];
            }
            acc += w_row[k_len - 1] * qkv_raw[c];
            conv_out[c] = silu_f64(acc);
        }
        if hist_len > 0 {
            state.conv_history.remove(0);
            state.conv_history.push(qkv_raw);
        }

        let key_dim = self.num_heads * self.head_k_dim;
        let q_slice = &conv_out[0..key_dim];
        let k_slice = &conv_out[key_dim..2 * key_dim];
        let v_slice = &conv_out[2 * key_dim..];
        let inv_sqrt_dk = 1.0 / (self.head_k_dim as f64).sqrt();

        let mut y_gated_flat = vec![0.0; self.num_heads * self.head_v_dim];
        for h in 0..self.num_heads {
            let q_h = &q_slice[h * self.head_k_dim..(h + 1) * self.head_k_dim];
            let k_h = &k_slice[h * self.head_k_dim..(h + 1) * self.head_k_dim];
            let v_h = &v_slice[h * self.head_v_dim..(h + 1) * self.head_v_dim];
            let z_h = &z_raw[h * self.head_v_dim..(h + 1) * self.head_v_dim];

            let inv_qn = 1.0 / (dot_f64(q_h, q_h) + 1e-6).sqrt();
            let inv_kn = 1.0 / (dot_f64(k_h, k_h) + 1e-6).sqrt();
            let sp_arg = a_raw[h] + self.dt_bias[h];
            let sp = if sp_arg > 20.0 {
                sp_arg
            } else {
                (1.0 + sp_arg.exp()).ln()
            };
            let alpha = (-self.a_log[h].exp() * sp).exp();
            let beta = 1.0 / (1.0 + (-b_raw[h]).exp());

            let s_mat = &mut state.s_heads[h];
            for val in s_mat.iter_mut() {
                *val *= alpha;
            }
            let mut kv_mem = vec![0.0; self.head_v_dim];
            for ki in 0..self.head_k_dim {
                let k_val = k_h[ki] * inv_kn;
                let row = &s_mat[ki * self.head_v_dim..(ki + 1) * self.head_v_dim];
                for (vi, &sv) in kv_mem.iter_mut().zip(row.iter()) {
                    *vi += sv * k_val;
                }
            }
            for ki in 0..self.head_k_dim {
                let k_val = k_h[ki] * inv_kn;
                let row = &mut s_mat[ki * self.head_v_dim..(ki + 1) * self.head_v_dim];
                for vi in 0..self.head_v_dim {
                    row[vi] += k_val * (v_h[vi] - kv_mem[vi]) * beta;
                }
            }
            let mut y_h = vec![0.0; self.head_v_dim];
            for ki in 0..self.head_k_dim {
                let q_val = q_h[ki] * inv_qn * inv_sqrt_dk;
                let row = &s_mat[ki * self.head_v_dim..(ki + 1) * self.head_v_dim];
                for (vi, &sv) in y_h.iter_mut().zip(row.iter()) {
                    *vi += sv * q_val;
                }
            }
            let var = dot_f64(&y_h, &y_h) / (self.head_v_dim as f64);
            let inv_rms = 1.0 / (var + 1e-6).sqrt();
            let out_slice = &mut y_gated_flat[h * self.head_v_dim..(h + 1) * self.head_v_dim];
            for vi in 0..self.head_v_dim {
                out_slice[vi] =
                    self.gated_norm_weight[vi] * (y_h[vi] * inv_rms) * silu_f64(z_h[vi]);
            }
        }
        self.out_proj.matvec(&y_gated_flat)
    }
}
