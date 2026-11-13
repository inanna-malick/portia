//! Exact Qwen3.5 GQA multi-candidate forward-pass evaluator & static `OV` copy-head discovery
//! (`v2_pointer_bridge::gqa_eval`).

use portia_compiler::discovery::gqa_bus::GqaPointerLayer;
use portia_compiler::numerics::{dot_f64, l2_norm, rms_norm_slice, StateMatrix};
use portia_compiler::V2CompilerError;

/// Full single-layer Qwen3.5 GQA evaluation block (`GqaPointerLayer` + `o_proj` + `final_norm`).
#[derive(Debug, Clone)]
pub struct Qwen35GqaEvalBlock {
    pub gqa: GqaPointerLayer,
    pub w_o_proj: StateMatrix,
    pub final_norm_scale: Vec<f64>,
    pub rope_theta: f64,
}

/// Forward-pass metrics on a single multi-candidate SQL schema/prose/scratchpad world.
#[derive(Debug, Clone)]
pub struct GqaCandidateForwardResult {
    /// Unembedding logits `<final_norm(h_cursor + attn_out), w_u_cand>` per candidate token.
    pub candidate_logits: Vec<f64>,
    /// Mean attention probability across all `num_q_heads` per context key position.
    pub mean_all_head_attn: Vec<f64>,
    /// Mean attention probability across `copy_heads` per context key position.
    pub mean_copy_head_attn: Vec<f64>,
}

/// Apply HuggingFace half-rotated partial RoPE (`rotate_half` on `[0 .. rotary_dim)`) in place.
pub fn apply_partial_rope_in_place(
    vec: &mut [f64],
    rotary_dim: usize,
    pos: usize,
    rope_theta: f64,
) {
    let rot = rotary_dim.min(vec.len());
    let half = rot / 2;
    for i in 0..half {
        let inv_freq = rope_theta.powf(-(2.0 * (i as f64)) / (rot as f64));
        let ang = (pos as f64) * inv_freq;
        let (s, c) = ang.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * c - x1 * s;
        vec[i + half] = x1 * c + x0 * s;
    }
}

impl Qwen35GqaEvalBlock {
    pub fn try_new(
        gqa: GqaPointerLayer,
        w_o_proj: StateMatrix,
        final_norm_scale: Vec<f64>,
        rope_theta: f64,
    ) -> Result<Self, V2CompilerError> {
        let expected_o_cols = gqa.num_q_heads * gqa.head_dim;
        if w_o_proj.nrows() != gqa.hidden_dim || w_o_proj.hidden_dim() != expected_o_cols {
            return Err(V2CompilerError::DimensionMismatch {
                context: "Qwen35GqaEvalBlock::w_o_proj",
                expected: expected_o_cols,
                actual: w_o_proj.hidden_dim(),
            });
        }
        if final_norm_scale.len() != gqa.hidden_dim {
            return Err(V2CompilerError::DimensionMismatch {
                context: "Qwen35GqaEvalBlock::final_norm_scale",
                expected: gqa.hidden_dim,
                actual: final_norm_scale.len(),
            });
        }
        Ok(Self {
            gqa,
            w_o_proj,
            final_norm_scale,
            rope_theta,
        })
    }

    /// Rank GQA query heads on FIT calibration worlds (`0%` eval overlap) by contrastive
    /// token-identity `OV` copy score `<e_fk - e_decoy, final_norm ⊙ (sigmoid(gate_h) * W_O^(h) W_V^(g) RMSNorm(h_fk))>`.
    #[must_use]
    pub fn discover_fit_token_copy_heads(
        &self,
        fit_cursor_h: &[f64],
        fit_fk_states_and_unembeds: &[(&[f64], &[f64], &[f64])],
        num_copy: usize,
        num_distractor: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<(usize, f64)>) {
        let d = self.gqa.hidden_dim;
        let hd = self.gqa.head_dim;
        let grp = self.gqa.kv_group_size();
        let q_in = rms_norm_slice(fit_cursor_h, &self.gqa.input_norm_scale, 1e-6);
        let mut scores: Vec<(usize, f64)> = (0..self.gqa.num_q_heads)
            .map(|h| {
                let g = h / grp;
                let gate_r0 = h * 2 * hd + hd;
                let gates: Vec<f64> = (0..hd)
                    .map(|i| {
                        1.0 / (1.0 + (-dot_f64(self.gqa.w_q_proj.row(gate_r0 + i), &q_in)).exp())
                    })
                    .collect();
                let mut sum_s = 0.0_f64;
                for &(h_fk, e_fk, e_decoy) in fit_fk_states_and_unembeds {
                    let k_in = rms_norm_slice(h_fk, &self.gqa.input_norm_scale, 1e-6);
                    let v_vec: Vec<f64> = (0..hd)
                        .map(|i| gates[i] * dot_f64(self.gqa.w_v_proj.row(g * hd + i), &k_in))
                        .collect();
                    let mut ov_score = 0.0_f64;
                    for c in 0..d {
                        let o_row = &self.w_o_proj.row(c)[h * hd..(h + 1) * hd];
                        let delta_c = self.final_norm_scale[c] * dot_f64(o_row, &v_vec);
                        ov_score += delta_c * (e_fk[c] - e_decoy[c]);
                    }
                    sum_s += ov_score;
                }
                (h, sum_s / (fit_fk_states_and_unembeds.len().max(1) as f64))
            })
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let copy_heads: Vec<usize> = scores
            .iter()
            .filter(|&&(_, s)| s > 0.0)
            .take(num_copy.min(self.gqa.num_q_heads))
            .map(|&(h, _)| h)
            .collect();
        let copy_heads = if copy_heads.is_empty() {
            vec![scores[0].0]
        } else {
            copy_heads
        };
        let distractor_heads: Vec<usize> = scores
            .iter()
            .rev()
            .filter(|&&(h, _)| !copy_heads.contains(&h))
            .take(num_distractor)
            .map(|&(h, _)| h)
            .collect();
        (copy_heads, distractor_heads, scores)
    }

    /// Rank GQA query heads purely from static pretrained weights via diagonal `OV` copy trace:
    /// $\operatorname{tr}(\operatorname{diag}(\gamma_{\text{final}})\,\mathbf{W}_O^{(h)}\,\mathbf{W}_V^{(g)}\,\operatorname{diag}(\gamma_{\text{in}}))$.
    #[must_use]
    pub fn discover_ov_copy_and_distractor_heads(
        &self,
        num_copy: usize,
        num_distractor: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<(usize, f64)>) {
        let d = self.gqa.hidden_dim;
        let hd = self.gqa.head_dim;
        let grp = self.gqa.kv_group_size();
        let mut head_traces: Vec<(usize, f64)> = (0..self.gqa.num_q_heads)
            .map(|h| {
                let g = h / grp;
                let mut tr = 0.0_f64;
                for r in 0..hd {
                    let v_row = self.gqa.w_v_proj.row(g * hd + r);
                    let o_col = h * hd + r;
                    for (c, &v_val) in v_row.iter().enumerate().take(d) {
                        tr += self.final_norm_scale[c]
                            * self.w_o_proj.row(c)[o_col]
                            * v_val
                            * self.gqa.input_norm_scale[c];
                    }
                }
                (h, tr)
            })
            .collect();
        head_traces.sort_by(|a, b| b.1.total_cmp(&a.1));
        let copy_heads: Vec<usize> = head_traces
            .iter()
            .take(num_copy.min(self.gqa.num_q_heads))
            .map(|&(h, _)| h)
            .collect();
        let distractor_heads: Vec<usize> = head_traces
            .iter()
            .rev()
            .take(num_distractor.min(self.gqa.num_q_heads.saturating_sub(copy_heads.len())))
            .map(|&(h, _)| h)
            .collect();
        (copy_heads, distractor_heads, head_traces)
    }

    /// Exact KV-value cache linear contamination norm `||W_V · diag(input_norm_scale) · delta_h||_2`.
    #[must_use]
    pub fn wv_cache_contamination_l2(&self, delta_h: &[f64]) -> f64 {
        let scaled: Vec<f64> = delta_h
            .iter()
            .zip(&self.gqa.input_norm_scale)
            .map(|(&x, &s)| x * s)
            .collect();
        let v_out: Vec<f64> = (0..self.gqa.w_v_proj.nrows())
            .map(|r| dot_f64(self.gqa.w_v_proj.row(r), &scaled))
            .collect();
        l2_norm(&v_out)
    }

    /// Evaluate the full Qwen3.5 GQA layer (`RMSNorm -> q_proj/k_proj/v_proj -> q_norm/k_norm ->
    /// partial RoPE -> softmax(QK^T / sqrt(d_h)) -> sigmoid(W_Q_gate) * V -> o_proj -> final_norm -> W_U`).
    #[must_use]
    pub fn eval_multi_candidate_forward(
        &self,
        query_h: &[f64],
        query_pos: usize,
        context_keys: &[(usize, Vec<f64>)],
        copy_heads: &[usize],
        unembed_candidates: &[&[f64]],
    ) -> GqaCandidateForwardResult {
        let d = self.gqa.hidden_dim;
        let hd = self.gqa.head_dim;
        let n_q = self.gqa.num_q_heads;
        let n_kv = self.gqa.num_kv_heads;
        let grp = self.gqa.kv_group_size();
        let n_keys = context_keys.len();
        let inv_sqrt_hd = 1.0 / (hd as f64).sqrt();

        let q_in = rms_norm_slice(query_h, &self.gqa.input_norm_scale, 1e-6);

        // Precompute `K` (normed + partial RoPE) and `V` per `(key_idx, kv_head)`:
        let mut k_cache = vec![vec![vec![0.0_f64; hd]; n_kv]; n_keys];
        let mut v_cache = vec![vec![vec![0.0_f64; hd]; n_kv]; n_keys];
        for (j, (pos_j, key_h)) in context_keys.iter().enumerate() {
            let k_in = rms_norm_slice(key_h, &self.gqa.input_norm_scale, 1e-6);
            for g in 0..n_kv {
                let r0 = g * hd;
                let k_raw: Vec<f64> = (0..hd)
                    .map(|i| dot_f64(self.gqa.w_k_proj.row(r0 + i), &k_in))
                    .collect();
                let mut k_normed = rms_norm_slice(&k_raw, &self.gqa.k_norm_scale, 1e-6);
                apply_partial_rope_in_place(
                    &mut k_normed,
                    self.gqa.rotary_dim,
                    *pos_j,
                    self.rope_theta,
                );
                k_cache[j][g] = k_normed;
                v_cache[j][g] = (0..hd)
                    .map(|i| dot_f64(self.gqa.w_v_proj.row(r0 + i), &k_in))
                    .collect();
            }
        }

        let mut concat_heads = vec![0.0_f64; n_q * hd];
        let mut mean_all_head_attn = vec![0.0_f64; n_keys];
        let mut mean_copy_head_attn = vec![0.0_f64; n_keys];
        let copy_denom = (copy_heads.len().max(1)) as f64;

        for h in 0..n_q {
            let g = h / grp;
            let q_r0 = h * 2 * hd;
            let gate_r0 = q_r0 + hd;

            let q_raw: Vec<f64> = (0..hd)
                .map(|i| dot_f64(self.gqa.w_q_proj.row(q_r0 + i), &q_in))
                .collect();
            let mut q_vec = rms_norm_slice(&q_raw, &self.gqa.q_norm_scale, 1e-6);
            apply_partial_rope_in_place(
                &mut q_vec,
                self.gqa.rotary_dim,
                query_pos,
                self.rope_theta,
            );

            let logits: Vec<f64> = (0..n_keys)
                .map(|j| dot_f64(&q_vec, &k_cache[j][g]) * inv_sqrt_hd)
                .collect();
            let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|&z| (z - max_l).exp()).collect();
            let sum_e: f64 = exps.iter().sum::<f64>().max(1e-30);
            let probs: Vec<f64> = exps.into_iter().map(|e| e / sum_e).collect();

            let is_copy = copy_heads.contains(&h);
            for j in 0..n_keys {
                mean_all_head_attn[j] += probs[j] / (n_q as f64);
                if is_copy {
                    mean_copy_head_attn[j] += probs[j] / copy_denom;
                }
            }

            let mut head_val = vec![0.0_f64; hd];
            for (j, &p) in probs.iter().enumerate() {
                for i in 0..hd {
                    head_val[i] += p * v_cache[j][g][i];
                }
            }
            for i in 0..hd {
                let z_gate = dot_f64(self.gqa.w_q_proj.row(gate_r0 + i), &q_in);
                let sig = 1.0 / (1.0 + (-z_gate).exp());
                concat_heads[h * hd + i] = sig * head_val[i];
            }
        }

        let mut h_post = query_h.to_vec();
        for (c, hp) in h_post.iter_mut().enumerate().take(d) {
            *hp += dot_f64(self.w_o_proj.row(c), &concat_heads);
        }
        let h_final = rms_norm_slice(&h_post, &self.final_norm_scale, 1e-6);
        let candidate_logits = unembed_candidates
            .iter()
            .map(|u| dot_f64(&h_final, u))
            .collect();

        GqaCandidateForwardResult {
            candidate_logits,
            mean_all_head_attn,
            mean_copy_head_attn,
        }
    }
}
