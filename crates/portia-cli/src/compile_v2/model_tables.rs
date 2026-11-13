//! `ModelGeometryTables` (`W_U_top`, `DownstreamSwigluBlock`, `DeltaNetGateLayer`) population
//! from `PreparedV2CompileSpec` (`compile_v2::model_tables`).

use crate::compile_v2::obligations::PreparedV2CompileSpec;
use anyhow::{bail, Result};
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::lowering::ModelGeometryTables;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::numerics::StateMatrix;
use portia_compiler::solvers::deltanet_scrub::DeltaNetGateLayer;
use portia_compiler::wire::parse_mount_policy_and_pen;
use portia_core::{CartridgeBank, MountPolicy};
use std::collections::BTreeMap;

fn build_wu_top_basis(
    vocab_top_k_eigvecs: &[Vec<f64>],
    d: usize,
) -> Result<Option<OrthonormalSubspace>> {
    if vocab_top_k_eigvecs.is_empty() {
        return Ok(None);
    }
    let rows_mat = if vocab_top_k_eigvecs.len() == d && vocab_top_k_eigvecs[0].len() != d {
        let k = vocab_top_k_eigvecs[0].len();
        if k == 0 {
            return Ok(None);
        }
        let mut flat = vec![0.0; k * d];
        for (r_d, row) in vocab_top_k_eigvecs.iter().enumerate() {
            if row.len() != k {
                bail!("vocab_top_k_eigvecs row {r_d} length mismatch");
            }
            for (c_k, &val) in row.iter().enumerate() {
                flat[c_k * d + r_d] = val;
            }
        }
        StateMatrix::try_new("wu_eigvecs_t", k, d, flat)?
    } else {
        StateMatrix::from_rows("wu_eigvecs_rows", vocab_top_k_eigvecs)?
    };
    let rank_cap = rows_mat.nrows().min(64).min(d.saturating_sub(16));
    if rank_cap == 0 {
        return Ok(None);
    }
    Ok(Some(OrthonormalSubspace::from_row_space_svd(
        "wu_top_basis",
        &rows_mat,
        1e-4,
        rank_cap,
    )?))
}

pub(crate) fn populate_model_geometry_tables(
    prep: &PreparedV2CompileSpec,
) -> Result<ModelGeometryTables> {
    let d = prep.hidden_dim;
    let router_layer = prep
        .obligations
        .iter()
        .filter(|ob| {
            parse_mount_policy_and_pen(&ob.bank_name).is_ok_and(|(_, mp, _)| {
                matches!(mp, MountPolicy::DynamicBank(CartridgeBank::OrientTrunk))
            })
        })
        .map(|ob| ob.layer)
        .max()
        .unwrap_or_else(|| {
            prep.obligations
                .iter()
                .map(|ob| ob.layer)
                .max()
                .unwrap_or(22)
        });

    let mut geom = ModelGeometryTables {
        model_id: prep.model_family.clone(),
        base_intermediate_size: prep.base_inter,
        widened_slot_capacity: prep.widened_slots.max(2),
        w_u_top_basis: None,
        downstream_mlps: BTreeMap::new(),
        gqa_layers: BTreeMap::new(),
        deltanet_layers: BTreeMap::new(),
    };

    if let Some(ref g_spec) = prep.geometry {
        geom.w_u_top_basis = build_wu_top_basis(&g_spec.vocab_top_k_eigvecs, d)?;

        if let Some(ref ds) = g_spec.downstream_mlp_anti_echo {
            let m = ds.w_gate.len();
            if m > 0 && ds.w_down.len() == d {
                let mut wd_cols_flat = vec![0.0; m * d];
                for (r_d, down_row) in ds.w_down.iter().enumerate() {
                    if down_row.len() != m {
                        bail!("downstream_mlp_anti_echo.w_down row {r_d} len != {m}");
                    }
                    for (c_m, &val) in down_row.iter().enumerate() {
                        wd_cols_flat[c_m * d + r_d] = val;
                    }
                }
                let post_scale: Vec<f64> =
                    ds.post_attn_norm_weight.iter().map(|&w| 1.0 + w).collect();
                let block = DownstreamSwigluBlock::try_new(
                    router_layer + 1,
                    post_scale,
                    StateMatrix::from_rows("ds_w_gate", &ds.w_gate)?,
                    StateMatrix::from_rows("ds_w_up", &ds.w_up)?,
                    StateMatrix::try_new("ds_w_down_cols", m, d, wd_cols_flat)?,
                )?;
                geom.downstream_mlps.insert(router_layer + 1, block);
            }
        }

        for (l_str, dn_spec) in &g_spec.deltanet_layer_weights {
            if let Ok(upstream_l) = l_str.parse::<usize>() {
                if !dn_spec.in_proj_a.is_empty() && !dn_spec.in_proj_b.is_empty() {
                    let norm_src = if dn_spec.input_layernorm_weight.len() == d {
                        &dn_spec.input_layernorm_weight
                    } else {
                        &g_spec.norm_weight
                    };
                    let input_norm_scale: Vec<f64> = norm_src.iter().map(|&w| 1.0 + w).collect();
                    let w_a_proj = StateMatrix::from_rows("dn_w_a", &dn_spec.in_proj_a)?;
                    let w_b_proj = StateMatrix::from_rows("dn_w_b", &dn_spec.in_proj_b)?;
                    let num_heads = w_a_proj.nrows();
                    geom.deltanet_layers.insert(
                        upstream_l + 1,
                        DeltaNetGateLayer {
                            layer_idx: upstream_l + 1,
                            hidden_dim: d,
                            num_heads,
                            input_norm_scale,
                            w_a_proj,
                            w_b_proj,
                        },
                    );
                }
            }
        }
    }
    Ok(geom)
}
