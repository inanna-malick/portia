//! Foreign-key RoPE-passthrough delta synthesis and `RelationalSchemaGraph` pointer lowering
//! (`v2_pointer_bridge::fk_route`).

use crate::grammar_trie::compile_sqlite_lark_grammar_for_schemas;
use crate::relational_ast::{ForeignKeyEdge, RelationalSchemaGraph};
use crate::v2_pointer_bridge::{build_paired_asid_coloring_intent, SqlPointerBridgeConfig};
use portia_compiler::discovery::gqa_bus::GqaPointerLayer;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::numerics::{dot_f64, normalize_unit_vec, rms_norm_slice, StateMatrix};
use portia_compiler::V2CompilerError;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, MountPolicy, TableSchema};
use std::collections::BTreeMap;

fn mean_passthrough_key_for_states(
    gqa: &GqaPointerLayer,
    copy_heads: &[usize],
    states: &StateMatrix,
) -> Result<Vec<f64>, V2CompilerError> {
    if states.hidden_dim() != gqa.hidden_dim {
        return Err(V2CompilerError::DimensionMismatch {
            context: "mean_passthrough_key_for_states",
            expected: gqa.hidden_dim,
            actual: states.hidden_dim(),
        });
    }
    let pass_d = gqa.passthrough_dim();
    let mut kv_heads = Vec::new();
    for &h in copy_heads {
        let kv_h = h / gqa.kv_group_size();
        if !kv_heads.contains(&kv_h) {
            kv_heads.push(kv_h);
        }
    }
    let scale = 1.0 / ((states.nrows() * kv_heads.len().max(1)) as f64);
    let mut acc = vec![0.0_f64; pass_d];
    for r in 0..states.nrows() {
        let h_norm = rms_norm_slice(states.row(r), &gqa.input_norm_scale, 1e-6);
        for &kv_h in &kv_heads {
            let r0 = kv_h * gqa.head_dim;
            let k_raw: Vec<f64> = (0..gqa.head_dim)
                .map(|i| dot_f64(gqa.w_k_proj.row(r0 + i), &h_norm))
                .collect();
            let k_normed = rms_norm_slice(&k_raw, &gqa.k_norm_scale, 1e-6);
            for p in 0..pass_d {
                acc[p] += k_normed[gqa.rotary_dim + p] * scale;
            }
        }
    }
    Ok(acc)
}

/// Extract a unit RoPE-passthrough (`[rotary_dim .. head_dim)`) structural foreign-key contrast
/// vector across `copy_heads` from FIT calibration states with **zero `v_ov_role` training-token leak**.
pub fn synthesize_fk_passthrough_query_delta(
    gqa: &GqaPointerLayer,
    copy_heads: &[usize],
    fk_target_states: &StateMatrix,
    competitor_col_states: &StateMatrix,
    prose_decoy_states: Option<&StateMatrix>,
    scratchpad_decoy_states: Option<&StateMatrix>,
) -> Result<Vec<f64>, V2CompilerError> {
    if copy_heads.is_empty() {
        return Err(V2CompilerError::InvalidConfig {
            context: "synthesize_fk_passthrough_query_delta",
            reason: "copy_heads must not be empty".to_string(),
        });
    }
    let k_fk = mean_passthrough_key_for_states(gqa, copy_heads, fk_target_states)?;
    let k_comp = mean_passthrough_key_for_states(gqa, copy_heads, competitor_col_states)?;
    let k_prose = match prose_decoy_states {
        Some(s) => Some(mean_passthrough_key_for_states(gqa, copy_heads, s)?),
        None => None,
    };
    let k_think = match scratchpad_decoy_states {
        Some(s) => Some(mean_passthrough_key_for_states(gqa, copy_heads, s)?),
        None => None,
    };

    let (w_comp, w_prose, w_think) = match (k_prose.is_some(), k_think.is_some()) {
        (true, true) => (0.40, 0.30, 0.30),
        (true, false) => (0.60, 0.40, 0.0),
        (false, true) => (0.60, 0.0, 0.40),
        (false, false) => (1.00, 0.0, 0.0),
    };

    let mut delta = vec![0.0_f64; gqa.passthrough_dim()];
    for i in 0..delta.len() {
        let mut v = k_fk[i] - w_comp * k_comp[i];
        if let Some(ref kp) = k_prose {
            v -= w_prose * kp[i];
        }
        if let Some(ref kt) = k_think {
            v -= w_think * kt[i];
        }
        delta[i] = v;
    }
    normalize_unit_vec(&delta).ok_or_else(|| V2CompilerError::DegenerateProjectedTarget {
        rule: "fk_passthrough_query_delta".to_string(),
        norm: 0.0,
    })
}

/// Build a [`HardwareIntentKind::SubmergedPointerRoute`] rule for a SQLite [`ForeignKeyEdge`].
pub fn build_foreign_key_pointer_route_intent(
    edge: &ForeignKeyEdge,
    config: &SqlPointerBridgeConfig,
    query_passthrough_delta: Vec<f64>,
    guard: RegimeExpr,
    grammar: GrammarCartridgeSpec,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    if query_passthrough_delta.is_empty() {
        return Err(V2CompilerError::InvalidConfig {
            context: "build_foreign_key_pointer_route_intent",
            reason: "query_passthrough_delta must be non-empty".to_string(),
        });
    }
    let rule_name = format!(
        "fk_ptr_{}_{}_to_{}_{}",
        edge.child_table, edge.child_col, edge.parent_table, edge.parent_col
    );
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name,
        guard,
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::ModePayload(config.read_mode)),
        grammar,
        kind: HardwareIntentKind::SubmergedPointerRoute {
            upstream_mlp_layer: config.upstream_mlp_layer,
            target_gqa_layer: config.target_gqa_layer,
            copy_heads: config.copy_heads.clone(),
            distractor_heads: config.distractor_heads.clone(),
            query_passthrough_delta,
            angle: config.pointer_route_angle,
        },
    }))
}

/// Validate that `edge`'s child and parent `(table, column)` endpoints exist in `schema`.
pub fn validate_fk_edge_in_schema(
    schema: &RelationalSchemaGraph,
    edge: &ForeignKeyEdge,
) -> Result<(), V2CompilerError> {
    let check_col = |tbl: &str, col: &str, role: &str| -> Result<(), V2CompilerError> {
        let Some(t) = schema.tables.get(tbl) else {
            return Err(V2CompilerError::InvalidConfig {
                context: "validate_fk_edge_in_schema",
                reason: format!("{role} table `{tbl}` missing in RelationalSchemaGraph"),
            });
        };
        if !t.columns.iter().any(|c| c.name.eq_ignore_ascii_case(col)) {
            return Err(V2CompilerError::InvalidConfig {
                context: "validate_fk_edge_in_schema",
                reason: format!("{role} column `{tbl}.{col}` missing in RelationalSchemaGraph"),
            });
        }
        Ok(())
    };
    check_col(&edge.child_table, &edge.child_col, "child")?;
    check_col(&edge.parent_table, &edge.parent_col, "parent")
}

/// Compile an entire [`RelationalSchemaGraph`] into a declarative `v2` Pointer-ISA [`IntentExpr`]
/// combining Paired `ASID` Symbol-Table Coloring with Submerged Foreign-Key Pointer Routing.
pub fn build_relational_schema_pointer_program(
    schema: &RelationalSchemaGraph,
    config: &SqlPointerBridgeConfig,
    fk_passthrough_deltas: &BTreeMap<ForeignKeyEdge, Vec<f64>>,
    read_guard: RegimeExpr,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    let tables_vec: Vec<TableSchema> = schema.tables.values().cloned().collect();
    let lark_grammar = if tables_vec.is_empty() {
        GrammarCartridgeSpec::Unconstrained
    } else {
        compile_sqlite_lark_grammar_for_schemas(&tables_vec, config.read_mode)
    };

    let asid_prog =
        build_paired_asid_coloring_intent(config, "sql_schema", read_guard.clone(), lark_grammar)?;
    if fk_passthrough_deltas.is_empty() {
        return Ok(asid_prog);
    }

    let mut fk_expr_opt: Option<IntentExpr> = None;
    for (edge, delta) in fk_passthrough_deltas {
        validate_fk_edge_in_schema(schema, edge)?;
        let rule_expr = build_foreign_key_pointer_route_intent(
            edge,
            config,
            delta.clone(),
            read_guard.clone(),
            GrammarCartridgeSpec::Unconstrained,
        )?;
        fk_expr_opt = Some(match fk_expr_opt {
            None => rule_expr,
            Some(acc) => acc & rule_expr,
        });
    }
    Ok(match fk_expr_opt {
        Some(fk_expr) => asid_prog & fk_expr,
        None => asid_prog,
    })
}
