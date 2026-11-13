//! Dual-Span (`OrientTrunk` `Plan:`/`<think>` + `ModePayload(Read)` `<sql>`) and Per-Mode
//! Two-Cartridge (`ThinkCartridge(Mode)` -> `</think>` swap -> `PayloadCartridge(Mode)`) `ASID`
//! & Submerged Foreign-Key Pointer Lowering (`v2_pointer_bridge::dual_span`).

use crate::grammar_trie::compile_sqlite_lark_grammar_for_schemas;
use crate::relational_ast::{ForeignKeyEdge, RelationalSchemaGraph};
use crate::v2_pointer_bridge::{
    build_foreign_key_pointer_route_intent, build_paired_asid_coloring_intent,
    build_schema_read_asid_intent, build_schema_write_asid_intent,
    build_scratchpad_write_asid_intent, validate_fk_edge_in_schema, SqlPointerBridgeConfig,
};
use portia_compiler::discovery::gqa_bus::MemorySegmentAsid;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::V2CompilerError;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, MountPolicy, TableSchema};
use std::collections::BTreeMap;

/// Disjoint per-mode Two-Cartridge lowering:
/// - `think_program`: mounted during `RelayPhase::PlanBody` (`[MODE: <mode>]` already known),
///   stamps `+AuthoritativeSchema` on `<schema>` (`PermanentStatic`), reads `+AuthoritativeSchema`
///   at `<think>`/`Plan:` join cursors, and stamps `+EphemeralScratchpad` (`-ASID`) onto generated
///   `<think>` tokens inside its dynamic bank (automatically detached at `</think>`).
/// - `payload_program`: mounted right after `</think>` during `RelayPhase::PayloadBody` (`<sql>`),
///   retaining the identical `PermanentStatic` `<schema>` stamp while querying
///   `+AuthoritativeSchema - 0.75 * EphemeralScratchpad` + `SubmergedPointerRoute`.
#[derive(Debug, Clone, PartialEq)]
pub struct PerModeThinkAndSqlPrograms {
    pub think_program: IntentExpr,
    pub payload_program: IntentExpr,
}

/// Plan-phase (`MountPolicy::DynamicBank(CartridgeBank::OrientTrunk)`) `ASID` query selector rule
/// driving `+AuthoritativeSchema` across `copy_heads` inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$
/// so `<think>` / `Plan:` dereferences the true `FOREIGN KEY` column during planning.
pub fn build_orient_trunk_schema_read_asid_intent(
    config: &SqlPointerBridgeConfig,
    rule_name: impl Into<String>,
    guard: RegimeExpr,
    grammar: GrammarCartridgeSpec,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name.into(),
        guard,
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
        grammar,
        kind: HardwareIntentKind::AsidSegmentColor {
            upstream_mlp_layer: config.upstream_mlp_layer,
            target_gqa_layer: config.target_gqa_layer,
            copy_heads: config.copy_heads.clone(),
            segment: MemorySegmentAsid::AuthoritativeSchema,
            is_write_side_key_stamp: false,
            angle: config.read_asid_angle,
        },
    }))
}

/// Plan-phase (`MountPolicy::DynamicBank(bank)`) `SubmergedPointerRoute` rule opening `copy_heads`
/// and closing `distractor_heads` at the `<think>` / `Plan:` join-key cursor.
pub fn build_bank_foreign_key_pointer_route_intent(
    bank: CartridgeBank,
    rule_prefix: &str,
    edge: &ForeignKeyEdge,
    config: &SqlPointerBridgeConfig,
    query_passthrough_delta: Vec<f64>,
    guard: RegimeExpr,
    grammar: GrammarCartridgeSpec,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    if query_passthrough_delta.is_empty() {
        return Err(V2CompilerError::InvalidConfig {
            context: "build_bank_foreign_key_pointer_route_intent",
            reason: "query_passthrough_delta must be non-empty".to_string(),
        });
    }
    let rule_name = format!(
        "{rule_prefix}_{}_{}_to_{}_{}",
        edge.child_table, edge.child_col, edge.parent_table, edge.parent_col
    );
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name,
        guard,
        mount_policy: MountPolicy::DynamicBank(bank),
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

/// Dynamic `<think>`-span `EphemeralScratchpad` (`-ASID`) write rule mounted only while the
/// thinking cartridge is active (detaching cleanly at `</think>` so `<sql>` tokens are never
/// misclassified as scratchpad).
pub fn build_dynamic_scratchpad_write_asid_intent(
    bank: CartridgeBank,
    config: &SqlPointerBridgeConfig,
    rule_name: impl Into<String>,
    guard: RegimeExpr,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name.into(),
        guard,
        mount_policy: MountPolicy::DynamicBank(bank),
        grammar: GrammarCartridgeSpec::Unconstrained,
        kind: HardwareIntentKind::AsidSegmentColor {
            upstream_mlp_layer: config.upstream_mlp_layer,
            target_gqa_layer: config.target_gqa_layer,
            copy_heads: config.copy_heads.clone(),
            segment: MemorySegmentAsid::EphemeralScratchpad,
            is_write_side_key_stamp: true,
            angle: config.write_asid_angle,
        },
    }))
}

/// Compile a [`RelationalSchemaGraph`] into a Dual-Span `v2` Pointer-ISA [`IntentExpr`] mounting:
/// 1. `PermanentStatic`: `+AuthoritativeSchema` (`<schema>`) & `+EphemeralScratchpad` (`<think>`) writes,
/// 2. `DynamicBank(OrientTrunk)`: `+AuthoritativeSchema` read + `SubmergedPointerRoute` during `<think>`/`Plan:`,
/// 3. `DynamicBank(ModePayload(config.read_mode))`: `+AuthoritativeSchema - EphemeralScratchpad` read
///    + `SubmergedPointerRoute` (+ SQLite Lark grammar) during `<sql>`.
pub fn build_dual_span_plan_and_sql_pointer_program(
    schema: &RelationalSchemaGraph,
    config: &SqlPointerBridgeConfig,
    fk_passthrough_deltas: &BTreeMap<ForeignKeyEdge, Vec<f64>>,
    plan_guard: RegimeExpr,
    read_guard: RegimeExpr,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    let tables_vec: Vec<TableSchema> = schema.tables.values().cloned().collect();
    let lark_grammar = if tables_vec.is_empty() {
        GrammarCartridgeSpec::Unconstrained
    } else {
        compile_sqlite_lark_grammar_for_schemas(&tables_vec, config.read_mode)
    };

    let sql_asid =
        build_paired_asid_coloring_intent(config, "sql_schema", read_guard.clone(), lark_grammar)?;
    let plan_read = build_orient_trunk_schema_read_asid_intent(
        config,
        "plan_orient_asid_read_schema",
        plan_guard.clone(),
        GrammarCartridgeSpec::Unconstrained,
    )?;
    let mut prog = sql_asid & plan_read;

    for (edge, delta) in fk_passthrough_deltas {
        validate_fk_edge_in_schema(schema, edge)?;
        let plan_fk = build_bank_foreign_key_pointer_route_intent(
            CartridgeBank::OrientTrunk,
            "plan_fk_ptr",
            edge,
            config,
            delta.clone(),
            plan_guard.clone(),
            GrammarCartridgeSpec::Unconstrained,
        )?;
        let sql_fk = build_foreign_key_pointer_route_intent(
            edge,
            config,
            delta.clone(),
            read_guard.clone(),
            GrammarCartridgeSpec::Unconstrained,
        )?;
        prog = prog & plan_fk & sql_fk;
    }
    Ok(prog)
}

/// Build disjoint Per-Mode Thinking (`ThinkCartridge(read_mode)`) and SQL Payload
/// (`PayloadCartridge(read_mode)`) pointer programs swapped cleanly at `</think>`.
pub fn build_per_mode_think_and_sql_pointer_programs(
    schema: &RelationalSchemaGraph,
    config: &SqlPointerBridgeConfig,
    fk_passthrough_deltas: &BTreeMap<ForeignKeyEdge, Vec<f64>>,
    plan_guard: RegimeExpr,
    read_guard: RegimeExpr,
) -> Result<PerModeThinkAndSqlPrograms, V2CompilerError> {
    config.validate()?;
    let mode_bank = CartridgeBank::ModePayload(config.read_mode);
    let tables_vec: Vec<TableSchema> = schema.tables.values().cloned().collect();
    let lark_grammar = if tables_vec.is_empty() {
        GrammarCartridgeSpec::Unconstrained
    } else {
        compile_sqlite_lark_grammar_for_schemas(&tables_vec, config.read_mode)
    };

    let w_schema_think = build_schema_write_asid_intent(config, "sql_schema_asid_write_schema")?;
    let w_user_think =
        build_scratchpad_write_asid_intent(config, "user_prose_asid_write_scratchpad")?;
    let r_plan = build_schema_read_asid_intent(
        config,
        "think_mode_asid_read_schema",
        plan_guard.clone(),
        GrammarCartridgeSpec::Unconstrained,
    )?;
    let w_scratch_dyn = build_dynamic_scratchpad_write_asid_intent(
        mode_bank,
        config,
        "think_mode_asid_write_scratchpad",
        plan_guard.clone(),
    )?;
    let mut think_program = (w_schema_think & w_user_think) >> (r_plan & w_scratch_dyn);

    let w_schema_sql = build_schema_write_asid_intent(config, "sql_schema_asid_write_schema")?;
    let w_user_sql =
        build_scratchpad_write_asid_intent(config, "user_prose_asid_write_scratchpad")?;
    let r_sql = build_schema_read_asid_intent(
        config,
        "sql_schema_asid_read_schema",
        read_guard.clone(),
        lark_grammar,
    )?;
    let mut payload_program = (w_schema_sql & w_user_sql) >> r_sql;

    for (edge, delta) in fk_passthrough_deltas {
        validate_fk_edge_in_schema(schema, edge)?;
        let think_fk = build_bank_foreign_key_pointer_route_intent(
            mode_bank,
            "think_fk_ptr",
            edge,
            config,
            delta.clone(),
            plan_guard.clone(),
            GrammarCartridgeSpec::Unconstrained,
        )?;
        let sql_fk = build_foreign_key_pointer_route_intent(
            edge,
            config,
            delta.clone(),
            read_guard.clone(),
            GrammarCartridgeSpec::Unconstrained,
        )?;
        think_program = think_program & think_fk;
        payload_program = payload_program & sql_fk;
    }

    Ok(PerModeThinkAndSqlPrograms {
        think_program,
        payload_program,
    })
}
