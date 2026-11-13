//! `portia-sql` `v2` Submerged Pointer-ISA & Paired `ASID` Symbol-Table Bridge (`v2_pointer_bridge`).
//!
//! Bridges live SQLite relational schemas ([`crate::relational_ast::RelationalSchemaGraph`],
//! [`crate::relational_ast::ForeignKeyEdge`]) to `portia-compiler`'s declarative [`IntentExpr`] ISA:
//! 1. **Paired `ASID` Symbol-Table Coloring ([`HardwareIntentKind::AsidSegmentColor`]):**
//!    - Write-side ([`MountPolicy::PermanentStatic`]) stamps [`MemorySegmentAsid::AuthoritativeSchema`]
//!      onto `<schema>`/`CREATE TABLE` tokens and [`MemorySegmentAsid::EphemeralScratchpad`] onto
//!      `<think>` scratchpad tokens inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$
//!      (`0.000` `W_V` KV-value payload corruption!).
//!    - Read-side (`MountPolicy::DynamicBank(CartridgeBank::ModePayload(Mode::Read))`) queries
//!      `+1.0 * AuthoritativeSchema - 0.75 * EphemeralScratchpad - 0.35 * others`.
//! 2. **Submerged Foreign-Key Pointer Routing ([`HardwareIntentKind::SubmergedPointerRoute`]):**
//!    - Opens polymorphic `copy_heads` (`W_Q_gate` sigmoid) and closes `distractor_heads`
//!      (prompt-noun / recency echo heads) while aligning `query_passthrough_delta` in
//!      `[rotary_dim .. head_dim)` strictly inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$
//!      (`GCAD` zero KV-value contamination, zero `v_ov_role` training-token leak!).

mod dual_span;
mod fk_route;
mod gqa_eval;
mod qwen_loader;

pub use dual_span::{
    build_bank_foreign_key_pointer_route_intent, build_dual_span_plan_and_sql_pointer_program,
    build_dynamic_scratchpad_write_asid_intent, build_orient_trunk_schema_read_asid_intent,
    build_per_mode_think_and_sql_pointer_programs, PerModeThinkAndSqlPrograms,
};
pub use fk_route::{
    build_foreign_key_pointer_route_intent, build_relational_schema_pointer_program,
    synthesize_fk_passthrough_query_delta, validate_fk_edge_in_schema,
};
pub use gqa_eval::{apply_partial_rope_in_place, GqaCandidateForwardResult, Qwen35GqaEvalBlock};
pub use qwen_loader::{
    find_latest_hf_snapshot, load_qwen35_gqa_eval_block, load_selective_bf16_rows,
    rotate_chordal_submerged,
};

use portia_compiler::discovery::gqa_bus::MemorySegmentAsid;
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::V2CompilerError;
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{CartridgeBank, GrammarCartridgeSpec, Mode, MountPolicy};
use serde::{Deserialize, Serialize};

/// Hardware layer & head-routing parameters for SQLite `ASID` + Submerged Pointer compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SqlPointerBridgeConfig {
    pub upstream_mlp_layer: usize,
    pub target_gqa_layer: usize,
    pub copy_heads: Vec<usize>,
    pub distractor_heads: Vec<usize>,
    pub write_asid_angle: ChordalAngleRad,
    pub read_asid_angle: ChordalAngleRad,
    pub pointer_route_angle: ChordalAngleRad,
    pub read_mode: Mode,
}

impl SqlPointerBridgeConfig {
    /// Canonical `Qwen3.5-0.8B-Base` (`d=1024`, `24`L) `L14 -> L15` mid-stack GQA configuration.
    #[must_use]
    pub fn qwen35_08b_l14_to_l15(copy_heads: Vec<usize>, distractor_heads: Vec<usize>) -> Self {
        Self {
            upstream_mlp_layer: 14,
            target_gqa_layer: 15,
            copy_heads,
            distractor_heads,
            write_asid_angle: ChordalAngleRad::new(0.42),
            read_asid_angle: ChordalAngleRad::new(0.45),
            pointer_route_angle: ChordalAngleRad::new(0.48),
            read_mode: Mode::Read,
        }
    }

    /// Canonical `Qwen3.5-0.8B-Base` (`d=1024`, `24`L) `L18 -> L19` (`83%` depth) penultimate GQA
    /// copy-head homolog (`H1, H5, H7, H3` direct token-copy readout; `0% -> 97.5%` unseen hex FK).
    #[must_use]
    pub fn qwen35_08b_l18_to_l19(copy_heads: Vec<usize>, distractor_heads: Vec<usize>) -> Self {
        Self {
            upstream_mlp_layer: 18,
            target_gqa_layer: 19,
            copy_heads,
            distractor_heads,
            write_asid_angle: ChordalAngleRad::new(0.45),
            read_asid_angle: ChordalAngleRad::new(0.45),
            pointer_route_angle: ChordalAngleRad::new(0.48),
            read_mode: Mode::Read,
        }
    }

    /// Canonical `Qwen3.5-4B-Base` (`d=2560`, `32`L) `L22 -> L23` (`75%` depth) GQA configuration.
    #[must_use]
    pub fn qwen35_4b_l22_to_l23(copy_heads: Vec<usize>, distractor_heads: Vec<usize>) -> Self {
        Self {
            upstream_mlp_layer: 22,
            target_gqa_layer: 23,
            copy_heads,
            distractor_heads,
            write_asid_angle: ChordalAngleRad::new(0.42),
            read_asid_angle: ChordalAngleRad::new(0.45),
            pointer_route_angle: ChordalAngleRad::new(0.48),
            read_mode: Mode::Read,
        }
    }

    /// Canonical `Qwen3.5-4B-Base` (`d=2560`, `32`L) `L26 -> L27` (`87.5%` depth) penultimate GQA
    /// copy-head homolog (`H8 = +4.865, H6 = +3.066` direct token-copy readout).
    #[must_use]
    pub fn qwen35_4b_l26_to_l27(copy_heads: Vec<usize>, distractor_heads: Vec<usize>) -> Self {
        Self {
            upstream_mlp_layer: 26,
            target_gqa_layer: 27,
            copy_heads,
            distractor_heads,
            write_asid_angle: ChordalAngleRad::new(0.45),
            read_asid_angle: ChordalAngleRad::new(0.45),
            pointer_route_angle: ChordalAngleRad::new(0.48),
            read_mode: Mode::Read,
        }
    }

    /// Validate causal layer hop (`upstream < downstream`), non-empty `copy_heads`, and disjoint head sets.
    pub fn validate(&self) -> Result<(), V2CompilerError> {
        if self.upstream_mlp_layer >= self.target_gqa_layer {
            return Err(V2CompilerError::NonCausalLayerHop {
                rule: "SqlPointerBridgeConfig".to_string(),
                upstream: self.upstream_mlp_layer,
                downstream: self.target_gqa_layer,
            });
        }
        if self.copy_heads.is_empty() {
            return Err(V2CompilerError::InvalidConfig {
                context: "SqlPointerBridgeConfig::validate",
                reason: "copy_heads must contain >= 1 GQA head".to_string(),
            });
        }
        if self
            .copy_heads
            .iter()
            .any(|h| self.distractor_heads.contains(h))
        {
            return Err(V2CompilerError::InvalidConfig {
                context: "SqlPointerBridgeConfig::validate",
                reason: "copy_heads and distractor_heads must be disjoint".to_string(),
            });
        }
        Ok(())
    }
}

/// Write-side (`MountPolicy::PermanentStatic`) `ASID` key-stamp rule coloring `<schema>` / `CREATE TABLE`
/// tokens with [`MemorySegmentAsid::AuthoritativeSchema`] inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$.
pub fn build_schema_write_asid_intent(
    config: &SqlPointerBridgeConfig,
    rule_name: impl Into<String>,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name.into(),
        guard: RegimeExpr::ConstBool(true),
        mount_policy: MountPolicy::PermanentStatic,
        grammar: GrammarCartridgeSpec::Unconstrained,
        kind: HardwareIntentKind::AsidSegmentColor {
            upstream_mlp_layer: config.upstream_mlp_layer,
            target_gqa_layer: config.target_gqa_layer,
            copy_heads: config.copy_heads.clone(),
            segment: MemorySegmentAsid::AuthoritativeSchema,
            is_write_side_key_stamp: true,
            angle: config.write_asid_angle,
        },
    }))
}

/// Write-side (`MountPolicy::PermanentStatic`) `ASID` key-stamp rule coloring `<think>` scratchpad
/// tokens with [`MemorySegmentAsid::EphemeralScratchpad`] inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$.
pub fn build_scratchpad_write_asid_intent(
    config: &SqlPointerBridgeConfig,
    rule_name: impl Into<String>,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name.into(),
        guard: RegimeExpr::ConstBool(true),
        mount_policy: MountPolicy::PermanentStatic,
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

/// Read-side (`DynamicBank(ModePayload(read_mode))`) `ASID` query selector rule driving
/// `+AuthoritativeSchema - 0.75 * EphemeralScratchpad` across `copy_heads` inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$.
pub fn build_schema_read_asid_intent(
    config: &SqlPointerBridgeConfig,
    rule_name: impl Into<String>,
    guard: RegimeExpr,
    grammar: GrammarCartridgeSpec,
) -> Result<IntentExpr, V2CompilerError> {
    config.validate()?;
    Ok(IntentExpr::rule(IntentRule {
        name: rule_name.into(),
        guard,
        mount_policy: MountPolicy::DynamicBank(CartridgeBank::ModePayload(config.read_mode)),
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

/// Build the complete Paired `ASID` Symbol-Table Coloring program:
/// `(write_authoritative_schema & write_ephemeral_scratchpad) >> read_authoritative_schema`.
pub fn build_paired_asid_coloring_intent(
    config: &SqlPointerBridgeConfig,
    name_prefix: &str,
    read_guard: RegimeExpr,
    read_grammar: GrammarCartridgeSpec,
) -> Result<IntentExpr, V2CompilerError> {
    let w_schema =
        build_schema_write_asid_intent(config, format!("{name_prefix}_asid_write_schema"))?;
    let w_think =
        build_scratchpad_write_asid_intent(config, format!("{name_prefix}_asid_write_scratchpad"))?;
    let r_schema = build_schema_read_asid_intent(
        config,
        format!("{name_prefix}_asid_read_schema"),
        read_guard,
        read_grammar,
    )?;
    Ok((w_schema & w_think) >> r_schema)
}
