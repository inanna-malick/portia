//! 7-dimensional `RogueliteStressVector`, combat-strategy BPE token estimation,
//! and empirical JSON/SQL subword counters.

use portia_core::domain::{BiasedScaleBand, CollisionLocus, CombatStrategyClass, NeedleDepthBand};
use serde::{Deserialize, Serialize};

use super::dimensions::RogueliteDimension;

/// Diagnosis of how a model's context/attention state behaves at a given [`RogueliteStressVector`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombatStrategyAssessment {
    pub recommended_strategy: CombatStrategyClass,
    pub raw_tanking_bpe_tokens: usize,
    pub finesse_surgical_bpe_tokens: usize,
    pub context_compression_ratio_x100: usize,
    pub deltanet_washout_risk_under_tanking: bool,
    pub gqa_induction_collision_risk: bool,
    pub tactical_prescription: String,
}

/// Complete coordinate in the 7-dimensional Roguelite Stress Lattice plus sub-axis modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RogueliteStressVector {
    pub row_haystack_size: usize,
    pub needle_depth: NeedleDepthBand,
    pub identifier_swarm_collision: usize,
    pub collision_locus: CollisionLocus,
    pub column_width_arity: usize,
    pub deprecated_decoy_columns: usize,
    pub schema_graph_topology: usize,
    pub fk_chain_depth: usize,
    pub policy_predicate_arity: usize,
    pub episode_turn_horizon: usize,
    pub chitchat_carry_distance: usize,
    pub steward_negotiation_rounds: usize,
    pub ast_symbol_haystack: usize,
    pub homonym_overload_count: usize,
}

impl RogueliteStressVector {
    /// Minimal sweet-spot baseline (`rows=5, swarm=3, cols=6, tables=2, clauses=2, turns=3, ast=8`).
    pub const fn sweet_spot_baseline() -> Self {
        Self {
            row_haystack_size: 5,
            needle_depth: NeedleDepthBand::Top10Pct,
            identifier_swarm_collision: 3,
            collision_locus: CollisionLocus::FinalCharDiff,
            column_width_arity: 6,
            deprecated_decoy_columns: 0,
            schema_graph_topology: 2,
            fk_chain_depth: 1,
            policy_predicate_arity: 2,
            episode_turn_horizon: 3,
            chitchat_carry_distance: 0,
            steward_negotiation_rounds: 1,
            ast_symbol_haystack: 8,
            homonym_overload_count: 1,
        }
    }

    /// Absolute minimum floor across all 7 dimensions (`rows=1, swarm=1, cols=4, tables=1, clauses=1, turns=1, ast=3`).
    pub const fn absolute_minimum() -> Self {
        Self {
            row_haystack_size: 1,
            needle_depth: NeedleDepthBand::Top10Pct,
            identifier_swarm_collision: 1,
            collision_locus: CollisionLocus::PrefixDiff,
            column_width_arity: 4,
            deprecated_decoy_columns: 0,
            schema_graph_topology: 1,
            fk_chain_depth: 1,
            policy_predicate_arity: 1,
            episode_turn_horizon: 1,
            chitchat_carry_distance: 0,
            steward_negotiation_rounds: 1,
            ast_symbol_haystack: 3,
            homonym_overload_count: 1,
        }
    }

    /// Boss-floor apex across all 7 dimensions (`rows=1000, swarm=250, cols=100, tables=30, clauses=16, turns=50, ast=300`).
    pub const fn boss_floor_apex() -> Self {
        Self {
            row_haystack_size: 1000,
            needle_depth: NeedleDepthBand::Bot90Pct,
            identifier_swarm_collision: 250,
            collision_locus: CollisionLocus::FinalCharDiff,
            column_width_arity: 100,
            deprecated_decoy_columns: 25,
            schema_graph_topology: 30,
            fk_chain_depth: 8,
            policy_predicate_arity: 16,
            episode_turn_horizon: 50,
            chitchat_carry_distance: 25,
            steward_negotiation_rounds: 8,
            ast_symbol_haystack: 300,
            homonym_overload_count: 12,
        }
    }

    /// Read any of the 7 primary continuous dimensions.
    pub const fn get_dimension(&self, dim: RogueliteDimension) -> usize {
        match dim {
            RogueliteDimension::RowHaystackSize => self.row_haystack_size,
            RogueliteDimension::IdentifierSwarmCollision => self.identifier_swarm_collision,
            RogueliteDimension::ColumnWidthArity => self.column_width_arity,
            RogueliteDimension::SchemaGraphTopology => self.schema_graph_topology,
            RogueliteDimension::PolicyPredicateArity => self.policy_predicate_arity,
            RogueliteDimension::EpisodeTurnHorizon => self.episode_turn_horizon,
            RogueliteDimension::AstSymbolHaystack => self.ast_symbol_haystack,
        }
    }

    /// Return a copy with `dim` updated and clamped to valid bounds.
    pub const fn with_dimension(&self, dim: RogueliteDimension, raw_value: usize) -> Self {
        let clamped = dim.band_spec().clamp(raw_value);
        let mut next = *self;
        match dim {
            RogueliteDimension::RowHaystackSize => next.row_haystack_size = clamped,
            RogueliteDimension::IdentifierSwarmCollision => {
                next.identifier_swarm_collision = clamped;
            }
            RogueliteDimension::ColumnWidthArity => {
                next.column_width_arity = clamped;
                if next.deprecated_decoy_columns > clamped / 4 {
                    next.deprecated_decoy_columns = clamped / 4;
                }
            }
            RogueliteDimension::SchemaGraphTopology => {
                next.schema_graph_topology = clamped;
                if next.fk_chain_depth > clamped {
                    next.fk_chain_depth = if clamped > 8 { 8 } else { clamped };
                }
            }
            RogueliteDimension::PolicyPredicateArity => next.policy_predicate_arity = clamped,
            RogueliteDimension::EpisodeTurnHorizon => {
                next.episode_turn_horizon = clamped;
                if next.chitchat_carry_distance > clamped / 2 {
                    next.chitchat_carry_distance = clamped / 2;
                }
            }
            RogueliteDimension::AstSymbolHaystack => next.ast_symbol_haystack = clamped,
        }
        next
    }

    /// Classify the scale band of a single dimension.
    pub const fn band_for_dimension(&self, dim: RogueliteDimension) -> BiasedScaleBand {
        dim.band_spec().classify(self.get_dimension(dim))
    }

    /// Highest [`BiasedScaleBand`] reached across all 7 dimensions.
    pub fn highest_band(&self) -> BiasedScaleBand {
        RogueliteDimension::ALL
            .into_iter()
            .map(|d| self.band_for_dimension(d))
            .max()
            .unwrap_or(BiasedScaleBand::TacticalSweetSpot)
    }

    /// Count how many dimensions currently sit in each [`BiasedScaleBand`].
    pub fn band_histogram(&self) -> (usize, usize, usize) {
        let mut sweet = 0;
        let mut deep = 0;
        let mut boss = 0;
        for dim in RogueliteDimension::ALL {
            match self.band_for_dimension(dim) {
                BiasedScaleBand::SubSweetSpotFloor | BiasedScaleBand::TacticalSweetSpot => {
                    sweet += 1;
                }
                BiasedScaleBand::DeepDungeon => deep += 1,
                BiasedScaleBand::ExtremeBossFloor => boss += 1,
            }
        }
        (sweet, deep, boss)
    }

    /// Partial order check: `self` dominates `other` iff every primary dimension of `self` is `>= other`.
    pub fn dominates(&self, other: &Self) -> bool {
        RogueliteDimension::ALL
            .into_iter()
            .all(|d| self.get_dimension(d) >= other.get_dimension(d))
    }

    /// Componentwise maximum (`join` in the product lattice).
    pub fn join(&self, other: &Self) -> Self {
        let mut out = *self;
        for dim in RogueliteDimension::ALL {
            let v = self.get_dimension(dim).max(other.get_dimension(dim));
            out = out.with_dimension(dim, v);
        }
        if other.needle_depth > out.needle_depth {
            out.needle_depth = other.needle_depth;
        }
        if other.collision_locus > out.collision_locus {
            out.collision_locus = other.collision_locus;
        }
        out.deprecated_decoy_columns = out
            .deprecated_decoy_columns
            .max(other.deprecated_decoy_columns);
        out.fk_chain_depth = out.fk_chain_depth.max(other.fk_chain_depth);
        out.chitchat_carry_distance = out
            .chitchat_carry_distance
            .max(other.chitchat_carry_distance);
        out.steward_negotiation_rounds = out
            .steward_negotiation_rounds
            .max(other.steward_negotiation_rounds);
        out.homonym_overload_count = out.homonym_overload_count.max(other.homonym_overload_count);
        out
    }

    /// Estimate the BPE token footprint required under [`CombatStrategyClass::RawTanking`] vs [`CombatStrategyClass::FinesseSurgical`].
    pub const fn estimate_bpe_tokens(&self, strategy: CombatStrategyClass) -> usize {
        match strategy {
            CombatStrategyClass::RawTanking => {
                let table_dump_tokens = self.row_haystack_size * self.column_width_arity * 5;
                let swarm_tokens = self.identifier_swarm_collision * 7;
                let schema_tokens = self.schema_graph_topology * self.column_width_arity * 4;
                let policy_tokens = self.policy_predicate_arity * 18;
                let turn_tokens = (self.episode_turn_horizon + self.chitchat_carry_distance) * 95;
                let ast_tokens = self.ast_symbol_haystack * 42;
                table_dump_tokens
                    + swarm_tokens
                    + schema_tokens
                    + policy_tokens
                    + turn_tokens
                    + ast_tokens
            }
            CombatStrategyClass::FinesseSurgical => {
                let surgical_row_tokens = 3 * self.column_width_arity * 5;
                let surgical_swarm_tokens = 24;
                let surgical_schema_tokens =
                    (self.fk_chain_depth + 1) * self.column_width_arity * 3;
                let policy_tokens = self.policy_predicate_arity * 12;
                let turn_tokens =
                    (self.steward_negotiation_rounds + 3) * 55 + self.chitchat_carry_distance * 15;
                let ast_lens_tokens = (self.homonym_overload_count + 1) * 45;
                surgical_row_tokens
                    + surgical_swarm_tokens
                    + surgical_schema_tokens
                    + policy_tokens
                    + turn_tokens
                    + ast_lens_tokens
            }
        }
    }

    /// Evaluate whether `RawTanking` or `FinesseSurgical` is required given a model's effective context budget.
    pub fn assess_combat_strategy(
        &self,
        model_effective_bpe_budget: usize,
    ) -> CombatStrategyAssessment {
        let raw_tokens = self.estimate_bpe_tokens(CombatStrategyClass::RawTanking);
        let finesse_tokens = self.estimate_bpe_tokens(CombatStrategyClass::FinesseSurgical);
        let ratio_x100 = (raw_tokens * 100) / finesse_tokens.max(1);
        let deltanet_washout = raw_tokens > model_effective_bpe_budget
            || (self.row_haystack_size > 60 && self.needle_depth == NeedleDepthBand::Bot90Pct)
            || self.chitchat_carry_distance >= 8;
        let gqa_collision = self.identifier_swarm_collision >= 20
            && self.collision_locus == CollisionLocus::FinalCharDiff;
        let recommended = if deltanet_washout || gqa_collision || raw_tokens > finesse_tokens * 3 {
            CombatStrategyClass::FinesseSurgical
        } else {
            CombatStrategyClass::RawTanking
        };
        let prescription = match recommended {
            CombatStrategyClass::FinesseSurgical => format!(
                "Use FinesseSurgical: push WHERE pk predicate ({} rows -> 1 row), scope PRAGMA to {} FK hops, and isolate AST lens ({}x token reduction).",
                self.row_haystack_size,
                self.fk_chain_depth,
                ratio_x100 / 100
            ),
            CombatStrategyClass::RawTanking => format!(
                "Sweet-spot floor ({} BPE tokens <= {} budget): either RawTanking or FinesseSurgical succeeds.",
                raw_tokens, model_effective_bpe_budget
            ),
        };
        CombatStrategyAssessment {
            recommended_strategy: recommended,
            raw_tanking_bpe_tokens: raw_tokens,
            finesse_surgical_bpe_tokens: finesse_tokens,
            context_compression_ratio_x100: ratio_x100,
            deltanet_washout_risk_under_tanking: deltanet_washout,
            gqa_induction_collision_risk: gqa_collision,
            tactical_prescription: prescription,
        }
    }

    /// Measure empirical Qwen3.5 BPE token counts directly from materialized `<tool_response>` JSON
    /// payloads using a default `4,096`-token context budget.
    pub fn measure_empirical_bpe_tokens(
        &self,
        raw_payload_json: &str,
        finesse_payload_json: &str,
    ) -> CombatStrategyAssessment {
        self.measure_empirical_bpe_tokens_with_budget(raw_payload_json, finesse_payload_json, 4_096)
    }

    /// Measure empirical Qwen3.5 BPE token counts from materialized `<tool_response>` JSON strings
    /// against an explicit `model_effective_bpe_budget`.
    pub fn measure_empirical_bpe_tokens_with_budget(
        &self,
        raw_payload_json: &str,
        finesse_payload_json: &str,
        model_effective_bpe_budget: usize,
    ) -> CombatStrategyAssessment {
        let raw_tokens = count_empirical_json_bpe_tokens(raw_payload_json).max(1);
        let finesse_tokens = count_empirical_json_bpe_tokens(finesse_payload_json).max(1);
        let ratio_x100 = (raw_tokens * 100) / finesse_tokens.max(1);
        let deltanet_washout = raw_tokens > model_effective_bpe_budget
            || (self.row_haystack_size > 60 && self.needle_depth == NeedleDepthBand::Bot90Pct)
            || self.chitchat_carry_distance >= 8;
        let gqa_collision = self.identifier_swarm_collision >= 20
            && self.collision_locus == CollisionLocus::FinalCharDiff;
        let recommended = if deltanet_washout || gqa_collision || raw_tokens > finesse_tokens * 3 {
            CombatStrategyClass::FinesseSurgical
        } else {
            CombatStrategyClass::RawTanking
        };
        let prescription = match recommended {
            CombatStrategyClass::FinesseSurgical => format!(
                "Empirical SQLite payload calibration: RawTanking emitted {} bytes ({} BPE tokens) vs FinesseSurgical {} bytes ({} BPE tokens, {:.2}x compression); use FinesseSurgical to prevent DeltaNet washout.",
                raw_payload_json.len(),
                raw_tokens,
                finesse_payload_json.len(),
                finesse_tokens,
                (ratio_x100 as f64) / 100.0
            ),
            CombatStrategyClass::RawTanking => format!(
                "Empirical SQLite payload within budget: RawTanking emitted {} bytes ({} BPE tokens <= {} budget) vs FinesseSurgical {} bytes ({} BPE tokens).",
                raw_payload_json.len(),
                raw_tokens,
                model_effective_bpe_budget,
                finesse_payload_json.len(),
                finesse_tokens
            ),
        };
        CombatStrategyAssessment {
            recommended_strategy: recommended,
            raw_tanking_bpe_tokens: raw_tokens,
            finesse_surgical_bpe_tokens: finesse_tokens,
            context_compression_ratio_x100: ratio_x100,
            deltanet_washout_risk_under_tanking: deltanet_washout,
            gqa_induction_collision_risk: gqa_collision,
            tactical_prescription: prescription,
        }
    }
}

/// Deterministic Qwen3.5 BPE subword token counter for serialized `<tool_response>` JSON and SQL strings.
pub fn count_empirical_json_bpe_tokens(payload: &str) -> usize {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return 0;
    }

    const SPECIAL_TAGS: [&str; 8] = [
        "<|im_start|>",
        "<|im_end|>",
        "<tool_response>",
        "</tool_response>",
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
    ];

    let mut token_count = 0usize;
    let mut alnum_run_bytes = 0usize;
    let mut rest = trimmed;

    while !rest.is_empty() {
        if rest.starts_with('<') {
            if let Some(tag) = SPECIAL_TAGS.iter().copied().find(|t| rest.starts_with(t)) {
                if alnum_run_bytes > 0 {
                    token_count += alnum_run_bytes.div_ceil(4);
                    alnum_run_bytes = 0;
                }
                token_count += 1;
                rest = &rest[tag.len()..];
                continue;
            }
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        rest = &rest[ch.len_utf8()..];
        if ch.is_ascii_alphanumeric() {
            alnum_run_bytes += 1;
        } else {
            if alnum_run_bytes > 0 {
                token_count += alnum_run_bytes.div_ceil(4);
                alnum_run_bytes = 0;
            }
            if !ch.is_whitespace() {
                token_count += 1;
            }
        }
    }
    if alnum_run_bytes > 0 {
        token_count += alnum_run_bytes.div_ceil(4);
    }

    token_count.max(1)
}
