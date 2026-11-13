//! Specification, builder fluent API, and instance execution helpers for unified SQL scenarios.

use std::path::Path;

use portia_core::domain::{BiasedScaleBand, LockState, PolicyState};
use portia_nursery::nursery_scenario::{
    build_default_steward_agenda, DomainSkin, UnifiedScenarioTrapModifier,
};
use portia_nursery::roguelite_lattice::{
    roguelite_stress_vector_strategy, CompoundPolicyMatrix, IdentifierSwarm, IntentTrajectoryKind,
    PolicyLockSurfaceIso, PrimaryKeyIsomorphism, RogueliteStressVector, StewardAgendaStep,
    WavePermutation,
};
use proptest::prelude::*;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

use crate::engine::{configure_hardened_sqlite, EngineError};

use super::super::postcondition_oracle::sql_tokens_reference_identifier;
use super::super::schema_gen::{open_seeded_in_memory_connection, FkEdgeAction};

/// Unified tensor-product SQL scenario specification that compiles a [`RogueliteStressVector`]
/// into **one cohesive SQLite database** simultaneously combining:
/// 1. `row_haystack_size` (`1..=1000`) rows with `needle_depth` (`Top10Pct`, `Mid50Pct`, `Bot90Pct`)
///    and $\ge 95\%$ Majority-Vote `"ALLOW"` / `"CLEAR"` decoys,
/// 2. `identifier_swarm_collision` (`1..=250`) Hamming-1 / Hamming-2 near-colliding slugs
///    at `collision_locus` (`PrefixDiff`, `MiddleHexDiff`, `FinalCharDiff`),
/// 3. `column_width_arity` (`6..=100`) columns interleaving `deprecated_decoy_columns` (`0..=25`),
///    split policy guard columns (`policy_guard_c0..c{K-1}`), and telemetry noise columns,
/// 4. `policy_predicate_arity` (`1..=16`) coupled guard clauses,
/// 5. `schema_graph_topology` (`1..=30`) tables in `sqlite_master` with a `fk_chain_depth` (`1..=8`)
///    foreign-key chain rooted at the primary haystack table (`roguelite_primary_ledger`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedSqlScenarioSpec {
    pub primary_table: String,
    pub domain_skin: DomainSkin,
    pub stress_vector: RogueliteStressVector,
    pub seed: u64,
    pub needle_policy_target: PolicyState,
    pub needle_lock_target: LockState,
    pub needle_initial_balance: i64,
    pub fk_leaf_action: FkEdgeAction,
    pub fog_page_limit: usize,
    pub primary_key_iso: PrimaryKeyIsomorphism,
    pub policy_lock_iso: PolicyLockSurfaceIso,
    pub permute_core_columns_sk: bool,
    pub has_upfront_schema: bool,
    pub conversational_preamble_turns: usize,
    pub intent_trajectory: IntentTrajectoryKind,
    pub steward_agenda: Vec<StewardAgendaStep>,
    pub trap_modifiers: Vec<UnifiedScenarioTrapModifier>,
    pub wave_permutation: WavePermutation,
}

/// Materialized Unified SQL Scenario (`UnifiedSqlScenarioInstance`) with exact oracle coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedSqlScenarioInstance {
    pub spec: UnifiedSqlScenarioSpec,
    pub highest_band: BiasedScaleBand,
    pub needle_row_index: usize,
    pub needle_primary_key: i64,
    pub needle_surface_key: String,
    pub needle_entity_slug: String,
    pub archived_honeypot_pk: Option<i64>,
    pub duplicate_candidate_pk: Option<i64>,
    pub topic_switch_secondary_table: Option<String>,
    pub web_homonym_target_table: Option<String>,
    pub identifier_swarm: IdentifierSwarm,
    pub policy_matrix: CompoundPolicyMatrix,
    pub canonical_columns: Vec<String>,
    pub policy_guard_columns: Vec<String>,
    pub deceptive_decoy_columns: Vec<String>,
    pub homonym_lookalike_columns: Vec<String>,
    pub noise_telemetry_columns: Vec<String>,
    pub all_primary_columns: Vec<String>,
    pub fk_chain_tables: Vec<String>,
    pub distractor_tables: Vec<String>,
    pub allow_clear_decoy_count: usize,
    pub setup_sql: Vec<String>,
}

impl UnifiedSqlScenarioSpec {
    /// Construct a [`UnifiedSqlScenarioSpec`] directly from any [`RogueliteStressVector`],
    /// seeding `steward_agenda` from `steward_negotiation_rounds`, `conversational_preamble_turns`
    /// from `chitchat_carry_distance`, and lookalike columns from `homonym_overload_count`.
    pub fn from_stress_vector(stress_vector: RogueliteStressVector, seed: u64) -> Self {
        let primary_table = "roguelite_primary_ledger".to_string();
        let steward_agenda = build_default_steward_agenda(&stress_vector, &primary_table, seed);
        let wave_permutation = WavePermutation::ALL[(seed as usize) % WavePermutation::ALL.len()];
        Self {
            primary_table,
            domain_skin: DomainSkin::BgpNoc,
            stress_vector,
            seed,
            needle_policy_target: PolicyState::Ask,
            needle_lock_target: LockState::Locked,
            needle_initial_balance: 4200,
            fk_leaf_action: FkEdgeAction::Restrict,
            fog_page_limit: 20,
            primary_key_iso: PrimaryKeyIsomorphism::default(),
            policy_lock_iso: PolicyLockSurfaceIso::StandardLiterals,
            permute_core_columns_sk: false,
            has_upfront_schema: true,
            conversational_preamble_turns: stress_vector.chitchat_carry_distance,
            intent_trajectory: IntentTrajectoryKind::TransactionalWrite {
                relative_delta_arithmetic: false,
            },
            steward_agenda,
            trap_modifiers: Vec::new(),
            wave_permutation,
        }
    }

    pub fn with_domain_skin(mut self, domain_skin: DomainSkin) -> Self {
        self.domain_skin = domain_skin;
        self
    }

    pub fn with_primary_key_iso(mut self, iso: PrimaryKeyIsomorphism) -> Self {
        self.primary_key_iso = iso;
        self
    }

    pub fn with_policy_lock_iso(mut self, iso: PolicyLockSurfaceIso) -> Self {
        self.policy_lock_iso = iso;
        self
    }

    pub fn with_permute_core_columns_sk(mut self, permute: bool) -> Self {
        self.permute_core_columns_sk = permute;
        self
    }

    pub fn with_upfront_schema(mut self, has_upfront_schema: bool) -> Self {
        self.has_upfront_schema = has_upfront_schema;
        self
    }

    pub fn with_conversational_preamble_turns(mut self, turns: usize) -> Self {
        self.conversational_preamble_turns = turns;
        self
    }

    pub fn with_intent_trajectory(mut self, trajectory: IntentTrajectoryKind) -> Self {
        self.intent_trajectory = trajectory;
        self
    }

    pub fn with_steward_agenda(mut self, agenda: Vec<StewardAgendaStep>) -> Self {
        self.steward_agenda = agenda;
        self
    }

    pub fn with_trap_modifiers(mut self, modifiers: Vec<UnifiedScenarioTrapModifier>) -> Self {
        self.trap_modifiers = modifiers;
        self
    }

    pub fn with_trap_modifier(mut self, modifier: UnifiedScenarioTrapModifier) -> Self {
        if !self.trap_modifiers.contains(&modifier) {
            self.trap_modifiers.push(modifier);
        }
        if modifier == UnifiedScenarioTrapModifier::BlindSchemaProbeFirstH2 {
            self.has_upfront_schema = false;
        }
        self
    }

    pub fn with_wave_permutation(mut self, perm: WavePermutation) -> Self {
        self.wave_permutation = perm;
        self
    }
}

impl UnifiedSqlScenarioInstance {
    /// Open a fresh hardened in-memory `rusqlite::Connection` seeded with `self.setup_sql`.
    pub fn open_in_memory_connection(&self) -> Result<Connection, EngineError> {
        open_seeded_in_memory_connection(&self.setup_sql)
    }

    /// Seed an on-disk SQLite file at `path` by executing `self.setup_sql` under hardened settings.
    pub fn seed_sqlite_file(&self, path: &Path) -> Result<(), EngineError> {
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;
        let conn = Connection::open_with_flags(path, flags)?;
        configure_hardened_sqlite(&conn)?;
        for stmt in &self.setup_sql {
            conn.execute_batch(stmt)?;
        }
        Ok(())
    }

    /// Return the canonical `WHERE` predicate required to uniquely identify the active needle row
    /// under the scenario's [`PrimaryKeyIsomorphism`] and [`UnifiedScenarioTrapModifier`] set.
    pub fn expected_target_where_clause(&self) -> String {
        let base_where = match &self.spec.primary_key_iso {
            PrimaryKeyIsomorphism::ShiftedInteger { .. } => {
                format!("record_id = {}", self.needle_primary_key)
            }
            PrimaryKeyIsomorphism::SwarmUuidSlugPrimaryKey => {
                if self.archived_honeypot_pk.is_some() || self.duplicate_candidate_pk.is_some() {
                    format!(
                        "entity_slug = '{}' AND record_id = {}",
                        self.needle_entity_slug, self.needle_primary_key
                    )
                } else {
                    format!("entity_slug = '{}'", self.needle_entity_slug)
                }
            }
            PrimaryKeyIsomorphism::CompositeTenantKey { tenant_code, .. } => {
                format!(
                    "tenant_code = '{}' AND record_id = {}",
                    tenant_code, self.needle_primary_key
                )
            }
        };
        if self.archived_honeypot_pk.is_some()
            && !sql_tokens_reference_identifier(&base_where, "lifecycle_status")
        {
            format!("{base_where} AND lifecycle_status = 'ACTIVE'")
        } else {
            base_where
        }
    }

    /// Return the canonical `SELECT` statement for inspecting the needle row under the active surface isomorphisms.
    pub fn expected_select_needle_sql(&self) -> String {
        let mut cols = vec![
            "record_id",
            "entity_slug",
            "balance_credits",
            "policy_decision",
            "lock_state",
        ];
        if matches!(
            self.spec.primary_key_iso,
            PrimaryKeyIsomorphism::CompositeTenantKey { .. }
        ) {
            cols.insert(0, "tenant_code");
        }
        if self.spec.policy_lock_iso == PolicyLockSurfaceIso::SplitTwoColumnBoolean {
            cols.push("policy_tier_bit");
            cols.push("lock_hold_bit");
        }
        if self.archived_honeypot_pk.is_some() {
            cols.push("lifecycle_status");
        }
        format!(
            "SELECT {} FROM {} WHERE {};",
            cols.join(", "),
            self.spec.primary_table,
            self.expected_target_where_clause()
        )
    }

    /// Return the canonical `UPDATE` statement for setting the needle row's `balance_credits`
    /// under the active [`PrimaryKeyIsomorphism`].
    pub fn expected_update_needle_sql(&self, new_balance: i64) -> String {
        format!(
            "UPDATE {} SET balance_credits = {} WHERE {};",
            self.spec.primary_table,
            new_balance,
            self.expected_target_where_clause()
        )
    }
}

/// Biased `proptest` strategy for [`UnifiedSqlScenarioSpec`] driven by [`roguelite_stress_vector_strategy`].
pub fn arb_unified_sql_scenario_spec() -> impl Strategy<Value = UnifiedSqlScenarioSpec> {
    (roguelite_stress_vector_strategy(), any::<u64>())
        .prop_map(|(vec, seed)| UnifiedSqlScenarioSpec::from_stress_vector(vec, seed))
}
