//! Dimension 4 (`MultiTableFkGraphDungeon`) generator and coalgebraic FK subtree verification.

use portia_core::domain::{DungeonDifficultyBand, LockState, PolicyState};
use portia_nursery::roguelite_lattice::{arb_biased_three_band_range, RogueliteStressVector};
use proptest::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::engine::EngineError;

use super::super::seed_builder::UnifiedSqlScenarioSpec;
use super::ddl_shared::{
    append_shared_fk_chain_and_distractors_ddl, open_seeded_in_memory_connection,
};

/// Foreign-key edge policy in Dimension 4 (`MultiTableFkGraphDungeon`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FkEdgeAction {
    Restrict,
    Cascade,
}

impl FkEdgeAction {
    pub fn as_sql_clause(self) -> &'static str {
        match self {
            Self::Restrict => "ON DELETE RESTRICT",
            Self::Cascade => "ON DELETE CASCADE",
        }
    }
}

/// Specification for Dimension 4 (`MultiTableFkGraphDungeon`: `2..=6` Sweet Spot $\to$ `30` Tables + Depth `1..=8` FK Chains).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiTableFkGraphDungeonSpec {
    pub total_tables: usize,
    pub fk_chain_depth: usize,
    pub leaf_action: FkEdgeAction,
}

/// Instantiated Dimension 4 (`MultiTableFkGraphDungeon`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiTableFkGraphDungeonInstance {
    pub spec: MultiTableFkGraphDungeonSpec,
    pub band: DungeonDifficultyBand,
    /// Ordered FK chain $T_1 \leftarrow T_2 \leftarrow \dots \leftarrow T_d$ (`chain_tables[0]` is root $T_1$).
    pub chain_tables: Vec<String>,
    pub distractor_tables: Vec<String>,
    pub root_blocked_pk: i64,
    pub root_clear_pk: i64,
    pub setup_sql: Vec<String>,
}

impl MultiTableFkGraphDungeonSpec {
    pub fn new(total_tables: usize, fk_chain_depth: usize, leaf_action: FkEdgeAction) -> Self {
        let clamped_tables = total_tables.clamp(2, 30);
        let clamped_depth = fk_chain_depth.clamp(2, clamped_tables.min(8));
        Self {
            total_tables: clamped_tables,
            fk_chain_depth: clamped_depth,
            leaf_action,
        }
    }

    /// Project this multi-table FK graph preset into a [`UnifiedSqlScenarioSpec`].
    pub fn to_unified_scenario_spec(&self) -> UnifiedSqlScenarioSpec {
        let mut sv = RogueliteStressVector::sweet_spot_baseline();
        sv.schema_graph_topology = self.total_tables.clamp(2, 30);
        sv.fk_chain_depth = self
            .fk_chain_depth
            .clamp(2, sv.schema_graph_topology.min(8));
        let mut spec = UnifiedSqlScenarioSpec::from_stress_vector(sv, 0x4000_0004);
        spec.primary_table = "fk_stage_01_ledger".to_string();
        spec.needle_policy_target = PolicyState::Allow;
        spec.needle_lock_target = LockState::Clear;
        spec.needle_initial_balance = 1000;
        spec.fk_leaf_action = self.leaf_action;
        spec
    }

    pub fn build(&self) -> MultiTableFkGraphDungeonInstance {
        let total_tables = self.total_tables.clamp(2, 30);
        let depth = self.fk_chain_depth.clamp(2, total_tables.min(8));
        let distractor_count = total_tables.saturating_sub(depth);

        let root_blocked_pk = 101i64;
        let root_clear_pk = 102i64;
        let root_table = "fk_stage_01_ledger";
        let mut setup_sql = vec![
            format!(
                "CREATE TABLE {} (
                    stage_01_id INTEGER PRIMARY KEY,
                    entity_slug TEXT NOT NULL,
                    policy_decision TEXT NOT NULL,
                    lock_state TEXT NOT NULL
                );",
                root_table
            ),
            format!(
                "INSERT INTO {} VALUES
                    ({}, 'chain_root_blocked', 'ALLOW', 'CLEAR'),
                    ({}, 'chain_root_clear', 'ALLOW', 'CLEAR');",
                root_table, root_blocked_pk, root_clear_pk
            ),
        ];

        let (chain_tables, distractor_tables) = append_shared_fk_chain_and_distractors_ddl(
            &mut setup_sql,
            root_table,
            "stage_01_id",
            root_blocked_pk,
            depth,
            "fk_stage",
            "ledger",
            1000,
            self.leaf_action,
            distractor_count,
        );

        MultiTableFkGraphDungeonInstance {
            spec: self.clone(),
            band: DungeonDifficultyBand::classify_table_count(total_tables),
            chain_tables,
            distractor_tables,
            root_blocked_pk,
            root_clear_pk,
            setup_sql,
        }
    }

    /// Build and open a hardened in-memory `rusqlite::Connection` populated with this multi-table FK graph dungeon.
    pub fn open_in_memory_connection(
        &self,
    ) -> Result<(Connection, MultiTableFkGraphDungeonInstance), EngineError> {
        let instance = self.build();
        let conn = instance.open_in_memory_connection()?;
        Ok((conn, instance))
    }
}

impl MultiTableFkGraphDungeonInstance {
    /// Open a hardened in-memory `rusqlite::Connection` seeded with `self.setup_sql`.
    pub fn open_in_memory_connection(&self) -> Result<Connection, EngineError> {
        open_seeded_in_memory_connection(&self.setup_sql)
    }

    /// Inspect the live SQLite schema via [`crate::relational_ast::RelationalSchemaGraph`] and
    /// verify the multi-hop foreign-key dependency subtree rooted at `chain_tables[0]`.
    pub fn verify_fk_subtree_via_hylo(
        &self,
        conn: &Connection,
    ) -> Result<(portia_core::hylo::RoseTree<String>, usize), EngineError> {
        let graph = crate::relational_ast::RelationalSchemaGraph::inspect_from_connection(conn)?;
        let root_table = self
            .chain_tables
            .first()
            .map(String::as_str)
            .unwrap_or("fk_stage_01_ledger");
        let subtree = graph.fk_reachable_subtree(root_table);
        let depth = graph.fk_subtree_depth(root_table);
        Ok((subtree, depth))
    }
}

/// Biased `proptest` strategy for Dimension 4 table counts (`65%` in `2..=6`, `25%` in `7..=15`, `10%` in `16..=30`).
pub fn arb_multi_table_count() -> impl Strategy<Value = usize> {
    arb_biased_three_band_range(2..=6, 7..=15, 16..=30)
}

/// Biased `proptest` strategy for [`MultiTableFkGraphDungeonSpec`].
pub fn arb_multi_table_fk_graph_spec() -> impl Strategy<Value = MultiTableFkGraphDungeonSpec> {
    (
        arb_multi_table_count(),
        2usize..=8,
        prop_oneof![Just(FkEdgeAction::Restrict), Just(FkEdgeAction::Cascade)],
    )
        .prop_map(|(tables, depth, action)| {
            MultiTableFkGraphDungeonSpec::new(tables, depth, action)
        })
}
