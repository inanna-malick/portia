//! Dimension 3 (`WideTableDungeon`) generator and column-projection authorizer comparison.

use portia_core::domain::{DungeonDifficultyBand, LockState, PolicyState};
use portia_nursery::roguelite_lattice::{arb_biased_three_band_range, RogueliteStressVector};
use proptest::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::engine::EngineError;

use super::super::seed_builder::UnifiedSqlScenarioSpec;
use super::ddl_shared::{
    build_shared_column_layout, build_shared_create_table_ddl, open_seeded_in_memory_connection,
    query_rows_with_authorizer,
};

/// Specification for Dimension 3 (`WideTableDungeon`: `6..=20` Sweet Spot $\to$ `100` Columns).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WideTableDungeonSpec {
    pub table_name: String,
    pub total_columns: usize,
    pub deceptive_legacy_columns: usize,
    pub row_count: usize,
}

/// Instantiated Dimension 3 (`WideTableDungeon`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WideTableDungeonInstance {
    pub spec: WideTableDungeonSpec,
    pub band: DungeonDifficultyBand,
    pub canonical_columns: Vec<String>,
    pub deceptive_decoy_columns: Vec<String>,
    pub noise_telemetry_columns: Vec<String>,
    pub all_columns_in_order: Vec<String>,
    pub target_entity_slug: String,
    pub setup_sql: Vec<String>,
}

impl WideTableDungeonInstance {
    /// Open a hardened in-memory `rusqlite::Connection` seeded with `self.setup_sql`.
    pub fn open_in_memory_connection(&self) -> Result<Connection, EngineError> {
        open_seeded_in_memory_connection(&self.setup_sql)
    }
}

/// Comparison between Raw `SELECT *` Wide-Row Bloat and Column-Projection Finesse (`SELECT c1, c2, c3`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WideTableFinesseComparison {
    pub total_table_columns: usize,
    pub select_star_columns_read: usize,
    pub projected_columns_read: usize,
    pub authorizer_columns_saved: usize,
    pub select_star_serialized_bytes: usize,
    pub projected_serialized_bytes: usize,
    pub bpe_token_compression_factor_x100: usize,
    pub touched_deceptive_columns_in_projection: usize,
}

impl WideTableDungeonSpec {
    pub fn new(total_columns: usize, deceptive_legacy_columns: usize, row_count: usize) -> Self {
        let clamped_cols = total_columns.clamp(6, 100);
        let max_deceptive = clamped_cols.saturating_sub(6).min(25);
        Self {
            table_name: "wide_telemetry_vault".to_string(),
            total_columns: clamped_cols,
            deceptive_legacy_columns: deceptive_legacy_columns.min(max_deceptive),
            row_count: row_count.clamp(3, 250),
        }
    }

    /// Project this wide-table preset into a [`UnifiedSqlScenarioSpec`].
    pub fn to_unified_scenario_spec(&self) -> UnifiedSqlScenarioSpec {
        let mut sv = RogueliteStressVector::sweet_spot_baseline();
        sv.row_haystack_size = self.row_count.clamp(3, 250);
        sv.column_width_arity = self.total_columns.clamp(6, 100);
        sv.deprecated_decoy_columns = self.deceptive_legacy_columns.min(25);
        sv.schema_graph_topology = 1;
        sv.fk_chain_depth = 1;
        sv.policy_predicate_arity = 1;
        let mut spec = UnifiedSqlScenarioSpec::from_stress_vector(sv, 0x3000_0003);
        spec.primary_table = self.table_name.clone();
        spec.needle_policy_target = PolicyState::Allow;
        spec.needle_lock_target = LockState::Clear;
        spec.needle_initial_balance = 1850;
        spec
    }

    pub fn build(&self) -> WideTableDungeonInstance {
        let total_cols = self.total_columns.clamp(6, 100);
        let (
            canonical_columns,
            deceptive_decoy_columns,
            noise_telemetry_columns,
            all_columns_in_order,
        ) = build_shared_column_layout("record_id", total_cols, self.deceptive_legacy_columns, &[]);

        let mut setup_sql = vec![build_shared_create_table_ddl(
            &self.table_name,
            "record_id",
            &all_columns_in_order,
        )];

        let target_entity_slug = "vault_sector_04".to_string();
        let rows_n = self.row_count.clamp(3, 250);
        let mut row_tuples = Vec::with_capacity(rows_n);
        for r_idx in 0..rows_n {
            let pk = 500 + r_idx as i64;
            let slug = if r_idx == rows_n / 2 {
                target_entity_slug.clone()
            } else {
                format!("vault_decoy_{:03}", r_idx)
            };
            let mut vals = Vec::with_capacity(all_columns_in_order.len());
            for c in &all_columns_in_order {
                if c == "record_id" {
                    vals.push(pk.to_string());
                } else if c == "entity_slug" {
                    vals.push(format!("'{}'", slug));
                } else if c == "balance_credits" {
                    vals.push("1850".to_string());
                } else if c == "policy_decision" {
                    vals.push("'ALLOW'".to_string());
                } else if c == "lock_state" {
                    vals.push("'CLEAR'".to_string());
                } else if c == "steward_handle" {
                    vals.push("'steward_core'".to_string());
                } else if c.contains("balance") || c.contains("quota") || c.contains("timer") {
                    vals.push("999999".to_string());
                } else {
                    vals.push("'DEPRECATED_NOISE_PAYLOAD_XXXX'".to_string());
                }
            }
            row_tuples.push(format!("({})", vals.join(", ")));
        }

        setup_sql.push(format!(
            "INSERT INTO {} VALUES {};",
            self.table_name,
            row_tuples.join(", ")
        ));

        WideTableDungeonInstance {
            spec: self.clone(),
            band: DungeonDifficultyBand::classify_column_count(total_cols),
            canonical_columns,
            deceptive_decoy_columns,
            noise_telemetry_columns,
            all_columns_in_order,
            target_entity_slug,
            setup_sql,
        }
    }

    /// Compare Raw `SELECT *` Wide-Row Bloat against Column-Projection Finesse directly on `rusqlite::Connection`.
    pub fn compare_select_star_vs_projection_finesse(
        &self,
    ) -> Result<(WideTableDungeonInstance, WideTableFinesseComparison), EngineError> {
        let instance = self.build();
        let conn = instance.open_in_memory_connection()?;

        let star_sql = format!("SELECT * FROM {};", instance.spec.table_name);
        let (star_rows, star_cols) = query_rows_with_authorizer(&conn, &star_sql)?;
        let star_bytes = serde_json::to_string(&star_rows).unwrap_or_default().len();
        let star_cols_read = star_cols.len();

        let proj_sql = format!(
            "SELECT record_id, entity_slug, balance_credits, policy_decision, lock_state FROM {} WHERE entity_slug = '{}';",
            instance.spec.table_name, instance.target_entity_slug
        );
        let (proj_rows, proj_cols) = query_rows_with_authorizer(&conn, &proj_sql)?;
        let proj_bytes = serde_json::to_string(&proj_rows).unwrap_or_default().len();
        let proj_cols_read = proj_cols.len();

        let touched_deceptive = proj_cols
            .iter()
            .filter(|cr| {
                instance
                    .deceptive_decoy_columns
                    .iter()
                    .any(|dc| dc.eq_ignore_ascii_case(&cr.column))
            })
            .count();

        let compression_x100 = (star_bytes * 100) / proj_bytes.max(1);

        Ok((
            instance,
            WideTableFinesseComparison {
                total_table_columns: self.total_columns.clamp(6, 100),
                select_star_columns_read: star_cols_read,
                projected_columns_read: proj_cols_read,
                authorizer_columns_saved: star_cols_read.saturating_sub(proj_cols_read),
                select_star_serialized_bytes: star_bytes,
                projected_serialized_bytes: proj_bytes,
                bpe_token_compression_factor_x100: compression_x100,
                touched_deceptive_columns_in_projection: touched_deceptive,
            },
        ))
    }
}

/// Biased `proptest` strategy for Dimension 3 column counts (`65%` in `6..=20`, `25%` in `21..=50`, `10%` in `51..=100`).
pub fn arb_wide_column_count() -> impl Strategy<Value = usize> {
    arb_biased_three_band_range(6..=20, 21..=50, 51..=100)
}

/// Biased `proptest` strategy for [`WideTableDungeonSpec`].
pub fn arb_wide_table_dungeon_spec() -> impl Strategy<Value = WideTableDungeonSpec> {
    (arb_wide_column_count(), 1usize..=25, 5usize..=40)
        .prop_map(|(cols, decoys, rows)| WideTableDungeonSpec::new(cols, decoys, rows))
}
