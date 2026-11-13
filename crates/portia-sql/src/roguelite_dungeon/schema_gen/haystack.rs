//! Dimension 1 (`RowHaystackDungeon`) generator and Fog-of-War pagination comparison.

use portia_core::domain::{DungeonDifficultyBand, LockState, NeedleDepth, PolicyState};
use portia_nursery::roguelite_lattice::{
    arb_biased_three_band_range, arb_needle_depth, RogueliteStressVector,
};
use proptest::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::engine::{EngineError, FogOfWarPageMetadata, SqlRow};

use super::super::seed_builder::UnifiedSqlScenarioSpec;
use super::ddl_shared::{
    build_fog_page_metadata, build_shared_column_layout, build_shared_create_table_ddl,
    open_seeded_in_memory_connection, query_rows_with_authorizer,
};

/// Specification for Dimension 1 (`RowHaystackDungeon`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowHaystackDungeonSpec {
    pub table_name: String,
    pub total_rows: usize,
    pub needle_depth: NeedleDepth,
    pub near_miss_decoy_count: usize,
    pub target_entity_prefix: String,
    pub target_needle_suffix: usize,
    pub needle_policy: String,
    pub needle_lock: String,
    pub needle_balance: i64,
}

/// Instantiated Dimension 1 (`RowHaystackDungeon`) with exact needle coordinates and decoy telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowHaystackDungeonInstance {
    pub spec: RowHaystackDungeonSpec,
    pub band: DungeonDifficultyBand,
    pub needle_row_index: usize,
    pub needle_primary_key: i64,
    pub needle_entity_slug: String,
    pub near_miss_entity_slugs: Vec<String>,
    pub allow_clear_decoy_count: usize,
    pub setup_sql: Vec<String>,
}

impl RowHaystackDungeonInstance {
    /// Open a hardened in-memory `rusqlite::Connection` seeded with `self.setup_sql`.
    pub fn open_in_memory_connection(&self) -> Result<Connection, EngineError> {
        open_seeded_in_memory_connection(&self.setup_sql)
    }
}

impl RowHaystackDungeonSpec {
    pub fn new(total_rows: usize, needle_depth: NeedleDepth, near_miss_decoy_count: usize) -> Self {
        let clamped_rows = total_rows.clamp(3, 1000);
        let max_near_miss = clamped_rows.saturating_sub(1).clamp(1, 15);
        Self {
            table_name: "haystack_accounts".to_string(),
            total_rows: clamped_rows,
            needle_depth,
            near_miss_decoy_count: near_miss_decoy_count.clamp(1, max_near_miss),
            target_entity_prefix: "acct_aurora".to_string(),
            target_needle_suffix: 7,
            needle_policy: "ASK".to_string(),
            needle_lock: "LOCKED".to_string(),
            needle_balance: 4200,
        }
    }

    /// Project this 1D haystack preset into a full [`UnifiedSqlScenarioSpec`].
    pub fn to_unified_scenario_spec(&self) -> UnifiedSqlScenarioSpec {
        let mut sv = RogueliteStressVector::sweet_spot_baseline();
        sv.row_haystack_size = self.total_rows.clamp(3, 1000);
        sv.needle_depth = self.needle_depth;
        sv.identifier_swarm_collision = (self.near_miss_decoy_count + 1).clamp(2, 250);
        sv.column_width_arity = 6;
        sv.deprecated_decoy_columns = 0;
        sv.schema_graph_topology = 1;
        sv.fk_chain_depth = 1;
        sv.policy_predicate_arity = 1;
        let mut spec = UnifiedSqlScenarioSpec::from_stress_vector(sv, 0x1000_0001);
        spec.primary_table = self.table_name.clone();
        spec.needle_policy_target =
            PolicyState::parse_flexible(&self.needle_policy).unwrap_or(PolicyState::Ask);
        spec.needle_lock_target =
            LockState::parse_flexible(&self.needle_lock).unwrap_or(LockState::Locked);
        spec.needle_initial_balance = self.needle_balance;
        spec
    }

    /// Compile the full SQLite DDL + batched `INSERT` statements for this haystack dungeon
    /// using the shared DDL builder.
    pub fn build(&self) -> RowHaystackDungeonInstance {
        let total_rows = self.total_rows.clamp(3, 1000);
        let needle_idx = self.needle_depth.needle_row_index(total_rows);
        let needle_pk = 1000 + needle_idx as i64;
        let needle_slug = format!(
            "{}_{:02}",
            self.target_entity_prefix, self.target_needle_suffix
        );

        let max_near = total_rows.saturating_sub(1).min(15);
        let target_near_count = self.near_miss_decoy_count.clamp(1, max_near.max(1));

        let (_, _, _, all_cols) = build_shared_column_layout("account_id", 6, 0, &[]);
        let mut setup_sql = vec![build_shared_create_table_ddl(
            &self.table_name,
            "account_id",
            &all_cols,
        )];

        let mut near_miss_slugs = Vec::new();
        let mut allow_clear_decoy_count = 0usize;
        let non_needle_total = total_rows.saturating_sub(1);
        let max_non_allow_decoys = non_needle_total / 20;
        let mut non_allow_decoys_emitted = 0usize;
        let mut decoy_idx = 0usize;

        for chunk_start in (0..total_rows).step_by(250) {
            let chunk_end = (chunk_start + 250).min(total_rows);
            let mut values_clauses = Vec::with_capacity(chunk_end - chunk_start);

            for row_i in chunk_start..chunk_end {
                let pk = 1000 + row_i as i64;
                if row_i == needle_idx {
                    values_clauses.push(format!(
                        "({}, '{}', {}, '{}', '{}', 'steward_ops')",
                        pk, needle_slug, self.needle_balance, self.needle_policy, self.needle_lock
                    ));
                } else {
                    let is_near_miss = near_miss_slugs.len()
                        < target_near_miss_for_row(
                            row_i,
                            needle_idx,
                            total_rows,
                            target_near_count,
                        );
                    let slug = if is_near_miss {
                        let suffix_num = near_miss_slugs.len() + 1;
                        let candidate_num = if suffix_num >= self.target_needle_suffix {
                            suffix_num + 1
                        } else {
                            suffix_num
                        };
                        let nm_slug = format!("{}_{:02}", self.target_entity_prefix, candidate_num);
                        near_miss_slugs.push(nm_slug.clone());
                        nm_slug
                    } else {
                        format!("decoy_node_{:04}", row_i)
                    };

                    let (pol, lck) = if non_allow_decoys_emitted < max_non_allow_decoys
                        && decoy_idx % 23 == 19
                    {
                        non_allow_decoys_emitted += 1;
                        ("DENY", "LOCKED")
                    } else {
                        allow_clear_decoy_count += 1;
                        ("ALLOW", "CLEAR")
                    };
                    let decoy_bal = 100 + ((row_i as i64 * 37) % 900);
                    values_clauses.push(format!(
                        "({}, '{}', {}, '{}', '{}', 'steward_auto')",
                        pk, slug, decoy_bal, pol, lck
                    ));
                    decoy_idx += 1;
                }
            }

            setup_sql.push(format!(
                "INSERT INTO {} VALUES {};",
                self.table_name,
                values_clauses.join(", ")
            ));
        }

        RowHaystackDungeonInstance {
            spec: self.clone(),
            band: DungeonDifficultyBand::classify_row_count(total_rows),
            needle_row_index: needle_idx,
            needle_primary_key: needle_pk,
            needle_entity_slug: needle_slug,
            near_miss_entity_slugs: near_miss_slugs,
            allow_clear_decoy_count,
            setup_sql,
        }
    }

    /// Build and open a hardened in-memory `rusqlite::Connection` populated with this haystack dungeon.
    pub fn open_in_memory_connection(
        &self,
    ) -> Result<(Connection, RowHaystackDungeonInstance), EngineError> {
        let instance = self.build();
        let conn = instance.open_in_memory_connection()?;
        Ok((conn, instance))
    }
}

fn target_near_miss_for_row(
    row_i: usize,
    needle_idx: usize,
    total_rows: usize,
    target_near_count: usize,
) -> usize {
    if row_i + target_near_count >= total_rows {
        return target_near_count;
    }
    let dist = row_i.abs_diff(needle_idx);
    if dist <= target_near_count || row_i < target_near_count {
        target_near_count
    } else {
        0
    }
}

/// Comparison between Mode A (Raw Haystack Tanking) and Mode B (Fog-of-War + `WHERE` Narrowing / `LIMIT` Pagination).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HaystackStrategyComparison {
    pub total_table_rows: usize,
    pub needle_row_index: usize,
    pub raw_tanking_returned_rows: usize,
    pub raw_tanking_serialized_bytes: usize,
    pub fog_unfiltered_returned_rows: usize,
    pub fog_unfiltered_metadata: FogOfWarPageMetadata,
    pub where_narrowed_returned_rows: usize,
    pub where_narrowed_serialized_bytes: usize,
    pub where_narrowed_metadata: FogOfWarPageMetadata,
    pub paginated_offset_returned_rows: usize,
    pub needle_recovered_by_where: bool,
    pub needle_recovered_by_pagination: bool,
}

/// Evaluate Mode A (Raw Tanking) vs. Mode B (Paginated Fog-of-War & `WHERE` Narrowing) directly on a
/// hardened `rusqlite::Connection` seeded from [`RowHaystackDungeonInstance`].
pub fn evaluate_raw_tanking_vs_fog_finesse(
    dungeon: &RowHaystackDungeonInstance,
    fog_page_size: usize,
) -> Result<HaystackStrategyComparison, EngineError> {
    let page_sz = fog_page_size.max(1);
    let conn = dungeon.open_in_memory_connection()?;

    let select_all_sql = format!(
        "SELECT * FROM {} ORDER BY account_id ASC;",
        dungeon.spec.table_name
    );
    let (raw_rows, _) = query_rows_with_authorizer(&conn, &select_all_sql)?;
    let raw_bytes = serde_json::to_string(&raw_rows).unwrap_or_default().len();
    let raw_returned_rows = raw_rows.len();

    let mut fog_rows = raw_rows;
    let fog_meta = build_fog_page_metadata(&mut fog_rows, page_sz);
    let fog_unfiltered_returned_rows = fog_rows.len();

    let where_sql = format!(
        "SELECT * FROM {} WHERE entity_slug = '{}';",
        dungeon.spec.table_name, dungeon.needle_entity_slug
    );
    let (mut where_rows, _) = query_rows_with_authorizer(&conn, &where_sql)?;
    let where_meta = build_fog_page_metadata(&mut where_rows, page_sz);
    let where_bytes = serde_json::to_string(&where_rows).unwrap_or_default().len();
    let needle_recovered_by_where = where_rows.len() == 1
        && where_rows[0].get("account_id").and_then(|v| v.as_i64())
            == Some(dungeon.needle_primary_key);

    let page_offset = (dungeon.needle_row_index / page_sz) * page_sz;
    let page_sql = format!(
        "SELECT * FROM {} ORDER BY account_id ASC LIMIT {} OFFSET {};",
        dungeon.spec.table_name, page_sz, page_offset
    );
    let (mut page_rows, _) = query_rows_with_authorizer(&conn, &page_sql)?;
    let _ = build_fog_page_metadata(&mut page_rows, page_sz);
    let needle_recovered_by_pagination = page_rows.iter().any(|r: &SqlRow| {
        r.get("account_id").and_then(|v| v.as_i64()) == Some(dungeon.needle_primary_key)
    });

    Ok(HaystackStrategyComparison {
        total_table_rows: dungeon.spec.total_rows,
        needle_row_index: dungeon.needle_row_index,
        raw_tanking_returned_rows: raw_returned_rows,
        raw_tanking_serialized_bytes: raw_bytes,
        fog_unfiltered_returned_rows,
        fog_unfiltered_metadata: fog_meta,
        where_narrowed_returned_rows: where_rows.len(),
        where_narrowed_serialized_bytes: where_bytes,
        where_narrowed_metadata: where_meta,
        paginated_offset_returned_rows: page_rows.len(),
        needle_recovered_by_where,
        needle_recovered_by_pagination,
    })
}

/// Biased `proptest` strategy for Dimension 1 row counts (`65%` in `5..=50`, `25%` in `51..=250`, `10%` in `251..=1000`).
pub fn arb_row_haystack_count() -> impl Strategy<Value = usize> {
    arb_biased_three_band_range(5..=50, 51..=250, 251..=1000)
}

/// Biased `proptest` strategy for [`RowHaystackDungeonSpec`].
pub fn arb_row_haystack_dungeon_spec() -> impl Strategy<Value = RowHaystackDungeonSpec> {
    (arb_row_haystack_count(), arb_needle_depth(), 3usize..=15).prop_map(
        |(total_rows, depth, near_miss)| RowHaystackDungeonSpec::new(total_rows, depth, near_miss),
    )
}
