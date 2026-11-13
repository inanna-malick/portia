use portia_core::domain::{DungeonDifficultyBand, NeedleDepth};
use proptest::prelude::*;

use super::*;
use crate::engine::list_sqlite_user_tables;

#[test]
fn row_haystack_1000_rows_places_needle_across_depths_and_enforces_majority_vote_trap() {
    for &cliff_n in &HAYSTACK_CLIFF_POINTS {
        for depth in [
            NeedleDepth::Top10Pct,
            NeedleDepth::Mid50Pct,
            NeedleDepth::Bot90Pct,
        ] {
            let spec = RowHaystackDungeonSpec::new(cliff_n, depth, 8);
            let (conn, instance) = spec.open_in_memory_connection().unwrap();

            let cnt: i64 = conn
                .query_row("SELECT COUNT(*) FROM haystack_accounts;", [], |r| r.get(0))
                .unwrap();
            assert_eq!(cnt, cliff_n as i64);

            let non_needle = cliff_n - 1;
            assert!(
                instance.allow_clear_decoy_count * 100 >= non_needle * 95,
                "expected >=95% ALLOW/CLEAR decoys, got {}/{}",
                instance.allow_clear_decoy_count,
                non_needle
            );
        }
    }

    let boss_spec = RowHaystackDungeonSpec::new(1000, NeedleDepth::Mid50Pct, 12);
    let boss_instance = boss_spec.build();
    assert_eq!(boss_instance.band, DungeonDifficultyBand::BossFloor10Pct);
    assert_eq!(boss_instance.needle_row_index, 500);
    assert_eq!(boss_instance.near_miss_entity_slugs.len(), 12);

    let cmp = evaluate_raw_tanking_vs_fog_finesse(&boss_instance, 20).unwrap();
    assert_eq!(cmp.raw_tanking_returned_rows, 1000);
    assert_eq!(cmp.fog_unfiltered_returned_rows, 20);
    assert_eq!(cmp.fog_unfiltered_metadata.total_matching_rows, 1000);
    assert_eq!(cmp.fog_unfiltered_metadata.returned_rows, 20);
    assert!(cmp.fog_unfiltered_metadata.has_more);
    assert_eq!(cmp.fog_unfiltered_metadata.next_offset, Some(20));
    assert!(cmp
        .fog_unfiltered_metadata
        .hint
        .as_deref()
        .unwrap()
        .contains("Use WHERE predicate or LIMIT 20 OFFSET 20"));

    assert_eq!(cmp.where_narrowed_returned_rows, 1);
    assert!(!cmp.where_narrowed_metadata.has_more);
    assert!(cmp.needle_recovered_by_where);
    assert!(cmp.needle_recovered_by_pagination);
    assert!(cmp.raw_tanking_serialized_bytes > cmp.where_narrowed_serialized_bytes * 500);
}

#[test]
fn wide_table_100_columns_with_25_deceptive_legacy_columns_rewards_projection_finesse() {
    let spec = WideTableDungeonSpec::new(100, 25, 40);
    let (instance, cmp) = spec.compare_select_star_vs_projection_finesse().unwrap();

    assert_eq!(instance.band, DungeonDifficultyBand::BossFloor10Pct);
    assert_eq!(instance.all_columns_in_order.len(), 100);
    assert_eq!(instance.deceptive_decoy_columns.len(), 25);
    assert_eq!(cmp.total_table_columns, 100);
    assert_eq!(cmp.select_star_columns_read, 100);
    assert_eq!(cmp.projected_columns_read, 5);
    assert_eq!(cmp.authorizer_columns_saved, 95);
    assert_eq!(cmp.touched_deceptive_columns_in_projection, 0);
    assert!(cmp.select_star_serialized_bytes > cmp.projected_serialized_bytes * 20);
}

#[test]
fn multi_table_30_tables_and_depth_8_fk_chain_enforces_foreign_key_constraints() {
    let spec = MultiTableFkGraphDungeonSpec::new(30, 8, FkEdgeAction::Restrict);
    let (conn, instance) = spec.open_in_memory_connection().unwrap();

    assert_eq!(instance.band, DungeonDifficultyBand::BossFloor10Pct);
    assert_eq!(instance.chain_tables.len(), 8);
    assert_eq!(instance.distractor_tables.len(), 22);
    assert_eq!(list_sqlite_user_tables(&conn).unwrap().len(), 30);

    let (_subtree, fk_depth) = instance.verify_fk_subtree_via_hylo(&conn).unwrap();
    assert_eq!(fk_depth, 8);

    let blocked_del = conn.execute(
        "DELETE FROM fk_stage_01_ledger WHERE stage_01_id = 101;",
        [],
    );
    assert!(blocked_del.is_err());

    let clear_del = conn
        .execute(
            "DELETE FROM fk_stage_01_ledger WHERE stage_01_id = 102;",
            [],
        )
        .unwrap();
    assert_eq!(clear_del, 1);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn proptest_biased_preset_dungeons_instantiate_in_sqlite(
        haystack_spec in arb_row_haystack_dungeon_spec(),
        wide_spec in arb_wide_table_dungeon_spec(),
        fk_spec in arb_multi_table_fk_graph_spec(),
    ) {
        let h_inst = haystack_spec.build();
        let h_cmp = evaluate_raw_tanking_vs_fog_finesse(&h_inst, 15).unwrap();
        prop_assert!(h_cmp.needle_recovered_by_where);
        prop_assert!(h_cmp.needle_recovered_by_pagination);

        let (w_inst, w_cmp) = wide_spec.compare_select_star_vs_projection_finesse().unwrap();
        prop_assert_eq!(w_inst.all_columns_in_order.len(), wide_spec.total_columns);
        prop_assert_eq!(w_cmp.touched_deceptive_columns_in_projection, 0);
        prop_assert!(w_cmp.select_star_columns_read >= w_cmp.projected_columns_read);

        let (fk_conn, fk_inst) = fk_spec.open_in_memory_connection().unwrap();
        let tables = list_sqlite_user_tables(&fk_conn).unwrap();
        prop_assert_eq!(tables.len(), fk_spec.total_tables);
        prop_assert_eq!(
            fk_inst.chain_tables.len() + fk_inst.distractor_tables.len(),
            fk_spec.total_tables
        );
    }
}
