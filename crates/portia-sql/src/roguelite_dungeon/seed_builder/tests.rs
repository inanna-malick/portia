use portia_core::domain::{BiasedScaleBand, CollisionLocus, NeedleDepth};
use portia_nursery::roguelite_lattice::{
    PolicyLockSurfaceIso, PrimaryKeyIsomorphism, RogueliteStressVector,
};
use rusqlite::Connection;

use super::*;

#[test]
fn unified_sql_scenario_compiles_stress_vector_and_seeds_memory_and_file_sqlite() {
    let mut vec = RogueliteStressVector::sweet_spot_baseline();
    vec.row_haystack_size = 120;
    vec.needle_depth = NeedleDepth::Mid50Pct;
    vec.identifier_swarm_collision = 18;
    vec.collision_locus = CollisionLocus::FinalCharDiff;
    vec.column_width_arity = 28;
    vec.deprecated_decoy_columns = 6;
    vec.schema_graph_topology = 10;
    vec.fk_chain_depth = 4;
    vec.policy_predicate_arity = 5;

    let spec = UnifiedSqlScenarioSpec::from_stress_vector(vec, 0xCAFE_BABE);
    let inst = spec.build();
    let conn = inst.open_in_memory_connection().unwrap();

    assert_eq!(inst.highest_band, BiasedScaleBand::DeepDungeon);
    assert_eq!(inst.all_primary_columns.len(), 28);
    assert_eq!(inst.policy_guard_columns.len(), 5);
    assert_eq!(inst.deceptive_decoy_columns.len(), 6);
    assert_eq!(inst.identifier_swarm.identifiers.len(), 18);
    assert_eq!(inst.fk_chain_tables.len(), 4);
    assert_eq!(inst.distractor_tables.len(), 6);

    let total_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM roguelite_primary_ledger;", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(total_rows, 120);

    let found_pk: i64 = conn
        .query_row(
            &format!(
                "SELECT record_id FROM roguelite_primary_ledger WHERE entity_slug = '{}';",
                inst.needle_entity_slug
            ),
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(found_pk, inst.needle_primary_key);

    assert!(conn
        .execute(
            &format!(
                "DELETE FROM roguelite_primary_ledger WHERE record_id = {};",
                inst.needle_primary_key
            ),
            [],
        )
        .is_err());

    let tmp_path = std::env::temp_dir().join(format!(
        "portia_w5_seed_builder_test_{}.sqlite",
        std::process::id()
    ));
    inst.seed_sqlite_file(&tmp_path).unwrap();
    let file_conn = Connection::open(&tmp_path).unwrap();
    let file_rows: i64 = file_conn
        .query_row("SELECT COUNT(*) FROM roguelite_primary_ledger;", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(file_rows, 120);
    drop(file_conn);
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn surface_isomorphisms_and_column_permutations_execute_cleanly_in_sqlite() {
    let base_stress = RogueliteStressVector {
        row_haystack_size: 12,
        needle_depth: NeedleDepth::Mid50Pct,
        identifier_swarm_collision: 5,
        collision_locus: CollisionLocus::MiddleHexDiff,
        column_width_arity: 16,
        deprecated_decoy_columns: 2,
        schema_graph_topology: 3,
        fk_chain_depth: 3,
        policy_predicate_arity: 3,
        episode_turn_horizon: 6,
        chitchat_carry_distance: 2,
        steward_negotiation_rounds: 3,
        ast_symbol_haystack: 12,
        homonym_overload_count: 3,
    };

    for (idx, policy_iso) in [
        PolicyLockSurfaceIso::StandardLiterals,
        PolicyLockSurfaceIso::NumericTierBoolean,
        PolicyLockSurfaceIso::DomainClearanceHold,
        PolicyLockSurfaceIso::SplitTwoColumnBoolean,
    ]
    .into_iter()
    .enumerate()
    {
        let pk_iso = match idx % 3 {
            0 => PrimaryKeyIsomorphism::ShiftedInteger {
                base_offset: 5000,
                stride: 10,
            },
            1 => PrimaryKeyIsomorphism::SwarmUuidSlugPrimaryKey,
            _ => PrimaryKeyIsomorphism::CompositeTenantKey {
                tenant_code: "T-09".to_string(),
                base_offset: 9000,
            },
        };
        let is_composite_pk = matches!(pk_iso, PrimaryKeyIsomorphism::CompositeTenantKey { .. });
        let is_uuid_pk = matches!(pk_iso, PrimaryKeyIsomorphism::SwarmUuidSlugPrimaryKey);
        let iso_spec = UnifiedSqlScenarioSpec::from_stress_vector(base_stress, 100 + idx as u64)
            .with_primary_key_iso(pk_iso)
            .with_policy_lock_iso(policy_iso)
            .with_permute_core_columns_sk(idx % 2 == 1);
        let iso_inst = iso_spec.build();
        let mut expected_cols = 16;
        if policy_iso == PolicyLockSurfaceIso::SplitTwoColumnBoolean {
            expected_cols += 2;
        }
        if is_composite_pk {
            expected_cols += 1;
        }
        assert_eq!(iso_inst.all_primary_columns.len(), expected_cols);
        assert_eq!(iso_inst.homonym_lookalike_columns.len(), 3);
        if is_uuid_pk {
            assert!(iso_inst.setup_sql[0].contains("entity_slug TEXT PRIMARY KEY"));
        }
        if is_composite_pk {
            assert!(iso_inst.setup_sql[0].contains("PRIMARY KEY (tenant_code, record_id)"));
        }

        let conn = iso_inst.open_in_memory_connection().unwrap();
        let mut stmt = conn
            .prepare(&iso_inst.expected_select_needle_sql())
            .unwrap();
        let count = stmt.query_map([], |_| Ok(())).unwrap().count();
        assert_eq!(count, 1);
    }
}
