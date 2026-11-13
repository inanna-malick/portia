use super::proptest_support::*;
use super::*;
use proptest::prelude::*;
use proptest::strategy::ValueTree;

#[test]
fn empirical_65_25_10_band_distribution_holds_across_all_7_dimensions() {
    for dim in RogueliteDimension::ALL {
        let spec = dim.band_spec();
        let mut sweet = 0usize;
        let mut deep = 0usize;
        let mut boss = 0usize;

        for ticket in 0u8..20 {
            for seed in 0u64..50 {
                let v = spec.sample_from_ticket(ticket, seed * 97);
                assert!(
                    v >= spec.sweet_spot.0 && v <= spec.boss_floor.1,
                    "Out-of-range value {v} on {dim}"
                );
                match spec.classify(v) {
                    BiasedScaleBand::TacticalSweetSpot => sweet += 1,
                    BiasedScaleBand::DeepDungeon => deep += 1,
                    BiasedScaleBand::ExtremeBossFloor => boss += 1,
                    BiasedScaleBand::SubSweetSpotFloor => {
                        panic!("Unexpected SubSweetSpotFloor {v} on {dim}")
                    }
                }
            }
        }

        assert_eq!(sweet, 650, "Sweet-spot weight mismatch on {dim}");
        assert_eq!(deep, 250, "Deep-dungeon weight mismatch on {dim}");
        assert_eq!(boss, 100, "Boss-floor weight mismatch on {dim}");
    }
}

#[test]
fn identifier_swarm_generator_preserves_uniqueness_and_hamming_bounds() {
    for locus in CollisionLocus::ALL {
        let swarm = generate_identifier_swarm("inc-7f3a-01a", 120, locus, 42);
        assert_eq!(swarm.identifiers.len(), 120);
        assert_eq!(
            swarm.identifiers[swarm.target_index],
            "inc-7f3a-01a".to_string()
        );
        assert!(swarm.verify_exact_target_match("inc-7f3a-01a"));
        assert_eq!(swarm.min_hamming_distance, 1);

        let mut unique = swarm.identifiers.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 120);
        for (idx, id) in swarm.identifiers.iter().enumerate() {
            if idx != swarm.target_index {
                assert!(!swarm.verify_exact_target_match(id));
                let dist = char_hamming_distance("inc-7f3a-01a", id);
                assert!(
                    (1..=2).contains(&dist),
                    "Expected Hamming distance 1..=2 for near-colliding distractor {id}, got {dist}"
                );
            }
        }
    }
}

#[test]
fn compound_policy_matrix_truth_table_and_witness_synthesis_are_exact() {
    for arity in [1usize, 2, 5, 10, 16] {
        let matrix = generate_compound_policy_matrix(arity, 0xCAFE_BABE);
        assert_eq!(matrix.clauses.len(), arity);

        let allow_row = matrix
            .synthesize_witness_row(RowPolicyDecision::Allow)
            .expect("Allow witness must exist");
        assert_eq!(matrix.evaluate(&allow_row), RowPolicyDecision::Allow);

        let ask_row = matrix
            .synthesize_witness_row(RowPolicyDecision::Ask)
            .expect("Ask witness must exist");
        assert_eq!(matrix.evaluate(&ask_row), RowPolicyDecision::Ask);

        if arity >= 2 {
            let deny_row = matrix
                .synthesize_witness_row(RowPolicyDecision::Deny)
                .expect("Deny witness must exist for arity >= 2");
            assert_eq!(matrix.evaluate(&deny_row), RowPolicyDecision::Deny);
        }
    }
}

#[test]
fn find_exact_wipeout_cliff_isolates_unit_integer_thresholds_and_contrasts_strategies() {
    let base = RogueliteStressVector::sweet_spot_baseline()
        .with_dimension(RogueliteDimension::ColumnWidthArity, 12);

    let bpe_budget = 4_200usize;

    let tank_receipt = find_exact_wipeout_cliff(
        &base,
        RogueliteDimension::RowHaystackSize,
        CombatStrategyClass::RawTanking,
        |v| v.estimate_bpe_tokens(CombatStrategyClass::RawTanking) <= bpe_budget,
    );
    assert!(tank_receipt.is_exact_unit_cliff());
    let n_pass = tank_receipt.max_passing_value.unwrap();
    let n_fail = tank_receipt.min_failing_value.unwrap();
    assert_eq!(n_pass + 1, n_fail);
    assert!(
        base.with_dimension(RogueliteDimension::RowHaystackSize, n_pass)
            .estimate_bpe_tokens(CombatStrategyClass::RawTanking)
            <= bpe_budget
    );
    assert!(
        base.with_dimension(RogueliteDimension::RowHaystackSize, n_fail)
            .estimate_bpe_tokens(CombatStrategyClass::RawTanking)
            > bpe_budget
    );
    assert_eq!(tank_receipt.cliff_band, Some(BiasedScaleBand::DeepDungeon));

    let finesse_receipt = find_exact_wipeout_cliff(
        &base,
        RogueliteDimension::RowHaystackSize,
        CombatStrategyClass::FinesseSurgical,
        |v| v.estimate_bpe_tokens(CombatStrategyClass::FinesseSurgical) <= bpe_budget,
    );
    assert_eq!(finesse_receipt.max_passing_value, Some(1000));
    assert_eq!(finesse_receipt.min_failing_value, None);
    assert_eq!(finesse_receipt.cliff_band, None);
}

#[test]
fn value_tree_shrinks_boss_floor_vector_to_exact_planted_multi_dimension_threshold() {
    let start = RogueliteStressVector::boss_floor_apex();
    let mut tree = RogueliteValueTree::new(start);

    let fails = |v: RogueliteStressVector| -> bool {
        v.row_haystack_size >= 137
            && v.identifier_swarm_collision >= 42
            && v.policy_predicate_arity >= 9
    };
    assert!(fails(tree.current()));

    while tree.simplify() {
        if !fails(tree.current()) {
            assert!(tree.complicate());
        }
    }

    let shrunk = tree.current();
    assert!(fails(shrunk));
    assert_eq!(shrunk.row_haystack_size, 137);
    assert_eq!(shrunk.identifier_swarm_collision, 42);
    assert_eq!(shrunk.policy_predicate_arity, 9);
    let base = RogueliteStressVector::sweet_spot_baseline();
    assert_eq!(shrunk.column_width_arity, base.column_width_arity);
    assert_eq!(shrunk.schema_graph_topology, base.schema_graph_topology);
    assert_eq!(shrunk.episode_turn_horizon, base.episode_turn_horizon);
    assert_eq!(shrunk.ast_symbol_haystack, base.ast_symbol_haystack);
}

#[test]
fn measure_empirical_bpe_tokens_and_cli_dimension_parser_verify_compression_and_washout() {
    for dim in RogueliteDimension::ALL {
        assert_eq!(
            RogueliteDimension::from_cli_name(dim.short_name()),
            Some(dim)
        );
    }
    assert_eq!(
        RogueliteDimension::from_cli_name("rows"),
        Some(RogueliteDimension::RowHaystackSize)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("swarm"),
        Some(RogueliteDimension::IdentifierSwarmCollision)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("cols"),
        Some(RogueliteDimension::ColumnWidthArity)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("tables"),
        Some(RogueliteDimension::SchemaGraphTopology)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("clauses"),
        Some(RogueliteDimension::PolicyPredicateArity)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("turns"),
        Some(RogueliteDimension::EpisodeTurnHorizon)
    );
    assert_eq!(
        RogueliteDimension::from_cli_name("ast"),
        Some(RogueliteDimension::AstSymbolHaystack)
    );

    let mut vec = RogueliteStressVector::sweet_spot_baseline();
    vec.row_haystack_size = 200;
    vec.column_width_arity = 24;
    vec.needle_depth = NeedleDepthBand::Bot90Pct;
    vec.identifier_swarm_collision = 25;
    vec.collision_locus = CollisionLocus::FinalCharDiff;

    let mut raw_rows = String::from("<tool_response>\n[");
    for i in 0..200 {
        if i > 0 {
            raw_rows.push(',');
        }
        raw_rows.push_str(&format!(
            "{{\"record_id\":{},\"entity_slug\":\"acct-7f3a-{:03}a\",\"balance_credits\":1850,\"policy_decision\":\"ALLOW\",\"lock_state\":\"CLEAR\",\"legacy_balance_deprecated\":999999,\"telemetry_01\":\"OK\"}}",
            1000 + i,
            i
        ));
    }
    raw_rows.push_str("]\n</tool_response>");
    let finesse_row = "<tool_response>\n[{\"record_id\":1180,\"entity_slug\":\"acct-7f3a-07a\",\"balance_credits\":4200,\"policy_decision\":\"ASK\",\"lock_state\":\"LOCKED\"}]\n</tool_response>";

    let empirical = vec.measure_empirical_bpe_tokens(&raw_rows, finesse_row);
    assert_eq!(
        empirical.recommended_strategy,
        CombatStrategyClass::FinesseSurgical
    );
    assert!(empirical.deltanet_washout_risk_under_tanking);
    assert!(empirical.gqa_induction_collision_risk);
    assert!(empirical.raw_tanking_bpe_tokens > empirical.finesse_surgical_bpe_tokens * 50);
    assert!(empirical.context_compression_ratio_x100 >= 5000);
    assert!(empirical
        .tactical_prescription
        .contains("Empirical SQLite payload calibration"));
}

#[test]
fn nursery_isomorphisms_trajectory_types_and_hylomorphic_shrinker_verify_all_variants() {
    let canonical = [
        "WAVE_A".to_string(),
        "WAVE_B".to_string(),
        "WAVE_C".to_string(),
    ];
    for perm in WavePermutation::ALL {
        let inv = perm.inverse();
        let roundtrip = inv.apply(perm.apply(canonical.clone()));
        assert_eq!(roundtrip, canonical);
        assert_eq!(perm.compose(inv), WavePermutation::Identity012);
        assert_eq!(inv.compose(perm), WavePermutation::Identity012);
        assert_eq!(WavePermutation::from_indices(perm.indices()), Some(perm));
        assert_eq!(perm.code().len(), 3);
    }
    assert_eq!(
        WavePermutation::SwapLast021.apply([10, 20, 30]),
        [10, 30, 20]
    );
    assert_eq!(
        WavePermutation::SwapFirst102.apply([10, 20, 30]),
        [20, 10, 30]
    );
    assert_eq!(WavePermutation::Cycle120.apply([10, 20, 30]), [20, 30, 10]);
    assert_eq!(WavePermutation::Cycle201.apply([10, 20, 30]), [30, 10, 20]);
    assert_eq!(
        WavePermutation::Reverse210.apply([10, 20, 30]),
        [30, 20, 10]
    );

    let shifted = PrimaryKeyIsomorphism::ShiftedInteger {
        base_offset: 8420,
        stride: 17,
    };
    let uuid_iso = PrimaryKeyIsomorphism::SwarmUuidSlugPrimaryKey;
    let tenant_iso = PrimaryKeyIsomorphism::CompositeTenantKey {
        tenant_code: "sre-zurich".to_string(),
        base_offset: 5000,
    };
    assert_eq!(shifted.resolve_row_pk(2), 8454);
    assert!(uuid_iso
        .format_surface_key(1, "acct-7f3a-07a")
        .contains("acct-7f3a-07a"));
    assert!(tenant_iso
        .format_surface_key(3, "acct-7f3a-07a")
        .starts_with("sre-zurich:"));

    for iso in PolicyLockSurfaceIso::ALL {
        for p in PolicyState::ALL {
            let enc = iso.encode_policy(p);
            assert_eq!(iso.decode_policy(&enc), Some(p));
            let (b0, b1) = iso.encode_split_policy_bits(p);
            assert!((0..=1).contains(&b0) && (0..=1).contains(&b1));
        }
        for l in LockState::ALL {
            let enc = iso.encode_lock(l);
            assert_eq!(iso.decode_lock(&enc), Some(l));
            let (b0, b1) = iso.encode_split_lock_bits(l);
            assert!((0..=1).contains(&b0) && (0..=1).contains(&b1));
        }
    }

    let trajectories = [
        IntentTrajectoryKind::TransactionalWrite {
            relative_delta_arithmetic: true,
        },
        IntentTrajectoryKind::ReadOnlyDiagnostic,
        IntentTrajectoryKind::MidFlightMindChangeAbort,
        IntentTrajectoryKind::ValueConditionalBranch {
            min_balance_threshold: 2500,
        },
    ];
    assert_eq!(trajectories.len(), 4);

    let agenda = [
        StewardAgendaStep::RequestAdditionalRead {
            required_table: "fk_child_stage_02_locks".to_string(),
            required_column: "dependency_status".to_string(),
        },
        StewardAgendaStep::ConditionalAllowWithOverride {
            token: "TOK-1".to_string(),
            overridden_post_balance: Some(3900),
            overridden_target_pk: Some(1007),
            overridden_status_note: Some("CAPPED".to_string()),
        },
        StewardAgendaStep::RevisedOverride {
            token: "TOK-2".to_string(),
            revised_post_balance: 3750,
        },
        StewardAgendaStep::UnconditionalAllow {
            token: "TOK-3".to_string(),
        },
        StewardAgendaStep::ExplicitDeny {
            token: "TOK-DENY".to_string(),
        },
    ];
    assert_eq!(agenda.len(), 5);

    let start = RogueliteStressVector::boss_floor_apex();
    let receipt = shrink_stress_vector_to_irreducible_cliff(start, |v| {
        v.row_haystack_size >= 142 && v.column_width_arity >= 33 && v.episode_turn_horizon >= 19
    });
    assert_eq!(receipt.irreducible_failing.row_haystack_size, 142);
    assert_eq!(receipt.irreducible_failing.column_width_arity, 33);
    assert_eq!(receipt.irreducible_failing.episode_turn_horizon, 19);
    assert_eq!(
        receipt.implicated_dimensions,
        vec![
            RogueliteDimension::RowHaystackSize,
            RogueliteDimension::ColumnWidthArity,
            RogueliteDimension::EpisodeTurnHorizon,
        ]
    );
    assert_eq!(receipt.contrastive_lower_covers.len(), 3);
    for cover in &receipt.contrastive_lower_covers {
        assert_eq!(cover.passing_value + 1, cover.failing_value);
        assert_eq!(cover.failing_upper_cover, receipt.irreducible_failing);
    }

    let mut non_rect_start = RogueliteStressVector::sweet_spot_baseline();
    non_rect_start.row_haystack_size = 20;
    non_rect_start.column_width_arity = 20;
    non_rect_start.schema_graph_topology = 10;
    let non_rect_receipt = shrink_stress_vector_to_irreducible_cliff(non_rect_start, |v| {
        v.row_haystack_size >= 7
            && v.column_width_arity >= 8
            && if v.schema_graph_topology >= 5 {
                v.row_haystack_size + v.column_width_arity >= 32
            } else {
                v.row_haystack_size + v.column_width_arity >= 15
            }
    });
    assert_eq!(non_rect_receipt.irreducible_failing.row_haystack_size, 7);
    assert_eq!(non_rect_receipt.irreducible_failing.column_width_arity, 8);
    assert_eq!(
        non_rect_receipt.irreducible_failing.schema_graph_topology,
        2
    );
    assert_eq!(non_rect_receipt.contrastive_lower_covers.len(), 2);
    for dim in RogueliteDimension::ALL {
        let val = non_rect_receipt.irreducible_failing.get_dimension(dim);
        let floor_val = RogueliteStressVector::sweet_spot_baseline().get_dimension(dim);
        if val > floor_val {
            assert!(non_rect_receipt
                .contrastive_lower_covers
                .iter()
                .any(|c| c.dimension == dim && c.passing_value == val - 1));
        }
    }
}

proptest! {
    #[test]
    fn prop_shrink_stress_vector_to_irreducible_cliff_isolates_exact_planted_3way_interaction(
        raw_vec in roguelite_stress_vector_strategy(),
        t_rows in 12usize..=500,
        t_cols in 10usize..=70,
        t_clauses in 4usize..=14,
    ) {
        let mut planted = RogueliteStressVector::sweet_spot_baseline();
        planted.row_haystack_size = t_rows;
        planted.column_width_arity = t_cols;
        planted.policy_predicate_arity = t_clauses;

        let start = raw_vec.join(&planted);
        let is_failing = |v: &RogueliteStressVector| -> bool {
            v.row_haystack_size >= t_rows
                && v.column_width_arity >= t_cols
                && v.policy_predicate_arity >= t_clauses
        };
        prop_assert!(is_failing(&start));

        let receipt = shrink_stress_vector_to_irreducible_cliff(start, is_failing);
        prop_assert!(is_failing(&receipt.irreducible_failing));
        prop_assert_eq!(receipt.irreducible_failing.row_haystack_size, t_rows);
        prop_assert_eq!(receipt.irreducible_failing.column_width_arity, t_cols);
        prop_assert_eq!(receipt.irreducible_failing.policy_predicate_arity, t_clauses);
        prop_assert_eq!(receipt.contrastive_lower_covers.len(), 3);
        for cover in &receipt.contrastive_lower_covers {
            prop_assert!(!is_failing(&cover.passing_lower_cover));
            prop_assert!(is_failing(&cover.failing_upper_cover));
            prop_assert_eq!(cover.passing_value + 1, cover.failing_value);
        }
    }

    #[test]
    fn prop_find_exact_wipeout_cliff_pinpoints_any_planted_threshold(
        dim_idx in 0usize..NUM_ROGUELITE_DIMENSIONS,
        planted_frac in 0u64..1000,
    ) {
        let dim = RogueliteDimension::ALL[dim_idx];
        let spec = dim.band_spec();
        let span = (spec.boss_floor.1 - spec.sweet_spot.0) as u64;
        let threshold = spec.sweet_spot.0 + ((planted_frac * span) / 1000) as usize;
        let threshold = threshold.clamp(spec.sweet_spot.0 + 1, spec.boss_floor.1);

        let base = RogueliteStressVector::sweet_spot_baseline();
        let receipt = find_exact_wipeout_cliff(
            &base,
            dim,
            CombatStrategyClass::RawTanking,
            |v| v.get_dimension(dim) < threshold,
        );

        prop_assert!(receipt.is_exact_unit_cliff());
        prop_assert_eq!(receipt.max_passing_value, Some(threshold - 1));
        prop_assert_eq!(receipt.min_failing_value, Some(threshold));
    }

    #[test]
    fn prop_roguelite_value_tree_shrinks_arbitrary_planted_pair_to_exact_coordinate(
        raw_vec in roguelite_stress_vector_strategy(),
        t_rows in 10usize..=600,
        t_swarm in 8usize..=180,
    ) {
        let mut planted = RogueliteStressVector::sweet_spot_baseline();
        planted.row_haystack_size = t_rows;
        planted.identifier_swarm_collision = t_swarm;

        let start = raw_vec.join(&planted);
        let mut tree = RogueliteValueTree::new(start);
        let fails = |v: RogueliteStressVector| -> bool {
            v.row_haystack_size >= t_rows && v.identifier_swarm_collision >= t_swarm
        };
        prop_assert!(fails(tree.current()));

        while tree.simplify() {
            if !fails(tree.current()) {
                prop_assert!(tree.complicate());
            }
        }

        let minimal = tree.current();
        prop_assert!(fails(minimal));
        prop_assert_eq!(minimal.row_haystack_size, t_rows);
        prop_assert_eq!(minimal.identifier_swarm_collision, t_swarm);
    }

    #[test]
    fn prop_multi_axis_ray_wipeout_cliff_isolates_exact_permille_boundary(
        threshold_permille in 1usize..=999,
    ) {
        let start = RogueliteStressVector::sweet_spot_baseline();
        let target = RogueliteStressVector::boss_floor_apex();
        let threshold_rows = interpolate_stress_ray(&start, &target, threshold_permille).row_haystack_size;
        prop_assume!(threshold_rows > start.row_haystack_size);

        let receipt = find_multi_axis_ray_wipeout_cliff(
            &start,
            &target,
            |v| v.row_haystack_size < threshold_rows,
        );
        let p_pass = receipt.max_passing_permille.unwrap();
        let p_fail = receipt.min_failing_permille.unwrap();
        prop_assert_eq!(p_pass + 1, p_fail);
        prop_assert!(receipt.max_passing_vector.unwrap().row_haystack_size < threshold_rows);
        prop_assert!(receipt.min_failing_vector.unwrap().row_haystack_size >= threshold_rows);
    }

    #[test]
    fn prop_dimension_biased_strategy_respects_bounds_and_combat_strategy_compresses_boss_floors(
        rows in dimension_biased_strategy(RogueliteDimension::RowHaystackSize),
        swarm in dimension_biased_strategy(RogueliteDimension::IdentifierSwarmCollision),
        cols in dimension_biased_strategy(RogueliteDimension::ColumnWidthArity),
        tables in dimension_biased_strategy(RogueliteDimension::SchemaGraphTopology),
        clauses in dimension_biased_strategy(RogueliteDimension::PolicyPredicateArity),
        turns in dimension_biased_strategy(RogueliteDimension::EpisodeTurnHorizon),
        ast_syms in dimension_biased_strategy(RogueliteDimension::AstSymbolHaystack),
    ) {
        let vec = RogueliteStressVector::sweet_spot_baseline()
            .with_dimension(RogueliteDimension::RowHaystackSize, rows)
            .with_dimension(RogueliteDimension::IdentifierSwarmCollision, swarm)
            .with_dimension(RogueliteDimension::ColumnWidthArity, cols)
            .with_dimension(RogueliteDimension::SchemaGraphTopology, tables)
            .with_dimension(RogueliteDimension::PolicyPredicateArity, clauses)
            .with_dimension(RogueliteDimension::EpisodeTurnHorizon, turns)
            .with_dimension(RogueliteDimension::AstSymbolHaystack, ast_syms);

        prop_assert!((5..=1000).contains(&vec.row_haystack_size));
        prop_assert!((3..=250).contains(&vec.identifier_swarm_collision));
        prop_assert!((6..=100).contains(&vec.column_width_arity));
        prop_assert!((2..=30).contains(&vec.schema_graph_topology));
        prop_assert!((2..=16).contains(&vec.policy_predicate_arity));
        prop_assert!((3..=50).contains(&vec.episode_turn_horizon));
        prop_assert!((8..=300).contains(&vec.ast_symbol_haystack));

        let boss = RogueliteStressVector::boss_floor_apex();
        let assessment = boss.assess_combat_strategy(4_096);
        prop_assert_eq!(assessment.recommended_strategy, CombatStrategyClass::FinesseSurgical);
        prop_assert!(assessment.deltanet_washout_risk_under_tanking);
        prop_assert!(assessment.gqa_induction_collision_risk);
        prop_assert!(assessment.raw_tanking_bpe_tokens > assessment.finesse_surgical_bpe_tokens * 10);
    }
}
