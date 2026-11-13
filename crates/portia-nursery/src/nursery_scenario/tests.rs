use super::*;
use crate::roguelite_lattice::RogueliteStressVector;
use proptest::prelude::*;

#[test]
fn app_type_and_three_tuple_coordinate_build_cleanroom_rust_hylo_and_sql_steward_specs() {
    let stress = RogueliteStressVector::sweet_spot_baseline();
    for app_type in AppType::ALL {
        for language in ScenarioLanguage::ALL {
            for skin in DomainSkin::ALL {
                let coord = ScenarioCoordinate::new(app_type, language, skin, stress, 0xCAFE_0042);
                match coord.materialize_domain() {
                    MaterializedScenarioDomain::SqlSteward(sql_vocab) => {
                        assert_eq!(sql_vocab.coordinate.app_type, AppType::SqlSteward);
                        assert_eq!(sql_vocab.primary_table, skin.primary_sql_table());
                        assert!(!sql_vocab.canonical_columns.is_empty());
                        assert!(!sql_vocab.steward_agenda.is_empty());
                    }
                    MaterializedScenarioDomain::RustAstHylo(rust_spec) => {
                        assert_eq!(rust_spec.coordinate.app_type, AppType::RustAstHylo);
                        assert!(rust_spec.verify_ast_valid_from_token_zero());
                        assert!(!rust_spec.has_contrived_decoy_functions);
                        assert_eq!(rust_spec.pipeline_schedule.len(), 6);
                        assert_eq!(rust_spec.dialogic_affordances.len(), 3);

                        let merged = rust_spec.execute_hylo_fanout_and_octomerge();
                        assert!(!merged.contains("todo!(\""));
                        assert!(merged.contains(
                            "approved_crate::eval_domain_step(input_metric, guard_clause, 0usize)"
                        ));

                        let clean_diags = rust_spec.diagnose_episode(true, true, true);
                        assert!(clean_diags.is_empty());

                        let wipeout_diags = rust_spec.diagnose_episode(false, false, false);
                        assert_eq!(wipeout_diags.len(), 3);
                        assert!(wipeout_diags
                            .iter()
                            .all(|d| d.app_type() == AppType::RustAstHylo));
                    }
                }
            }
        }
    }
}

#[test]
fn sealed_witness_rejects_direct_mint_from_approved_crate_bypass_with_rustc_privacy_error() {
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::OrbitalConstellation,
        RogueliteStressVector::sweet_spot_baseline(),
        0x4200_0005,
    );
    let spec = coord.build_cleanroom_rust_hylo_spec();

    let bypass_patches: Vec<(String, String)> = spec
        .todo_holes
        .iter()
        .map(|hole| {
            (
                hole.fn_name.clone(),
                format!(
                    "Ok(sealed_witness::VerifiedDomainOutput::mint_from_approved_crate(input_metric.saturating_sub(guard_clause as u64) + {}u64))",
                    hole.hole_index
                ),
            )
        })
        .collect();
    let (bypass_cleared, bypass_diag) =
        evaluate_nursery_candidate_with_rustc(&spec, true, &bypass_patches);
    assert!(
        !bypass_cleared,
        "direct call to private `mint_from_approved_crate` must fail rustc privacy check, got: {bypass_diag}"
    );
    assert!(
        bypass_diag.contains("private") || bypass_diag.contains("E0624"),
        "expected rustc privacy error on `mint_from_approved_crate`, got: {bypass_diag}"
    );

    let valid_patches: Vec<(String, String)> = spec
        .todo_holes
        .iter()
        .map(|hole| {
            (
                hole.fn_name.clone(),
                format!(
                    "approved_crate::eval_domain_step(input_metric, guard_clause, {}usize)",
                    hole.hole_index
                ),
            )
        })
        .collect();
    let (valid_cleared, valid_diag) =
        evaluate_nursery_candidate_with_rustc(&spec, true, &valid_patches);
    assert!(
        valid_cleared,
        "calling `approved_crate::eval_domain_step` must compile and pass under rustc --test: {valid_diag}"
    );
}

#[test]
fn domain_skin_and_trap_modifier_provide_topic_switch_m5_and_web_homonym_m6_pairs_across_all_skins()
{
    assert_eq!(
        DomainSkin::BgpNoc.cross_table_topic_switch_pair(),
        (
            "bgp_incidents",
            "oncall_sre_roster",
            "show me the oncall rotations"
        )
    );
    assert_eq!(
        DomainSkin::BgpNoc.domain_noun_web_homonym_pair(),
        (
            "pop sites",
            "pop_facility_sites",
            "show me the pop sites",
            &["google search", "youtube", "facebook", "popular websites"][..]
        )
    );
    for skin in DomainSkin::ALL {
        let (primary_tbl, secondary_tbl, switch_prompt) =
            UnifiedScenarioTrapModifier::cross_table_topic_switch_pair(skin);
        assert!(!primary_tbl.is_empty());
        assert!(!secondary_tbl.is_empty());
        assert_ne!(primary_tbl, secondary_tbl);
        assert!(switch_prompt.starts_with("show me the "));

        let (homonym_phrase, target_tbl, homonym_prompt, markers) =
            UnifiedScenarioTrapModifier::domain_noun_web_homonym_pair(skin);
        assert!(!homonym_phrase.is_empty());
        assert!(!target_tbl.is_empty());
        assert!(homonym_prompt.contains(homonym_phrase));
        assert!(!markers.is_empty());
        assert!(skin.contains_web_homonym_hallucination(markers[0]));
        assert!(!skin.contains_web_homonym_hallucination(target_tbl));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_arb_scenario_coordinate_materializes_valid_sql_and_cleanroom_rust_hylo(
        coord in arb_scenario_coordinate()
    ) {
        let rust_spec = coord.build_cleanroom_rust_hylo_spec();
        prop_assert!(rust_spec.verify_ast_valid_from_token_zero());
        prop_assert!(!rust_spec.has_contrived_decoy_functions);
        let expected_holes = coord.stress_vector.ast_symbol_haystack.clamp(3, 16);
        prop_assert_eq!(rust_spec.todo_holes.len(), expected_holes);

        let merged = rust_spec.execute_hylo_fanout_and_octomerge();
        prop_assert!(!merged.contains("todo!(\""));

        let sql_vocab = coord.build_sql_steward_vocabulary();
        prop_assert_eq!(
            sql_vocab.primary_table.as_str(),
            coord.domain_skin.primary_sql_table()
        );
        prop_assert_eq!(
            sql_vocab.steward_agenda.len(),
            coord.stress_vector.steward_negotiation_rounds.clamp(1, 8)
        );
    }
}
