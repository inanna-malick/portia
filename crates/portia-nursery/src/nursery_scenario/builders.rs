//! Deterministic domain builders and `proptest` strategies for [`ScenarioCoordinate`].

use proptest::prelude::*;
use serde::{Deserialize, Serialize};

use crate::roguelite_lattice::{roguelite_stress_vector_strategy, RogueliteStressVector};

use super::coordinate::{AppType, DomainSkin, ScenarioCoordinate, ScenarioLanguage};
use super::coroner::build_default_steward_agenda;
use super::rust_hylo_spec::{
    render_sealed_witness_and_approved_crate_block, CleanroomRustHyloSpec,
    DialogicCodingAffordance, HyloCartridgeLevel, HyloCartridgePhase, RustHyloPipelineStage,
    RustTodoHoleSpec, SqlStewardCoordinateVocabulary,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializedScenarioDomain {
    SqlSteward(SqlStewardCoordinateVocabulary),
    RustAstHylo(CleanroomRustHyloSpec),
}

impl ScenarioCoordinate {
    pub const fn new(
        app_type: AppType,
        language: ScenarioLanguage,
        domain_skin: DomainSkin,
        stress_vector: RogueliteStressVector,
        seed: u64,
    ) -> Self {
        Self {
            app_type,
            language,
            domain_skin,
            stress_vector,
            seed,
        }
    }

    /// Deterministic builder for the `SqlSteward` domain table/column vocabulary at this coordinate.
    pub fn build_sql_steward_vocabulary(&self) -> SqlStewardCoordinateVocabulary {
        let primary_table = self.domain_skin.primary_sql_table().to_string();
        let canonical_columns = vec![
            "record_id".to_string(),
            "entity_slug".to_string(),
            "balance_credits".to_string(),
            "policy_decision".to_string(),
            "lock_state".to_string(),
            "steward_handle".to_string(),
        ];
        let decoy_count = self.stress_vector.deprecated_decoy_columns.min(25);
        let deceptive_decoy_columns = (0..decoy_count)
            .map(|idx| {
                format!(
                    "legacy_{}_col_{:02}_deprecated",
                    self.domain_skin.slug(),
                    idx + 1
                )
            })
            .collect();
        let fk_depth = self.stress_vector.fk_chain_depth.clamp(1, 8);
        let fk_child_tables = (1..fk_depth)
            .map(|hop| format!("{}_fk_stage_{:02}_locks", self.domain_skin.slug(), hop + 1))
            .collect();
        let steward_agenda =
            build_default_steward_agenda(&self.stress_vector, &primary_table, self.seed);

        SqlStewardCoordinateVocabulary {
            coordinate: *self,
            primary_table,
            canonical_columns,
            deceptive_decoy_columns,
            fk_child_tables,
            localized_preamble: self.language.localized_task_preamble().to_string(),
            steward_agenda,
        }
    }

    /// Deterministic builder for a `RustAstHylo` specification (`AST-valid from Token 0`,
    /// $2 \times 2$ Cartridge Matrix, `N` `todo!("...")` typed holes, and `Vec<DialogicCodingAffordance>`).
    pub fn build_cleanroom_rust_hylo_spec(&self) -> CleanroomRustHyloSpec {
        let module_name = self.domain_skin.primary_rust_module().to_string();
        let trait_name = self.domain_skin.primary_rust_trait().to_string();
        let state_struct_name = format!("{}State", trait_name);
        let hole_count = self.stress_vector.ast_symbol_haystack.clamp(3, 16);

        let mut todo_holes = Vec::with_capacity(hole_count);
        let mut trait_methods = Vec::with_capacity(hole_count);
        let mut impl_methods = Vec::with_capacity(hole_count);
        let mut hole_names = Vec::with_capacity(hole_count);
        let mut test_assertions = Vec::with_capacity(hole_count);

        for idx in 0..hole_count {
            let fn_name = format!("{}_step_{:02}", self.domain_skin.slug(), idx + 1);
            let signature = format!(
                "fn {}(&self, input_metric: u64, guard_clause: u32) -> Result<VerifiedDomainOutput, &'static str>",
                fn_name
            );
            let todo_stub = format!("todo!(\"{}::{}\")", module_name, fn_name);
            let invariant_contract = format!(
                "[{}] Enforce conservation clause #{} over `{}`",
                self.language.iso_code(),
                idx + 1,
                fn_name
            );
            trait_methods.push(format!("    {};", signature));
            impl_methods.push(format!(
                "    {} {{\n        {}\n    }}",
                signature, todo_stub
            ));
            test_assertions.push(format!(
                "        assert_eq!(state.{}(100, 10).map(|v| v.metric()), Ok(90 + {}u64));\n\
                 \x20       assert_eq!(state.{}(5, 20).map(|v| v.metric()), Ok({}u64));",
                fn_name, idx, fn_name, idx
            ));
            hole_names.push(fn_name.clone());
            todo_holes.push(RustTodoHoleSpec {
                hole_index: idx,
                fn_name,
                signature,
                todo_stub_literal: todo_stub,
                invariant_contract,
            });
        }

        let ast_valid_token_zero_skeleton = format!(
            "// AST-valid-from-token-0 skeleton ({} / {})\n\
             {}\n\n\
             pub struct {} {{\n\
             \x20   pub seed: u64,\n\
             \x20   pub predicate_arity: usize,\n\
             }}\n\n\
             pub trait {} {{\n\
             {}\n\
             }}\n\n\
             impl {} for {} {{\n\
             {}\n\
             }}\n\n\
             #[cfg(test)]\n\
             mod tests {{\n\
             \x20   use super::*;\n\n\
             \x20   #[test]\n\
             \x20   fn verify_cleanroom_holes_and_clarified_invariants() {{\n\
             \x20       let state = {} {{ seed: {}, predicate_arity: {} }};\n\
             {}\n\
             \x20   }}\n\
             }}\n",
            self.domain_skin.slug(),
            self.language.iso_code(),
            render_sealed_witness_and_approved_crate_block(),
            state_struct_name,
            trait_name,
            trait_methods.join("\n"),
            trait_name,
            state_struct_name,
            impl_methods.join("\n\n"),
            state_struct_name,
            self.seed,
            hole_count,
            test_assertions.join("\n")
        );

        let (crate_name, crate_purpose) = self.domain_skin.approved_ecosystem_crate();
        let dialogic_affordances = vec![
            DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                ambiguous_aspect: format!(
                    "{} boundary rounding & saturating underflow policy (seed={})",
                    self.domain_skin.slug(),
                    self.seed & 0xFFFF
                ),
                user_clarification: format!(
                    "{} Use saturating floor at 0 and reject negative drift with Err(\"underflow\").",
                    self.language.localized_task_preamble()
                ),
            },
            DialogicCodingAffordance::RequestEcosystemCrate {
                crate_name: crate_name.to_string(),
                purpose: crate_purpose.to_string(),
                approved: true,
            },
            DialogicCodingAffordance::ReviewScaffoldBeforeFanout {
                todo_hole_names: hole_names,
            },
        ];

        let pipeline_schedule = vec![
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::ForkFanout { hole_count },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::OctoMerge {
                merged_hole_count: hole_count,
            },
        ];

        CleanroomRustHyloSpec {
            coordinate: *self,
            module_name,
            trait_name,
            state_struct_name,
            ast_valid_token_zero_skeleton,
            todo_holes,
            pipeline_schedule,
            dialogic_affordances,
            has_contrived_decoy_functions: false,
        }
    }

    /// Materialize the domain specification corresponding to `self.app_type`.
    pub fn materialize_domain(&self) -> MaterializedScenarioDomain {
        match self.app_type {
            AppType::SqlSteward => {
                MaterializedScenarioDomain::SqlSteward(self.build_sql_steward_vocabulary())
            }
            AppType::RustAstHylo => {
                MaterializedScenarioDomain::RustAstHylo(self.build_cleanroom_rust_hylo_spec())
            }
        }
    }
}

/// `proptest` strategy for [`AppType`].
pub fn arb_app_type() -> impl Strategy<Value = AppType> {
    prop_oneof![Just(AppType::SqlSteward), Just(AppType::RustAstHylo)]
}

/// `proptest` strategy for [`ScenarioLanguage`].
pub fn arb_scenario_language() -> impl Strategy<Value = ScenarioLanguage> {
    prop_oneof![
        Just(ScenarioLanguage::English),
        Just(ScenarioLanguage::Japanese),
        Just(ScenarioLanguage::Russian),
        Just(ScenarioLanguage::Spanish),
        Just(ScenarioLanguage::German),
        Just(ScenarioLanguage::Indonesian),
    ]
}

/// `proptest` strategy for [`DomainSkin`].
pub fn arb_domain_skin() -> impl Strategy<Value = DomainSkin> {
    prop_oneof![
        Just(DomainSkin::BgpNoc),
        Just(DomainSkin::AlpineLevain),
        Just(DomainSkin::OrbitalConstellation),
        Just(DomainSkin::SolarisVault),
    ]
}

/// Canonical `proptest` strategy for [`ScenarioCoordinate`] spanning `AppType` ×
/// `(ScenarioLanguage, DomainSkin, RogueliteStressVector)` × `seed`.
pub fn arb_scenario_coordinate() -> impl Strategy<Value = ScenarioCoordinate> {
    (
        arb_app_type(),
        arb_scenario_language(),
        arb_domain_skin(),
        roguelite_stress_vector_strategy(),
        any::<u64>(),
    )
        .prop_map(|(app_type, language, domain_skin, stress_vector, seed)| {
            ScenarioCoordinate::new(app_type, language, domain_skin, stress_vector, seed)
        })
}
