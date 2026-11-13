//! `RustAstHylo` 2×2 cartridge matrix stages, dialogic affordances, sealed-witness harness,
//! and `rustc --test` candidate evaluator.

use serde::{Deserialize, Serialize};

use crate::roguelite_lattice::StewardAgendaStep;

use super::coordinate::ScenarioCoordinate;
use super::coroner::ScenarioWipeoutDiagnosis;

/// Phase axis of the $2 \times 2$ `RustAstHylo` Cartridge Matrix (`Scaffold ⊥ Impl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HyloCartridgePhase {
    /// Emit compiling trait/struct skeleton with `N` `todo!("...")` typed holes.
    Scaffold,
    /// Fill a single forked `todo!("...")` leaf body in isolation before `OctoMerge`.
    Impl,
}

/// Level axis of the $2 \times 2$ `RustAstHylo` Cartridge Matrix (`Type ⊥ Value`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HyloCartridgeLevel {
    /// Type-level signatures, associated types, and parametric bounds.
    TypeCart,
    /// Value-level operational expressions, control flow, and `todo!("...")` hole bodies.
    ValueCart,
}

/// Complete stage in the `RustAstHylo` hylomorphic pipeline:
/// `Scaffold::TypeCart` -> `Scaffold::ValueCart` -> `Fork × N`
/// -> `Impl::TypeCart` -> `Impl::ValueCart` -> `OctoMerge`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RustHyloPipelineStage {
    CartridgeCell {
        phase: HyloCartridgePhase,
        level: HyloCartridgeLevel,
    },
    ForkFanout {
        hole_count: usize,
    },
    OctoMerge {
        merged_hole_count: usize,
    },
}

impl RustHyloPipelineStage {
    /// The canonical $2 \times 2$ cartridge matrix cells in execution order.
    pub const MATRIX_2X2: [(HyloCartridgePhase, HyloCartridgeLevel); 4] = [
        (HyloCartridgePhase::Scaffold, HyloCartridgeLevel::TypeCart),
        (HyloCartridgePhase::Scaffold, HyloCartridgeLevel::ValueCart),
        (HyloCartridgePhase::Impl, HyloCartridgeLevel::TypeCart),
        (HyloCartridgePhase::Impl, HyloCartridgeLevel::ValueCart),
    ];
}

/// Dialogic `Mode::Ask` affordances answered by the simulated steward during nursery episodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DialogicCodingAffordance {
    /// Agent asks for clarification on an underspecified invariant before mutating.
    ClarifyUnderspecifiedInvariant {
        ambiguous_aspect: String,
        user_clarification: String,
    },
    /// Agent requests approval for an ecosystem crate instead of writing an ad-hoc implementation.
    RequestEcosystemCrate {
        crate_name: String,
        purpose: String,
        approved: bool,
    },
    /// Agent presents the trait + `N` `todo!("...")` stubs for review before `Fork × N` fanout.
    ReviewScaffoldBeforeFanout { todo_hole_names: Vec<String> },
}

/// Specification for a single `todo!("...")` typed hole inside a [`CleanroomRustHyloSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RustTodoHoleSpec {
    pub hole_index: usize,
    pub fn_name: String,
    pub signature: String,
    pub todo_stub_literal: String,
    pub invariant_contract: String,
}

/// Deterministic `SqlSteward` domain table and column vocabulary projected from a [`ScenarioCoordinate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlStewardCoordinateVocabulary {
    pub coordinate: ScenarioCoordinate,
    pub primary_table: String,
    pub canonical_columns: Vec<String>,
    pub deceptive_decoy_columns: Vec<String>,
    pub fk_child_tables: Vec<String>,
    pub localized_preamble: String,
    pub steward_agenda: Vec<StewardAgendaStep>,
}

/// `RustAstHylo` crate specification (`AST-valid from Token 0`, zero contrived decoy functions)
/// projected from a [`ScenarioCoordinate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanroomRustHyloSpec {
    pub coordinate: ScenarioCoordinate,
    pub module_name: String,
    pub trait_name: String,
    pub state_struct_name: String,
    pub ast_valid_token_zero_skeleton: String,
    pub todo_holes: Vec<RustTodoHoleSpec>,
    pub pipeline_schedule: Vec<RustHyloPipelineStage>,
    pub dialogic_affordances: Vec<DialogicCodingAffordance>,
    pub has_contrived_decoy_functions: bool,
}

impl CleanroomRustHyloSpec {
    /// Verify `AST-valid from Token 0` via `tree-sitter-rust` CST and `syn` AST parsing,
    /// confirming `self.trait_name` and `self.state_struct_name` exist in the parsed `syn::File`,
    /// all `self.todo_holes` match `extract_rust_todo_holes`, and zero `legacy_*_deprecated` items exist.
    pub fn verify_ast_valid_from_token_zero(&self) -> bool {
        if self.has_contrived_decoy_functions {
            return false;
        }
        let Ok((_, syn_file)) = crate::rust_ast_lens::parse_and_validate_rust_cst_and_ast(
            &self.ast_valid_token_zero_skeleton,
        ) else {
            return false;
        };
        let mut found_trait = false;
        let mut found_state_struct = false;
        for item in &syn_file.items {
            match item {
                syn::Item::Trait(item_trait) => {
                    let ident = item_trait.ident.to_string();
                    if ident.contains("legacy_") || ident.contains("_deprecated") {
                        return false;
                    }
                    if ident == self.trait_name {
                        found_trait = true;
                    }
                }
                syn::Item::Struct(item_struct) => {
                    let ident = item_struct.ident.to_string();
                    if ident.contains("legacy_") || ident.contains("_deprecated") {
                        return false;
                    }
                    if ident == self.state_struct_name {
                        found_state_struct = true;
                    }
                }
                syn::Item::Fn(item_fn) => {
                    let ident = item_fn.sig.ident.to_string();
                    if ident.contains("legacy_") || ident.contains("_deprecated") {
                        return false;
                    }
                }
                _ => {}
            }
        }
        if !found_trait || !found_state_struct {
            return false;
        }
        let Ok(extracted_holes) =
            crate::rust_ast_lens::extract_rust_todo_holes(&self.ast_valid_token_zero_skeleton)
        else {
            return false;
        };
        if extracted_holes.len() != self.todo_holes.len() {
            return false;
        }
        self.todo_holes
            .iter()
            .zip(extracted_holes.iter())
            .all(|(spec_hole, ext_hole)| {
                spec_hole.fn_name == ext_hole.fn_name
                    && spec_hole
                        .todo_stub_literal
                        .contains(&ext_hole.intent_message)
            })
    }

    /// Render the compact `[AST ...]` header for this specification.
    pub fn format_ast_outline(&self) -> Result<String, portia_core::CoreProtocolError> {
        crate::rust_ast_lens::format_rust_ast_outline(
            &self.module_name,
            &self.ast_valid_token_zero_skeleton,
        )
    }

    /// Instantiate an interactive [`crate::rust_ast_lens::UnifiedRustAstScenarioSession`].
    pub fn into_session(
        self,
    ) -> Result<crate::rust_ast_lens::UnifiedRustAstScenarioSession, portia_core::CoreProtocolError>
    {
        crate::rust_ast_lens::UnifiedRustAstScenarioSession::new(self)
    }

    /// Execute the `Scaffold -> Fork × N -> OctoMerge` hylomorphism over reference hole patches.
    pub fn execute_hylo_fanout_and_octomerge(&self) -> String {
        let patches: Vec<(String, String)> = self
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
        crate::rust_ast_lens::hylo_octo_merge_rust_holes(
            &self.ast_valid_token_zero_skeleton,
            &patches,
        )
        .expect("CleanroomRustHyloSpec::execute_hylo_fanout_and_octomerge must produce a valid Rust AST")
    }

    /// Attribute coroner [`ScenarioWipeoutDiagnosis`] failures on an episode summary.
    pub fn diagnose_episode(
        &self,
        asked_clarification: bool,
        requested_ecosystem_crate: bool,
        scaffold_ast_valid: bool,
    ) -> Vec<ScenarioWipeoutDiagnosis> {
        let mut diags = Vec::new();
        if !scaffold_ast_valid {
            diags.push(
                ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                    detail: format!(
                        "module `{}` failed AST-valid-from-token-0 check at Scaffold::ValueCart",
                        self.module_name
                    ),
                },
            );
        }
        for affordance in &self.dialogic_affordances {
            match affordance {
                DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                    ambiguous_aspect,
                    ..
                } if !asked_clarification => {
                    diags.push(
                        ScenarioWipeoutDiagnosis::GuessedOnAmbiguousSpecWithoutAsking {
                            ambiguous_aspect: ambiguous_aspect.clone(),
                        },
                    );
                }
                DialogicCodingAffordance::RequestEcosystemCrate {
                    crate_name,
                    approved: true,
                    ..
                } if !requested_ecosystem_crate => {
                    diags.push(
                        ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate {
                            available_crate: crate_name.clone(),
                        },
                    );
                }
                _ => {}
            }
        }
        diags
    }
}

/// Render the sealed `VerifiedDomainOutput` witness and child `#[cfg(approved_crate)] pub mod approved_crate`
/// block for offline nursery `rustc --test` episodes.
///
/// Keeping `const fn mint_from_approved_crate` private inside `mod sealed_witness` ensures that
/// only `sealed_witness::approved_crate::eval_domain_step` can construct `VerifiedDomainOutput`,
/// while direct calls from outer functions fail `rustc` privacy checking (`E0624`).
pub const fn render_sealed_witness_and_approved_crate_block() -> &'static str {
    "pub use sealed_witness::VerifiedDomainOutput;\n\
     #[cfg(approved_crate)]\n\
     pub use sealed_witness::approved_crate;\n\n\
     mod sealed_witness {\n\
     \x20   #[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
     \x20   pub struct VerifiedDomainOutput {\n\
     \x20       metric: u64,\n\
     \x20       _sealed_crate_witness: (),\n\
     \x20   }\n\n\
     \x20   impl VerifiedDomainOutput {\n\
     \x20       pub const fn metric(self) -> u64 {\n\
     \x20           self.metric\n\
     \x20       }\n\n\
     \x20       #[cfg(approved_crate)]\n\
     \x20       const fn mint_from_approved_crate(metric: u64) -> Self {\n\
     \x20           Self {\n\
     \x20               metric,\n\
     \x20               _sealed_crate_witness: (),\n\
     \x20           }\n\
     \x20       }\n\
     \x20   }\n\n\
     \x20   #[cfg(approved_crate)]\n\
     \x20   pub mod approved_crate {\n\
     \x20       use super::VerifiedDomainOutput;\n\n\
     \x20       pub fn eval_domain_step(\n\
     \x20           input_metric: u64,\n\
     \x20           guard_clause: u32,\n\
     \x20           hole_index: usize,\n\
     \x20       ) -> Result<VerifiedDomainOutput, &'static str> {\n\
     \x20           let floor_saturated = input_metric.saturating_sub(guard_clause as u64);\n\
     \x20           Ok(VerifiedDomainOutput::mint_from_approved_crate(\n\
     \x20               floor_saturated.saturating_add(hole_index as u64),\n\
     \x20           ))\n\
     \x20       }\n\
     \x20   }\n\
     }"
}

/// Compile and execute a candidate leaf patch set on a [`CleanroomRustHyloSpec`] using real
/// `rustc --test --edition 2021` (with `--cfg approved_crate` enabled when `requested_ecosystem_crate`
/// is `true`), returning `(cleared, compiler_or_test_diagnostic)`.
pub fn evaluate_nursery_candidate_with_rustc(
    spec: &CleanroomRustHyloSpec,
    requested_ecosystem_crate: bool,
    leaf_patches: &[(String, String)],
) -> (bool, String) {
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(1);

    let merged_source = match crate::rust_ast_lens::hylo_octo_merge_rust_holes(
        &spec.ast_valid_token_zero_skeleton,
        leaf_patches,
    ) {
        Ok(src) => src,
        Err(err) => return (false, format!("AST splice failed: {err}")),
    };

    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir().join(format!(
        "portia_nursery_rustc_{}_{}",
        std::process::id(),
        seq
    ));
    if let Err(err) = std::fs::create_dir_all(&tmp_dir) {
        return (false, format!("failed to create tempdir: {err}"));
    }
    let src_path = tmp_dir.join("lib.rs");
    let bin_path = tmp_dir.join("cleanroom_test_bin");
    if let Err(err) = std::fs::write(&src_path, &merged_source) {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return (false, format!("failed to write lib.rs: {err}"));
    }

    let mut cmd = Command::new("rustc");
    cmd.arg("--test")
        .arg("--edition")
        .arg("2021")
        .arg("-A")
        .arg("dead_code")
        .arg("-A")
        .arg("unused_variables");
    if requested_ecosystem_crate {
        cmd.arg("--cfg").arg("approved_crate");
    }
    let compile_out = match cmd.arg(&src_path).arg("-o").arg(&bin_path).output() {
        Ok(out) => out,
        Err(err) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return (false, format!("failed to invoke rustc: {err}"));
        }
    };
    if !compile_out.status.success() {
        let stderr = String::from_utf8_lossy(&compile_out.stderr).to_string();
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return (false, stderr);
    }

    let test_out = Command::new(&bin_path).output();
    let _ = std::fs::remove_dir_all(&tmp_dir);
    match test_out {
        Ok(out) if out.status.success() => (
            true,
            String::from_utf8_lossy(&out.stdout).trim().to_string(),
        ),
        Ok(out) => (
            false,
            format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        ),
        Err(err) => (false, format!("failed to run compiled test binary: {err}")),
    }
}
