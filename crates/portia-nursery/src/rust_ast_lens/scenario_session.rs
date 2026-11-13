//! Interactive `UnifiedRustAstScenarioSession` state machine driving `CleanroomRustHyloSpec`
//! episodes across `Mode::Probe`, `Mode::Read`, `Mode::Ask`, `Mode::Write`, `Mode::Result`,
//! and `Mode::Abstain`.

use std::collections::BTreeSet;

use portia_core::{CoreProtocolError, Mode};
use serde::{Deserialize, Serialize};

use crate::nursery_scenario::{
    CleanroomRustHyloSpec, DialogicCodingAffordance, HyloCartridgeLevel, HyloCartridgePhase,
    RustHyloPipelineStage, ScenarioEpisodeSummary, ScenarioWipeoutDiagnosis,
};

use super::cst_extract::{
    extract_rust_todo_holes, format_rust_ast_outline, slice_rust_fn_window,
    verify_balanced_rust_delimiters,
};
use super::grammar::RustAstHoleGrammarCartridge;
use super::splice_merge::{splice_rust_todo_hole_body, validate_hole_replacement_block_ast};

/// Per-turn step outcome returned by [`UnifiedRustAstScenarioSession::step_turn`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustAstTurnOutcome {
    pub turn_index: usize,
    pub mode: Mode,
    pub ok: bool,
    pub active_pipeline_stage: RustHyloPipelineStage,
    pub tool_response: Option<String>,
    pub remaining_todo_holes: usize,
    pub response_bytes: usize,
    pub estimated_bpe_tokens: usize,
    pub turn_wipeout_diagnoses: Vec<ScenarioWipeoutDiagnosis>,
    pub episode_summary: Option<ScenarioEpisodeSummary>,
    pub error_message: Option<String>,
}

/// Interactive step-by-step coalgebra driving a [`CleanroomRustHyloSpec`] across `Mode::Probe`,
/// `Mode::Read`, `Mode::Ask`, `Mode::Write`, `Mode::Result`, and `Mode::Abstain`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedRustAstScenarioSession {
    pub spec: CleanroomRustHyloSpec,
    pub current_source: String,
    pub turn_index: usize,
    pub pipeline_stage_cursor: usize,
    pub probed_ast_outline: bool,
    pub inspected_fn_holes: BTreeSet<String>,
    pub filled_fn_holes: BTreeSet<String>,
    pub approved_crates: BTreeSet<String>,
    pub asked_clarification: bool,
    pub requested_ecosystem_crate: bool,
    pub reviewed_scaffold_before_fanout: bool,
    pub scaffold_ast_valid: bool,
    pub cumulative_response_bytes: usize,
    pub cumulative_estimated_bpe_tokens: usize,
    pub wipeout_diagnoses: Vec<ScenarioWipeoutDiagnosis>,
}

/// Alias for [`UnifiedRustAstScenarioSession`].
pub type RustAstHyloSession = UnifiedRustAstScenarioSession;

impl UnifiedRustAstScenarioSession {
    /// Create a new interactive [`UnifiedRustAstScenarioSession`] from a [`CleanroomRustHyloSpec`].
    pub fn new(spec: CleanroomRustHyloSpec) -> Result<Self, CoreProtocolError> {
        verify_balanced_rust_delimiters(&spec.ast_valid_token_zero_skeleton)?;
        let current_source = spec.ast_valid_token_zero_skeleton.clone();
        Ok(Self {
            spec,
            current_source,
            turn_index: 0,
            pipeline_stage_cursor: 0,
            probed_ast_outline: false,
            inspected_fn_holes: BTreeSet::new(),
            filled_fn_holes: BTreeSet::new(),
            approved_crates: BTreeSet::new(),
            asked_clarification: false,
            requested_ecosystem_crate: false,
            reviewed_scaffold_before_fanout: false,
            scaffold_ast_valid: true,
            cumulative_response_bytes: 0,
            cumulative_estimated_bpe_tokens: 0,
            wipeout_diagnoses: Vec::new(),
        })
    }

    /// Return the currently active [`RustHyloPipelineStage`] in the $2 \times 2$ Cartridge Matrix schedule.
    pub fn active_pipeline_stage(&self) -> RustHyloPipelineStage {
        self.spec
            .pipeline_schedule
            .get(self.pipeline_stage_cursor)
            .copied()
            .unwrap_or(RustHyloPipelineStage::OctoMerge {
                merged_hole_count: self.spec.todo_holes.len(),
            })
    }

    /// Return the live [`RustAstHoleGrammarCartridge`] for the remaining `todo!("...")` holes
    /// in `self.current_source` and any approved ecosystem crate.
    pub fn hole_grammar_cartridge(&self) -> RustAstHoleGrammarCartridge {
        let hole_fn_names = extract_rust_todo_holes(&self.current_source)
            .map(|holes| holes.into_iter().map(|h| h.fn_name).collect())
            .unwrap_or_else(|_| {
                self.spec
                    .todo_holes
                    .iter()
                    .map(|h| h.fn_name.clone())
                    .collect()
            });
        let approved_crate_name = self.approved_crates.iter().next().cloned();
        RustAstHoleGrammarCartridge {
            hole_fn_names,
            approved_crate_name,
        }
    }

    fn record_diagnosis(
        &mut self,
        turn_diags: &mut Vec<ScenarioWipeoutDiagnosis>,
        diag: ScenarioWipeoutDiagnosis,
    ) {
        if !turn_diags.contains(&diag) {
            turn_diags.push(diag.clone());
        }
        if !self.wipeout_diagnoses.contains(&diag) {
            self.wipeout_diagnoses.push(diag);
        }
    }

    fn step_write_turn(
        &mut self,
        payload: &str,
        turn_diags: &mut Vec<ScenarioWipeoutDiagnosis>,
    ) -> (Option<String>, Option<String>) {
        let Some((target_fn, new_body)) = Self::parse_write_payload(payload) else {
            self.scaffold_ast_valid = false;
            let detail =
                "invalid Mode::Write payload format (expected `<fn_name> => <body>`)".to_string();
            self.record_diagnosis(
                turn_diags,
                ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                    detail: detail.clone(),
                },
            );
            return (None, Some(detail));
        };

        if !self.probed_ast_outline && !self.inspected_fn_holes.contains(&target_fn) {
            self.record_diagnosis(
                turn_diags,
                ScenarioWipeoutDiagnosis::MissingReadBeforeWriteProvenance,
            );
        }

        if !self.asked_clarification {
            let missing_aspects: Vec<String> = self
                .spec
                .dialogic_affordances
                .iter()
                .filter_map(|affordance| match affordance {
                    DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                        ambiguous_aspect,
                        ..
                    } => Some(ambiguous_aspect.clone()),
                    _ => None,
                })
                .collect();
            for ambiguous_aspect in missing_aspects {
                self.record_diagnosis(
                    turn_diags,
                    ScenarioWipeoutDiagnosis::GuessedOnAmbiguousSpecWithoutAsking {
                        ambiguous_aspect,
                    },
                );
            }
        }

        let required_crates: Vec<String> = self
            .spec
            .dialogic_affordances
            .iter()
            .filter_map(|affordance| match affordance {
                DialogicCodingAffordance::RequestEcosystemCrate {
                    crate_name,
                    approved: true,
                    ..
                } => Some(crate_name.clone()),
                _ => None,
            })
            .collect();
        for available_crate in required_crates {
            if (!self.requested_ecosystem_crate && Self::looks_like_hand_rolled_parser(&new_body))
                || new_body.contains("mint_from_approved_crate")
            {
                self.record_diagnosis(
                    turn_diags,
                    ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate {
                        available_crate,
                    },
                );
            }
        }

        let mut error_message = None;
        if let Err(err) = validate_hole_replacement_block_ast(&target_fn, &new_body) {
            self.scaffold_ast_valid = false;
            let detail = err.to_string();
            self.record_diagnosis(
                turn_diags,
                ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                    detail: detail.clone(),
                },
            );
            error_message = Some(detail);
        }

        if !turn_diags.is_empty() {
            let msg = error_message.unwrap_or_else(|| {
                format!(
                    "rejected Mode::Write to `{target_fn}` due to validation failures: {:?}",
                    turn_diags
                )
            });
            return (None, Some(msg));
        }

        match splice_rust_todo_hole_body(&self.current_source, &target_fn, &new_body) {
            Ok(spliced) => {
                self.current_source = spliced;
                self.filled_fn_holes.insert(target_fn.clone());
                let remaining = extract_rust_todo_holes(&self.current_source)
                    .map(|h| h.len())
                    .unwrap_or(0);
                if remaining == 0 {
                    self.pipeline_stage_cursor =
                        self.spec.pipeline_schedule.len().saturating_sub(1);
                } else {
                    self.pipeline_stage_cursor = self
                        .spec
                        .pipeline_schedule
                        .iter()
                        .position(|s| {
                            matches!(
                                s,
                                RustHyloPipelineStage::CartridgeCell {
                                    phase: HyloCartridgePhase::Impl,
                                    level: HyloCartridgeLevel::ValueCart,
                                }
                            )
                        })
                        .unwrap_or(4);
                }
                (
                    Some(format!(
                        "{{\"status\": \"SPLICED\", \"fn_name\": \"{target_fn}\", \"remaining_todo_holes\": {remaining}}}"
                    )),
                    None,
                )
            }
            Err(err) => {
                self.scaffold_ast_valid = false;
                let detail = err.to_string();
                self.record_diagnosis(
                    turn_diags,
                    ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                        detail: detail.clone(),
                    },
                );
                (None, Some(detail))
            }
        }
    }

    /// Execute one interactive OODA turn (`Probe`, `Read`, `Ask`, `Write`, `Result`, `Abstain`).
    pub fn step_turn(&mut self, mode: Mode, payload: &str) -> RustAstTurnOutcome {
        let turn_index = self.turn_index;
        self.turn_index += 1;
        let mut turn_diags = Vec::new();
        let mut tool_response = None;
        let mut error_message = None;
        let mut episode_summary = None;

        match mode {
            Mode::Probe => {
                match format_rust_ast_outline(&self.spec.module_name, &self.current_source) {
                    Ok(outline) => {
                        self.probed_ast_outline = true;
                        if self.pipeline_stage_cursor == 0 {
                            self.pipeline_stage_cursor = 1;
                        }
                        tool_response = Some(outline);
                    }
                    Err(err) => {
                        self.scaffold_ast_valid = false;
                        let detail = err.to_string();
                        self.record_diagnosis(
                            &mut turn_diags,
                            ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                                detail: detail.clone(),
                            },
                        );
                        error_message = Some(detail);
                    }
                }
            }
            Mode::Read => {
                let target_fn = payload
                    .trim()
                    .strip_prefix("fn ")
                    .unwrap_or(payload.trim())
                    .trim_matches('`')
                    .trim();
                match slice_rust_fn_window(&self.current_source, target_fn) {
                    Ok(window) => {
                        self.inspected_fn_holes.insert(target_fn.to_string());
                        if matches!(
                            self.active_pipeline_stage(),
                            RustHyloPipelineStage::CartridgeCell {
                                phase: HyloCartridgePhase::Scaffold,
                                ..
                            }
                        ) {
                            self.pipeline_stage_cursor =
                                2.min(self.spec.pipeline_schedule.len() - 1);
                        }
                        tool_response = Some(window);
                    }
                    Err(err) => {
                        error_message = Some(err.to_string());
                    }
                }
            }
            Mode::Ask => {
                let reply_parts = self.apply_matching_dialogic_affordances(payload, true);
                if !reply_parts.is_empty() {
                    tool_response = Some(reply_parts.join("\n"));
                }
            }
            Mode::Write => {
                let (resp, err) = self.step_write_turn(payload, &mut turn_diags);
                tool_response = resp;
                error_message = err;
            }
            Mode::Result | Mode::Abstain => {
                let remaining = extract_rust_todo_holes(&self.current_source)
                    .map(|h| h.len())
                    .unwrap_or(0);
                if mode == Mode::Result && remaining > 0 {
                    let detail = format!(
                        "terminated with Mode::Result while {remaining} todo! holes remain unfilled"
                    );
                    self.record_diagnosis(
                        &mut turn_diags,
                        ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary {
                            detail: detail.clone(),
                        },
                    );
                    error_message = Some(detail);
                }

                let cleared = self.wipeout_diagnoses.is_empty()
                    && turn_diags.is_empty()
                    && (mode == Mode::Abstain || remaining == 0);
                episode_summary = Some(ScenarioEpisodeSummary {
                    cleared,
                    wipeout_diagnoses: self.wipeout_diagnoses.clone(),
                    total_turns: self.turn_index,
                    total_response_bytes: self.cumulative_response_bytes,
                    total_bpe_tokens: self.cumulative_estimated_bpe_tokens,
                });
                tool_response = Some(format!(
                    "{{\"status\": \"{}\", \"cleared\": {}}}",
                    mode.as_str(),
                    cleared
                ));
            }
        }

        let response_bytes = tool_response.as_ref().map(|s| s.len()).unwrap_or(0);
        let estimated_bpe_tokens = response_bytes.div_ceil(4);
        self.cumulative_response_bytes += response_bytes;
        self.cumulative_estimated_bpe_tokens += estimated_bpe_tokens;

        let remaining_todo_holes = extract_rust_todo_holes(&self.current_source)
            .map(|h| h.len())
            .unwrap_or(0);
        let ok = turn_diags.is_empty()
            && error_message.is_none()
            && episode_summary.as_ref().map(|s| s.cleared).unwrap_or(true);

        RustAstTurnOutcome {
            turn_index,
            mode,
            ok,
            active_pipeline_stage: self.active_pipeline_stage(),
            tool_response,
            remaining_todo_holes,
            response_bytes,
            estimated_bpe_tokens,
            turn_wipeout_diagnoses: turn_diags,
            episode_summary,
            error_message,
        }
    }
}
