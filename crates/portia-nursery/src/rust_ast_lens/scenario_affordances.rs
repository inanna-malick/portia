//! Payload parsing, dialogic affordance matching, and sealed-witness crate verification
//! for [`UnifiedRustAstScenarioSession`].

use portia_core::Mode;

use crate::nursery_scenario::DialogicCodingAffordance;

use super::cst_extract::{extract_rust_todo_holes, parse_rust_fn_items};
use super::scenario_session::{RustAstTurnOutcome, UnifiedRustAstScenarioSession};

/// Format the Lark `hole_body` rule for `Mode::Write`, attaching the optional nursery
/// sealed-witness namespace prefix when `approved_crate_name` is active.
pub(crate) fn format_write_body_lark_rule(approved_crate_name: Option<&str>) -> String {
    if let Some(crate_name) = approved_crate_name.map(str::trim).filter(|s| !s.is_empty()) {
        let crate_ident = crate_name
            .replace('-', "_")
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!("(\"approved_crate::\" | \"{crate_ident}::\")? /[^\\r\\n]+/")
    } else {
        "/[^\\r\\n]+/".to_string()
    }
}

impl UnifiedRustAstScenarioSession {
    /// Parse a `Mode::Write` payload of the form `"<fn_name> => <replacement_body>"` or
    /// `"fn <fn_name>(...) { <replacement_body> }"` / `"fn <fn_name> { <replacement_body> }"`.
    pub fn parse_write_payload(payload: &str) -> Option<(String, String)> {
        if let Some((lhs, rhs)) = payload.split_once("=>") {
            let fn_name = lhs
                .trim()
                .trim_start_matches("fn ")
                .trim_matches('`')
                .trim()
                .to_string();
            let body = rhs.trim().to_string();
            if !fn_name.is_empty() && !body.is_empty() {
                return Some((fn_name, body));
            }
        }
        let trimmed = payload.trim();
        if let Ok(parsed_items) = parse_rust_fn_items(trimmed) {
            if let Some(first_fn) = parsed_items.into_iter().next() {
                if first_fn.body_close_brace > first_fn.body_open_brace {
                    let body = trimmed[first_fn.body_open_brace + 1..first_fn.body_close_brace]
                        .trim()
                        .to_string();
                    if !body.is_empty() {
                        return Some((first_fn.fn_name, body));
                    }
                }
            }
        }
        let after_fn = trimmed.strip_prefix("fn ").unwrap_or(trimmed);
        let (ident_part, rest_from_brace) = after_fn.split_once('{')?;
        let synthetic_fn = if ident_part.contains('(') {
            format!("fn {after_fn}")
        } else {
            format!("fn {}() {{{rest_from_brace}", ident_part.trim())
        };
        if let Ok(parsed_items) = parse_rust_fn_items(&synthetic_fn) {
            if let Some(first_fn) = parsed_items.into_iter().next() {
                if first_fn.body_close_brace > first_fn.body_open_brace {
                    let body = synthetic_fn
                        [first_fn.body_open_brace + 1..first_fn.body_close_brace]
                        .trim()
                        .to_string();
                    if !body.is_empty() {
                        return Some((first_fn.fn_name, body));
                    }
                }
            }
        }
        None
    }

    /// Check via `syn::visit::Visit` whether a parsed replacement block constructs its sealed
    /// `VerifiedDomainOutput` via the `#[cfg(approved_crate)]` module (`approved_crate::eval_domain_step`
    /// or the approved domain crate path) without calling the private `mint_from_approved_crate` constructor.
    pub(crate) fn block_calls_approved_ecosystem_crate_ast(
        body: &str,
        approved_crate_name: &str,
    ) -> bool {
        use syn::visit::Visit;

        struct ApprovedCratePathVisitor<'a> {
            approved_crate_ident: &'a str,
            called_approved_crate: bool,
            called_forbidden_mint: bool,
        }

        impl<'ast> Visit<'ast> for ApprovedCratePathVisitor<'_> {
            fn visit_path(&mut self, i: &'ast syn::Path) {
                if let Some(first_seg) = i.segments.first() {
                    let seg_name = first_seg.ident.to_string();
                    if seg_name == "approved_crate" || seg_name == self.approved_crate_ident {
                        self.called_approved_crate = true;
                    }
                }
                if i.segments
                    .iter()
                    .any(|seg| seg.ident == "mint_from_approved_crate")
                {
                    self.called_forbidden_mint = true;
                }
                syn::visit::visit_path(self, i);
            }
        }

        let clean_ident = approved_crate_name.replace('-', "_");
        let wrapped = format!("{{ {body} }}");
        if let Ok(block) = syn::parse_str::<syn::Block>(&wrapped) {
            let mut visitor = ApprovedCratePathVisitor {
                approved_crate_ident: &clean_ident,
                called_approved_crate: false,
                called_forbidden_mint: false,
            };
            visitor.visit_block(&block);
            visitor.called_approved_crate && !visitor.called_forbidden_mint
        } else {
            false
        }
    }

    /// Check whether a candidate replacement body bypasses the approved ecosystem crate.
    pub fn looks_like_hand_rolled_parser(body: &str) -> bool {
        !Self::block_calls_approved_ecosystem_crate_ast(body, "approved_crate")
    }

    pub(crate) fn apply_matching_dialogic_affordances(
        &mut self,
        question: &str,
        emit_nursery_replies: bool,
    ) -> Vec<String> {
        let q_lower = question.to_ascii_lowercase();
        let mut reply_parts = Vec::new();
        let mut has_explicit_clarify_affordance = false;

        for affordance in &self.spec.dialogic_affordances {
            match affordance {
                DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                    ambiguous_aspect,
                    user_clarification,
                } => {
                    has_explicit_clarify_affordance = true;
                    if q_lower.contains("clarif")
                        || q_lower.contains("underflow")
                        || q_lower.contains("invariant")
                        || q_lower.contains("round")
                        || q_lower.contains("ambiguous")
                        || q_lower.contains("floor")
                    {
                        self.asked_clarification = true;
                        if emit_nursery_replies {
                            reply_parts.push(format!(
                                "[CLARIFIED `{ambiguous_aspect}`]: {user_clarification}"
                            ));
                        }
                    }
                }
                DialogicCodingAffordance::RequestEcosystemCrate {
                    crate_name,
                    purpose,
                    approved,
                } => {
                    if q_lower.contains("crate")
                        || q_lower.contains(&crate_name.to_ascii_lowercase())
                        || q_lower.contains("dependency")
                    {
                        self.requested_ecosystem_crate = true;
                        if *approved {
                            self.approved_crates.insert(crate_name.clone());
                            if emit_nursery_replies {
                                reply_parts.push(format!(
                                    "[CRATE_APPROVED `{crate_name}`]: Approved for {purpose}."
                                ));
                            }
                        }
                    }
                }
                DialogicCodingAffordance::ReviewScaffoldBeforeFanout { todo_hole_names } => {
                    if q_lower.contains("scaffold")
                        || q_lower.contains("fanout")
                        || q_lower.contains("review")
                        || q_lower.contains("todo")
                    {
                        self.reviewed_scaffold_before_fanout = true;
                        if emit_nursery_replies {
                            reply_parts.push(format!(
                                "[SCAFFOLD_REVIEWED]: Approved Fork x {} fanout over {:?}.",
                                todo_hole_names.len(),
                                todo_hole_names
                            ));
                        }
                    }
                }
            }
        }

        if !has_explicit_clarify_affordance && !question.trim().is_empty() {
            self.asked_clarification = true;
        }

        reply_parts
    }

    /// Record a live human clarification turn on `self` without inventing simulated steward
    /// answers (`tool_response: None`) or blanket-completing unrelated dialogic flags.
    pub fn record_live_user_clarification(&mut self, question: &str) -> RustAstTurnOutcome {
        let turn_index = self.turn_index;
        self.turn_index += 1;
        let _ = self.apply_matching_dialogic_affordances(question, false);
        let remaining_todo_holes = extract_rust_todo_holes(&self.current_source)
            .map(|h| h.len())
            .unwrap_or(0);
        RustAstTurnOutcome {
            turn_index,
            mode: Mode::Ask,
            ok: true,
            active_pipeline_stage: self.active_pipeline_stage(),
            tool_response: None,
            remaining_todo_holes,
            response_bytes: 0,
            estimated_bpe_tokens: 0,
            turn_wipeout_diagnoses: Vec::new(),
            episode_summary: None,
            error_message: None,
        }
    }

    /// Explicit offline nursery steward simulation for `Mode::Ask`, answering only configured
    /// [`DialogicCodingAffordance`]s that match `question`.
    pub fn simulate_nursery_steward_ask(&mut self, question: &str) -> RustAstTurnOutcome {
        self.step_turn(Mode::Ask, question)
    }
}
