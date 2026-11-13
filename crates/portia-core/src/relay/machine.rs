//! `RelayMachine` runtime state automaton implementing `HarnessPen -> RouterPen -> PayloadPen`.

use super::scanners::find_any_close_tag_outside_literals;
use super::{
    json_has_unclosed_string, sql_has_unclosed_quote, verify_single_pen_schedule,
    CompletedRelayTurn, GrammarCartridgeSpec, PayloadGrammarAutomaton, PenSpanRecord,
    PlannedTrunkStep, RelayPhase, SpanCartridge, TypedTurnAction, PLAN_BODY_SINGLE_LINE_LARK,
    POST_MODE_FRAMING, TURN_CLOSE_SUFFIX, TURN_OPEN_PREFILL,
};
use crate::bpe::{heal_delimiter_crossing, heal_mode_label_buffer, is_valid_mode_label_prefix};
use crate::{CartridgeBank, CoreProtocolError, Mode, PenOwner};

/// Deterministic `HarnessPen -> RouterPen -> PayloadPen` state machine.
#[derive(Debug, Clone)]
pub struct RelayMachine {
    phase: RelayPhase,
    selected_mode: Option<Mode>,
    mode_buf: String,
    plan_buf: String,
    payload_buf: String,
    pending_harness_prefill: String,
    full_wire_text: String,
    spans: Vec<PenSpanRecord>,
}

impl Default for RelayMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl RelayMachine {
    pub fn new() -> Self {
        Self {
            phase: RelayPhase::TurnOpenPrefill,
            selected_mode: None,
            mode_buf: String::new(),
            plan_buf: String::new(),
            payload_buf: String::new(),
            pending_harness_prefill: TURN_OPEN_PREFILL.to_string(),
            full_wire_text: String::new(),
            spans: Vec::new(),
        }
    }

    pub fn phase(&self) -> RelayPhase {
        self.phase
    }

    pub fn selected_mode(&self) -> Option<Mode> {
        self.selected_mode
    }

    pub fn plan_buf(&self) -> &str {
        &self.plan_buf
    }

    pub fn payload_buf(&self) -> &str {
        &self.payload_buf
    }

    /// Return the active [`SpanCartridge`] (`(weight_bank, grammar_spec)`) at the current phase across
    /// **all three** trunk-generated phases (`ModeLabel`, `PlanBody`, and `PayloadBody`), and
    /// `Unconstrained` on deterministic `HarnessPen` prefill phases.
    pub fn active_span_cartridge<G: PayloadGrammarAutomaton + ?Sized>(
        &self,
        grammar: &G,
    ) -> SpanCartridge {
        let weight_bank = self.active_cartridge_bank();
        let grammar_spec = match self.phase {
            RelayPhase::ModeLabel => {
                let literals: Vec<String> = grammar
                    .legal_modes()
                    .into_iter()
                    .map(|m| format!(" {}]", m.as_str()))
                    .collect();
                if literals.is_empty() {
                    GrammarCartridgeSpec::Unconstrained
                } else {
                    GrammarCartridgeSpec::LiteralSet(literals)
                }
            }
            RelayPhase::PlanBody => {
                GrammarCartridgeSpec::LarkGrammar(PLAN_BODY_SINGLE_LINE_LARK.to_string())
            }
            RelayPhase::PayloadBody => self
                .selected_mode
                .map(|m| grammar.grammar_cartridge_spec(m))
                .unwrap_or(GrammarCartridgeSpec::Unconstrained),
            RelayPhase::TurnOpenPrefill
            | RelayPhase::PostModePrefill
            | RelayPhase::PostPlanPrefill
            | RelayPhase::TurnComplete => GrammarCartridgeSpec::Unconstrained,
        };
        SpanCartridge::new(weight_bank, grammar_spec)
    }

    /// Which pen exclusively holds the cursor (`|ActivePen(t)| == 1`) right now.
    pub fn active_pen(&self) -> PenOwner {
        match self.phase {
            RelayPhase::TurnOpenPrefill
            | RelayPhase::PostModePrefill
            | RelayPhase::PostPlanPrefill
            | RelayPhase::TurnComplete => PenOwner::HarnessPen,
            RelayPhase::ModeLabel => PenOwner::RouterPen,
            RelayPhase::PlanBody | RelayPhase::PayloadBody => PenOwner::PayloadPen,
        }
    }

    /// Which cartridge bank must be mounted in the model rack at the current phase.
    ///
    /// `PostModePrefill` (`\nPlan:`) and `PlanBody` run under [`CartridgeBank::Base`] so free-form
    /// natural-language planning stays on the untouched pretrain, scoping [`CartridgeBank::ModePayload`]
    /// strictly to `PostPlanPrefill` (`</think>\n<sql>\n` / `</think>\n<to_user>\n`) and `PayloadBody`.
    pub fn active_cartridge_bank(&self) -> CartridgeBank {
        match self.phase {
            RelayPhase::TurnOpenPrefill | RelayPhase::ModeLabel => CartridgeBank::OrientTrunk,
            RelayPhase::PostModePrefill | RelayPhase::PlanBody | RelayPhase::TurnComplete => {
                CartridgeBank::Base
            }
            RelayPhase::PostPlanPrefill | RelayPhase::PayloadBody => {
                let mode = self
                    .selected_mode
                    .expect("selected_mode must be locked before PostPlanPrefill");
                CartridgeBank::ModePayload(mode)
            }
        }
    }

    /// True if `PayloadPen` is currently inside an unclosed single-quoted SQL literal or
    /// double-quoted JSON string inside `<sql>` / `<to_user>`, so control tags like `</sql>` or
    /// `<|im_end|>` appearing inside literals must be treated as source data, not stop tags.
    pub fn control_tokens_are_source(&self) -> bool {
        if self.phase != RelayPhase::PayloadBody {
            return false;
        }
        let Some(mode) = self.selected_mode else {
            return false;
        };
        match mode {
            Mode::Probe | Mode::Read | Mode::Write => sql_has_unclosed_quote(&self.payload_buf),
            Mode::Result | Mode::Abstain => json_has_unclosed_string(&self.payload_buf),
            Mode::Ask => false,
        }
    }

    /// Consume the pending `H = 0` `HarnessPen` prefill chunk (updating KV/DeltaNet state under
    /// `self.active_cartridge_bank()`) and advance to the next `RouterPen` or `PayloadPen` phase.
    pub fn flush_harness_prefill(
        &mut self,
        encode_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<Option<PenSpanRecord>, CoreProtocolError> {
        if self.active_pen() != PenOwner::HarnessPen || self.pending_harness_prefill.is_empty() {
            return Ok(None);
        }
        let text = std::mem::take(&mut self.pending_harness_prefill);
        let token_ids = encode_fn(&text);
        let consumer_bank = self.active_cartridge_bank();
        let phase_at_record = self.phase;

        let record = PenSpanRecord {
            step_index: self.spans.len(),
            phase: phase_at_record,
            pen: PenOwner::HarnessPen,
            producer_bank: None,
            consumer_bank,
            text: text.clone(),
            token_ids,
        };
        self.full_wire_text.push_str(&text);
        self.spans.push(record.clone());

        self.phase = match self.phase {
            RelayPhase::TurnOpenPrefill => RelayPhase::ModeLabel,
            RelayPhase::PostModePrefill => RelayPhase::PlanBody,
            RelayPhase::PostPlanPrefill => RelayPhase::PayloadBody,
            RelayPhase::TurnComplete => RelayPhase::TurnComplete,
            other => other,
        };

        Ok(Some(record))
    }

    /// Plan the exact `consume_bank` handoff for a newly sampled trunk token (`RouterPen` or
    /// `PayloadPen`) **before** the runtime worker forwards `token_id` into `past_key_values`.
    pub fn plan_sampled_trunk_token(
        &mut self,
        token_id: u32,
        token_piece: &str,
    ) -> Result<PlannedTrunkStep, CoreProtocolError> {
        let span = match self.phase {
            RelayPhase::ModeLabel => self.step_router_mode_token(token_id, token_piece)?,
            RelayPhase::PlanBody => self.step_payload_plan_token(token_id, token_piece)?,
            RelayPhase::PayloadBody => self.step_payload_body_token(token_id, token_piece)?,
            other => {
                return Err(CoreProtocolError::SinglePenViolation {
                    step: self.spans.len(),
                    detail: format!(
                        "plan_sampled_trunk_token called in phase {:?} owned by {:?}",
                        other,
                        self.active_pen()
                    ),
                });
            }
        };
        let producer_bank = span
            .producer_bank
            .expect("trunk token step must always record producer_bank");
        let consume_bank = span.consumer_bank;
        Ok(PlannedTrunkStep {
            producer_bank,
            consume_bank,
            phase_after: self.phase,
            span,
        })
    }

    pub fn step_trunk_token(
        &mut self,
        token_id: u32,
        token_piece: &str,
    ) -> Result<PenSpanRecord, CoreProtocolError> {
        self.plan_sampled_trunk_token(token_id, token_piece)
            .map(|planned| planned.span)
    }

    fn step_router_mode_token(
        &mut self,
        token_id: u32,
        token_piece: &str,
    ) -> Result<PenSpanRecord, CoreProtocolError> {
        let mut candidate = self.mode_buf.clone();
        candidate.push_str(token_piece);

        if let Some(split) = heal_delimiter_crossing(&candidate, "]", POST_MODE_FRAMING)? {
            let mode = heal_mode_label_buffer(&split.head_before_delimiter)?;
            self.selected_mode = Some(mode);
            self.mode_buf = split.head_before_delimiter;

            let record = PenSpanRecord {
                step_index: self.spans.len(),
                phase: RelayPhase::ModeLabel,
                pen: PenOwner::RouterPen,
                producer_bank: Some(CartridgeBank::OrientTrunk),
                // `]` (`tid = 60`) is consumed under `OrientTrunk`, switching banks on `\n` (`tid = 198`).
                consumer_bank: CartridgeBank::OrientTrunk,
                text: token_piece.to_string(),
                token_ids: vec![token_id],
            };
            self.full_wire_text.push_str(token_piece);
            self.spans.push(record.clone());

            self.pending_harness_prefill = split.remaining_framing_to_prefill;
            self.phase = if self.pending_harness_prefill.is_empty() {
                RelayPhase::PlanBody
            } else {
                RelayPhase::PostModePrefill
            };
            return Ok(record);
        }

        if !is_valid_mode_label_prefix(&candidate) {
            return Err(CoreProtocolError::InvalidModeLabel(candidate));
        }

        self.mode_buf = candidate;
        let record = PenSpanRecord {
            step_index: self.spans.len(),
            phase: RelayPhase::ModeLabel,
            pen: PenOwner::RouterPen,
            producer_bank: Some(CartridgeBank::OrientTrunk),
            consumer_bank: CartridgeBank::OrientTrunk,
            text: token_piece.to_string(),
            token_ids: vec![token_id],
        };
        self.full_wire_text.push_str(token_piece);
        self.spans.push(record.clone());
        Ok(record)
    }

    fn step_payload_plan_token(
        &mut self,
        token_id: u32,
        token_piece: &str,
    ) -> Result<PenSpanRecord, CoreProtocolError> {
        let mode = self
            .selected_mode
            .expect("selected_mode must be set in PlanBody");
        let bank = CartridgeBank::Base;
        let upcoming_framing = format!("</think>\n{}\n", mode.route_tag().open_tag());

        let mut candidate = self.plan_buf.clone();
        candidate.push_str(token_piece);

        if let Some(split) = heal_delimiter_crossing(&candidate, "\n", &upcoming_framing)? {
            if split.head_before_delimiter.trim().is_empty() {
                return Err(CoreProtocolError::BoundaryViolation(
                    "empty Plan: line is not permitted".to_string(),
                ));
            }
            let clean_plan = split
                .head_before_delimiter
                .strip_prefix(' ')
                .unwrap_or(&split.head_before_delimiter)
                .to_string();
            self.plan_buf = clean_plan;
            let wire_piece = if split.healed_double_newline {
                let first_nl = token_piece.find('\n').unwrap_or(token_piece.len() - 1);
                &token_piece[..=first_nl]
            } else {
                token_piece
            };

            let record = PenSpanRecord {
                step_index: self.spans.len(),
                phase: RelayPhase::PlanBody,
                pen: PenOwner::PayloadPen,
                producer_bank: Some(bank),
                consumer_bank: bank,
                text: wire_piece.to_string(),
                token_ids: vec![token_id],
            };
            self.full_wire_text.push_str(wire_piece);
            self.spans.push(record.clone());

            self.pending_harness_prefill = split.remaining_framing_to_prefill;
            self.phase = if self.pending_harness_prefill.is_empty() {
                RelayPhase::PayloadBody
            } else {
                RelayPhase::PostPlanPrefill
            };
            return Ok(record);
        }

        self.plan_buf = candidate;
        let record = PenSpanRecord {
            step_index: self.spans.len(),
            phase: RelayPhase::PlanBody,
            pen: PenOwner::PayloadPen,
            producer_bank: Some(bank),
            consumer_bank: bank,
            text: token_piece.to_string(),
            token_ids: vec![token_id],
        };
        self.full_wire_text.push_str(token_piece);
        self.spans.push(record.clone());
        Ok(record)
    }

    fn step_payload_body_token(
        &mut self,
        token_id: u32,
        token_piece: &str,
    ) -> Result<PenSpanRecord, CoreProtocolError> {
        let mode = self
            .selected_mode
            .expect("selected_mode must be set in PayloadBody");
        let bank = CartridgeBank::ModePayload(mode);

        let mut candidate = self.payload_buf.clone();
        candidate.push_str(token_piece);

        let mut completed = false;
        let mut wire_piece = token_piece;
        if let Some((tag_pos, matched_tag)) =
            find_any_close_tag_outside_literals(&candidate, mode, mode.route_tag())
        {
            let after_tag = &candidate[tag_pos + matched_tag.len()..];
            if !after_tag.trim().is_empty() {
                return Err(CoreProtocolError::BoundaryViolation(format!(
                    "trailing bytes {:?} after closing route tag {}",
                    after_tag, matched_tag
                )));
            }
            let body_only = candidate[..tag_pos]
                .strip_suffix('\n')
                .unwrap_or(&candidate[..tag_pos])
                .to_string();
            if body_only.trim().is_empty() {
                return Err(CoreProtocolError::BoundaryViolation(
                    "empty payload body before closing route tag".to_string(),
                ));
            }
            if !after_tag.is_empty() && token_piece.len() >= after_tag.len() {
                wire_piece = &token_piece[..token_piece.len() - after_tag.len()];
            }
            self.payload_buf = body_only;
            completed = true;
        } else {
            self.payload_buf = candidate;
        }

        self.full_wire_text.push_str(wire_piece);
        let consumer_bank = if completed { CartridgeBank::Base } else { bank };

        let record = PenSpanRecord {
            step_index: self.spans.len(),
            phase: RelayPhase::PayloadBody,
            pen: PenOwner::PayloadPen,
            producer_bank: Some(bank),
            consumer_bank,
            text: wire_piece.to_string(),
            token_ids: vec![token_id],
        };
        self.spans.push(record.clone());

        if completed {
            self.pending_harness_prefill = TURN_CLOSE_SUFFIX.to_string();
            self.phase = RelayPhase::TurnComplete;
        }

        Ok(record)
    }

    /// Finalize the turn once `RelayPhase::TurnComplete` is reached (flushing `<|im_end|>\n` if still pending).
    pub fn into_completed_turn(
        mut self,
        encode_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<CompletedRelayTurn, CoreProtocolError> {
        if self.phase != RelayPhase::TurnComplete {
            return Err(CoreProtocolError::BoundaryViolation(format!(
                "cannot finalize turn in incomplete phase {:?}",
                self.phase
            )));
        }
        let _ = self.flush_harness_prefill(encode_fn)?;
        verify_single_pen_schedule(&self.spans)?;
        let mode = self.selected_mode.expect("completed turn must have mode");
        let action = TypedTurnAction::parse_for_mode(mode, &self.payload_buf)?;
        Ok(CompletedRelayTurn {
            mode,
            route: mode.route_tag(),
            plan: self.plan_buf,
            payload: self.payload_buf,
            action,
            full_wire_text: self.full_wire_text,
            spans: self.spans,
        })
    }
}
