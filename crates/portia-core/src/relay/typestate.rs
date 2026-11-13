//! Compile-time typestate Single-Pen turn cursors (`TurnAtOpenPrefill` .. `TurnAtPayloadBody`)
//! and span-schedule invariant verifier (`verify_single_pen_schedule`).

use super::{CompletedRelayTurn, PenSpanRecord, RelayMachine, RelayPhase};
use crate::{CartridgeBank, CoreProtocolError, Mode, PenOwner};

pub struct TurnAtOpenPrefill {
    machine: RelayMachine,
}

pub struct TurnAtRouterDecision {
    machine: RelayMachine,
}

pub struct TurnAtPostModePrefill {
    machine: RelayMachine,
    mode: Mode,
}

pub struct TurnAtPlanBody {
    machine: RelayMachine,
    mode: Mode,
}

pub struct TurnAtPostPlanPrefill {
    machine: RelayMachine,
    mode: Mode,
}

pub struct TurnAtPayloadBody {
    machine: RelayMachine,
    mode: Mode,
}

impl Default for TurnAtOpenPrefill {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnAtOpenPrefill {
    pub fn new() -> Self {
        Self {
            machine: RelayMachine::new(),
        }
    }

    /// `HarnessPen` (`H = 0`) prefills `<|im_start|>assistant\n<think>\n[MODE:` under `OrientTrunk`
    /// and hands the single pen to `RouterPen`.
    pub fn harness_open(
        mut self,
        encode_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<TurnAtRouterDecision, CoreProtocolError> {
        self.machine.flush_harness_prefill(encode_fn)?;
        Ok(TurnAtRouterDecision {
            machine: self.machine,
        })
    }
}

impl TurnAtRouterDecision {
    /// `RouterPen` selects `mode` (`" <MODE>]"`) under `OrientTrunk` (consuming `]` = 60 under `OrientTrunk`)
    /// and hands the pen back to `HarnessPen` for `\nPlan:`.
    pub fn router_select(mut self, mode: Mode) -> Result<TurnAtPostModePrefill, CoreProtocolError> {
        let first_tid = mode.qwen35_first_token_id();
        let chunk = format!(" {}]", mode.as_str());
        self.machine.step_trunk_token(first_tid, &chunk)?;
        Ok(TurnAtPostModePrefill {
            machine: self.machine,
            mode,
        })
    }
}

impl TurnAtPostModePrefill {
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// `HarnessPen` (`H = 0`) fast-forwards `\nPlan:` starting on `\n` (`tid = 198`) under
    /// `CartridgeBank::Base` and hands the pen to `PayloadPen`.
    pub fn harness_plan_header(
        mut self,
        encode_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<TurnAtPlanBody, CoreProtocolError> {
        self.machine.flush_harness_prefill(encode_fn)?;
        Ok(TurnAtPlanBody {
            machine: self.machine,
            mode: self.mode,
        })
    }
}

impl TurnAtPlanBody {
    /// `PayloadPen` emits the single-line `Plan:` text ending in `\n` under `CartridgeBank::Base`
    /// and hands the pen back to `HarnessPen` for `</think>\n<sql>\n` or `</think>\n<to_user>\n`.
    pub fn payload_plan(
        mut self,
        plan_token_id: u32,
        plan_line: &str,
    ) -> Result<TurnAtPostPlanPrefill, CoreProtocolError> {
        let with_nl = if plan_line.ends_with('\n') {
            plan_line.to_string()
        } else {
            format!("{}\n", plan_line)
        };
        self.machine.step_trunk_token(plan_token_id, &with_nl)?;
        Ok(TurnAtPostPlanPrefill {
            machine: self.machine,
            mode: self.mode,
        })
    }
}

impl TurnAtPostPlanPrefill {
    /// `HarnessPen` (`H = 0`) fast-forwards `</think>\n<sql>\n` or `</think>\n<to_user>\n` under `ModePayload(self.mode)`
    /// and hands the pen to `PayloadPen` for the body.
    pub fn harness_route_open(
        mut self,
        encode_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<TurnAtPayloadBody, CoreProtocolError> {
        self.machine.flush_harness_prefill(encode_fn)?;
        Ok(TurnAtPayloadBody {
            machine: self.machine,
            mode: self.mode,
        })
    }
}

impl TurnAtPayloadBody {
    /// `PayloadPen` emits the payload body and closing route tag (`</sql>` or `</to_user>`),
    /// and `HarnessPen` seals `<|im_end|>\n` under `CartridgeBank::Base`, returning a verified [`CompletedRelayTurn`].
    pub fn payload_body(
        mut self,
        body_token_id: u32,
        payload_body_without_close_tag: &str,
        encode_close_fn: impl FnOnce(&str) -> Vec<u32>,
    ) -> Result<CompletedRelayTurn, CoreProtocolError> {
        let close_tag = self.mode.route_tag().close_tag();
        let full_chunk = format!(
            "{}\n{}",
            payload_body_without_close_tag.trim_end(),
            close_tag
        );
        self.machine.step_trunk_token(body_token_id, &full_chunk)?;
        self.machine.into_completed_turn(encode_close_fn)
    }
}

/// Verify the Single-Pen handoff law across a recorded span trace:
/// 1. Turn must open with `HarnessPen` at `TurnOpenPrefill` under `OrientTrunk`.
/// 2. `RouterPen` spans must only appear in `ModeLabel` with `producer_bank == Some(OrientTrunk)`
///    and `consumer_bank == OrientTrunk` (ensuring `]` is consumed under `OrientTrunk`).
/// 3. `PostModePrefill` (`\nPlan:`) must be owned by `HarnessPen` and consumed under `Base`, and
///    `PlanBody` must be owned by `PayloadPen` with `producer_bank == Some(Base)` and `consumer_bank == Base`.
/// 4. `PostPlanPrefill` must be owned by `HarnessPen` under `ModePayload(mode)`, and `PayloadBody`
///    spans must be owned by `PayloadPen` with `producer_bank == Some(ModePayload(mode))`.
pub fn verify_single_pen_schedule(spans: &[PenSpanRecord]) -> Result<(), CoreProtocolError> {
    if spans.is_empty() {
        return Err(CoreProtocolError::SinglePenViolation {
            step: 0,
            detail: "empty span schedule".to_string(),
        });
    }
    for (idx, span) in spans.iter().enumerate() {
        if span.step_index != idx {
            return Err(CoreProtocolError::SinglePenViolation {
                step: idx,
                detail: format!("non-contiguous step_index {}", span.step_index),
            });
        }
        match span.phase {
            RelayPhase::TurnOpenPrefill => {
                if span.pen != PenOwner::HarnessPen
                    || span.producer_bank.is_some()
                    || span.consumer_bank != CartridgeBank::OrientTrunk
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid TurnOpenPrefill span: {:?}", span),
                    });
                }
            }
            RelayPhase::ModeLabel => {
                if span.pen != PenOwner::RouterPen
                    || span.producer_bank != Some(CartridgeBank::OrientTrunk)
                    || span.consumer_bank != CartridgeBank::OrientTrunk
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!(
                            "ModeLabel span must be produced and consumed under OrientTrunk: {:?}",
                            span
                        ),
                    });
                }
            }
            RelayPhase::PostModePrefill => {
                if span.pen != PenOwner::HarnessPen
                    || span.producer_bank.is_some()
                    || span.consumer_bank != CartridgeBank::Base
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid PostModePrefill span: {:?}", span),
                    });
                }
            }
            RelayPhase::PostPlanPrefill => {
                if span.pen != PenOwner::HarnessPen
                    || span.producer_bank.is_some()
                    || !matches!(span.consumer_bank, CartridgeBank::ModePayload(_))
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid PostPlanPrefill span: {:?}", span),
                    });
                }
            }
            RelayPhase::PlanBody => {
                if span.pen != PenOwner::PayloadPen
                    || span.producer_bank != Some(CartridgeBank::Base)
                    || span.consumer_bank != CartridgeBank::Base
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid PlanBody span: {:?}", span),
                    });
                }
            }
            RelayPhase::PayloadBody => {
                if span.pen != PenOwner::PayloadPen
                    || !matches!(span.producer_bank, Some(CartridgeBank::ModePayload(_)))
                    || !(Some(span.consumer_bank) == span.producer_bank
                        || span.consumer_bank == CartridgeBank::Base)
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid PayloadBody span: {:?}", span),
                    });
                }
            }
            RelayPhase::TurnComplete => {
                if span.pen != PenOwner::HarnessPen
                    || span.producer_bank.is_some()
                    || span.consumer_bank != CartridgeBank::Base
                {
                    return Err(CoreProtocolError::SinglePenViolation {
                        step: idx,
                        detail: format!("invalid TurnComplete span: {:?}", span),
                    });
                }
            }
        }
    }
    Ok(())
}
