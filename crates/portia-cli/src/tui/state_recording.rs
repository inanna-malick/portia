//! Streaming relay span recording, context-window tracking, and causal `Mode::Ask`
//! classification methods for [`TwoPaneTuiState`].

use super::state::{
    AnnotatedContextSpan, ContextSpanKind, InteractionItem, PaneScrollViewport, TwoPaneTuiState,
};
use super::tokens::{estimate_token_count, extract_control_token_badges};
use crate::DriverObservation;
use portia_core::relay::{CompletedRelayTurn, PenSpanRecord, RelayStreamEvent};
use portia_core::{CartridgeBank, Mode, PenOwner, RouteTag};

struct TaskAskObservationState {
    saw_steward_ask_row: bool,
    latest_multirow_choices: Option<Vec<String>>,
    latest_read_rows: Vec<Vec<serde_json::Value>>,
    has_task_db_read: bool,
}

impl TwoPaneTuiState {
    #[must_use]
    pub fn has_pending_steward_ask_row(&self) -> bool {
        self.resolve_task_ask_observation_state(usize::MAX)
            .saw_steward_ask_row
    }

    fn resolve_task_ask_observation_state(&self, turn_index: usize) -> TaskAskObservationState {
        let mut task_start_idx = 0usize;
        let mut session_fallback_rows: Vec<Vec<serde_json::Value>> =
            self.session_preview_rows.clone();

        for (idx, item) in self.interactions.iter().enumerate() {
            match item {
                InteractionItem::UserPrompt {
                    turn_index: p_idx, ..
                } => {
                    if *p_idx > turn_index {
                        break;
                    }
                    task_start_idx = idx;
                }
                InteractionItem::Observation {
                    turn_index: obs_turn,
                    observation,
                } if *obs_turn <= turn_index
                    && observation.ok
                    && observation.mode == Mode::Read
                    && !observation.rows.is_empty() =>
                {
                    session_fallback_rows = observation.rows.clone();
                }
                InteractionItem::AssistantTurn {
                    turn_index: t_idx, ..
                }
                | InteractionItem::Observation {
                    turn_index: t_idx, ..
                } if *t_idx > turn_index => {
                    break;
                }
                _ => {}
            }
        }

        let mut latest_steward_auth = portia_core::StewardAuthState::NotRequired;
        let mut latest_multirow_choices: Option<Vec<String>> = None;
        let mut latest_read_rows: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut has_task_db_read = false;

        for item in &self.interactions[task_start_idx..] {
            match item {
                InteractionItem::Observation {
                    turn_index: obs_turn,
                    observation,
                } if *obs_turn <= turn_index => {
                    if observation.mode != Mode::Ask || *obs_turn < turn_index {
                        latest_steward_auth = observation.steward_auth;
                    }
                    if observation.ok && matches!(observation.mode, Mode::Probe | Mode::Read) {
                        has_task_db_read = true;
                        if observation.mode == Mode::Read {
                            latest_read_rows = observation.rows.clone();
                            if self.candidate_disambiguation_enabled && observation.rows.len() >= 2
                            {
                                let choices: Vec<String> = observation
                                    .rows
                                    .iter()
                                    .take(6)
                                    .filter_map(|row| {
                                        crate::driver::summarize_candidate_row_choice(row)
                                    })
                                    .collect();
                                latest_multirow_choices = (choices.len() >= 2).then_some(choices);
                            } else {
                                latest_multirow_choices = None;
                            }
                        }
                    } else if *obs_turn < turn_index
                        && observation.ok
                        && observation.mode == Mode::Ask
                    {
                        latest_multirow_choices = None;
                    }
                }
                InteractionItem::AssistantTurn {
                    turn_index: t_idx, ..
                }
                | InteractionItem::Observation {
                    turn_index: t_idx, ..
                } if *t_idx > turn_index => {
                    break;
                }
                _ => {}
            }
        }

        if latest_read_rows.is_empty() && !session_fallback_rows.is_empty() {
            latest_read_rows = session_fallback_rows;
        }

        TaskAskObservationState {
            saw_steward_ask_row: latest_steward_auth.requires_ask(),
            latest_multirow_choices,
            latest_read_rows,
            has_task_db_read,
        }
    }

    #[must_use]
    pub fn ask_interaction_mode(&self, question: &str) -> crate::AskInteractionMode {
        self.classify_turn_ask_interaction_mode(usize::MAX, question)
    }

    #[must_use]
    pub fn classify_turn_ask_interaction_mode(
        &self,
        turn_index: usize,
        payload: &str,
    ) -> crate::AskInteractionMode {
        let TaskAskObservationState {
            saw_steward_ask_row,
            latest_multirow_choices,
            latest_read_rows,
            has_task_db_read,
        } = self.resolve_task_ask_observation_state(turn_index);

        let (choices, active_rows): (&[String], &[Vec<serde_json::Value>]) =
            if self.candidate_disambiguation_enabled {
                (
                    latest_multirow_choices.as_deref().unwrap_or(&[]),
                    latest_read_rows.as_slice(),
                )
            } else {
                (&[], &[])
            };

        crate::driver::resolve_ask_interaction_mode_from_observation_state(
            saw_steward_ask_row,
            choices,
            active_rows,
            has_task_db_read,
            payload,
        )
    }

    #[must_use]
    pub fn classify_coroutine_step_effect(
        &self,
        turn_index: usize,
        mode: Mode,
        payload: &str,
    ) -> crate::driver::CoroutineStepEffect {
        match mode {
            Mode::Probe | Mode::Read | Mode::Write => {
                crate::driver::CoroutineStepEffect::AutoContinueDb
            }
            Mode::Result | Mode::Abstain => crate::driver::CoroutineStepEffect::YieldToPrompt,
            Mode::Ask => {
                let gui_mode = self.classify_turn_ask_interaction_mode(turn_index, payload);
                match gui_mode {
                    crate::AskInteractionMode::StewardPermissionChoice { .. }
                    | crate::AskInteractionMode::MultiSelect { .. } => {
                        crate::driver::CoroutineStepEffect::SuspendForGuiChoice {
                            lead_in_prose: payload.trim().to_string(),
                            gui_mode,
                        }
                    }
                    crate::AskInteractionMode::ConversationalClarify => {
                        crate::driver::CoroutineStepEffect::YieldToPrompt
                    }
                }
            }
        }
    }

    pub(crate) fn push_context_span(
        &mut self,
        kind: ContextSpanKind,
        pen: PenOwner,
        producer_bank: Option<CartridgeBank>,
        consumer_bank: CartridgeBank,
        raw_text: String,
        token_ids: Vec<u32>,
    ) {
        let transition = match self.last_mounted_bank {
            Some(prev) if prev != consumer_bank => {
                self.cartridge_switch_count += 1;
                Some((prev, consumer_bank))
            }
            _ => None,
        };
        self.last_mounted_bank = Some(consumer_bank);

        let token_count = if token_ids.is_empty() {
            estimate_token_count(&raw_text)
        } else {
            token_ids.len()
        };
        let token_start = self.total_context_tokens;
        let token_end = token_start + token_count;
        self.total_context_tokens = token_end;

        let control_badges = extract_control_token_badges(&raw_text, &token_ids, kind.clone());
        let global_span_index = self.context_spans.len();
        self.context_spans.push(AnnotatedContextSpan {
            global_span_index,
            kind,
            pen,
            producer_bank,
            consumer_bank,
            cartridge_transition: transition,
            token_start,
            token_end,
            token_ids,
            raw_text,
            control_badges,
        });
    }

    pub fn record_user_prompt_envelope(
        &mut self,
        turn_index: usize,
        schema_header: &str,
        user_prompt: &str,
        full_envelope: &str,
        token_ids: Vec<u32>,
    ) {
        self.interactions.push(InteractionItem::UserPrompt {
            turn_index,
            schema_header: schema_header.to_string(),
            user_text: user_prompt.to_string(),
        });
        self.push_context_span(
            ContextSpanKind::UserPromptEnvelope { turn_index },
            PenOwner::HarnessPen,
            None,
            CartridgeBank::Base,
            full_envelope.to_string(),
            token_ids,
        );
    }

    pub fn record_completed_relay_turn(&mut self, turn_index: usize, turn: &CompletedRelayTurn) {
        let single_pen_ok = portia_core::relay::verify_single_pen_schedule(&turn.spans).is_ok();
        self.interactions.push(InteractionItem::AssistantTurn {
            turn_index,
            mode: turn.mode,
            route: turn.route,
            plan: turn.plan.clone(),
            payload: turn.payload.clone(),
            single_pen_verified: single_pen_ok,
        });
        for (idx, span) in turn.spans.iter().enumerate() {
            self.record_pen_span_record(turn_index, idx, span);
        }
    }

    pub fn begin_streaming_assistant_turn(&mut self, turn_index: usize) {
        self.interactions.push(InteractionItem::AssistantTurn {
            turn_index,
            mode: Mode::Read,
            route: RouteTag::ToDatabase,
            plan: String::new(),
            payload: String::new(),
            single_pen_verified: true,
        });
        self.left_viewport = PaneScrollViewport::new_follow_tail();
        self.right_viewport = PaneScrollViewport::new_follow_tail();
    }

    pub fn apply_relay_stream_event(&mut self, turn_index: usize, event: &RelayStreamEvent) {
        match event {
            RelayStreamEvent::HarnessPrefillSpan { span } => {
                self.record_pen_span_record(turn_index, span.step_index, span);
            }
            RelayStreamEvent::TrunkTokenStepped {
                selected_mode,
                span,
                plan_so_far,
                payload_so_far,
                ..
            } => {
                self.record_pen_span_record(turn_index, span.step_index, span);
                if let Some(InteractionItem::AssistantTurn {
                    mode,
                    route,
                    plan,
                    payload,
                    ..
                }) = self.interactions.iter_mut().rev().find(|item| {
                    matches!(
                        item,
                        InteractionItem::AssistantTurn {
                            turn_index: idx,
                            ..
                        } if *idx == turn_index
                    )
                }) {
                    if let Some(m) = selected_mode {
                        *mode = *m;
                        *route = m.route_tag();
                    }
                    *plan = plan_so_far.trim().to_string();
                    *payload = payload_so_far.trim().to_string();
                }
            }
        }
        self.left_viewport = PaneScrollViewport::new_follow_tail();
        self.right_viewport = PaneScrollViewport::new_follow_tail();
    }

    pub fn finalize_streaming_assistant_turn(
        &mut self,
        turn_index: usize,
        turn: &CompletedRelayTurn,
    ) {
        let single_pen_ok = portia_core::relay::verify_single_pen_schedule(&turn.spans).is_ok();
        if let Some(InteractionItem::AssistantTurn {
            mode,
            route,
            plan,
            payload,
            single_pen_verified,
            ..
        }) = self.interactions.iter_mut().rev().find(|item| {
            matches!(
                item,
                InteractionItem::AssistantTurn {
                    turn_index: idx,
                    ..
                } if *idx == turn_index
            )
        }) {
            *mode = turn.mode;
            *route = turn.route;
            *plan = turn.plan.clone();
            *payload = turn.payload.clone();
            *single_pen_verified = single_pen_ok;
        } else {
            self.interactions.push(InteractionItem::AssistantTurn {
                turn_index,
                mode: turn.mode,
                route: turn.route,
                plan: turn.plan.clone(),
                payload: turn.payload.clone(),
                single_pen_verified: single_pen_ok,
            });
        }

        for (idx, span) in turn.spans.iter().enumerate() {
            let already_recorded = self.context_spans.iter().any(|s| {
                matches!(
                    s.kind,
                    ContextSpanKind::RelayPenSpan {
                        turn_index: t_idx,
                        span_index: s_idx,
                        ..
                    } if t_idx == turn_index && s_idx == idx
                )
            });
            if !already_recorded {
                self.record_pen_span_record(turn_index, idx, span);
            }
        }
        self.left_viewport = PaneScrollViewport::new_follow_tail();
        self.right_viewport = PaneScrollViewport::new_follow_tail();
    }

    pub fn record_pen_span_record(
        &mut self,
        turn_index: usize,
        span_index: usize,
        span: &PenSpanRecord,
    ) {
        self.push_context_span(
            ContextSpanKind::RelayPenSpan {
                turn_index,
                span_index,
                phase: span.phase,
            },
            span.pen,
            span.producer_bank,
            span.consumer_bank,
            span.text.clone(),
            span.token_ids.clone(),
        );
    }

    pub fn record_observation(
        &mut self,
        turn_index: usize,
        route: RouteTag,
        obs: &DriverObservation,
        wire_chunk: &str,
        token_ids: Vec<u32>,
    ) {
        self.interactions.push(InteractionItem::Observation {
            turn_index,
            observation: obs.clone(),
        });
        self.push_context_span(
            ContextSpanKind::ToolResponseEnvelope {
                turn_index,
                event_id: obs.event_id.clone(),
                route,
            },
            PenOwner::HarnessPen,
            None,
            CartridgeBank::Base,
            wire_chunk.to_string(),
            token_ids,
        );
    }
}
