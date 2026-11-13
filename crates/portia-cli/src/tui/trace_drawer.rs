//! Power-user BPE token & `CartridgeBank` boundary inspector (`SidebarMode::RawTokenTrace`, `Ctrl+T` / `/trace`).

use super::model_scale::TuiModelScaleSpec;
use super::{
    annotate_raw_text_with_control_tokens, AnnotatedContextSpan, ContextSpanKind, TwoPaneTuiState,
};
use iocraft::prelude::{Color, MixedTextContent, Weight};
use portia_core::relay::RelayPhase;
use portia_core::{CartridgeBank, Mode};
use serde::{Deserialize, Serialize};

/// A contiguous epoch of tokens in the KV/DeltaNet context window consumed under a single
/// mounted [`CartridgeBank`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CartridgeEpoch {
    pub epoch_number: usize,
    pub turn_index: usize,
    pub bank: CartridgeBank,
    pub previous_bank: Option<CartridgeBank>,
    pub token_start: usize,
    pub token_end: usize,
    pub mount_reason: String,
    pub summary_action: String,
    pub spans: Vec<AnnotatedContextSpan>,
}

/// Summary card for a slot bank installed in the widened `M -> M + widened_slots` rack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadedCartridgeBankCard {
    pub bank_name: String,
    pub physical_slots: String,
    pub mechanism: String,
    pub is_currently_mounted: bool,
    pub mount_count: usize,
    pub tokens_processed: usize,
}

fn truncate_preview(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

fn describe_mount_trigger(
    first_span: &AnnotatedContextSpan,
    prev_bank: Option<CartridgeBank>,
    scale: &TuiModelScaleSpec,
) -> String {
    match (&first_span.kind, first_span.consumer_bank) {
        (ContextSpanKind::UserPromptEnvelope { turn_index }, CartridgeBank::Base) => {
            format!("Turn {turn_index} open: Prefilled user prompt under Base W_0")
        }
        (
            ContextSpanKind::RelayPenSpan {
                turn_index,
                phase: RelayPhase::TurnOpenPrefill,
                ..
            },
            CartridgeBank::OrientTrunk,
        ) => format!(
            "Turn {turn_index} @ '[MODE:': Mounted OrientTrunk {} router slots",
            scale.format_orient_slots()
        ),
        (
            ContextSpanKind::RelayPenSpan {
                turn_index,
                phase: RelayPhase::PostModePrefill,
                ..
            },
            CartridgeBank::ModePayload(mode),
        ) => format!(
            "Turn {turn_index} @ '\\n' (tid=198 after '{}]'): Switched {:?} -> ModePayload({}) ({})",
            mode.as_str(),
            prev_bank.unwrap_or(CartridgeBank::OrientTrunk),
            mode.as_str(),
            scale.format_payload_slots()
        ),
        (
            ContextSpanKind::RelayPenSpan {
                turn_index,
                phase: RelayPhase::TurnComplete,
                ..
            },
            CartridgeBank::Base,
        ) => format!(
            "Turn {turn_index} @ closing route tag: Detached {:?} -> Base W_0 for <|im_end|>",
            prev_bank.unwrap_or(CartridgeBank::Base)
        ),
        (
            ContextSpanKind::ToolResponseEnvelope {
                turn_index,
                event_id,
                ..
            },
            CartridgeBank::Base,
        ) => format!(
            "Turn {turn_index} obs {event_id}: Consumed <tool_response> into KV/DeltaNet under Base W_0"
        ),
        (_, CartridgeBank::OrientTrunk) => format!(
            "Mounted OrientTrunk ({})",
            scale.format_orient_slots()
        ),
        (_, CartridgeBank::ModePayload(mode)) => format!(
            "Mounted ModePayload({}) ({})",
            mode.as_str(),
            scale.format_payload_slots()
        ),
        (_, bank) => format!("Mounted {bank:?}"),
    }
}

fn describe_epoch_summary(spans: &[AnnotatedContextSpan], bank: CartridgeBank) -> String {
    match bank {
        CartridgeBank::OrientTrunk => spans
            .iter()
            .find(|s| {
                matches!(
                    s.kind,
                    ContextSpanKind::RelayPenSpan {
                        phase: RelayPhase::ModeLabel,
                        ..
                    }
                )
            })
            .map_or_else(
                || "Prefilled '[MODE:' under OrientTrunk".to_string(),
                |s| {
                    format!(
                        "Prefilled '[MODE:' (H=0) -> RouterPen emitted '{}'",
                        s.raw_text.trim()
                    )
                },
            ),
        CartridgeBank::ModePayload(mode) => {
            let (mut plan_p, mut pay_p) = (String::new(), String::new());
            for s in spans {
                if let ContextSpanKind::RelayPenSpan { phase, .. } = s.kind {
                    if phase == RelayPhase::PlanBody {
                        plan_p = s.raw_text.trim().to_string();
                    } else if phase == RelayPhase::PayloadBody {
                        pay_p = s.raw_text.lines().next().unwrap_or("").trim().to_string();
                    }
                }
            }
            if !pay_p.is_empty() {
                format!(
                    "{}: Plan {:?} -> {:?}",
                    mode.as_str(),
                    truncate_preview(&plan_p, 24),
                    truncate_preview(&pay_p, 34)
                )
            } else {
                format!("ModePayload({}) active", mode.as_str())
            }
        }
        CartridgeBank::Base => {
            if spans
                .iter()
                .any(|s| matches!(s.kind, ContextSpanKind::UserPromptEnvelope { .. }))
            {
                "User prompt prefill under Base W_0 (0 active slots)".to_string()
            } else if spans
                .iter()
                .any(|s| matches!(s.kind, ContextSpanKind::ToolResponseEnvelope { .. }))
            {
                "Turn close <|im_end|> + SQLite <tool_response> prefill (0 active slots)"
                    .to_string()
            } else {
                "Turn close <|im_end|> under un-steered Base W_0".to_string()
            }
        }
    }
}

/// Coalesce fine-grained [`AnnotatedContextSpan`]s into [`CartridgeEpoch`] blocks annotated with
/// `scale`'s physical layer & slot coordinates.
#[must_use]
pub fn coalesce_cartridge_epochs_for_scale(
    spans: &[AnnotatedContextSpan],
    scale: &TuiModelScaleSpec,
) -> Vec<CartridgeEpoch> {
    let mut epochs: Vec<CartridgeEpoch> = Vec::new();
    for span in spans {
        let span_turn = match &span.kind {
            ContextSpanKind::UserPromptEnvelope { turn_index }
            | ContextSpanKind::RelayPenSpan { turn_index, .. }
            | ContextSpanKind::ToolResponseEnvelope { turn_index, .. } => *turn_index,
        };
        if epochs
            .last()
            .is_some_and(|last| last.bank == span.consumer_bank && last.turn_index == span_turn)
        {
            let last = epochs.last_mut().expect("checked non-empty");
            last.token_end = span.token_end;
            last.spans.push(span.clone());
            last.summary_action = describe_epoch_summary(&last.spans, last.bank);
        } else {
            let prev_bank = epochs.last().map(|e| e.bank);
            let epoch_number = epochs.len() + 1;
            let mount_reason = describe_mount_trigger(span, prev_bank, scale);
            let sub_spans = vec![span.clone()];
            let summary_action = describe_epoch_summary(&sub_spans, span.consumer_bank);
            epochs.push(CartridgeEpoch {
                epoch_number,
                turn_index: span_turn,
                bank: span.consumer_bank,
                previous_bank: prev_bank,
                token_start: span.token_start,
                token_end: span.token_end,
                mount_reason,
                summary_action,
                spans: sub_spans,
            });
        }
    }
    epochs
}

/// Coalesce fine-grained [`AnnotatedContextSpan`]s using canonical `0.8B-Base` coordinates.
#[must_use]
pub fn coalesce_cartridge_epochs(spans: &[AnnotatedContextSpan]) -> Vec<CartridgeEpoch> {
    coalesce_cartridge_epochs_for_scale(spans, &TuiModelScaleSpec::default_08b_base())
}

/// Summarize the installed Cartridge Rack (`M -> M + widened_slots`) for `state`'s active
/// `Qwen3.5` model scale (`0.8B`, `2B`, `4B`, `9B`, `27B`) and live usage statistics.
#[must_use]
pub fn summarize_loaded_cartridge_rack(state: &TwoPaneTuiState) -> Vec<LoadedCartridgeBankCard> {
    let scale = state.model_scale_spec();
    let (mut b_m, mut b_t, mut o_m, mut o_t, mut p_m, mut p_t) = (0, 0, 0, 0, 0, 0);
    for ep in coalesce_cartridge_epochs_for_scale(&state.context_spans, &scale) {
        let n_tok = ep.token_end.saturating_sub(ep.token_start);
        match ep.bank {
            CartridgeBank::Base => {
                b_m += 1;
                b_t += n_tok;
            }
            CartridgeBank::OrientTrunk => {
                o_m += 1;
                o_t += n_tok;
            }
            CartridgeBank::ModePayload(_) => {
                p_m += 1;
                p_t += n_tok;
            }
        }
    }
    let last = state.last_mounted_bank;
    vec![
        LoadedCartridgeBankCard {
            bank_name: format!("BANK 0 · Base W_0 ({} Un-Steered Trunk)", scale.scale_badge),
            physical_slots: format!(
                "Slots: none ({} native SwiGLU + {} DeltaNet + {} GQA)",
                scale.base_intermediate_size, scale.deltanet_layers, scale.gqa_layers
            ),
            mechanism: "Carries user prompts, <tool_response>, and <|im_end|> with 0 drift"
                .to_string(),
            is_currently_mounted: last == Some(CartridgeBank::Base) || last.is_none(),
            mount_count: b_m,
            tokens_processed: b_t,
        },
        LoadedCartridgeBankCard {
            bank_name: "BANK 1 · OrientTrunk (Pointer-ISA Epistemic Mode Router)".to_string(),
            physical_slots: format!(
                "{} (Stiefel SO(d={}) coupled diodes -> BPE Mode Toks)",
                scale.format_orient_slots(),
                scale.hidden_dim
            ),
            mechanism:
                "Mounted at '[MODE:' -> routes PROBE/READ/ASK(Structured-GUI vs Chitchat)/WRITE/RESULT/ABSTAIN"
                    .to_string(),
            is_currently_mounted: last == Some(CartridgeBank::OrientTrunk),
            mount_count: o_m,
            tokens_processed: o_t,
        },
        LoadedCartridgeBankCard {
            bank_name: "BANK 2 · ModePayload (SQL / ModePayload(Ask · Structured-GUI) / ModePayload(Ask · Chitchat -> Base W_0))"
                .to_string(),
            physical_slots: format!(
                "{} (M: {} -> {}, Pointer-ISA Payload & Anti-Echo)",
                scale.format_payload_slots(),
                scale.base_intermediate_size,
                scale.widened_intermediate_size()
            ),
            mechanism: "Mounted at '\\n' (tid=198) -> detaches at '</sql>'/'</to_user>'"
                .to_string(),
            is_currently_mounted: matches!(last, Some(CartridgeBank::ModePayload(_))),
            mount_count: p_m,
            tokens_processed: p_t,
        },
    ]
}

fn style_control_token_spans(text: &str) -> Vec<MixedTextContent> {
    let (mut spans, mut rest) = (Vec::new(), text);
    while let Some(open_idx) = rest.find('«') {
        if open_idx > 0 {
            spans.push(MixedTextContent::new(&rest[..open_idx]));
        }
        let after_open = &rest[open_idx..];
        let Some(close_rel) = after_open.find('»') else {
            spans.push(MixedTextContent::new(after_open));
            return spans;
        };
        let badge_end = close_rel + '»'.len_utf8();
        spans.push(MixedTextContent::new(&after_open[..badge_end]).color(Color::Cyan));
        rest = &after_open[badge_end..];
    }
    if !rest.is_empty() || spans.is_empty() {
        spans.push(MixedTextContent::new(rest));
    }
    spans
}

/// Build styled [`MixedTextContent`] lines for `SidebarMode::RawTokenTrace` (`PANE 2: CONTEXT WINDOW`).
#[must_use]
pub fn build_right_context_window_lines(
    state: &TwoPaneTuiState,
    inner_width: usize,
) -> Vec<Vec<MixedTextContent>> {
    let scale = state.model_scale_spec();
    let sep_w = inner_width.saturating_sub(2).max(20);

    let mut lines = vec![
        vec![
            MixedTextContent::new(format!("Scale: {} ", scale.scale_badge))
                .color(Color::Magenta)
                .weight(Weight::Bold),
            MixedTextContent::new(format!(
                "({}) | Tokens: {} | Swaps: {}",
                scale.format_architecture_summary(),
                state.total_context_tokens,
                state.cartridge_switch_count
            ))
            .color(Color::White),
        ],
        vec![MixedTextContent::new(format!(
            "Rack: OrientTrunk {} · ModePayload {} · Bundle: {}",
            scale.format_orient_slots(),
            scale.format_payload_slots(),
            state.cartridge_label
        ))
        .color(Color::Grey)],
        vec![MixedTextContent::new("─".repeat(sep_w)).color(Color::Grey)],
    ];

    if state.context_spans.is_empty() {
        lines.push(vec![MixedTextContent::new(
            "Context window is empty (0 tokens).",
        )
        .color(Color::Grey)]);
        return lines;
    }

    for ep in &coalesce_cartridge_epochs_for_scale(&state.context_spans, &scale) {
        let banner = match (ep.previous_bank, ep.bank) {
            (None, CartridgeBank::Base) => format!(
                "──[>> LOADED Base W_0 ({} SwiGLU + {} DeltaNet + {} GQA) @ tok {:04}: {}]──",
                scale.base_intermediate_size,
                scale.deltanet_layers,
                scale.gqa_layers,
                ep.token_start,
                ep.mount_reason
            ),
            (Some(from), CartridgeBank::OrientTrunk) => format!(
                "──[>> SWAPPED {from:?} ──> OrientTrunk ({}) @ tok {:04}: {}]──",
                scale.format_orient_slots(),
                ep.token_start,
                ep.mount_reason
            ),
            (Some(from), CartridgeBank::ModePayload(m)) => format!(
                "──[>> SWAPPED {from:?} ──> ModePayload({}) ({}) @ tok {:04}: {}]──",
                m.as_str(),
                scale.format_payload_slots(),
                ep.token_start,
                ep.mount_reason
            ),
            (Some(from), CartridgeBank::Base) => format!(
                "──[<< DETACHED {from:?} ──> Base W_0 @ tok {:04}: {}]──",
                ep.token_start, ep.mount_reason
            ),
            (None, other) => format!(
                "──[>> LOADED {other:?} @ tok {:04}: {}]──",
                ep.token_start, ep.mount_reason
            ),
        };
        lines.push(vec![MixedTextContent::new(banner)
            .color(Color::Yellow)
            .weight(Weight::Bold)]);

        let (mut epoch_text, mut router_note) = (String::new(), None);
        for span in &ep.spans {
            if matches!(
                span.kind,
                ContextSpanKind::RelayPenSpan {
                    phase: RelayPhase::ModeLabel,
                    ..
                }
            ) {
                if let Some(m) = Mode::parse_flexible(span.raw_text.trim().trim_end_matches(']')) {
                    router_note = Some(format!(
                        "  <-- (RouterPen: first_tok={} | ']'=60)",
                        m.qwen35_first_token_id()
                    ));
                }
            }
            epoch_text.push_str(&span.raw_text);
        }
        let rendered =
            annotate_raw_text_with_control_tokens(&epoch_text, state.show_inline_token_ids);
        for (line_idx, raw_line) in rendered.lines().filter(|l| !l.is_empty()).enumerate() {
            let indented = format!("  {raw_line}");
            let mut spans = if state.show_inline_token_ids {
                style_control_token_spans(&indented)
            } else {
                vec![MixedTextContent::new(indented)]
            };
            if line_idx == 0 {
                if let Some(note) = &router_note {
                    spans.push(
                        MixedTextContent::new(note.clone())
                            .color(Color::Magenta)
                            .weight(Weight::Bold),
                    );
                }
            }
            lines.push(spans);
        }
    }
    lines
}
