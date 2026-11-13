//! `Qwen3.5` BPE control-token badge extraction and inline raw-trace annotations (`Ctrl+T`).

use super::ContextSpanKind;
use portia_core::bpe::{
    TOKEN_IM_END, TOKEN_IM_START, TOKEN_THINK_CLOSE, TOKEN_THINK_OPEN, TURN_PREFIX_MODE_OPEN_IDS,
};
use portia_core::relay::RelayPhase;
use portia_core::Mode;
use serde::{Deserialize, Serialize};

/// Known `Qwen3.5` control / structural token annotation for the right-hand Context Window pane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlTokenBadge {
    pub token_id: u32,
    pub raw_piece: String,
    pub label: String,
    pub role: ControlTokenRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlTokenRole {
    ChatMlBoundary,
    ThinkDelimiter,
    ModeHeaderFraming,
    ModeDecisionToken,
    ModeContinuation,
    RouteDelimiter,
    NewlineAnchor,
    HealedMergedDelimiter,
}

pub(crate) fn estimate_token_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        (text.len().div_ceil(4)).max(1)
    }
}

/// Extract known `Qwen3.5` BPE control tokens and mode decision registers from a context span.
#[must_use]
pub fn extract_control_token_badges(
    raw_text: &str,
    token_ids: &[u32],
    kind: ContextSpanKind,
) -> Vec<ControlTokenBadge> {
    let mut badges = Vec::new();

    for &tid in token_ids {
        match tid {
            TOKEN_IM_START => badges.push(ControlTokenBadge {
                token_id: TOKEN_IM_START,
                raw_piece: "<|im_start|>".to_string(),
                label: "ChatML Turn Open".to_string(),
                role: ControlTokenRole::ChatMlBoundary,
            }),
            TOKEN_IM_END => badges.push(ControlTokenBadge {
                token_id: TOKEN_IM_END,
                raw_piece: "<|im_end|>".to_string(),
                label: "ChatML Turn Close".to_string(),
                role: ControlTokenRole::ChatMlBoundary,
            }),
            TOKEN_THINK_OPEN => badges.push(ControlTokenBadge {
                token_id: TOKEN_THINK_OPEN,
                raw_piece: "<think>".to_string(),
                label: "CoT Think Open".to_string(),
                role: ControlTokenRole::ThinkDelimiter,
            }),
            TOKEN_THINK_CLOSE => badges.push(ControlTokenBadge {
                token_id: TOKEN_THINK_CLOSE,
                raw_piece: "</think>".to_string(),
                label: "CoT Think Close".to_string(),
                role: ControlTokenRole::ThinkDelimiter,
            }),
            5140 => badges.push(ControlTokenBadge {
                token_id: 5140,
                raw_piece: " PRO".to_string(),
                label: "Mode::Probe First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            18104 => badges.push(ControlTokenBadge {
                token_id: 18104,
                raw_piece: " READ".to_string(),
                label: "Mode::Read First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            5574 => badges.push(ControlTokenBadge {
                token_id: 5574,
                raw_piece: " AS".to_string(),
                label: "Mode::Ask First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            37272 => badges.push(ControlTokenBadge {
                token_id: 37272,
                raw_piece: " WRITE".to_string(),
                label: "Mode::Write First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            27604 => badges.push(ControlTokenBadge {
                token_id: 27604,
                raw_piece: " RESULT".to_string(),
                label: "Mode::Result First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            13732 => badges.push(ControlTokenBadge {
                token_id: 13732,
                raw_piece: " AB".to_string(),
                label: "Mode::Abstain First-Tok".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            11261 | 42 | 759 | 6622 => badges.push(ControlTokenBadge {
                token_id: tid,
                raw_piece: match tid {
                    11261 => "BE",
                    42 => "K",
                    759 => "ST",
                    _ => "AIN",
                }
                .to_string(),
                label: "Mode Continuation".to_string(),
                role: ControlTokenRole::ModeContinuation,
            }),
            60 if matches!(
                kind,
                ContextSpanKind::RelayPenSpan {
                    phase: RelayPhase::ModeLabel,
                    ..
                }
            ) =>
            {
                badges.push(ControlTokenBadge {
                    token_id: 60,
                    raw_piece: "]".to_string(),
                    label: "Mode Header Close Bracket".to_string(),
                    role: ControlTokenRole::ModeHeaderFraming,
                })
            }
            271 => badges.push(ControlTokenBadge {
                token_id: 271,
                raw_piece: "\\n\\n".to_string(),
                label: "Merged Double-LF (BPE Healed -> 198)".to_string(),
                role: ControlTokenRole::HealedMergedDelimiter,
            }),
            248077..=248082 => badges.push(ControlTokenBadge {
                token_id: tid,
                raw_piece: format!("<|op_reg_{tid}|>"),
                label: "1-Token Orthogonal Opcode Register".to_string(),
                role: ControlTokenRole::ModeDecisionToken,
            }),
            _ => {}
        }
    }

    if badges.is_empty() {
        if raw_text.contains("<|im_start|>") {
            badges.push(ControlTokenBadge {
                token_id: TOKEN_IM_START,
                raw_piece: "<|im_start|>".to_string(),
                label: "ChatML Turn Open".to_string(),
                role: ControlTokenRole::ChatMlBoundary,
            });
        }
        if raw_text.contains("<think>") {
            badges.push(ControlTokenBadge {
                token_id: TOKEN_THINK_OPEN,
                raw_piece: "<think>".to_string(),
                label: "CoT Think Open".to_string(),
                role: ControlTokenRole::ThinkDelimiter,
            });
        }
        if raw_text.contains("[MODE:") {
            badges.push(ControlTokenBadge {
                token_id: TURN_PREFIX_MODE_OPEN_IDS[1],
                raw_piece: format!(
                    "[MODE: ({}, {}, {})",
                    TURN_PREFIX_MODE_OPEN_IDS[0],
                    TURN_PREFIX_MODE_OPEN_IDS[1],
                    TURN_PREFIX_MODE_OPEN_IDS[2]
                ),
                label: "HarnessPen H=0 Mode Anchor".to_string(),
                role: ControlTokenRole::ModeHeaderFraming,
            });
        }
        if matches!(
            kind,
            ContextSpanKind::RelayPenSpan {
                phase: RelayPhase::ModeLabel,
                ..
            }
        ) {
            let trimmed = raw_text.trim().trim_end_matches(']');
            if let Some(m) = Mode::parse_flexible(trimmed) {
                badges.push(ControlTokenBadge {
                    token_id: m.qwen35_first_token_id(),
                    raw_piece: format!(" {}", m.as_str()),
                    label: format!("Mode::{:?} (FirstTok={})", m, m.qwen35_first_token_id()),
                    role: ControlTokenRole::ModeDecisionToken,
                });
                badges.push(ControlTokenBadge {
                    token_id: 60,
                    raw_piece: "]".to_string(),
                    label: "Mode Close Bracket (tid=60)".to_string(),
                    role: ControlTokenRole::ModeHeaderFraming,
                });
            }
        }
        if raw_text.starts_with("\nPlan:") {
            badges.push(ControlTokenBadge {
                token_id: 198,
                raw_piece: "\\nPlan: (198, 19860, 25)".to_string(),
                label: "Cartridge-Switch Newline + Plan Anchor".to_string(),
                role: ControlTokenRole::NewlineAnchor,
            });
        }
        if raw_text.contains("</think>") {
            badges.push(ControlTokenBadge {
                token_id: TOKEN_THINK_CLOSE,
                raw_piece: "</think>".to_string(),
                label: "CoT Think Close".to_string(),
                role: ControlTokenRole::ThinkDelimiter,
            });
        }
        if raw_text.contains("<|im_end|>") {
            badges.push(ControlTokenBadge {
                token_id: TOKEN_IM_END,
                raw_piece: "<|im_end|>".to_string(),
                label: "ChatML Turn Close (Base Bank)".to_string(),
                role: ControlTokenRole::ChatMlBoundary,
            });
        }
    }

    badges
}

/// Annotate control tokens inline inside a raw context chunk so the user can visually see
/// exact BPE boundaries (`«248045:<|im_start|>»`, `«198:\n»`, `«60:]»`, etc.) in the Right Pane.
#[must_use]
pub fn annotate_raw_text_with_control_tokens(raw: &str, show_ids: bool) -> String {
    if !show_ids {
        return raw.to_string();
    }
    let mut s = raw.to_string();
    let mode_word_tid = TURN_PREFIX_MODE_OPEN_IDS[1];
    let replacements = [
        ("<|im_start|>", format!("«{TOKEN_IM_START}:<|im_start|>»")),
        ("<|im_end|>", format!("«{TOKEN_IM_END}:<|im_end|>»")),
        ("<think>", format!("«{TOKEN_THINK_OPEN}:<think>»")),
        ("</think>", format!("«{TOKEN_THINK_CLOSE}:</think>»")),
        ("[MODE:", format!("«58:[»«{mode_word_tid}:MODE»«25::»")),
        ("Plan:", "«19860:Plan»«25::»".to_string()),
    ];
    for (pat, rep) in replacements {
        s = s.replace(pat, &rep);
    }
    s.replace('\n', "«198:\\n»\n")
}
