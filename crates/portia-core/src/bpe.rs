//! BPE token-healing & Qwen3.5 boundary-crossing token table.
//!
//! Encodes the exact 248,320-token Qwen3.5 (0.8B / 4B / 27B) vocabulary constants at the
//! `[MODE:` commitment cursor, resolves space-prefixed (`" READ"`) vs bare (`"READ"`) mode
//! token emissions, and splits boundary-straddling tokens (such as merged `\n\n` = `271` at
//! the end of `Plan:`) cleanly without aborting the turn.

use crate::{CoreProtocolError, Mode};
use serde::{Deserialize, Serialize};

/// Exact Qwen3.5 BPE token IDs for `[MODE:` (`'[' = 58`, `'MODE' = 22098`, `':' = 25`).
/// Note: `vocab[": "]` is `None` in Qwen3.5; `[MODE:` must NOT include a trailing space
/// so that single-token space-prefixed mode IDs (`5140`, `18104`, `5574`, `37272`, `27604`, `13732`)
/// fire directly at cursor `0` of `RouterPen`.
pub const TURN_PREFIX_MODE_OPEN_IDS: [u32; 3] = [58, 22098, 25];

pub const TOKEN_OPEN_BRACKET: u32 = 58; // "["
pub const TOKEN_MODE_WORD: u32 = 22098; // "MODE"
pub const TOKEN_COLON: u32 = 25; // ":"
pub const TOKEN_SPACE: u32 = 220; // " "
pub const TOKEN_CLOSE_BRACKET: u32 = 60; // "]"
pub const TOKEN_SPACE_CLOSE_BRACKET: u32 = 2205; // " ]"
pub const TOKEN_NEWLINE: u32 = 198; // "\n"
pub const TOKEN_DOUBLE_NEWLINE: u32 = 271; // "\n\n"
pub const TOKEN_PLAN_WORD: u32 = 19860; // "Plan"
pub const TOKEN_COLON_CLOSE_TAG: u32 = 7518; // ":</"

/// Pretrain Living-Off-The-Land (`LOTL`) Qwen3.5 ChatML & Tool-Call single-token constants (`tokenizer_config.json`):
pub const TOKEN_IM_START: u32 = 248045; // "<|im_start|>"
pub const TOKEN_IM_END: u32 = 248046; // "<|im_end|>"
pub const TOKEN_SYSTEM: u32 = 8678; // "system"
pub const TOKEN_USER: u32 = 846; // "user"
pub const TOKEN_ASSISTANT: u32 = 74455; // "assistant"
pub const TOKEN_THINK_OPEN: u32 = 248068; // "<think>"
pub const TOKEN_THINK_CLOSE: u32 = 248069; // "</think>"
pub const TOKEN_TOOL_CALL_OPEN: u32 = 248058; // "<tool_call>"
pub const TOKEN_TOOL_CALL_CLOSE: u32 = 248059; // "</tool_call>"
pub const TOKEN_TOOL_RESPONSE_OPEN: u32 = 248066; // "<tool_response>"
pub const TOKEN_TOOL_RESPONSE_CLOSE: u32 = 248067; // "</tool_response>"

/// Space-prefixed canonical first-token IDs in Qwen3.5 when prefill ends at `[MODE:`.
pub const TOKEN_SPACE_PRO: u32 = 5140; // " PRO"
pub const TOKEN_SPACE_READ: u32 = 18104; // " READ"
pub const TOKEN_SPACE_AS: u32 = 5574; // " AS"
pub const TOKEN_SPACE_WRITE: u32 = 37272; // " WRITE"
pub const TOKEN_SPACE_RESULT: u32 = 27604; // " RESULT"
pub const TOKEN_SPACE_REPORT: u32 = 42616; // " REPORT" (legacy alias -> Mode::Result)
pub const TOKEN_SPACE_AB: u32 = 13732; // " AB"

/// Bare (no-leading-space) mode first-token IDs in Qwen3.5 (emitted if model omits space
/// after `[MODE:` or if a caller accidentally prefilled `[MODE: ` with `TOKEN_SPACE = 220`).
pub const TOKEN_BARE_PRO: u32 = 8844; // "PRO"
pub const TOKEN_BARE_READ: u32 = 9277; // "READ"
pub const TOKEN_BARE_ASK: u32 = 7150; // "ASK"
pub const TOKEN_BARE_AS: u32 = 1846; // "AS"
pub const TOKEN_BARE_WRITE: u32 = 31695; // "WRITE"
pub const TOKEN_BARE_RESULT: u32 = 13694; // "RESULT"
pub const TOKEN_BARE_REPORT: u32 = 75454; // "REPORT"
pub const TOKEN_BARE_AB: u32 = 1803; // "AB"

/// Mode continuation token IDs in Qwen3.5.
pub const TOKEN_CONT_BE: u32 = 11261; // "BE" (completes " PRO" + "BE" -> " PROBE")
pub const TOKEN_CONT_K: u32 = 42; // "K"  (completes " AS" + "K" -> " ASK")
pub const TOKEN_CONT_ST: u32 = 759; // "ST" (completes " AB" + "ST" + "AIN")
pub const TOKEN_CONT_AIN: u32 = 6622; // "AIN"

/// Canonical 6-mode first-token array in `Mode::ALL` order (`[Probe, Read, Ask, Write, Result, Abstain]`).
pub const CANONICAL_MODE_FIRST_TOKENS: [u32; 6] = [
    TOKEN_SPACE_PRO,
    TOKEN_SPACE_READ,
    TOKEN_SPACE_AS,
    TOKEN_SPACE_WRITE,
    TOKEN_SPACE_RESULT,
    TOKEN_SPACE_AB,
];

/// Resolves a Qwen3.5 first-token ID (either canonical space-prefixed or bare no-space)
/// after `[MODE:` into its target `Mode` and whether continuation tokens are still required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModeFirstTokenMatch {
    pub mode: Mode,
    pub had_leading_space: bool,
    pub expected_continuations: &'static [u32],
}

/// Resolve either a space-prefixed or bare Qwen3.5 token ID emitted at the `[MODE:` cursor.
pub fn resolve_qwen35_mode_first_token(token_id: u32) -> Option<ModeFirstTokenMatch> {
    match token_id {
        TOKEN_SPACE_PRO => Some(ModeFirstTokenMatch {
            mode: Mode::Probe,
            had_leading_space: true,
            expected_continuations: &[TOKEN_CONT_BE],
        }),
        TOKEN_BARE_PRO => Some(ModeFirstTokenMatch {
            mode: Mode::Probe,
            had_leading_space: false,
            expected_continuations: &[TOKEN_CONT_BE],
        }),
        TOKEN_SPACE_READ => Some(ModeFirstTokenMatch {
            mode: Mode::Read,
            had_leading_space: true,
            expected_continuations: &[],
        }),
        TOKEN_BARE_READ => Some(ModeFirstTokenMatch {
            mode: Mode::Read,
            had_leading_space: false,
            expected_continuations: &[],
        }),
        TOKEN_SPACE_AS => Some(ModeFirstTokenMatch {
            mode: Mode::Ask,
            had_leading_space: true,
            expected_continuations: &[TOKEN_CONT_K],
        }),
        TOKEN_BARE_AS => Some(ModeFirstTokenMatch {
            mode: Mode::Ask,
            had_leading_space: false,
            expected_continuations: &[TOKEN_CONT_K],
        }),
        TOKEN_BARE_ASK => Some(ModeFirstTokenMatch {
            mode: Mode::Ask,
            had_leading_space: false,
            expected_continuations: &[],
        }),
        TOKEN_SPACE_WRITE => Some(ModeFirstTokenMatch {
            mode: Mode::Write,
            had_leading_space: true,
            expected_continuations: &[],
        }),
        TOKEN_BARE_WRITE => Some(ModeFirstTokenMatch {
            mode: Mode::Write,
            had_leading_space: false,
            expected_continuations: &[],
        }),
        TOKEN_SPACE_RESULT | TOKEN_SPACE_REPORT => Some(ModeFirstTokenMatch {
            mode: Mode::Result,
            had_leading_space: true,
            expected_continuations: &[],
        }),
        TOKEN_BARE_RESULT | TOKEN_BARE_REPORT => Some(ModeFirstTokenMatch {
            mode: Mode::Result,
            had_leading_space: false,
            expected_continuations: &[],
        }),
        TOKEN_SPACE_AB => Some(ModeFirstTokenMatch {
            mode: Mode::Abstain,
            had_leading_space: true,
            expected_continuations: &[TOKEN_CONT_ST, TOKEN_CONT_AIN],
        }),
        TOKEN_BARE_AB => Some(ModeFirstTokenMatch {
            mode: Mode::Abstain,
            had_leading_space: false,
            expected_continuations: &[TOKEN_CONT_ST, TOKEN_CONT_AIN],
        }),
        _ => None,
    }
}

/// Complete canonical Qwen3.5 token sequence for `<space><MODE>]` (excluding `\n` = 198)
/// consumed under `CartridgeBank::OrientTrunk`.
pub fn canonical_mode_header_tokens_under_orient(mode: Mode) -> Vec<u32> {
    let mut out = Vec::with_capacity(4);
    out.push(mode.qwen35_first_token_id());
    out.extend_from_slice(mode.qwen35_continuation_token_ids());
    out.push(TOKEN_CLOSE_BRACKET);
    out
}

/// Result of healing a potentially boundary-crossing token or text fragment at a structural
/// delimiter (`']'` for mode close, or `'\n'` for plan line close).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryHealSplit {
    /// Bytes belonging to the current semantic span (up to and including the delimiter if applicable).
    pub head_before_delimiter: String,
    /// The matched delimiter itself (`"]"` or `"\n"`).
    pub delimiter: String,
    /// How many leading bytes of `upcoming_framing` were already absorbed by the tail of the emitted token,
    /// or healed via benign whitespace normalization (e.g. merged `\n\n` = `271` at the end of `Plan:`).
    pub consumed_framing_prefix_bytes: usize,
    /// Remaining suffix of `upcoming_framing` that `HarnessPen` must still prefill.
    pub remaining_framing_to_prefill: String,
    /// True if benign trailing whitespace (such as the second `\n` in token `271` `\n\n`) was absorbed.
    pub healed_double_newline: bool,
}

/// Heal a token emission that closes a delimiter (`delimiter` = `"]"` or `"\n"`) right before
/// a deterministic `HarnessPen` framing string (`upcoming_framing`, e.g. `"\nPlan:"` or `"</think>\n<sql>\n"`).
///
/// Handles two boundary-crossing BPE cases:
/// 1. If a token straddles `delimiter` and a prefix of `upcoming_framing` (e.g. `"]\n"` before `"\nPlan:"`),
///    it consumes the matching prefix of `upcoming_framing` and returns the remainder (`"Plan:"`).
/// 2. If `PayloadPen` ends the `Plan:` line by emitting Qwen3.5 token `271` (`"\n\n"`), where
///    `delimiter == "\n"` and `upcoming_framing` starts with `"</think>\n..."`, extra trailing
///    whitespace after the plan-closing `\n` is absorbed cleanly without failing boundary alignment.
pub fn heal_delimiter_crossing(
    emitted_chunk: &str,
    delimiter: &str,
    upcoming_framing: &str,
) -> Result<Option<BoundaryHealSplit>, CoreProtocolError> {
    let Some(pos) = emitted_chunk.find(delimiter) else {
        return Ok(None);
    };
    let head = &emitted_chunk[..pos];
    let tail = &emitted_chunk[pos + delimiter.len()..];

    if tail.is_empty() {
        return Ok(Some(BoundaryHealSplit {
            head_before_delimiter: head.to_string(),
            delimiter: delimiter.to_string(),
            consumed_framing_prefix_bytes: 0,
            remaining_framing_to_prefill: upcoming_framing.to_string(),
            healed_double_newline: false,
        }));
    }

    if let Some(stripped) = upcoming_framing.strip_prefix(tail) {
        return Ok(Some(BoundaryHealSplit {
            head_before_delimiter: head.to_string(),
            delimiter: delimiter.to_string(),
            consumed_framing_prefix_bytes: tail.len(),
            remaining_framing_to_prefill: stripped.to_string(),
            healed_double_newline: false,
        }));
    }

    // Special token-healing case: Qwen3.5 token 271 ("\n\n") or trailing whitespace right after
    // the newline closing `Plan: <text>\n` before `</think>\n<sql>` or `</think>\n<to_user>`.
    let trimmed_tail = tail.trim_start_matches(['\n', '\r', ' ', '\t']);
    if delimiter == "\n" && (trimmed_tail.is_empty() || upcoming_framing.starts_with(trimmed_tail))
    {
        let prefix_len = trimmed_tail.len();
        return Ok(Some(BoundaryHealSplit {
            head_before_delimiter: head.to_string(),
            delimiter: delimiter.to_string(),
            consumed_framing_prefix_bytes: prefix_len,
            remaining_framing_to_prefill: upcoming_framing[prefix_len..].to_string(),
            healed_double_newline: true,
        }));
    }

    Err(CoreProtocolError::BoundaryViolation(format!(
        "token tail {:?} after delimiter {:?} does not match upcoming framing {:?}",
        tail, delimiter, upcoming_framing
    )))
}

/// Heal a raw mode header buffer (between `[MODE:` and `]`) into a canonical `Mode`.
/// Accepts both canonical space-prefixed (`" READ"`, `" PROBE"`) and bare (`"READ"`, `"PROBE"`, `"REPORT"`)
/// emissions without failing on missing/extra space around the mode word.
pub fn heal_mode_label_buffer(label_before_bracket: &str) -> Result<Mode, CoreProtocolError> {
    Mode::parse_flexible(label_before_bracket)
        .ok_or_else(|| CoreProtocolError::InvalidModeLabel(label_before_bracket.to_string()))
}

/// Check whether a partial text buffer after `[MODE:` (before `]` has been seen) is still a valid
/// prefix of at least one canonical or bare mode label (including `"REPORT"`).
pub fn is_valid_mode_label_prefix(partial: &str) -> bool {
    if partial.is_empty() || partial == " " {
        return true;
    }
    let trimmed = partial.strip_prefix(' ').unwrap_or(partial);
    const CANDIDATES: [&str; 7] = [
        "PROBE", "READ", "ASK", "WRITE", "RESULT", "REPORT", "ABSTAIN",
    ];
    CANDIDATES.iter().any(|cand| cand.starts_with(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_first_token_resolution_handles_space_and_bare_tokens() {
        for mode in Mode::ALL {
            let tid = mode.qwen35_first_token_id();
            let resolved =
                resolve_qwen35_mode_first_token(tid).expect("canonical token must resolve");
            assert_eq!(resolved.mode, mode);
            assert!(resolved.had_leading_space);
            assert_eq!(
                resolved.expected_continuations,
                mode.qwen35_continuation_token_ids()
            );
        }

        // Bare tokens (when model omits space after `[MODE:`) also heal cleanly:
        assert_eq!(
            resolve_qwen35_mode_first_token(TOKEN_BARE_READ)
                .unwrap()
                .mode,
            Mode::Read
        );
        assert_eq!(
            resolve_qwen35_mode_first_token(TOKEN_BARE_PRO)
                .unwrap()
                .mode,
            Mode::Probe
        );
        assert_eq!(
            resolve_qwen35_mode_first_token(TOKEN_BARE_REPORT)
                .unwrap()
                .mode,
            Mode::Result
        );
        assert_eq!(
            resolve_qwen35_mode_first_token(TOKEN_SPACE_REPORT)
                .unwrap()
                .mode,
            Mode::Result
        );
    }

    #[test]
    fn boundary_healer_handles_merged_double_newline_271_after_plan() {
        let upcoming = "</think>\n<sql>\n";
        let split = heal_delimiter_crossing("Check candidate rows.\n\n", "\n", upcoming)
            .expect("must not fail on token 271 double newline")
            .expect("must find delimiter");
        assert_eq!(split.head_before_delimiter, "Check candidate rows.");
        assert!(split.healed_double_newline);
        assert_eq!(split.remaining_framing_to_prefill, upcoming);
    }

    #[test]
    fn boundary_healer_handles_straddled_bracket_newline() {
        let upcoming = crate::relay::POST_MODE_FRAMING;
        let split = heal_delimiter_crossing(" READ]\n", "]", upcoming)
            .expect("must succeed")
            .expect("must find bracket");
        assert_eq!(split.head_before_delimiter, " READ");
        assert_eq!(split.consumed_framing_prefix_bytes, 1);
        assert_eq!(split.remaining_framing_to_prefill, "Plan:");
    }
}
