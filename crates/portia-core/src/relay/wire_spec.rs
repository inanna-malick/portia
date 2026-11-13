//! Canonical wire chunk normalization witness (`LotlNormalizedWireChunk`), KV prefill
//! dispositions, and constrained grammar cartridge specifications (`GrammarCartridgeSpec`, `SpanCartridge`).

use super::normalize_lotl_wire_chunk;
use crate::{CartridgeBank, Mode};
use serde::{Deserialize, Serialize};

/// Opaque witness newtype guaranteeing that an outbound worker wire chunk has been
/// canonicalized through [`normalize_lotl_wire_chunk`] before crossing the JSON-over-stdio
/// boundary.
///
/// The inner `String` field is private so unnormalized strings cannot be wrapped directly
/// outside this module:
///
/// ```compile_fail
/// use portia_core::LotlNormalizedWireChunk;
///
/// let _unverified = LotlNormalizedWireChunk("unnormalized".to_string());
/// ```
///
/// Callers must construct values via [`LotlNormalizedWireChunk::normalize`] (and `serde`
/// deserialization likewise canonicalizes through [`LotlNormalizedWireChunk::normalize`]):
///
/// ```
/// use portia_core::{LotlNormalizedWireChunk, Mode};
///
/// let chunk = LotlNormalizedWireChunk::normalize("\nPlan: ", Some(Mode::Read));
/// assert_eq!(chunk.as_str(), "\nPlan:");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LotlNormalizedWireChunk(String);

pub(super) fn is_user_prompt_envelope(raw: &str) -> bool {
    raw.starts_with("<|im_start|>user\n") && !raw.starts_with("<|im_start|>user\n<tool_response>\n")
}

impl<'de> Deserialize<'de> for LotlNormalizedWireChunk {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        if is_user_prompt_envelope(&raw) {
            Ok(Self(raw))
        } else {
            Ok(Self::normalize(&raw, None))
        }
    }
}

impl LotlNormalizedWireChunk {
    /// Wrap a `<|im_start|>user\n...<|im_end|>\n` user prompt envelope **verbatim** without
    /// rewriting any tags or substrings inside user-supplied content.
    #[must_use]
    pub fn from_user_prompt_envelope(user_prompt_envelope: impl Into<String>) -> Self {
        Self(user_prompt_envelope.into())
    }

    /// Canonicalize `raw_text` under `selected_mode` into a verified Pretrain-LOTL wire chunk.
    pub fn normalize(raw_text: &str, selected_mode: Option<Mode>) -> Self {
        Self(normalize_lotl_wire_chunk(raw_text, selected_mode))
    }

    /// Borrow the canonicalized wire string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the witness wrapper and return the owned canonicalized `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl AsRef<str> for LotlNormalizedWireChunk {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Explicit KV / DeltaNet session cache disposition required on every worker `Prefill` request
/// so session reset vs. anaphoric follow-up retention is decided solely in Rust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvPrefillDisposition {
    /// Clear `past_key_values` before prefilling (fresh session or explicit `/reset`).
    ResetSessionKv,
    /// Preserve `past_key_values` across turns in a stateful REPL / TUI session.
    PreserveStatefulSessionKv,
    /// Append an intra-turn chunk (`HarnessPen` framing, `<tool_response>`, or continuation)
    /// into the active turn's live `past_key_values`.
    AppendIntraTurnKv,
}

impl KvPrefillDisposition {
    /// Whether the worker must clear `past_key_values` before consuming this prefill chunk.
    pub const fn should_reset_kv(self) -> bool {
        match self {
            Self::ResetSessionKv => true,
            Self::PreserveStatefulSessionKv | Self::AppendIntraTurnKv => false,
        }
    }
}

/// Prompt-blind `H = 0` syntax/schema grammar cartridge co-mounted alongside
/// [`CartridgeBank`] across `ModeLabel`, `PlanBody`, and `PayloadBody`.
pub trait PayloadGrammarAutomaton {
    /// Closed set of epistemically legal [`Mode`]s at `[MODE:` given current environment provenance
    /// (with zero access to `user_prompt`).
    fn legal_modes(&self) -> Vec<Mode> {
        Mode::ALL.to_vec()
    }

    /// Return the declarative [`GrammarCartridgeSpec`] (`LarkGrammar`, `JsonSchema`, `LiteralSet`,
    /// or `Unconstrained`) for `mode` during `PayloadBody`.
    fn grammar_cartridge_spec(&self, _mode: Mode) -> GrammarCartridgeSpec {
        GrammarCartridgeSpec::Unconstrained
    }
}

/// Declarative `H = 0` context-specific constrained grammar specification co-mounted with a
/// [`CartridgeBank`] (`H > 0` weight surgery) for the duration of a syntax span `[t_0, t_e)`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "spec", rename_all = "snake_case")]
pub enum GrammarCartridgeSpec {
    /// Free-form natural language (e.g. `PlanBody` or prose `Result` / `Ask` / `Abstain`).
    #[default]
    Unconstrained,
    /// Context-free grammar in `llguidance` Lark EBNF syntax (`start: ...`),
    /// loaded from a `.lark` grammar file and parameterized with live symbols (`PRAGMA table_info` / AST scope).
    LarkGrammar(String),
    /// JSON Schema constraint (e.g. `Ask · Structured-GUI` widgets or typed JSON reports).
    JsonSchema(serde_json::Value),
    /// Closed finite set of allowed literal strings/continuations (e.g. `ModeLabel`).
    LiteralSet(Vec<String>),
}

impl GrammarCartridgeSpec {
    pub const fn is_constrained(&self) -> bool {
        !matches!(self, Self::Unconstrained)
    }

    /// Lower any constrained [`GrammarCartridgeSpec`] into its canonical `llguidance` Lark string
    /// (`None` iff `Unconstrained`).
    #[must_use]
    pub fn to_llguidance_lark(&self) -> Option<String> {
        match self {
            Self::Unconstrained => None,
            Self::LarkGrammar(src) => Some(src.clone()),
            Self::JsonSchema(schema) => Some(format!("start: %json {schema}")),
            Self::LiteralSet(items) => {
                if items.is_empty() {
                    return None;
                }
                let alts: Vec<String> = items
                    .iter()
                    .map(|s| {
                        let escaped = s
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\r', "\\r");
                        format!("\"{escaped}\"")
                    })
                    .collect();
                Some(format!("start: {}", alts.join(" | ")))
            }
        }
    }
}

/// A coupled Portia Span Cartridge: `H > 0` Weight Surgery paired with an optional `H = 0`
/// Context-Specific Constrained Grammar (`llguidance` Lark EBNF / JSON Schema).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanCartridge {
    pub weight_bank: CartridgeBank,
    pub grammar: GrammarCartridgeSpec,
}

impl SpanCartridge {
    pub const fn new(weight_bank: CartridgeBank, grammar: GrammarCartridgeSpec) -> Self {
        Self {
            weight_bank,
            grammar,
        }
    }

    pub const fn unconstrained(weight_bank: CartridgeBank) -> Self {
        Self {
            weight_bank,
            grammar: GrammarCartridgeSpec::Unconstrained,
        }
    }
}

/// Unconstrained `H = 0` grammar cartridge (`grammar_cartridge_spec` always returns `Unconstrained`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnconstrainedGrammarCartridge;

impl PayloadGrammarAutomaton for UnconstrainedGrammarCartridge {}
