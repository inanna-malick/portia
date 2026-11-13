//! `lark-bpe-proptest` — Pure-Rust `llguidance` grammar verification and
//! straddled-boundary BPE `proptest` fuzzing library.
//!
//! ```
//! use lark_bpe_proptest::{GrammarUnderTest, GrammarVerifier, TokenizerHarness};
//! let tokenizer = TokenizerHarness::synthetic_byte_bpe(&[b"hello".to_vec()], b"<eos>")?;
//! let verifier = GrammarVerifier::new(tokenizer, GrammarUnderTest::Regex("hello".into()))?;
//! assert!(verifier.verify_string_greedy("hello")?.is_accepted());
//! let bytes: Vec<u32> = b"hello".iter().map(|&b| u32::from(b)).collect();
//! assert!(verifier.verify_string_under_partition("hello", &bytes)?.is_accepted());
//! # Ok::<(), lark_bpe_proptest::GrammarHarnessError>(())
//! ```

#![forbid(unsafe_code)]

#[cfg(test)]
mod properties;
pub mod strategies;
#[cfg(test)]
mod tests;
mod tokenizer;
mod verifier;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use toktrie::TokEnv;

/// Grammar specification language compiled via `llguidance`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarUnderTest {
    /// `.lark` EBNF grammar specification.
    Lark(String),
    /// JSON Schema subset supported by llguidance strict compilation.
    JsonSchema(serde_json::Value),
    /// Regular expression constraining the entire span.
    Regex(String),
}

/// Outcome of stepping a concrete token sequence through a compiled grammar matcher.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceVerdict {
    /// Every token in the sequence was allowed by the step mask AND EOS/stop was legal at the end.
    Accepted {
        steps: usize,
        max_step_micros: u64,
        total_micros: u64,
    },
    /// A token at `step_index` was disallowed by the grammar's token bitmask (or matcher already stopped).
    MaskRejectedAtStep {
        step_index: usize,
        token_id: u32,
        token_bytes: Vec<u8>,
    },
    /// All prefix tokens were allowed, but EOS/stop was not legal after the final token (truncated prefix).
    IncompleteAtEnd { steps: usize },
}

impl TraceVerdict {
    #[inline]
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

/// Declarative positive + negative test batch with per-step latency budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarAuditSuite {
    pub name: String,
    pub grammar: GrammarUnderTest,
    pub must_accept: Vec<String>,
    pub must_reject: Vec<String>,
    pub max_step_micros_budget: u64,
}

/// Summary receipt returned by [`GrammarVerifier::verify_batch_suite`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuiteAuditReceipt {
    pub name: String,
    pub accepted_verified: usize,
    pub rejected_verified: usize,
    pub total_mask_steps: usize,
    pub worst_step_micros: u64,
}

/// Errors returned during tokenizer construction, grammar compilation, or suite verification.
#[derive(Debug, Error)]
pub enum GrammarHarnessError {
    #[error("tokenizer initialization error: {0}")]
    TokenizerInit(String),

    #[error("invalid grammar specification: {0}")]
    InvalidGrammar(String),

    #[error("token partition does not decode to expected UTF-8 string bytes")]
    PartitionBytesMismatch,

    #[error("expected suite input {input:?} to be ACCEPTED, but got {verdict:?}")]
    UnexpectedRejection {
        input: String,
        verdict: TraceVerdict,
    },

    #[error("expected suite input {input:?} to be REJECTED, but it was ACCEPTED")]
    UnexpectedAcceptance { input: String },

    #[error("step mask latency {actual_micros}us exceeded budget {budget_micros}us on {input:?}")]
    StepLatencyExceeded {
        input: String,
        actual_micros: u64,
        budget_micros: u64,
    },
}

/// Wrapper around `toktrie::TokEnv` supporting both real HuggingFace `tokenizer.json` files
/// and synthetic byte+merge BPE vocabularies for zero-disk `proptest` shrinking.
#[derive(Clone)]
pub struct TokenizerHarness {
    tok_env: TokEnv,
}

/// Compiled `llguidance` grammar bound to a [`TokenizerHarness`].
pub struct GrammarVerifier {
    harness: TokenizerHarness,
    grammar: GrammarUnderTest,
    canonical: llguidance::Matcher,
    unforced: llguidance::Matcher,
}
