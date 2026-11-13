use crate::tokenizer::{is_eos_token, is_ordinary_payload_token};
use crate::*;
use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    Matcher, ParserFactory,
};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    time::Instant,
};

type Result<T> = std::result::Result<T, GrammarHarnessError>;
fn invalid(error: impl std::fmt::Display) -> GrammarHarnessError {
    GrammarHarnessError::InvalidGrammar(error.to_string())
}
fn strict_schema(schema_value: &serde_json::Value) -> Result<()> {
    match schema_value {
        serde_json::Value::Object(map) => {
            if map
                .get("x-guidance")
                .and_then(|options| options.get("lenient"))
                .and_then(|lenient| lenient.as_bool())
                == Some(true)
            {
                return Err(invalid("x-guidance.lenient=true is not permitted"));
            }
            for child in map.values() {
                strict_schema(child)?;
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                strict_schema(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn healthy(matcher: &Matcher) -> Result<()> {
    if matcher.is_error() || !matcher.stop_reason().is_ok() {
        return Err(invalid(
            matcher
                .get_error()
                .unwrap_or_else(|| matcher.stop_reason().to_string()),
        ));
    }
    Ok(())
}
#[derive(Default)]
struct Stats {
    calls: usize,
    max: u64,
    total: u64,
}
struct Trace {
    verdict: TraceVerdict,
    stats: Stats,
}

impl GrammarVerifier {
    /// Compile both production and tokenization-invariance modes eagerly.
    pub fn new(harness: TokenizerHarness, grammar: GrammarUnderTest) -> Result<Self> {
        let compiled = catch_unwind(AssertUnwindSafe(|| -> Result<(Matcher, Matcher)> {
            let top_level = match &grammar {
                GrammarUnderTest::Lark(source) => TopLevelGrammar::from_lark(source.clone()),
                GrammarUnderTest::Regex(pattern) => {
                    let trailing_backslashes = pattern
                        .bytes()
                        .rev()
                        .take_while(|&byte| byte == b'\\')
                        .count();
                    if trailing_backslashes % 2 == 1 {
                        return Err(invalid("unescaped trailing backslash in regex"));
                    }
                    TopLevelGrammar::from_regex(pattern)
                }
                GrammarUnderTest::JsonSchema(schema_value) => {
                    strict_schema(schema_value)?;
                    TopLevelGrammar::from_json_schema(schema_value.clone())
                }
            };
            let mut factory = ParserFactory::new_simple(harness.tok_env()).map_err(invalid)?;
            factory.quiet();
            let (grammar_ir, mut lexer) = GrammarInit::Serialized(top_level)
                .to_internal(Some(harness.tok_env().clone()), factory.limits().clone())
                .map_err(invalid)?;
            lexer.no_forcing = false;
            let canonical = Matcher::new(factory.create_parser_from_init_default(
                GrammarInit::Internal(grammar_ir.clone(), lexer.clone()),
            ));
            healthy(&canonical)?;
            lexer.no_forcing = true;
            let unforced = Matcher::new(
                factory.create_parser_from_init_default(GrammarInit::Internal(grammar_ir, lexer)),
            );
            healthy(&unforced)?;
            Ok((canonical, unforced))
        }))
        .map_err(|_| invalid("grammar compilation panicked"))??;
        Ok(Self {
            harness,
            grammar,
            canonical: compiled.0,
            unforced: compiled.1,
        })
    }
    /// Tokenizer used by both verification modes.
    #[must_use]
    pub const fn harness(&self) -> &TokenizerHarness {
        &self.harness
    }
    /// Original grammar specification.
    #[must_use]
    pub const fn grammar(&self) -> &GrammarUnderTest {
        &self.grammar
    }

    fn trace(
        &self,
        tokens: &[u32],
        canonical: bool,
        duration: &mut impl FnMut(u64) -> u64,
    ) -> Result<Trace> {
        let mut matcher = if canonical {
            self.canonical.deep_clone()
        } else {
            self.unforced.deep_clone()
        };
        let trie = self.harness.tok_env().tok_trie();
        let mut stats = Stats::default();
        let mut ended = false;
        for (step_index, &token_id) in tokens.iter().enumerate() {
            healthy(&matcher)?;
            let start = Instant::now();
            let mask = matcher.compute_mask_or_eos().map_err(invalid)?;
            let elapsed_micros = duration(start.elapsed().as_micros().min(u64::MAX as u128) as u64);
            stats.calls += 1;
            stats.max = stats.max.max(elapsed_micros);
            stats.total = stats.total.saturating_add(elapsed_micros);
            healthy(&matcher)?;
            if ended || token_id as usize >= trie.vocab_size() || !mask.is_allowed(token_id) {
                let token_bytes = if (token_id as usize) < trie.vocab_size() {
                    trie.token(token_id).to_vec()
                } else {
                    Vec::new()
                };
                return Ok(Trace {
                    verdict: TraceVerdict::MaskRejectedAtStep {
                        step_index,
                        token_id,
                        token_bytes,
                    },
                    stats,
                });
            }
            if is_eos_token(trie, token_id) {
                ended = true;
            } else {
                matcher.consume_token(token_id).map_err(invalid)?;
                healthy(&matcher)?;
            }
        }
        healthy(&matcher)?;
        let accepted = ended || matcher.is_accepting().map_err(invalid)?;
        healthy(&matcher)?;
        let verdict = if accepted {
            TraceVerdict::Accepted {
                steps: tokens.len(),
                max_step_micros: stats.max,
                total_micros: stats.total,
            }
        } else {
            TraceVerdict::IncompleteAtEnd {
                steps: tokens.len(),
            }
        };
        Ok(Trace { verdict, stats })
    }
    fn check_partition(&self, text: &str, tokens: &[u32]) -> Result<()> {
        let trie = self.harness.tok_env().tok_trie();
        let mut bytes = Vec::new();
        for &token_id in tokens {
            if !is_ordinary_payload_token(trie, token_id) {
                return Err(GrammarHarnessError::PartitionBytesMismatch);
            }
            bytes.extend_from_slice(trie.token(token_id));
        }
        if bytes != text.as_bytes() {
            return Err(GrammarHarnessError::PartitionBytesMismatch);
        }
        Ok(())
    }
    fn greedy_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let tokens =
            catch_unwind(AssertUnwindSafe(|| self.harness.tokenize_str(text))).map_err(|_| {
                GrammarHarnessError::TokenizerInit("tokenizer encoding panicked".into())
            })?;
        self.check_partition(text, &tokens)?;
        Ok(tokens)
    }
    /// Verify arbitrary token splits with native forcing disabled. One final EOS is optional.
    pub fn verify_token_sequence(&self, tokens: &[u32]) -> Result<TraceVerdict> {
        Ok(self
            .trace(tokens, false, &mut |elapsed_micros| elapsed_micros)?
            .verdict)
    }
    /// Verify explicit tokens with production canonical forcing enabled.
    pub fn verify_token_sequence_canonical(&self, tokens: &[u32]) -> Result<TraceVerdict> {
        Ok(self
            .trace(tokens, true, &mut |elapsed_micros| elapsed_micros)?
            .verdict)
    }
    /// Verify native greedy encoding, exact payload bytes, and production masks.
    pub fn verify_string_greedy(&self, text: &str) -> Result<TraceVerdict> {
        self.verify_token_sequence_canonical(&self.greedy_tokens(text)?)
    }
    /// Verify an exact payload partition with native token forcing disabled.
    pub fn verify_string_under_partition(
        &self,
        text: &str,
        token_partition: &[u32],
    ) -> Result<TraceVerdict> {
        self.check_partition(text, token_partition)?;
        self.verify_token_sequence(token_partition)
    }
    /// Audit canonical traces. Budgets apply to accepted positive cases only.
    pub fn verify_batch_suite(&self, suite: &GrammarAuditSuite) -> Result<SuiteAuditReceipt> {
        self.batch(suite, &mut |elapsed_micros| elapsed_micros)
    }
    fn batch(
        &self,
        suite: &GrammarAuditSuite,
        duration: &mut impl FnMut(u64) -> u64,
    ) -> Result<SuiteAuditReceipt> {
        if suite.grammar != self.grammar {
            return Err(invalid("suite.grammar does not match verifier.grammar"));
        }
        let mut receipt = SuiteAuditReceipt {
            name: suite.name.clone(),
            accepted_verified: 0,
            rejected_verified: 0,
            total_mask_steps: 0,
            worst_step_micros: 0,
        };
        for (inputs, positive) in [(&suite.must_accept, true), (&suite.must_reject, false)] {
            for input in inputs {
                let trace = self.trace(&self.greedy_tokens(input)?, true, duration)?;
                receipt.total_mask_steps += trace.stats.calls;
                receipt.worst_step_micros = receipt.worst_step_micros.max(trace.stats.max);
                if positive {
                    if !trace.verdict.is_accepted() {
                        return Err(GrammarHarnessError::UnexpectedRejection {
                            input: input.clone(),
                            verdict: trace.verdict,
                        });
                    }
                    if trace.stats.max > suite.max_step_micros_budget {
                        return Err(GrammarHarnessError::StepLatencyExceeded {
                            input: input.clone(),
                            actual_micros: trace.stats.max,
                            budget_micros: suite.max_step_micros_budget,
                        });
                    }
                    receipt.accepted_verified += 1;
                } else {
                    if trace.verdict.is_accepted() {
                        return Err(GrammarHarnessError::UnexpectedAcceptance {
                            input: input.clone(),
                        });
                    }
                    receipt.rejected_verified += 1;
                }
            }
        }
        Ok(receipt)
    }
    #[cfg(test)]
    fn verify_batch_suite_with_timer<F: FnMut() -> u64>(
        &self,
        suite: &GrammarAuditSuite,
        mut timer: F,
    ) -> Result<SuiteAuditReceipt> {
        self.batch(suite, &mut |_| timer())
    }
}

#[cfg(test)]
#[test]
fn deterministic_timing() {
    let grammar = GrammarUnderTest::Regex("a+".into());
    let verifier = GrammarVerifier::new(
        TokenizerHarness::synthetic_byte_bpe(&[], b"EOS").unwrap(),
        grammar.clone(),
    )
    .unwrap();
    let mut suite = GrammarAuditSuite {
        name: "timing".into(),
        grammar,
        must_accept: vec!["aaa".into()],
        must_reject: vec!["b".into()],
        max_step_micros_budget: 100,
    };
    let mut ticks = [15, 250, 40].into_iter();
    assert!(matches!(
        verifier.verify_batch_suite_with_timer(&suite, || ticks.next().unwrap()),
        Err(GrammarHarnessError::StepLatencyExceeded {
            actual_micros: 250,
            budget_micros: 100,
            ..
        })
    ));
    suite.max_step_micros_budget = 300;
    let mut ticks = [15, 250, 40, 290].into_iter();
    let receipt = verifier
        .verify_batch_suite_with_timer(&suite, || ticks.next().unwrap())
        .unwrap();
    assert_eq!(
        (receipt.total_mask_steps, receipt.worst_step_micros),
        (4, 290)
    );
    assert_eq!(
        (receipt.accepted_verified, receipt.rejected_verified),
        (1, 1)
    );
}
