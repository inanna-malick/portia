use crate::{GrammarHarnessError, TokenizerHarness};
use serde_json::Value;
use std::{collections::HashSet, path::Path, sync::Arc};
use toktrie::{TokRxInfo, TokTrie, TokenizerEnv};
use toktrie_hf_tokenizers::ByteTokenizer;

const EOS_ID: u32 = 256;
const EOS_NAMES: &[&str] = &[
    "</s>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<｜end▁of▁sentence｜>",
    "<eos>",
];

struct SyntheticBpeEnv {
    trie: TokTrie,
}

pub(crate) fn is_eos_token(trie: &TokTrie, token_id: u32) -> bool {
    trie.eos_tokens().contains(&token_id)
}

pub(crate) fn is_ordinary_payload_token(trie: &TokTrie, token_id: u32) -> bool {
    if token_id as usize >= trie.vocab_size() || is_eos_token(trie, token_id) {
        return false;
    }
    let token_bytes = trie.token(token_id);
    !(token_bytes.is_empty()
        || token_bytes.len() > 1 && token_bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER)
}

impl TokenizerEnv for SyntheticBpeEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, mut bytes: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        while !bytes.is_empty() {
            let (token, len) = self.trie.prefix_token_id(bytes);
            let (token, len) = if !is_ordinary_payload_token(&self.trie, token) {
                (u32::from(bytes[0]), 1)
            } else {
                (token, len)
            };
            tokens.push(token);
            bytes = &bytes[len..];
        }
        tokens
    }

    fn tokenize_is_canonical(&self) -> bool {
        // Longest-prefix matching defines this synthetic tokenizer's canonical split.
        true
    }
}

fn init_error(error: impl std::fmt::Display) -> GrammarHarnessError {
    GrammarHarnessError::TokenizerInit(error.to_string())
}

fn initialize<T>(
    f: impl FnOnce() -> Result<T, GrammarHarnessError>,
) -> Result<T, GrammarHarnessError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("tokenizer dependency panicked");
        init_error(message)
    })?
}

fn identified_eos(json: &Value) -> Result<u32, GrammarHarnessError> {
    let mut eos = None;
    for token in json["added_tokens"].as_array().into_iter().flatten() {
        if token["content"]
            .as_str()
            .is_some_and(|name| EOS_NAMES.contains(&name))
        {
            let id = token["id"]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| init_error("recognized EOS has no valid token ID"))?;
            // The adapter scans added tokens in ascending ID order: the last wins.
            eos = Some(eos.map_or(id, |previous: u32| previous.max(id)));
        }
    }
    eos.ok_or_else(|| init_error("tokenizer has no recognized EOS added token"))
}

impl TokenizerHarness {
    /// Access the underlying `TokEnv`.
    #[must_use]
    pub const fn tok_env(&self) -> &toktrie::TokEnv {
        &self.tok_env
    }

    /// Load an HF byte-level or byte-fallback tokenizer with an identifiable EOS.
    /// Padding and truncation are disabled so verification sees the complete input.
    pub fn from_hf_tokenizer_json(path: &Path) -> Result<Self, GrammarHarnessError> {
        let bytes = std::fs::read(path).map_err(init_error)?;
        let json: Value = serde_json::from_slice(&bytes).map_err(init_error)?;
        let eos = identified_eos(&json)?;
        initialize(|| {
            let mut tokenizer = ByteTokenizer::from_json_bytes(&bytes).map_err(init_error)?;
            let info = tokenizer.tokrx_info();
            if eos >= info.vocab_size || info.tok_eos != eos {
                return Err(init_error("tokenizer EOS metadata is inconsistent"));
            }
            tokenizer.hf_tokenizer.with_padding(None);
            tokenizer
                .hf_tokenizer
                .with_truncation(None)
                .map_err(init_error)?;
            let tok_env = tokenizer.into_tok_env(None).map_err(init_error)?;
            if !tok_env.tok_trie().is_special_token(eos) {
                return Err(init_error(
                    "EOS token must have a special-token representation",
                ));
            }
            Ok(Self { tok_env })
        })
    }

    /// Create a longest-prefix byte tokenizer. IDs 0..=255 are the corresponding
    /// bytes, EOS is 256, and distinct multi-byte merges follow in input order.
    /// Empty and single-byte merges are ignored; marker-prefixed merges are errors.
    pub fn synthetic_byte_bpe(
        extra_merged_tokens: &[Vec<u8>],
        eos_token_bytes: &[u8],
    ) -> Result<Self, GrammarHarnessError> {
        if eos_token_bytes.is_empty() {
            return Err(init_error("EOS spelling must not be empty"));
        }
        let mut eos = eos_token_bytes.to_vec();
        if eos[0] != TokTrie::SPECIAL_TOKEN_MARKER || eos.len() == 1 {
            eos.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
        }
        let mut words: Vec<_> = (0..=255).map(|byte| vec![byte]).collect();
        words.push(eos);
        let mut seen = HashSet::new();
        for token in extra_merged_tokens.iter().filter(|token| token.len() > 1) {
            if token[0] == TokTrie::SPECIAL_TOKEN_MARKER {
                return Err(init_error(
                    "merged tokens must not start with the special marker",
                ));
            }
            if seen.insert(token) {
                words.push(token.clone());
            }
        }
        let vocab_size = u32::try_from(words.len()).map_err(init_error)?;
        // TokTrie reserves the all-ones value of its 24-bit token-ID field.
        if vocab_size > 0x00ff_ffff {
            return Err(init_error(
                "vocabulary exceeds toktrie's token-ID representation",
            ));
        }
        words.iter().try_fold(0_u32, |total, word| {
            let len = u32::try_from(word.len()).map_err(init_error)?;
            total
                .checked_add(len)
                .ok_or_else(|| init_error("token bytes exceed toktrie's offset representation"))
        })?;
        initialize(|| {
            let trie = TokTrie::from(&TokRxInfo::new(vocab_size, EOS_ID), &words);
            Ok(Self {
                tok_env: Arc::new(SyntheticBpeEnv { trie }),
            })
        })
    }

    /// Encode using the environment's canonical tokenizer without adding EOS.
    /// HF normalization and literal special-token recognition remain observable;
    /// string verification rejects encodings that do not preserve ordinary bytes.
    pub fn tokenize_str(&self, text: &str) -> Vec<u32> {
        self.tok_env.tokenize(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    #[test]
    fn secondary_eos_is_terminal_and_never_ordinary_payload() {
        use crate::{GrammarUnderTest, GrammarVerifier, TraceVerdict};
        use proptest::{
            strategy::{Strategy, ValueTree},
            test_runner::{Config, TestRunner},
        };
        let base = TokenizerHarness::synthetic_byte_bpe(&[b"end".to_vec()], b"<eos>").unwrap();
        let trie = base.tok_env().tok_trie().with_eos_tokens(&[256, 257]);
        assert!(is_eos_token(&trie, 257));
        assert!(!is_ordinary_payload_token(&trie, 257));
        assert!(is_ordinary_payload_token(&trie, 255));
        assert!(!is_ordinary_payload_token(&trie, u32::MAX));
        let harness = TokenizerHarness {
            tok_env: Arc::new(toktrie::TokEnvWithTrie::new(base.tok_env().clone(), trie)),
        };
        let verifier =
            GrammarVerifier::new(harness.clone(), GrammarUnderTest::Regex("ok".into())).unwrap();
        for eos in [256, 257] {
            assert!(verifier
                .verify_token_sequence(&[111, 107, eos])
                .unwrap()
                .is_accepted());
            assert!(matches!(
                verifier
                    .verify_token_sequence(&[111, 107, eos, 97])
                    .unwrap(),
                TraceVerdict::MaskRejectedAtStep { step_index: 3, .. }
            ));
        }
        assert!(matches!(
            verifier.verify_string_under_partition("okend", &[111, 107, 257]),
            Err(GrammarHarnessError::PartitionBytesMismatch)
        ));
        assert!(matches!(
            verifier.verify_string_greedy("okend"),
            Err(GrammarHarnessError::PartitionBytesMismatch)
        ));
        let strategy = crate::strategies::arb_token_partition_for_bytes(harness, b"end".to_vec());
        let mut runner = TestRunner::new(Config {
            failure_persistence: None,
            ..Config::default()
        });
        for _ in 0..128 {
            assert_eq!(
                strategy.new_tree(&mut runner).unwrap().current(),
                vec![101, 110, 100]
            );
        }
    }

    fn hf_fixture() -> Value {
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": 3, "content": "<eos>", "special": true,
                "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false,
                "vocab": {"a": 0, "b": 1, "ab": 2, "<eos>": 3},
                "merges": ["a b"]}
        })
    }

    fn load_json(json: &Value) -> Result<TokenizerHarness, GrammarHarnessError> {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        serde_json::to_writer(&mut file, json).unwrap();
        TokenizerHarness::from_hf_tokenizer_json(file.path())
    }

    #[test]
    fn synthetic_bytes_ids_and_longest_prefix_are_stable() {
        let harness = TokenizerHarness::synthetic_byte_bpe(
            &[
                b"ab".to_vec(),
                b"abc".to_vec(),
                b"ab".to_vec(),
                vec![],
                vec![b'a'],
            ],
            b"<eos>",
        )
        .unwrap();
        let trie = harness.tok_env().tok_trie();
        assert_eq!(trie.vocab_size(), 259);
        assert_eq!(trie.eos_token(), EOS_ID);
        for byte in 0..=255 {
            assert_eq!(trie.token(byte), &[byte as u8]);
        }
        assert_eq!(harness.tokenize_str("abcabz"), [258, 257, b'z' as u32]);
        assert!(harness.tok_env().tokenize_is_canonical());
    }

    #[test]
    fn eos_spelling_is_not_content_and_marker_only_eos_is_distinct() {
        for eos in [&b"<eos>"[..], &b"\xff<eos>"[..], &b"\xff"[..]] {
            let harness = TokenizerHarness::synthetic_byte_bpe(&[], eos).unwrap();
            let trie = harness.tok_env().tok_trie();
            assert!(trie.is_special_token(EOS_ID));
            assert_ne!(trie.token(EOS_ID), trie.token(255));
            let encoded = harness.tokenize_str("<eos>");
            assert!(!encoded.contains(&EOS_ID));
            assert_eq!(trie.decode(&encoded), b"<eos>");
        }
        let harness = TokenizerHarness::synthetic_byte_bpe(&[b"abcd".to_vec()], b"ab").unwrap();
        assert_eq!(harness.tok_env().tokenize_bytes(b"\xffabcd"), [255, 257]);
    }

    #[test]
    fn synthetic_rejects_empty_eos_and_special_merges() {
        for result in [
            TokenizerHarness::synthetic_byte_bpe(&[], b""),
            TokenizerHarness::synthetic_byte_bpe(&[b"\xff<special>".to_vec()], b"<eos>"),
        ] {
            assert!(matches!(result, Err(GrammarHarnessError::TokenizerInit(_))));
        }
    }

    #[test]
    fn synthetic_merges_accept_partial_utf8() {
        let harness =
            TokenizerHarness::synthetic_byte_bpe(&[vec![0xf0, 0x9f], vec![0xa6, 0x80]], b"<eos>")
                .unwrap();
        assert_eq!(harness.tokenize_str("🦀"), [257, 258]);
    }

    #[test]
    fn hf_loads_real_bpe_and_retains_literal_special_tokens() {
        let harness = load_json(&hf_fixture()).unwrap();
        assert_eq!(harness.tokenize_str("abab"), [2, 2]);
        assert_eq!(harness.tokenize_str("<eos>"), [3]);
        assert_eq!(harness.tok_env().eos_token(), 3);
    }

    #[test]
    fn hf_verification_rejects_normalized_bytes_and_literal_eos() {
        let mut json = hf_fixture();
        json["normalizer"] = json!({"type": "Lowercase"});
        let harness = load_json(&json).unwrap();
        assert_eq!(harness.tokenize_str("AB"), [2]);
        let verifier = crate::GrammarVerifier::new(
            harness,
            crate::GrammarUnderTest::Regex("[a-z]+".to_string()),
        )
        .unwrap();
        for text in ["AB", "<eos>"] {
            assert!(matches!(
                verifier.verify_string_greedy(text),
                Err(GrammarHarnessError::PartitionBytesMismatch)
            ));
        }
    }

    #[test]
    fn hf_eos_choice_matches_the_adapters_deterministic_order() {
        let mut json = hf_fixture();
        json["model"]["vocab"]["</s>"] = json!(4);
        let mut additional = json["added_tokens"][0].clone();
        additional["id"] = json!(4);
        additional["content"] = json!("</s>");
        json["added_tokens"]
            .as_array_mut()
            .unwrap()
            .insert(0, additional);
        assert_eq!(load_json(&json).unwrap().tok_env().eos_token(), 4);
    }

    #[test]
    fn hf_disables_padding_and_truncation() {
        let mut json = hf_fixture();
        json["truncation"] = json!({"direction": "Right", "max_length": 1,
            "strategy": "LongestFirst", "stride": 0});
        json["padding"] = json!({"strategy": {"Fixed": 5}, "direction": "Right",
            "pad_to_multiple_of": null, "pad_id": 3, "pad_type_id": 0,
            "pad_token": "<eos>"});
        let harness = load_json(&json).unwrap();
        assert_eq!(harness.tokenize_str("abab"), [2, 2]);
        assert_eq!(harness.tokenize_str("a"), [0]);
    }

    #[test]
    fn hf_rejects_missing_eos_and_unsupported_decoder() {
        let mut missing = hf_fixture();
        missing["added_tokens"] = json!([]);
        let mut unsupported = hf_fixture();
        unsupported["decoder"] = json!({"type": "WordPiece", "prefix": "##", "cleanup": true});
        for json in [missing, unsupported] {
            assert!(matches!(
                load_json(&json),
                Err(GrammarHarnessError::TokenizerInit(_))
            ));
        }
    }

    #[test]
    fn hf_maps_file_and_native_panics_to_initialization_errors() {
        let mut malformed = tempfile::NamedTempFile::new().unwrap();
        malformed.write_all(b"not JSON").unwrap();
        assert!(matches!(
            TokenizerHarness::from_hf_tokenizer_json(malformed.path()),
            Err(GrammarHarnessError::TokenizerInit(_))
        ));
        let mut json = hf_fixture();
        json["decoder"] = json!({"type": "ByteFallback"});
        json["model"]["vocab"]["<0xZZ>"] = json!(4);
        assert!(matches!(
            load_json(&json),
            Err(GrammarHarnessError::TokenizerInit(_))
        ));
        assert!(matches!(
            TokenizerHarness::from_hf_tokenizer_json(Path::new("/nonexistent/tokenizer.json")),
            Err(GrammarHarnessError::TokenizerInit(_))
        ));
    }
}
