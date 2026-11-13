//! Shrinkable adversarial vocabularies and byte-exact vocabulary partitions.

use crate::{tokenizer::is_ordinary_payload_token, TokenizerHarness};
use proptest::prelude::*;
use proptest::strategy::NewTree;
use proptest::test_runner::TestRunner;

/// Generate up to `max_merges` distinct multi-byte tokens around delimiters.
///
/// The supplied delimiters supplement the standard JSON, statement, and closing-tag
/// boundaries. Empty delimiters are ignored. Direct spans and one-sided straddles
/// receive more weight than two-sided straddles; grammar-specific tests should force a
/// known crossing token into their tested sequence to establish actual coverage.
pub fn arb_adversarial_bpe_merges(
    delimiters: Vec<Vec<u8>>,
    max_merges: usize,
) -> impl Strategy<Value = Vec<Vec<u8>>> {
    let mut seeds: Vec<Vec<u8>> = ["{", "}", "\"", ":", ";", "\n", "</", "{\"", "\":", ";\n"]
        .into_iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    seeds.extend(delimiters.into_iter().filter(|s| !s.is_empty()));
    seeds.sort();
    seeds.dedup();
    let context: Vec<Vec<u8>> = ["a", "0", " ", "\"", "}", ";", "\n", "€", "🦀", "無"]
        .into_iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let direct: Vec<_> = seeds
        .iter()
        .filter(|seed| seed.len() >= 2)
        .cloned()
        .collect();
    let delimiter = proptest::sample::select(seeds);
    let flank = proptest::sample::select(context);
    let join = |(mut left, right): (Vec<u8>, Vec<u8>)| {
        left.extend(right);
        left
    };
    let merge = prop_oneof![
        5 => proptest::sample::select(direct),
        2 => (flank.clone(), delimiter.clone()).prop_map(join),
        2 => (delimiter.clone(), flank.clone()).prop_map(join),
        1 => (flank.clone(), delimiter, flank).prop_map(|(mut left, middle, right)| {
            left.extend(middle);
            left.extend(right);
            left
        }),
    ];
    proptest::collection::vec(merge, 0..=max_merges).prop_map(|mut merges| {
        merges.retain(|merge| merge.len() >= 2);
        merges.sort();
        merges.dedup();
        merges
    })
}

#[derive(Debug)]
struct Unpartitionable;

impl Strategy for Unpartitionable {
    type Tree = Just<Vec<u32>>;
    type Value = Vec<u32>;

    fn new_tree(&self, _: &mut TestRunner) -> NewTree<Self> {
        Err("input has no complete ordinary-token vocabulary partition".into())
    }
}

/// Generate complete non-EOS partitions of `bytes` using `harness`'s vocabulary.
///
/// Empty input produces an empty partition. If nonempty input cannot be covered,
/// [`Strategy::new_tree`] returns an error instead of inventing a partial result.
/// Special and empty tokens are excluded; the singleton raw byte `0xff` is kept.
/// Each visited offset chooses uniformly among its viable outgoing token edges.
/// Every shrink remains byte-exact because all choices reach the end of the input.
pub fn arb_token_partition_for_bytes(
    harness: TokenizerHarness,
    bytes: Vec<u8>,
) -> BoxedStrategy<Vec<u32>> {
    if bytes.is_empty() {
        return Just(Vec::new()).boxed();
    }
    let trie = harness.tok_env().tok_trie();
    let mut by_first: [Vec<(u32, &[u8])>; 256] = std::array::from_fn(|_| Vec::new());
    for id in 0..trie.vocab_size() as u32 {
        let token = trie.token(id);
        if is_ordinary_payload_token(trie, id) {
            by_first[token[0] as usize].push((id, token));
        }
    }
    let mut edges = vec![Vec::new(); bytes.len() + 1];
    let mut reachable = vec![false; bytes.len() + 1];
    reachable[bytes.len()] = true;
    for offset in (0..bytes.len()).rev() {
        for &(id, token) in &by_first[bytes[offset] as usize] {
            if bytes[offset..].starts_with(token) && reachable[offset + token.len()] {
                edges[offset].push((token.len(), id));
            }
        }
        edges[offset].sort_unstable();
        reachable[offset] = !edges[offset].is_empty();
    }
    if !reachable[0] {
        return Unpartitionable.boxed();
    }
    let len = bytes.len();
    let choices: Vec<_> = edges[..len]
        .iter()
        .map(|outgoing| 0..outgoing.len().max(1))
        .collect();
    choices
        .prop_map(move |choices| {
            let mut tokens = Vec::new();
            let mut offset = 0;
            while offset < len {
                let options = &edges[offset];
                let (length, token) = options[choices[offset]];
                tokens.push(token);
                offset += length;
            }
            tokens
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::ValueTree;
    use std::sync::Arc;
    use toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

    fn sparse_env(words: &[&[u8]]) -> TokenizerHarness {
        let mut words: Vec<_> = words.iter().map(|w| w.to_vec()).collect();
        let eos = words.len() as u32;
        words.push(b"\xff<eos>".to_vec());
        let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, eos), &words);
        TokenizerHarness {
            tok_env: Arc::new(ApproximateTokEnv::new(trie)),
        }
    }

    fn runner() -> TestRunner {
        TestRunner::new(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })
    }

    #[test]
    fn unreachable_input_fails_generation_and_empty_input_succeeds() {
        let env = sparse_env(&[b"ab"]);
        assert!(arb_token_partition_for_bytes(env.clone(), b"a".to_vec())
            .new_tree(&mut runner())
            .is_err());
        assert!(arb_token_partition_for_bytes(env, Vec::new())
            .new_tree(&mut runner())
            .expect("empty partition")
            .current()
            .is_empty());
    }

    #[test]
    fn dead_end_prefixes_are_removed_before_sampling() {
        let strategy =
            arb_token_partition_for_bytes(sparse_env(&[b"ab", b"a", b"bc"]), b"abc".to_vec());
        for _ in 0..128 {
            assert_eq!(
                strategy.new_tree(&mut runner()).unwrap().current(),
                vec![1, 2]
            );
        }
    }

    #[test]
    fn special_tokens_cannot_supply_ordinary_bytes() {
        let env = sparse_env(&[b"\xffhidden"]);
        assert!(arb_token_partition_for_bytes(env, b"\xffhidden".to_vec())
            .new_tree(&mut runner())
            .is_err());
    }

    #[test]
    fn singleton_ff_is_a_raw_byte_partition() {
        let env = TokenizerHarness::synthetic_byte_bpe(&[], b"<eos>").unwrap();
        let strategy = arb_token_partition_for_bytes(env, vec![0xff]);
        assert_eq!(
            strategy.new_tree(&mut runner()).unwrap().current(),
            vec![255]
        );
    }

    #[test]
    fn merge_samples_frequently_shorten_json_and_sql_partitions() {
        use proptest::test_runner::{RngAlgorithm, TestRng};
        let mut runner = TestRunner::new_with_rng(
            ProptestConfig {
                failure_persistence: None,
                ..ProptestConfig::default()
            },
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = arb_adversarial_bpe_merges(vec![b";\n</sql>".to_vec()], 12);
        for text in ["{\"a\":123}", "<sql>SELECT a FROM b;\n</sql>"] {
            let mut shortened = 0;
            for _ in 0..512 {
                let merges = strategy.new_tree(&mut runner).unwrap().current();
                assert!(merges.iter().all(|merge| merge.len() >= 2));
                let harness = TokenizerHarness::synthetic_byte_bpe(&merges, b"<eos>").unwrap();
                // Measure vocabulary hits independently of the random partitioner's
                // deliberate choice to sometimes use byte fallback despite a hit.
                let partition = harness.tokenize_str(text);
                shortened += usize::from(partition.len() < text.len());
            }
            assert!(
                shortened * 5 >= 512 * 3,
                "{text:?}: {shortened}/512 partitions used merges"
            );
        }
    }
}
