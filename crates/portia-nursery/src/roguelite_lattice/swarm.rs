//! Near-colliding entity identifier swarm generator and Hamming-distance oracle.

use portia_core::domain::CollisionLocus;
use serde::{Deserialize, Serialize};

/// A generated swarm of near-colliding entity identifiers (`IdentifierSwarmCollision`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentifierSwarm {
    pub target_identifier: String,
    pub target_index: usize,
    pub locus: CollisionLocus,
    pub identifiers: Vec<String>,
    pub min_hamming_distance: usize,
}

impl IdentifierSwarm {
    /// Returns `true` iff `candidate` matches `target_identifier` byte-for-byte.
    pub fn verify_exact_target_match(&self, candidate: &str) -> bool {
        candidate.trim() == self.target_identifier
    }
}

/// Compute character-level Hamming distance for equal-length strings (plus length delta if lengths differ).
pub fn char_hamming_distance(a: &str, b: &str) -> usize {
    let ca: Vec<char> = a.chars().collect();
    let cb: Vec<char> = b.chars().collect();
    let common = ca.len().min(cb.len());
    let mut diffs = ca.len().abs_diff(cb.len());
    for i in 0..common {
        if ca[i] != cb[i] {
            diffs += 1;
        }
    }
    diffs
}

/// Generate a swarm of `count` (`1..=250`) unique near-colliding slugs around `target_slug`
/// at the specified [`CollisionLocus`].
pub fn generate_identifier_swarm(
    target_slug: &str,
    count: usize,
    locus: CollisionLocus,
    seed: u64,
) -> IdentifierSwarm {
    let clamped_count = count.clamp(1, 250);
    let base = if target_slug.trim().is_empty() {
        "inc-7f3a-01a".to_string()
    } else {
        target_slug.trim().to_string()
    };

    let mut items = Vec::with_capacity(clamped_count);
    items.push(base.clone());

    let chars: Vec<char> = base.chars().collect();
    let len = chars.len();
    let alphabet: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    let mut step: u64 = 1;
    while items.len() < clamped_count {
        let mix = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(step.wrapping_mul(1442695040888963407));
        step += 1;

        let mut cand = chars.clone();
        let (pos_a, pos_b) = match locus {
            CollisionLocus::PrefixDiff => (0, (len / 4).max(1).min(len.saturating_sub(1))),
            CollisionLocus::MiddleHexDiff => {
                let mid = len / 2;
                (mid, (mid + 1).min(len.saturating_sub(1)))
            }
            CollisionLocus::FinalCharDiff => {
                let last = len.saturating_sub(1);
                (last, last.saturating_sub(1))
            }
        };

        let c1 = alphabet[((step as usize) + (mix as usize % 7)) % alphabet.len()] as char;
        cand[pos_a] = if c1 == chars[pos_a] {
            alphabet[((step as usize) + 1) % alphabet.len()] as char
        } else {
            c1
        };

        if step > 35 && pos_b != pos_a {
            let c2 = alphabet[((step as usize / 35) + ((mix >> 16) as usize % 11)) % alphabet.len()]
                as char;
            if c2 != chars[pos_b] {
                cand[pos_b] = c2;
            }
        }

        let s: String = cand.into_iter().collect();
        if s != base && !items.contains(&s) {
            items.push(s);
        }
    }

    let target_idx = if clamped_count == 1 {
        0
    } else {
        (seed as usize) % clamped_count
    };
    items.swap(0, target_idx);

    let min_hamming = if clamped_count == 1 {
        0
    } else {
        items
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != target_idx)
            .map(|(_, s)| char_hamming_distance(&base, s))
            .min()
            .unwrap_or(1)
    };

    IdentifierSwarm {
        target_identifier: base,
        target_index: target_idx,
        locus,
        identifiers: items,
        min_hamming_distance: min_hamming,
    }
}
