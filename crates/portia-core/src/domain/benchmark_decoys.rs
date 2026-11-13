//! Opt-in synthetic nursery benchmark decoy/honeypot column classification helpers.
//!
//! **Scope Notice:** These helpers exist strictly for synthetic nursery test-oracle grading
//! and controlled schema-decoy experiments. They are **never** active inside runtime SQL
//! `llguidance` Lark grammars, which accept every physical column present in `PRAGMA table_info`.

use serde::{Deserialize, Serialize};

/// Synthetic benchmark prefix patterns identifying deprecated decoy columns in nursery test suites.
pub const DEPRECATED_DECOY_PREFIXES: &[&str] = &[
    "legacy_",
    "old_",
    "deprecated_",
    "obsolete_",
    "shadow_",
    "decoy_",
];

/// Synthetic benchmark suffix patterns identifying deprecated decoy columns in nursery test suites.
pub const DEPRECATED_DECOY_SUFFIXES: &[&str] = &[
    "_deprecated",
    "_old_v1",
    "_old_v2",
    "_old",
    "_legacy",
    "_shadow_decoy",
    "_decoy",
    "_obsolete",
    "_bak",
    "_unused",
];

/// Synthetic nursery test-oracle classification of a column identifier into either a live
/// canonical column or a quarantined decoy alias (e.g. `legacy_budget_cents_deprecated`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ColumnAliasClass {
    CanonicalLive,
    QuarantinedLegacyDecoy,
}

impl ColumnAliasClass {
    /// Return `Some("prefix:...")` or `Some("suffix:...")` if `name` matches a synthetic
    /// benchmark decoy pattern, or `None` if it is a live column.
    pub fn matched_decoy_pattern(name: &str) -> Option<String> {
        let lower = name
            .trim()
            .trim_matches(['`', '"', '\''])
            .to_ascii_lowercase();
        for pfx in DEPRECATED_DECOY_PREFIXES {
            if lower.starts_with(pfx) {
                return Some(format!("prefix:{pfx}"));
            }
        }
        for sfx in DEPRECATED_DECOY_SUFFIXES {
            if lower.ends_with(sfx) {
                return Some(format!("suffix:{sfx}"));
            }
        }
        if lower.contains("_legacy_") {
            return Some("infix:_legacy_".to_string());
        }
        None
    }

    /// Classify a column identifier for synthetic nursery test-oracle grading by inspecting
    /// whether it carries a decoy marker (`legacy_`, `_deprecated`, `_obsolete`, `_shadow_decoy`, `old_`).
    pub fn classify_column_name(name: &str) -> Self {
        if Self::matched_decoy_pattern(name).is_some() {
            ColumnAliasClass::QuarantinedLegacyDecoy
        } else {
            ColumnAliasClass::CanonicalLive
        }
    }

    #[inline]
    pub const fn is_live(self) -> bool {
        matches!(self, ColumnAliasClass::CanonicalLive)
    }
}

/// Opt-in synthetic nursery test-oracle helper returning `true` iff `name` matches a synthetic
/// decoy column naming pattern (`legacy_*`, `*_deprecated`, `archive_*_v1`, etc.).
///
/// Never invoked by runtime SQL grammar specialization (`EpistemicSqlGrammarCartridge`).
#[inline]
pub fn is_quarantined_decoy_column_name(name: &str) -> bool {
    let low = name
        .trim()
        .trim_matches(['`', '"', '\''])
        .to_ascii_lowercase();
    if !ColumnAliasClass::classify_column_name(name).is_live() {
        return true;
    }
    (low.starts_with("archive_")
        && (low.ends_with("_v1")
            || low.ends_with("_v2")
            || low.ends_with("_deprecated")
            || low.ends_with("_decoy")
            || low.ends_with("_legacy")
            || low.ends_with("_old")))
        || low.ends_with("_archive_v1")
        || low.ends_with("_archive_v2")
}
