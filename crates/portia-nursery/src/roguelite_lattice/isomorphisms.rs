//! Qualitative algebraic surface isomorphisms, trajectory archetypes, and steward agenda steps.

use portia_core::domain::{LockState, PolicyState};
use serde::{Deserialize, Serialize};

/// Elements of the symmetric group $S_3$ acting on the 3 wave positions `(0, 1, 2)`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum WavePermutation {
    /// Identity permutation `(0, 1, 2)`.
    #[default]
    Identity012,
    /// Adjacent transposition `(0, 2, 1)`: swaps waves 2 and 3.
    SwapLast021,
    /// Adjacent transposition `(1, 0, 2)`: swaps waves 1 and 2.
    SwapFirst102,
    /// 3-cycle `(1, 2, 0)`.
    Cycle120,
    /// 3-cycle `(2, 0, 1)`.
    Cycle201,
    /// Full reversal transposition `(2, 1, 0)`.
    Reverse210,
}

impl WavePermutation {
    pub const ALL: [WavePermutation; 6] = [
        Self::Identity012,
        Self::SwapLast021,
        Self::SwapFirst102,
        Self::Cycle120,
        Self::Cycle201,
        Self::Reverse210,
    ];

    /// Permutation index array `[σ(0), σ(1), σ(2)]`.
    #[inline]
    pub const fn indices(&self) -> [usize; 3] {
        match self {
            Self::Identity012 => [0, 1, 2],
            Self::SwapLast021 => [0, 2, 1],
            Self::SwapFirst102 => [1, 0, 2],
            Self::Cycle120 => [1, 2, 0],
            Self::Cycle201 => [2, 0, 1],
            Self::Reverse210 => [2, 1, 0],
        }
    }

    /// Construct a [`WavePermutation`] from a 3-element index array in `{0, 1, 2}`.
    #[inline]
    pub const fn from_indices(indices: [usize; 3]) -> Option<Self> {
        match indices {
            [0, 1, 2] => Some(Self::Identity012),
            [0, 2, 1] => Some(Self::SwapLast021),
            [1, 0, 2] => Some(Self::SwapFirst102),
            [1, 2, 0] => Some(Self::Cycle120),
            [2, 0, 1] => Some(Self::Cycle201),
            [2, 1, 0] => Some(Self::Reverse210),
            _ => None,
        }
    }

    /// Canonical short code (`"012"`, `"021"`, `"102"`, `"120"`, `"201"`, `"210"`).
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Identity012 => "012",
            Self::SwapLast021 => "021",
            Self::SwapFirst102 => "102",
            Self::Cycle120 => "120",
            Self::Cycle201 => "201",
            Self::Reverse210 => "210",
        }
    }

    /// Apply this $S_3$ permutation to an arbitrary 3-element array `[T; 3]`.
    #[inline]
    pub fn apply<T>(&self, items: [T; 3]) -> [T; 3] {
        let [w0, w1, w2] = items;
        match self {
            Self::Identity012 => [w0, w1, w2],
            Self::SwapLast021 => [w0, w2, w1],
            Self::SwapFirst102 => [w1, w0, w2],
            Self::Cycle120 => [w1, w2, w0],
            Self::Cycle201 => [w2, w0, w1],
            Self::Reverse210 => [w2, w1, w0],
        }
    }

    /// Group composition `(self ∘ other)` in $S_3$.
    #[inline]
    pub const fn compose(self, other: Self) -> Self {
        let a = self.indices();
        let b = other.indices();
        let composed = [b[a[0]], b[a[1]], b[a[2]]];
        match Self::from_indices(composed) {
            Some(perm) => perm,
            None => Self::Identity012,
        }
    }

    /// Group inverse $\sigma^{-1} \in S_3$.
    #[inline]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Identity012 => Self::Identity012,
            Self::SwapLast021 => Self::SwapLast021,
            Self::SwapFirst102 => Self::SwapFirst102,
            Self::Cycle120 => Self::Cycle201,
            Self::Cycle201 => Self::Cycle120,
            Self::Reverse210 => Self::Reverse210,
        }
    }
}

/// Primary-key polymorphism isomorphism across nursery schemas.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrimaryKeyIsomorphism {
    /// Affine shifted integer lattice (`base_offset + row_index * stride`).
    ShiftedInteger { base_offset: i64, stride: i64 },
    /// Swarm UUID slug primary-key representation.
    SwarmUuidSlugPrimaryKey,
    /// Composite tenant-scoped key (`<tenant_code>:<base_offset + row_index>`).
    CompositeTenantKey {
        tenant_code: String,
        base_offset: i64,
    },
}

impl Default for PrimaryKeyIsomorphism {
    fn default() -> Self {
        Self::ShiftedInteger {
            base_offset: 1000,
            stride: 1,
        }
    }
}

impl PrimaryKeyIsomorphism {
    /// Compute the deterministic collision-free integer primary key for `row_index`.
    pub fn resolve_row_pk(&self, row_index: usize) -> i64 {
        let idx = row_index as i64;
        match self {
            Self::ShiftedInteger {
                base_offset,
                stride,
            } => {
                let eff_stride = stride.abs().max(1);
                let safe_base = if (80..=330).contains(base_offset) {
                    base_offset.saturating_add(7300)
                } else {
                    *base_offset
                };
                safe_base.saturating_add(idx.saturating_mul(eff_stride))
            }
            Self::SwarmUuidSlugPrimaryKey => 7000 + idx.saturating_mul(13),
            Self::CompositeTenantKey { base_offset, .. } => {
                let safe_base = (*base_offset).max(4000);
                safe_base.saturating_add(idx.saturating_mul(7))
            }
        }
    }

    /// Format the surface key token for `row_index` and `entity_slug`.
    pub fn format_surface_key(&self, row_index: usize, entity_slug: &str) -> String {
        let pk = self.resolve_row_pk(row_index);
        match self {
            Self::ShiftedInteger { .. } => pk.to_string(),
            Self::SwarmUuidSlugPrimaryKey => format!("uuid:{entity_slug}:{pk}"),
            Self::CompositeTenantKey { tenant_code, .. } => {
                let clean_tenant = if tenant_code.trim().is_empty() {
                    "org-core"
                } else {
                    tenant_code.trim()
                };
                format!("{clean_tenant}:{pk}")
            }
        }
    }
}

/// Bijective surface encoding isomorphism for `(PolicyState, LockState)`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum PolicyLockSurfaceIso {
    #[default]
    StandardLiterals,
    NumericTierBoolean,
    DomainClearanceHold,
    SplitTwoColumnBoolean,
}

impl PolicyLockSurfaceIso {
    pub const ALL: [PolicyLockSurfaceIso; 4] = [
        Self::StandardLiterals,
        Self::NumericTierBoolean,
        Self::DomainClearanceHold,
        Self::SplitTwoColumnBoolean,
    ];

    /// Bijectively encode [`PolicyState`] into its surface string representation under this isomorphism.
    pub fn encode_policy(&self, policy: PolicyState) -> String {
        match (self, policy) {
            (Self::StandardLiterals, PolicyState::Allow) => "ALLOW".to_string(),
            (Self::StandardLiterals, PolicyState::Ask) => "ASK".to_string(),
            (Self::StandardLiterals, PolicyState::Deny) => "DENY".to_string(),
            (Self::NumericTierBoolean, PolicyState::Allow) => "0".to_string(),
            (Self::NumericTierBoolean, PolicyState::Ask) => "1".to_string(),
            (Self::NumericTierBoolean, PolicyState::Deny) => "2".to_string(),
            (Self::DomainClearanceHold, PolicyState::Allow) => "GREEN_AUTO".to_string(),
            (Self::DomainClearanceHold, PolicyState::Ask) => "AMBER_STEWARD".to_string(),
            (Self::DomainClearanceHold, PolicyState::Deny) => "RED_VETO".to_string(),
            (Self::SplitTwoColumnBoolean, PolicyState::Allow) => "SPLIT_ALLOW_0_0".to_string(),
            (Self::SplitTwoColumnBoolean, PolicyState::Ask) => "SPLIT_ASK_1_0".to_string(),
            (Self::SplitTwoColumnBoolean, PolicyState::Deny) => "SPLIT_DENY_1_1".to_string(),
        }
    }

    /// Bijectively encode [`LockState`] into its surface string representation under this isomorphism.
    pub fn encode_lock(&self, lock: LockState) -> String {
        match (self, lock) {
            (Self::StandardLiterals, LockState::Clear) => "CLEAR".to_string(),
            (Self::StandardLiterals, LockState::Locked) => "LOCKED".to_string(),
            (Self::NumericTierBoolean, LockState::Clear) => "0".to_string(),
            (Self::NumericTierBoolean, LockState::Locked) => "1".to_string(),
            (Self::DomainClearanceHold, LockState::Clear) => "OPEN".to_string(),
            (Self::DomainClearanceHold, LockState::Locked) => "SEALED_HOLD".to_string(),
            (Self::SplitTwoColumnBoolean, LockState::Clear) => "SPLIT_CLEAR_0_0".to_string(),
            (Self::SplitTwoColumnBoolean, LockState::Locked) => "SPLIT_LOCKED_1_1".to_string(),
        }
    }

    /// Return the 2-column boolean bit pair `(requires_review_bit, hard_veto_bit)` for [`PolicyState`].
    pub const fn encode_split_policy_bits(&self, policy: PolicyState) -> (i64, i64) {
        match policy {
            PolicyState::Allow => (0, 0),
            PolicyState::Ask => (1, 0),
            PolicyState::Deny => (1, 1),
        }
    }

    /// Return the 2-column boolean bit pair `(hold_flag_bit, seal_interlock_bit)` for [`LockState`].
    pub const fn encode_split_lock_bits(&self, lock: LockState) -> (i64, i64) {
        match lock {
            LockState::Clear => (0, 0),
            LockState::Locked => (1, 1),
        }
    }

    /// Decode a surface policy string back to [`PolicyState`].
    pub fn decode_policy(&self, raw: &str) -> Option<PolicyState> {
        let norm = raw.trim().to_ascii_uppercase();
        match (self, norm.as_str()) {
            (Self::StandardLiterals, "ALLOW") => Some(PolicyState::Allow),
            (Self::StandardLiterals, "ASK") => Some(PolicyState::Ask),
            (Self::StandardLiterals, "DENY") => Some(PolicyState::Deny),
            (Self::NumericTierBoolean, "0") => Some(PolicyState::Allow),
            (Self::NumericTierBoolean, "1") => Some(PolicyState::Ask),
            (Self::NumericTierBoolean, "2") => Some(PolicyState::Deny),
            (Self::DomainClearanceHold, "GREEN_AUTO") => Some(PolicyState::Allow),
            (Self::DomainClearanceHold, "AMBER_STEWARD") => Some(PolicyState::Ask),
            (Self::DomainClearanceHold, "RED_VETO") => Some(PolicyState::Deny),
            (Self::SplitTwoColumnBoolean, "SPLIT_ALLOW_0_0" | "0:0" | "0,0") => {
                Some(PolicyState::Allow)
            }
            (Self::SplitTwoColumnBoolean, "SPLIT_ASK_1_0" | "1:0" | "1,0") => {
                Some(PolicyState::Ask)
            }
            (Self::SplitTwoColumnBoolean, "SPLIT_DENY_1_1" | "1:1" | "1,1") => {
                Some(PolicyState::Deny)
            }
            _ => match norm.as_str() {
                "SPLIT_ALLOW_0_0" | "0:0" => Some(PolicyState::Allow),
                "SPLIT_ASK_1_0" | "1:0" => Some(PolicyState::Ask),
                "SPLIT_DENY_1_1" | "1:1" => Some(PolicyState::Deny),
                _ => PolicyState::parse_flexible(&norm),
            },
        }
    }

    /// Decode a surface lock string back to [`LockState`].
    pub fn decode_lock(&self, raw: &str) -> Option<LockState> {
        let norm = raw.trim().to_ascii_uppercase();
        match (self, norm.as_str()) {
            (Self::StandardLiterals, "CLEAR") => Some(LockState::Clear),
            (Self::StandardLiterals, "LOCKED") => Some(LockState::Locked),
            (Self::NumericTierBoolean, "0") => Some(LockState::Clear),
            (Self::NumericTierBoolean, "1") => Some(LockState::Locked),
            (Self::DomainClearanceHold, "OPEN") => Some(LockState::Clear),
            (Self::DomainClearanceHold, "SEALED_HOLD") => Some(LockState::Locked),
            (Self::SplitTwoColumnBoolean, "SPLIT_CLEAR_0_0" | "0:0" | "0,0") => {
                Some(LockState::Clear)
            }
            (Self::SplitTwoColumnBoolean, "SPLIT_LOCKED_1_1" | "1:1" | "1,1") => {
                Some(LockState::Locked)
            }
            _ => match norm.as_str() {
                "SPLIT_CLEAR_0_0" | "0:0" => Some(LockState::Clear),
                "SPLIT_LOCKED_1_1" | "1:1" => Some(LockState::Locked),
                _ => LockState::parse_flexible(&norm),
            },
        }
    }
}

/// Epistemic intent trajectory archetype across nursery episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntentTrajectoryKind {
    TransactionalWrite { relative_delta_arithmetic: bool },
    ReadOnlyDiagnostic,
    MidFlightMindChangeAbort,
    ValueConditionalBranch { min_balance_threshold: i64 },
}

impl Default for IntentTrajectoryKind {
    fn default() -> Self {
        Self::TransactionalWrite {
            relative_delta_arithmetic: false,
        }
    }
}

/// Comonadic steward agenda step in a multi-round `Mode::Ask` negotiation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StewardAgendaStep {
    RequestAdditionalRead {
        required_table: String,
        required_column: String,
    },
    ConditionalAllowWithOverride {
        token: String,
        overridden_post_balance: Option<i64>,
        overridden_target_pk: Option<i64>,
        overridden_status_note: Option<String>,
    },
    RevisedOverride {
        token: String,
        revised_post_balance: i64,
    },
    UnconditionalAllow {
        token: String,
    },
    ExplicitDeny {
        token: String,
    },
}
