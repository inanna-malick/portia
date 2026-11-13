//! Relational governance and SQL mutation domain primitives (`PolicyState`, `LockState`,
//! `RowLifecycleStatus`, and `WriteOp`).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Canonical relational governance policy state $\in \{\text{Allow}, \text{Ask}, \text{Deny}\}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PolicyState {
    Allow,
    Ask,
    Deny,
}

/// Type alias for row-level policy decisions observed inside SQLite `<tool_response>` rows.
pub type RowPolicyDecision = PolicyState;

/// Type alias for steward inquiry decisions (`ALLOW` / `ASK` / `DENY`).
pub type StewardDecision = PolicyState;

impl PolicyState {
    pub const ALL: [PolicyState; 3] = [PolicyState::Allow, PolicyState::Ask, PolicyState::Deny];

    /// Canonical uppercase SQL/wire literal (`"ALLOW"`, `"ASK"`, `"DENY"`).
    #[inline]
    pub const fn as_canonical_str(self) -> &'static str {
        match self {
            PolicyState::Allow => "ALLOW",
            PolicyState::Ask => "ASK",
            PolicyState::Deny => "DENY",
        }
    }

    /// Alias for [`Self::as_canonical_str`].
    #[inline]
    pub const fn canonical_str(self) -> &'static str {
        self.as_canonical_str()
    }

    /// SQL literal alias for [`Self::as_canonical_str`].
    #[inline]
    pub const fn as_sql_str(self) -> &'static str {
        self.as_canonical_str()
    }

    /// Parse a policy literal across standard, numeric-tier (`0/1/2`), and clearance
    /// (`GREEN_AUTO` / `AMBER_STEWARD` / `RED_VETO`) surface encodings.
    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "ALLOW" | "0" | "GREEN_AUTO" | "APPROVED" => Some(PolicyState::Allow),
            "ASK" | "1" | "AMBER_STEWARD" | "STEWARD_REQUIRED" => Some(PolicyState::Ask),
            "DENY" | "2" | "RED_VETO" | "FORBIDDEN" => Some(PolicyState::Deny),
            _ => None,
        }
    }
}

impl fmt::Display for PolicyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_canonical_str())
    }
}

/// Canonical physical database interlock state $\in \{\text{Clear}, \text{Locked}\}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LockState {
    Clear,
    Locked,
}

/// Type alias for physical database interlock state.
pub type DbInterlockState = LockState;

impl LockState {
    pub const ALL: [LockState; 2] = [LockState::Clear, LockState::Locked];

    /// Canonical uppercase SQL/wire literal (`"CLEAR"`, `"LOCKED"`).
    #[inline]
    pub const fn as_canonical_str(self) -> &'static str {
        match self {
            LockState::Clear => "CLEAR",
            LockState::Locked => "LOCKED",
        }
    }

    /// Alias for [`Self::as_canonical_str`].
    #[inline]
    pub const fn canonical_str(self) -> &'static str {
        self.as_canonical_str()
    }

    /// SQL literal alias for [`Self::as_canonical_str`].
    #[inline]
    pub const fn as_sql_str(self) -> &'static str {
        self.as_canonical_str()
    }

    /// Whether the physical database interlock is currently engaged (`LockState::Locked`).
    #[inline]
    pub const fn is_locked(self) -> bool {
        matches!(self, LockState::Locked)
    }

    /// Parse a lock literal across standard (`CLEAR`/`LOCKED`), boolean (`0`/`1`), and
    /// hold-state (`OPEN`/`SEALED_HOLD`) surface encodings.
    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "CLEAR" | "0" | "OPEN" | "UNLOCKED" | "FALSE" => Some(LockState::Clear),
            "LOCKED" | "1" | "SEALED_HOLD" | "HOLD" | "TRUE" => Some(LockState::Locked),
            _ => None,
        }
    }
}

impl fmt::Display for LockState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_canonical_str())
    }
}

/// Canonical lifecycle status of a candidate row in a relational scenario table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowLifecycleStatus {
    Active,
    Reserve,
    Template,
    Followup,
    Standby,
    Archived,
    Obsolete,
}

impl RowLifecycleStatus {
    pub const ALL: [RowLifecycleStatus; 7] = [
        RowLifecycleStatus::Active,
        RowLifecycleStatus::Reserve,
        RowLifecycleStatus::Template,
        RowLifecycleStatus::Followup,
        RowLifecycleStatus::Standby,
        RowLifecycleStatus::Archived,
        RowLifecycleStatus::Obsolete,
    ];

    /// Canonical uppercase SQL literal (`"ACTIVE"`, `"RESERVE"`, `"TEMPLATE"`, etc.).
    #[inline]
    pub const fn as_canonical_str(self) -> &'static str {
        match self {
            RowLifecycleStatus::Active => "ACTIVE",
            RowLifecycleStatus::Reserve => "RESERVE",
            RowLifecycleStatus::Template => "TEMPLATE",
            RowLifecycleStatus::Followup => "FOLLOWUP",
            RowLifecycleStatus::Standby => "STANDBY",
            RowLifecycleStatus::Archived => "ARCHIVED",
            RowLifecycleStatus::Obsolete => "OBSOLETE",
        }
    }

    /// SQL literal alias for [`Self::as_canonical_str`].
    #[inline]
    pub const fn as_sql_str(self) -> &'static str {
        self.as_canonical_str()
    }

    /// Whether this row status represents a decommissioned synthetic decoy row (`Archived` or `Obsolete`).
    #[inline]
    pub const fn is_decommissioned_decoy(self) -> bool {
        matches!(
            self,
            RowLifecycleStatus::Archived | RowLifecycleStatus::Obsolete
        )
    }

    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "ACTIVE" => Some(RowLifecycleStatus::Active),
            "RESERVE" => Some(RowLifecycleStatus::Reserve),
            "TEMPLATE" => Some(RowLifecycleStatus::Template),
            "FOLLOWUP" | "FOLLOW_UP" => Some(RowLifecycleStatus::Followup),
            "STANDBY" => Some(RowLifecycleStatus::Standby),
            "ARCHIVED" => Some(RowLifecycleStatus::Archived),
            "OBSOLETE" | "DEPRECATED" => Some(RowLifecycleStatus::Obsolete),
            _ => None,
        }
    }
}

impl fmt::Display for RowLifecycleStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_canonical_str())
    }
}

/// Canonical SQL mutation verb (`UPDATE`, `INSERT`, `DELETE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WriteOp {
    Update,
    Insert,
    Delete,
}

/// Type alias for SQL mutation operations.
pub type SqlWriteOp = WriteOp;

impl WriteOp {
    pub const ALL: [WriteOp; 3] = [WriteOp::Update, WriteOp::Insert, WriteOp::Delete];

    /// Canonical SQL verb string (`"UPDATE"`, `"INSERT"`, `"DELETE"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            WriteOp::Update => "UPDATE",
            WriteOp::Insert => "INSERT",
            WriteOp::Delete => "DELETE",
        }
    }

    /// Alias for [`Self::as_str`] returning the SQL statement verb.
    #[inline]
    pub const fn sql_verb(self) -> &'static str {
        self.as_str()
    }

    pub fn parse_flexible(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "UPDATE" => Some(WriteOp::Update),
            "INSERT" | "INSERT INTO" => Some(WriteOp::Insert),
            "DELETE" | "DELETE FROM" => Some(WriteOp::Delete),
            _ => None,
        }
    }
}

impl fmt::Display for WriteOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
