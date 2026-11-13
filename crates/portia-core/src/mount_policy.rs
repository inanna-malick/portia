//! Slot mounting policies (`MountPolicy`) and wire bank key serialization (`WireBankKey`).

use crate::{CartridgeBank, Mode, PenOwner, PrefillSegmentKind};
use serde::{Deserialize, Serialize};
use std::fmt;

/// First-class slot mounting policy (`PermanentStatic`, `DynamicBank`, `ModePlan`, `PrefillSegment`),
/// keeping permanent static MLP widening distinct from deterministic host-phase bank switching
/// across compilation, verification, serialization, and runtime execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MountPolicy {
    /// Permanently mounted across all tokens and all dynamic banks (`"UnifiedStatic"`).
    PermanentStatic,
    /// Dynamically mounted when the given [`CartridgeBank`] (`Base`, `OrientTrunk`, `ModePayload`) is active.
    DynamicBank(CartridgeBank),
    /// Per-mode `<think>` / `PlanBody` thinking cartridge (`"ModePlan:<Mode>"`), swapped out at
    /// `</think>` for `DynamicBank(CartridgeBank::ModePayload(mode))`.
    ModePlan(Mode),
    /// Deterministic `HarnessPen` prefill segment cartridge (`"Prefill:Schema"`, `"Prefill:User"`, `"Prefill:ToolObs"`).
    PrefillSegment(PrefillSegmentKind),
}

impl MountPolicy {
    /// Canonical runtime wire key (`"UnifiedStatic"`, `"Base"`, `"OrientTrunk"`, `"ModePlan:<Mode>"`,
    /// `"ModePayload:<Mode>"`, `"Prefill:<Segment>"`).
    #[must_use]
    pub const fn wire_key(self) -> &'static str {
        match self {
            MountPolicy::PermanentStatic => "UnifiedStatic",
            MountPolicy::DynamicBank(bank) => bank.wire_key(),
            MountPolicy::ModePlan(Mode::Probe) => "ModePlan:Probe",
            MountPolicy::ModePlan(Mode::Read) => "ModePlan:Read",
            MountPolicy::ModePlan(Mode::Ask) => "ModePlan:Ask",
            MountPolicy::ModePlan(Mode::Write) => "ModePlan:Write",
            MountPolicy::ModePlan(Mode::Result) => "ModePlan:Result",
            MountPolicy::ModePlan(Mode::Abstain) => "ModePlan:Abstain",
            MountPolicy::PrefillSegment(seg) => seg.wire_key(),
        }
    }

    /// Resolve the active runtime wire bank key against `available_banks` in a loaded cartridge,
    /// falling back (`ModePlan:<M>` -> `ModePayload:<M>`, `Prefill:<S>` -> `Base`) when
    /// running cartridges that omit phase-specialized banks.
    #[must_use]
    pub fn resolve_fallback_wire_key(
        self,
        available_banks: &std::collections::BTreeSet<String>,
    ) -> &'static str {
        let primary = self.wire_key();
        if available_banks.contains(primary) {
            return primary;
        }
        match self {
            MountPolicy::ModePlan(m) => CartridgeBank::ModePayload(m).wire_key(),
            MountPolicy::PrefillSegment(_) => CartridgeBank::Base.wire_key(),
            MountPolicy::PermanentStatic | MountPolicy::DynamicBank(_) => primary,
        }
    }

    /// True iff two [`MountPolicy`] values can be physically co-mounted on the same token forward pass.
    /// `PermanentStatic` is co-mounted alongside every dynamic phase bank; distinct dynamic phase
    /// banks are mutually exclusive (`0` slot interference across phases!).
    #[must_use]
    pub const fn co_mounted_with(self, other: Self) -> bool {
        match (self, other) {
            (MountPolicy::PermanentStatic, _) | (_, MountPolicy::PermanentStatic) => true,
            (MountPolicy::DynamicBank(a), MountPolicy::DynamicBank(b)) => match (a, b) {
                (CartridgeBank::Base, CartridgeBank::Base)
                | (CartridgeBank::OrientTrunk, CartridgeBank::OrientTrunk) => true,
                (CartridgeBank::ModePayload(ma), CartridgeBank::ModePayload(mb)) => {
                    matches!(
                        (ma, mb),
                        (Mode::Probe, Mode::Probe)
                            | (Mode::Read, Mode::Read)
                            | (Mode::Ask, Mode::Ask)
                            | (Mode::Write, Mode::Write)
                            | (Mode::Result, Mode::Result)
                            | (Mode::Abstain, Mode::Abstain)
                    )
                }
                _ => false,
            },
            (MountPolicy::ModePlan(ma), MountPolicy::ModePlan(mb)) => matches!(
                (ma, mb),
                (Mode::Probe, Mode::Probe)
                    | (Mode::Read, Mode::Read)
                    | (Mode::Ask, Mode::Ask)
                    | (Mode::Write, Mode::Write)
                    | (Mode::Result, Mode::Result)
                    | (Mode::Abstain, Mode::Abstain)
            ),
            (MountPolicy::PrefillSegment(sa), MountPolicy::PrefillSegment(sb)) => matches!(
                (sa, sb),
                (PrefillSegmentKind::Schema, PrefillSegmentKind::Schema)
                    | (PrefillSegmentKind::User, PrefillSegmentKind::User)
                    | (PrefillSegmentKind::ToolObs, PrefillSegmentKind::ToolObs)
            ),
            _ => false,
        }
    }

    /// Parse a wire bank/mount key into `(MountPolicy, PenOwner)` without aliasing `"UnifiedStatic"` to `OrientTrunk`.
    pub fn from_wire_key(raw: &str) -> Option<(Self, PenOwner)> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("UnifiedStatic")
            || trimmed.eq_ignore_ascii_case("UNIFIED_STATIC")
            || trimmed.eq_ignore_ascii_case("PermanentStatic")
            || trimmed.eq_ignore_ascii_case("PERMANENT_STATIC")
        {
            return Some((MountPolicy::PermanentStatic, PenOwner::HarnessPen));
        }
        if trimmed.eq_ignore_ascii_case("OrientTrunk")
            || trimmed.eq_ignore_ascii_case("ORIENT_TRUNK")
        {
            return Some((
                MountPolicy::DynamicBank(CartridgeBank::OrientTrunk),
                PenOwner::RouterPen,
            ));
        }
        if trimmed.eq_ignore_ascii_case("Base") {
            return Some((
                MountPolicy::DynamicBank(CartridgeBank::Base),
                PenOwner::HarnessPen,
            ));
        }
        if let Some(rest) = trimmed.strip_prefix("ModePayload:") {
            let mode = Mode::parse_flexible(&rest.trim().to_ascii_uppercase())?;
            return Some((
                MountPolicy::DynamicBank(CartridgeBank::ModePayload(mode)),
                PenOwner::PayloadPen,
            ));
        }
        if let Some(rest) = trimmed.strip_prefix("ModePlan:") {
            let mode = Mode::parse_flexible(&rest.trim().to_ascii_uppercase())?;
            return Some((MountPolicy::ModePlan(mode), PenOwner::PayloadPen));
        }
        if let Some(rest) = trimmed.strip_prefix("Prefill:") {
            let seg = PrefillSegmentKind::parse_flexible(rest)?;
            return Some((MountPolicy::PrefillSegment(seg), PenOwner::HarnessPen));
        }
        None
    }
}

impl CartridgeBank {
    /// Parse a wire bank key (`"Base"`, `"OrientTrunk"`, `"ModePayload:Read"`) into [`CartridgeBank`].
    #[must_use]
    pub fn from_wire_key(raw: &str) -> Option<Self> {
        match MountPolicy::from_wire_key(raw)? {
            (MountPolicy::DynamicBank(bank), _) => Some(bank),
            (
                MountPolicy::PermanentStatic
                | MountPolicy::ModePlan(_)
                | MountPolicy::PrefillSegment(_),
                _,
            ) => None,
        }
    }
}

impl fmt::Display for CartridgeBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire_key())
    }
}

impl fmt::Display for MountPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire_key())
    }
}

/// Wire-string serialized wrapper around [`CartridgeBank`] (`"Base"`, `"OrientTrunk"`, `"ModePayload:Read"`),
/// preserving `CartridgeBank`'s tagged JSON-RPC representation while enabling typed string-keyed maps and vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WireBankKey(pub CartridgeBank);

impl Serialize for WireBankKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.wire_key())
    }
}

impl<'de> Deserialize<'de> for WireBankKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        CartridgeBank::from_wire_key(&s).map(Self).ok_or_else(|| {
            serde::de::Error::custom(format!("invalid CartridgeBank wire key: {s:?}"))
        })
    }
}
