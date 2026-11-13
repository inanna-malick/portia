//! Strongly-typed vocabulary token IDs, layer indices, and widened MLP slot offsets.

use crate::RouterSlotPair;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Temporal envelope kind for a compiled SwiGLU slot pair (`TRIG` vs `SUSTAIN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvelopeKind {
    /// 1-token impulse diode at a decision cursor (e.g. `[MODE:`).
    Trig,
    /// Multi-token sustain sidechain across `[span_start .. span_end)` with prefix `< span_start` clamped `<= -8.0`.
    Sustain { span_start: usize },
}

/// Strongly-typed vocabulary token ID (`0 .. vocab_size`), preventing local row/class offsets
/// (`0 .. 6`) from being confused with Qwen3.5 BPE token IDs (`5140`, `18104`, etc.).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);

impl TokenId {
    /// First special/chat-control token ID in Qwen3.5 (`<|endoftext|>` / `<|im_start|>` block).
    pub const QWEN35_NATURAL_VOCAB_LIMIT: u32 = 248_044;

    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// True iff `self` lies strictly below the Qwen3.5 special/reserved token block.
    #[inline]
    #[must_use]
    pub const fn is_qwen35_natural_token(self) -> bool {
        self.0 < Self::QWEN35_NATURAL_VOCAB_LIMIT
    }
}

impl From<u32> for TokenId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<TokenId> for u32 {
    #[inline]
    fn from(id: TokenId) -> Self {
        id.0
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transformer decoder layer index (`0 .. num_layers`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerIndex(pub usize);

impl LayerIndex {
    #[inline]
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for LayerIndex {
    #[inline]
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<LayerIndex> for usize {
    #[inline]
    fn from(index: LayerIndex) -> Self {
        index.0
    }
}

impl fmt::Display for LayerIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for LayerIndex {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.trim().parse::<usize>().map(Self)
    }
}

/// Stock MLP `intermediate_size` ($M_{\text{base}}$, e.g. `3584` on `Qwen3.5-0.8B`, `12288` on `9B`)
/// before appending reserved cartridge slots.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BaseIntermediateSize(pub usize);

impl BaseIntermediateSize {
    #[inline]
    #[must_use]
    pub const fn new(size: usize) -> Self {
        Self(size)
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Zero-based slot offset within the reserved `[0 .. widened_slots)` cartridge region.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RelativeSlotOffset(pub usize);

impl RelativeSlotOffset {
    #[inline]
    #[must_use]
    pub const fn new(offset: usize) -> Self {
        Self(offset)
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    #[inline]
    #[must_use]
    pub const fn to_physical(self, base: BaseIntermediateSize) -> PhysicalSlotIndex {
        PhysicalSlotIndex(base.0 + self.0)
    }
}

/// Physical MLP neuron index (`base_intermediate_size + relative_offset`) in the widened MLP.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhysicalSlotIndex(pub usize);

impl PhysicalSlotIndex {
    #[inline]
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    #[inline]
    #[must_use]
    pub const fn checked_to_relative(
        self,
        base: BaseIntermediateSize,
    ) -> Option<RelativeSlotOffset> {
        if self.0 >= base.0 {
            Some(RelativeSlotOffset(self.0 - base.0))
        } else {
            None
        }
    }

    /// Resolve a boundary integer (encoded either as a `< base` relative offset or `>= base`
    /// physical index) into canonical `(RelativeSlotOffset, PhysicalSlotIndex)` once at
    /// deserialization boundaries so internal code never guesses from magnitude.
    #[inline]
    #[must_use]
    pub const fn resolve_flexible_boundary(
        raw: usize,
        base: BaseIntermediateSize,
    ) -> (RelativeSlotOffset, Self) {
        if raw >= base.0 {
            (RelativeSlotOffset(raw - base.0), Self(raw))
        } else {
            (RelativeSlotOffset(raw), Self(base.0 + raw))
        }
    }
}

/// Canonical coupled 2-slot `(radial_slot_offset, tangential_slot_offset)` pair within `[0 .. reserved_slots)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotPairOffset {
    pub radial_slot_offset: usize,
    pub tangential_slot_offset: usize,
}

impl SlotPairOffset {
    #[inline]
    #[must_use]
    pub const fn new(radial_slot_offset: usize, tangential_slot_offset: usize) -> Self {
        Self {
            radial_slot_offset,
            tangential_slot_offset,
        }
    }

    #[inline]
    #[must_use]
    pub const fn radial(self) -> RelativeSlotOffset {
        RelativeSlotOffset(self.radial_slot_offset)
    }

    #[inline]
    #[must_use]
    pub const fn tangential(self) -> RelativeSlotOffset {
        RelativeSlotOffset(self.tangential_slot_offset)
    }

    #[inline]
    #[must_use]
    pub const fn overlaps_with(self, other: Self) -> bool {
        self.radial_slot_offset == other.radial_slot_offset
            || self.radial_slot_offset == other.tangential_slot_offset
            || self.tangential_slot_offset == other.radial_slot_offset
            || self.tangential_slot_offset == other.tangential_slot_offset
    }
}

impl From<(usize, usize)> for SlotPairOffset {
    #[inline]
    fn from(pair: (usize, usize)) -> Self {
        Self::new(pair.0, pair.1)
    }
}

impl From<RouterSlotPair> for SlotPairOffset {
    #[inline]
    fn from(pair: RouterSlotPair) -> Self {
        Self::new(pair.radial_slot, pair.tangential_slot)
    }
}

impl From<SlotPairOffset> for RouterSlotPair {
    #[inline]
    fn from(pair: SlotPairOffset) -> Self {
        Self {
            radial_slot: pair.radial_slot_offset,
            tangential_slot: pair.tangential_slot_offset,
        }
    }
}
