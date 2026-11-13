//! Stratified Roguelite stress scale bands, needle depths, collision loci, and biased samplers.

use serde::{Deserialize, Serialize};

/// Proptest sampling weight for the Tactical Sweet-Spot band (`13 / 20 = 65%`).
pub const SWEET_SPOT_WEIGHT: u32 = 13;

/// Proptest sampling weight for the Deep Dungeon band (`5 / 20 = 25%`).
pub const DEEP_DUNGEON_WEIGHT: u32 = 5;

/// Proptest sampling weight for the Extreme Boss-Floor band (`2 / 20 = 10%`).
pub const BOSS_FLOOR_WEIGHT: u32 = 2;

/// Canonical stratified scale band across all 7 Roguelite stress dimensions (`65% / 25% / 10%`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BiasedScaleBand {
    /// Values below the tactical sweet-spot floor (e.g. `1..=4` rows or 2-turn baseline).
    SubSweetSpotFloor,
    /// `65%` proptest weight (`13/20`): realistic day-to-day operational friction.
    TacticalSweetSpot,
    /// `25%` proptest weight (`5/20`): heavy multi-hop / multi-entity load.
    DeepDungeon,
    /// `10%` proptest weight (`2/20`): extreme stress ceiling designed to trigger DeltaNet/GQA wipeout.
    ExtremeBossFloor,
}

/// Alias used in SQL dungeon modules.
pub type DungeonDifficultyBand = BiasedScaleBand;

/// Alias used in horizon endurance modules.
pub type HorizonBand = BiasedScaleBand;

/// Alias used in AST roguelite modules.
pub type AstDungeonFloorBand = BiasedScaleBand;

#[allow(non_upper_case_globals)]
impl BiasedScaleBand {
    pub const SweetSpot65Pct: Self = Self::TacticalSweetSpot;
    pub const DeepDungeon25Pct: Self = Self::DeepDungeon;
    pub const BossFloor10Pct: Self = Self::ExtremeBossFloor;
    pub const TwoTurnBaseline: Self = Self::SubSweetSpotFloor;
    pub const SweetSpot: Self = Self::TacticalSweetSpot;
    pub const BossFloor: Self = Self::ExtremeBossFloor;

    /// Nominal sampling probability (`0.0`, `0.65`, `0.25`, `0.10`).
    #[inline]
    pub const fn nominal_probability(self) -> f64 {
        match self {
            Self::SubSweetSpotFloor => 0.0,
            Self::TacticalSweetSpot => 0.65,
            Self::DeepDungeon => 0.25,
            Self::ExtremeBossFloor => 0.10,
        }
    }

    /// Target sampling weight percentage (`0`, `65`, `25`, `10`).
    #[inline]
    pub const fn target_weight_pct(self) -> u8 {
        match self {
            Self::SubSweetSpotFloor => 0,
            Self::TacticalSweetSpot => 65,
            Self::DeepDungeon => 25,
            Self::ExtremeBossFloor => 10,
        }
    }

    /// Alias for [`Self::target_weight_pct`].
    #[inline]
    pub const fn weight_pct(self) -> u8 {
        self.target_weight_pct()
    }

    /// Classify any integer value given `(sweet_min, sweet_max)` and `deep_max`.
    #[inline]
    pub const fn classify_bounds(
        value: usize,
        sweet_spot: (usize, usize),
        deep_max: usize,
    ) -> Self {
        if value < sweet_spot.0 {
            Self::SubSweetSpotFloor
        } else if value <= sweet_spot.1 {
            Self::TacticalSweetSpot
        } else if value <= deep_max {
            Self::DeepDungeon
        } else {
            Self::ExtremeBossFloor
        }
    }

    /// Classify Dimension 0 row count (`5..=50` sweet, `51..=250` deep, `251..=1000` boss).
    #[inline]
    pub const fn classify_row_count(n: usize) -> Self {
        Self::classify_bounds(n, (5, 50), 250)
    }

    /// Classify Dimension 2 column count (`6..=20` sweet, `21..=50` deep, `51..=100` boss).
    #[inline]
    pub const fn classify_column_count(c: usize) -> Self {
        Self::classify_bounds(c, (6, 20), 50)
    }

    /// Classify Dimension 3 table count (`2..=6` sweet, `7..=15` deep, `16..=30` boss).
    #[inline]
    pub const fn classify_table_count(t: usize) -> Self {
        Self::classify_bounds(t, (2, 6), 15)
    }

    /// Classify Dimension 5 episode turn count (`2` baseline, `3..=10` sweet, `11..=25` deep, `26..=50` boss).
    #[inline]
    pub const fn from_turns(turns: usize) -> Option<Self> {
        match turns {
            2 => Some(Self::SubSweetSpotFloor),
            3..=10 => Some(Self::TacticalSweetSpot),
            11..=25 => Some(Self::DeepDungeon),
            26..=50 => Some(Self::ExtremeBossFloor),
            _ => None,
        }
    }

    /// Inclusive `(min_turns, max_turns)` range for episode horizon bands.
    #[inline]
    pub const fn turn_range(self) -> (usize, usize) {
        match self {
            Self::SubSweetSpotFloor => (2, 2),
            Self::TacticalSweetSpot => (3, 10),
            Self::DeepDungeon => (11, 25),
            Self::ExtremeBossFloor => (26, 50),
        }
    }

    /// Classify Dimension 6 AST symbol count (`5..=30` sweet, `31..=100` deep, `101..=300` boss).
    #[inline]
    pub const fn from_symbol_count(symbol_count: usize) -> Self {
        if symbol_count <= 30 {
            Self::TacticalSweetSpot
        } else if symbol_count <= 100 {
            Self::DeepDungeon
        } else {
            Self::ExtremeBossFloor
        }
    }

    /// Inclusive `(min_symbols, max_symbols)` range for AST symbol bands.
    #[inline]
    pub const fn symbol_range(self) -> (usize, usize) {
        match self {
            Self::SubSweetSpotFloor | Self::TacticalSweetSpot => (5, 30),
            Self::DeepDungeon => (31, 100),
            Self::ExtremeBossFloor => (101, 300),
        }
    }
}

/// Canonical relative depth of a target needle inside a row or symbol haystack (`Top10Pct`, `Mid50Pct`, `Bot90Pct`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeedleDepth {
    /// First `10%` of items (`index ≈ 0.10 * N`): tests DeltaNet recurrent washout across trailing `90%` decoys.
    Top10Pct,
    /// Middle `50%` of items (`index ≈ 0.50 * N`): tests GQA RoPE/softmax middle-context attention dilution.
    Mid50Pct,
    /// Final `90%` of items (`index ≈ 0.90 * N`): recency baseline near the end of the scan.
    Bot90Pct,
}

/// Alias used in `roguelite_lattice`.
pub type NeedleDepthBand = NeedleDepth;

/// Alias used in AST roguelite scenarios.
pub type NeedleDepthPosition = NeedleDepth;

impl NeedleDepth {
    pub const ALL: [NeedleDepth; 3] = [Self::Top10Pct, Self::Mid50Pct, Self::Bot90Pct];

    /// Zero-based index of the target needle inside a collection of `total_items` (`>= 1`).
    #[inline]
    pub const fn resolve_index(self, total_items: usize) -> usize {
        if total_items <= 1 {
            return 0;
        }
        let max_idx = total_items - 1;
        let raw = match self {
            Self::Top10Pct => total_items / 10,
            Self::Mid50Pct => total_items / 2,
            Self::Bot90Pct => (total_items * 9) / 10,
        };
        if raw > max_idx {
            max_idx
        } else {
            raw
        }
    }

    /// Alias for [`Self::resolve_index`] used on SQL row haystacks.
    #[inline]
    pub const fn needle_row_index(self, total_rows: usize) -> usize {
        self.resolve_index(total_rows)
    }
}

/// Subword BPE boundary locus where near-colliding identifiers differ (`IdentifierSwarmCollision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CollisionLocus {
    /// Differs in the leading prefix token (`"inc-7f3a-01a"` vs `"ind-7f3a-01a"`).
    PrefixDiff,
    /// Differs in the middle hex shard (`"inc-7f3a-01a"` vs `"inc-7f8a-01a"`).
    MiddleHexDiff,
    /// Differs only in the final character (`"inc-7f3a-01a"` vs `"inc-7f3a-01b"`), maximally stressing induction copy heads.
    FinalCharDiff,
}

impl CollisionLocus {
    pub const ALL: [CollisionLocus; 3] =
        [Self::PrefixDiff, Self::MiddleHexDiff, Self::FinalCharDiff];
}

/// Strategic execution class when confronting a high-stress dungeon floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CombatStrategyClass {
    /// Brute-force context ingestion (`SELECT * FROM tbl`, dumping all AST symbols or tables).
    RawTanking,
    /// Surgical predicate pushdown (`SELECT cols FROM tbl WHERE pk = ...`, `PRAGMA table_info`,
    /// symbol-scoped AST lens, Steward witness token), keeping active context $O(1)$ across haystacks.
    FinesseSurgical,
}

/// Deterministic `65% / 25% / 10%` sampler mapping a `u64` seed into `(sweet, deep, boss)` inclusive ranges.
pub const fn sample_biased_three_band_u64(
    seed: u64,
    sweet: (usize, usize),
    deep: (usize, usize),
    boss: (usize, usize),
) -> usize {
    let bucket = (seed % 100) as usize;
    let sub = ((seed / 100) % 100_000) as usize;
    let (lo, hi) = if bucket < 65 {
        sweet
    } else if bucket < 90 {
        deep
    } else {
        boss
    };
    let span = (hi - lo) + 1;
    lo + (sub % span)
}
