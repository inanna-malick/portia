//! Stress lattice dimensions, range specifications, and 3-band (`65% / 25% / 10%`) samplers.

use std::fmt;
use std::ops::RangeInclusive;

use portia_core::domain::{
    BiasedScaleBand, NeedleDepth, BOSS_FLOOR_WEIGHT, DEEP_DUNGEON_WEIGHT, SWEET_SPOT_WEIGHT,
};
use proptest::prelude::{prop_oneof, BoxedStrategy, Just, Strategy};
use serde::{Deserialize, Serialize};

/// Shared `proptest` strategy builder applying the `13 : 5 : 2` (`65% / 25% / 10%`)
/// band weighting across any `(sweet_spot, deep_dungeon, boss_floor)` range triple.
pub fn arb_biased_three_band_range(
    sweet_spot: RangeInclusive<usize>,
    deep_dungeon: RangeInclusive<usize>,
    boss_floor: RangeInclusive<usize>,
) -> BoxedStrategy<usize> {
    prop_oneof![
        SWEET_SPOT_WEIGHT => sweet_spot,
        DEEP_DUNGEON_WEIGHT => deep_dungeon,
        BOSS_FLOOR_WEIGHT => boss_floor,
    ]
    .boxed()
}

/// Shared `proptest` strategy for [`NeedleDepth`].
pub fn arb_needle_depth() -> BoxedStrategy<NeedleDepth> {
    prop_oneof![
        Just(NeedleDepth::Top10Pct),
        Just(NeedleDepth::Mid50Pct),
        Just(NeedleDepth::Bot90Pct),
    ]
    .boxed()
}

/// Number of continuous stress dimensions in the Roguelite Stress Lattice.
pub const NUM_ROGUELITE_DIMENSIONS: usize = 7;

/// Range specification for a single continuous stress dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionBandSpec {
    pub dimension: RogueliteDimension,
    pub absolute_min: usize,
    pub sweet_spot: (usize, usize),
    pub deep_dungeon: (usize, usize),
    pub boss_floor: (usize, usize),
}

impl DimensionBandSpec {
    /// Classify any integer value along this dimension into its [`BiasedScaleBand`].
    pub const fn classify(&self, value: usize) -> BiasedScaleBand {
        if value < self.sweet_spot.0 {
            BiasedScaleBand::SubSweetSpotFloor
        } else if value <= self.sweet_spot.1 {
            BiasedScaleBand::TacticalSweetSpot
        } else if value <= self.deep_dungeon.1 {
            BiasedScaleBand::DeepDungeon
        } else {
            BiasedScaleBand::ExtremeBossFloor
        }
    }

    /// Clamp a raw value into `[absolute_min, boss_floor.1]`.
    pub const fn clamp(&self, value: usize) -> usize {
        if value < self.absolute_min {
            self.absolute_min
        } else if value > self.boss_floor.1 {
            self.boss_floor.1
        } else {
            value
        }
    }

    /// Deterministic `65% / 25% / 10%` band sampler from a `[0..20)` ticket and a `u64` offset seed.
    pub const fn sample_from_ticket(&self, ticket_mod_20: u8, offset_seed: u64) -> usize {
        let (lo, hi) = if ticket_mod_20 < 13 {
            self.sweet_spot
        } else if ticket_mod_20 < 18 {
            self.deep_dungeon
        } else {
            self.boss_floor
        };
        let span = (hi - lo + 1) as u64;
        lo + (offset_seed % span) as usize
    }
}

/// The 7 continuous stress dimensions of the Roguelite Stress Lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(usize)]
pub enum RogueliteDimension {
    /// Dimension 0: Total rows in table haystack (`1..=1000`; sweet `5..=50`, deep `51..=250`, boss `251..=1000`).
    RowHaystackSize = 0,
    /// Dimension 1: Number of Hamming-1 / Levenshtein-1 near-colliding entity slugs (`1..=250`; sweet `3..=25`, deep `26..=100`, boss `101..=250`).
    IdentifierSwarmCollision = 1,
    /// Dimension 2: Total columns in table (`4..=100`; sweet `6..=20`, deep `21..=50`, boss `51..=100`).
    ColumnWidthArity = 2,
    /// Dimension 3: Total tables in schema graph (`1..=30`; sweet `2..=6`, deep `7..=15`, boss `16..=30`).
    SchemaGraphTopology = 3,
    /// Dimension 4: Coupled boolean/numeric guard clauses (`1..=16`; sweet `2..=5`, deep `6..=10`, boss `11..=16`).
    PolicyPredicateArity = 4,
    /// Dimension 5: Total episode turns / waves (`1..=50`; sweet `3..=10`, deep `11..=25`, boss `26..=50`).
    EpisodeTurnHorizon = 5,
    /// Dimension 6: Total AST functions/methods in module (`3..=300`; sweet `8..=35`, deep `36..=120`, boss `121..=300`).
    AstSymbolHaystack = 6,
}

impl RogueliteDimension {
    pub const ALL: [RogueliteDimension; NUM_ROGUELITE_DIMENSIONS] = [
        Self::RowHaystackSize,
        Self::IdentifierSwarmCollision,
        Self::ColumnWidthArity,
        Self::SchemaGraphTopology,
        Self::PolicyPredicateArity,
        Self::EpisodeTurnHorizon,
        Self::AstSymbolHaystack,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn short_name(self) -> &'static str {
        match self {
            Self::RowHaystackSize => "row_haystack_size",
            Self::IdentifierSwarmCollision => "identifier_swarm_collision",
            Self::ColumnWidthArity => "column_width_arity",
            Self::SchemaGraphTopology => "schema_graph_topology",
            Self::PolicyPredicateArity => "policy_predicate_arity",
            Self::EpisodeTurnHorizon => "episode_turn_horizon",
            Self::AstSymbolHaystack => "ast_symbol_haystack",
        }
    }

    /// Parse a CLI or shorthand dimension identifier (`"rows"`, `"swarm"`, `"cols"`, `"tables"`,
    /// `"clauses"`, `"turns"`, `"ast"`, etc.).
    pub fn from_cli_name(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "rows" | "row" | "row_haystack" | "row_haystack_size" | "haystack" => {
                Some(Self::RowHaystackSize)
            }
            "1" | "swarm" | "identifier_swarm" | "identifier_swarm_collision" | "slugs" => {
                Some(Self::IdentifierSwarmCollision)
            }
            "2" | "cols" | "columns" | "column_width" | "column_width_arity" => {
                Some(Self::ColumnWidthArity)
            }
            "3" | "tables" | "schema" | "schema_graph" | "schema_graph_topology" => {
                Some(Self::SchemaGraphTopology)
            }
            "4" | "clauses" | "policy" | "policy_predicate" | "policy_predicate_arity" => {
                Some(Self::PolicyPredicateArity)
            }
            "5" | "turns" | "horizon" | "episode_turn_horizon" | "waves" => {
                Some(Self::EpisodeTurnHorizon)
            }
            "6" | "ast" | "ast_symbols" | "ast_symbol_haystack" | "symbols" => {
                Some(Self::AstSymbolHaystack)
            }
            _ => None,
        }
    }

    /// Band specification (`absolute_min`, `sweet_spot`, `deep_dungeon`, `boss_floor`).
    pub const fn band_spec(self) -> DimensionBandSpec {
        match self {
            Self::RowHaystackSize => DimensionBandSpec {
                dimension: self,
                absolute_min: 1,
                sweet_spot: (5, 50),
                deep_dungeon: (51, 250),
                boss_floor: (251, 1000),
            },
            Self::IdentifierSwarmCollision => DimensionBandSpec {
                dimension: self,
                absolute_min: 1,
                sweet_spot: (3, 25),
                deep_dungeon: (26, 100),
                boss_floor: (101, 250),
            },
            Self::ColumnWidthArity => DimensionBandSpec {
                dimension: self,
                absolute_min: 4,
                sweet_spot: (6, 20),
                deep_dungeon: (21, 50),
                boss_floor: (51, 100),
            },
            Self::SchemaGraphTopology => DimensionBandSpec {
                dimension: self,
                absolute_min: 1,
                sweet_spot: (2, 6),
                deep_dungeon: (7, 15),
                boss_floor: (16, 30),
            },
            Self::PolicyPredicateArity => DimensionBandSpec {
                dimension: self,
                absolute_min: 1,
                sweet_spot: (2, 5),
                deep_dungeon: (6, 10),
                boss_floor: (11, 16),
            },
            Self::EpisodeTurnHorizon => DimensionBandSpec {
                dimension: self,
                absolute_min: 1,
                sweet_spot: (3, 10),
                deep_dungeon: (11, 25),
                boss_floor: (26, 50),
            },
            Self::AstSymbolHaystack => DimensionBandSpec {
                dimension: self,
                absolute_min: 3,
                sweet_spot: (8, 35),
                deep_dungeon: (36, 120),
                boss_floor: (121, 300),
            },
        }
    }
}

impl fmt::Display for RogueliteDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}
