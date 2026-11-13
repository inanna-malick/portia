//! Scenario stress modifiers, canonical SQL query shapes, and comparison operators.

use proptest::prelude::*;
use serde::{Deserialize, Serialize};

use super::coordinate::DomainSkin;

/// Composable nursery stress modifiers that can be attached to a scenario specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnifiedScenarioTrapModifier {
    /// Inserts a same-entity `'ARCHIVED'` decoy row with `policy_decision = 'ALLOW', lock_state = 'CLEAR'`
    /// alongside the active row.
    ArchDecoyHoneypotM2,
    /// Simulated steward claims a physical `'LOCKED'` database row is `'CLEAR'`; physical lock state governs.
    ConfusedStewardOnLockedRowH3,
    /// Omits the upfront `[Table ...]` schema header, requiring initial `Mode::Probe`
    /// (`PRAGMA table_info(...)`) before `Mode::Read`.
    BlindSchemaProbeFirstH2,
    /// Conservation underflow floor (`current_value - delta < min_floor -> Mode::Abstain`).
    ConservationUnderflowFloorH4 {
        min_floor: i64,
        requested_deduction: i64,
    },
    /// Mid-flight user intent pivot (`Mutation -> ReadOnly` abort at `[ASK]`, `0` writes).
    MidFlightIntentPivotToReadOnlyM3,
    /// Cross-wave causal value carry (`Wave 2 target = Wave 1 committed value + delta`).
    CrossWaveValueCarryH1a { wave2_delta: i64 },
    /// `ON DELETE RESTRICT` child-link check requiring child-table inspection before `DELETE`.
    OnDeleteRestrictChildCheckM4 { active_child_links: usize },
    /// Cross-wave outcome branching (`Wave 1 Abstained -> UPDATE promote reserve` vs `Wave 1 Applied -> DELETE reserve`).
    CrossWaveOutcomeBranchH1b,
    /// Concurrent `before_write` external `DELETE` race (`rows_affected == 0 -> Mode::Abstain`).
    BeforeWriteConcurrentDeleteRace,
    /// Ambiguous duplicate candidate rows sharing `needle_entity_slug`, requiring `Mode::Ask` disambiguation.
    AmbiguousEntityCandidateDisambiguation,
    /// Multi-turn cross-table topic switch (Turn 1 queries primary table `T_1`, Turn 2 switches to sibling table `T_2`).
    CrossTableTopicSwitchM5,
    /// Conversational domain noun with a high-frequency web homonym (`"pop sites"` -> `pop_facility_sites`).
    DomainNounWebHomonymM6,
}

impl UnifiedScenarioTrapModifier {
    /// Return the canonical `(primary_table, secondary_table, conversational_topic_switch_prompt)`
    /// pair for [`Self::CrossTableTopicSwitchM5`] under the given [`DomainSkin`].
    pub const fn cross_table_topic_switch_pair(
        skin: DomainSkin,
    ) -> (&'static str, &'static str, &'static str) {
        skin.cross_table_topic_switch_pair()
    }

    /// Return the canonical `(homonym_phrase, target_table, conversational_homonym_prompt, web_hallucination_markers)`
    /// tuple for [`Self::DomainNounWebHomonymM6`] under the given [`DomainSkin`].
    pub const fn domain_noun_web_homonym_pair(
        skin: DomainSkin,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        &'static [&'static str],
    ) {
        skin.domain_noun_web_homonym_pair()
    }
}

/// Canonical SQL query shapes spanning point lookups, projections with range/order/limit,
/// aggregates, and multi-table `INNER JOIN` / `LEFT JOIN` queries over foreign-key graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SqlQueryShape {
    /// `SELECT * FROM <t> WHERE <pk> = <val>;`
    PointLookupEq,
    /// `SELECT <col1>, <col2> FROM <t> WHERE <num_col> >= <threshold> ORDER BY <num_col> DESC LIMIT <k>;`
    ColumnProjectionRange,
    /// `SELECT COUNT(*) FROM <t> WHERE <col> = <val>;`
    AggregateCountWhere,
    /// `SELECT <t1>.<pk1>, <t1>.<slug>, <t2>.<child_col> FROM <t1> JOIN <t2> ON <t1>.<pk1> = <t2>.<fk_col> WHERE ...;`
    InnerJoinOnForeignKey,
    /// `SELECT ... FROM <t1> LEFT JOIN <t2> ON <t1>.<pk1> = <t2>.<fk_col> WHERE ...;`
    LeftJoinWithFilter,
}

impl SqlQueryShape {
    pub const ALL: [SqlQueryShape; 5] = [
        SqlQueryShape::PointLookupEq,
        SqlQueryShape::ColumnProjectionRange,
        SqlQueryShape::AggregateCountWhere,
        SqlQueryShape::InnerJoinOnForeignKey,
        SqlQueryShape::LeftJoinWithFilter,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            SqlQueryShape::PointLookupEq => "point_lookup_eq",
            SqlQueryShape::ColumnProjectionRange => "column_projection_range",
            SqlQueryShape::AggregateCountWhere => "aggregate_count_where",
            SqlQueryShape::InnerJoinOnForeignKey => "inner_join_on_fk",
            SqlQueryShape::LeftJoinWithFilter => "left_join_with_filter",
        }
    }

    pub const fn is_multi_table_join(self) -> bool {
        matches!(
            self,
            SqlQueryShape::InnerJoinOnForeignKey | SqlQueryShape::LeftJoinWithFilter
        )
    }
}

/// Comparison and pattern-matching operators exercised by [`SqlQueryShape::ColumnProjectionRange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SqlComparisonOperator {
    Gt,
    Gte,
    Lt,
    Lte,
    Ne,
    Like,
}

impl SqlComparisonOperator {
    pub const ALL: [SqlComparisonOperator; 6] = [
        SqlComparisonOperator::Gt,
        SqlComparisonOperator::Gte,
        SqlComparisonOperator::Lt,
        SqlComparisonOperator::Lte,
        SqlComparisonOperator::Ne,
        SqlComparisonOperator::Like,
    ];

    pub const fn sql_token(self) -> &'static str {
        match self {
            SqlComparisonOperator::Gt => ">",
            SqlComparisonOperator::Gte => ">=",
            SqlComparisonOperator::Lt => "<",
            SqlComparisonOperator::Lte => "<=",
            SqlComparisonOperator::Ne => "!=",
            SqlComparisonOperator::Like => "LIKE",
        }
    }
}

/// `proptest` strategy for [`SqlQueryShape`].
pub fn arb_sql_query_shape() -> impl Strategy<Value = SqlQueryShape> {
    prop_oneof![
        Just(SqlQueryShape::PointLookupEq),
        Just(SqlQueryShape::ColumnProjectionRange),
        Just(SqlQueryShape::AggregateCountWhere),
        Just(SqlQueryShape::InnerJoinOnForeignKey),
        Just(SqlQueryShape::LeftJoinWithFilter),
    ]
}

/// `proptest` strategy for [`SqlComparisonOperator`].
pub fn arb_sql_comparison_operator() -> impl Strategy<Value = SqlComparisonOperator> {
    prop_oneof![
        Just(SqlComparisonOperator::Gt),
        Just(SqlComparisonOperator::Gte),
        Just(SqlComparisonOperator::Lt),
        Just(SqlComparisonOperator::Lte),
        Just(SqlComparisonOperator::Ne),
        Just(SqlComparisonOperator::Like),
    ]
}
