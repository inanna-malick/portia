//! SQLite mutant-oracle evaluation and shrinkable `proptest` strategy for
//! [`super::opaque_gibberish::OpaqueGibberishDungeonInstance`].

use super::opaque_gibberish::{
    ForeignKeyDdlStyle, GibberishDialect, OpaqueGibberishDungeonInstance,
    OpaqueGibberishDungeonSpec, OpaqueGibberishVerdict,
};
use crate::engine::{configure_hardened_sqlite, EngineError};
use proptest::prelude::*;
use rusqlite::Connection;

impl OpaqueGibberishDungeonInstance {
    /// Open an in-memory `rusqlite::Connection` seeded with `self.setup_sql`.
    pub fn open_connection(&self) -> Result<Connection, EngineError> {
        let conn = Connection::open_in_memory()?;
        configure_hardened_sqlite(&conn)?;
        for stmt in &self.setup_sql {
            conn.execute_batch(stmt)?;
        }
        Ok(conn)
    }

    /// Execute `candidate_sql` against SQLite and classify against `gold_sum` & `decoy_mutant_sums`.
    pub fn evaluate_candidate_sql(&self, candidate_sql: &str) -> OpaqueGibberishVerdict {
        let Ok(conn) = self.open_connection() else {
            return OpaqueGibberishVerdict::SqlExecutionError {
                message: "failed to seed in-memory SQLite connection".to_string(),
            };
        };
        let trimmed = candidate_sql.trim().trim_end_matches("</sql>").trim();
        match conn.query_row(trimmed, [], |r| r.get::<_, Option<i64>>(0)) {
            Ok(Some(v)) if v == self.gold_sum => OpaqueGibberishVerdict::ExactGoldMatch { sum: v },
            Ok(Some(v))
                if self.decoy_mutant_sums.get(&self.think_scratchpad_decoy_col) == Some(&v) =>
            {
                OpaqueGibberishVerdict::CopiedThinkScratchpadDecoy {
                    got_sum: v,
                    expected_sum: self.gold_sum,
                }
            }
            Ok(Some(v)) if self.decoy_mutant_sums.get(&self.user_prose_decoy_col) == Some(&v) => {
                OpaqueGibberishVerdict::CopiedUserProseDecoy {
                    got_sum: v,
                    expected_sum: self.gold_sum,
                }
            }
            Ok(Some(v)) => {
                for silent in &self.silent_schema_decoy_cols {
                    if self.decoy_mutant_sums.get(silent) == Some(&v) {
                        return OpaqueGibberishVerdict::PickedSilentSchemaDecoy {
                            decoy_column: silent.clone(),
                            got_sum: v,
                            expected_sum: self.gold_sum,
                        };
                    }
                }
                OpaqueGibberishVerdict::OtherWrongResult {
                    got_sum: Some(v),
                    expected_sum: self.gold_sum,
                }
            }
            Ok(None) => OpaqueGibberishVerdict::OtherWrongResult {
                got_sum: None,
                expected_sum: self.gold_sum,
            },
            Err(e) => OpaqueGibberishVerdict::SqlExecutionError {
                message: e.to_string(),
            },
        }
    }
}

/// Shrinkable `proptest` strategy over [`OpaqueGibberishDungeonSpec`] across all 3 dialects & 2 DDL styles.
pub fn arb_opaque_gibberish_dungeon_spec() -> impl Strategy<Value = OpaqueGibberishDungeonSpec> {
    let dialect = prop_oneof![
        Just(GibberishDialect::VowelFreeConsonant),
        Just(GibberishDialect::HexAlphaNumeric),
        Just(GibberishDialect::SharedPrefixSwarm),
    ];
    let ddl_style = prop_oneof![
        Just(ForeignKeyDdlStyle::TableConstraint),
        Just(ForeignKeyDdlStyle::InlineColumnReferences),
    ];
    (any::<u64>(), dialect, ddl_style, 4usize..=10, any::<bool>()).prop_map(
        |(seed, dialect, ddl_style, child_candidate_columns, include_think_scratchpad_decoy)| {
            OpaqueGibberishDungeonSpec {
                seed,
                dialect,
                ddl_style,
                child_candidate_columns,
                include_think_scratchpad_decoy,
            }
        },
    )
}
