//! Opaque Gibberish / Hex-Obfuscated Relational Schema & `<think>`-Decoy `proptest` Generator
//! (`roguelite_dungeon::opaque_gibberish`).
//!
//! Strips 100% of English identifier semantics (`user_id`, `account_id`) by generating pairwise-disjoint
//! gibberish table and column identifiers (`t_qvxk`, `c_kpmz` or `t_9f2a`, `c_77b1`), shuffling column
//! declaration order per table, injecting `<user>` prose and `<think>` scratchpad wrong-column decoys,
//! and seeding a `rusqlite::Connection` whose binary-spaced weights guarantee strictly distinct `SUM`
//! results between the true `FOREIGN KEY` join and every decoy column (**Self-Diagnosing Mutant Oracle**).

pub use super::opaque_gibberish_eval::arb_opaque_gibberish_dungeon_spec;
use crate::engine::{configure_hardened_sqlite, EngineError};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const CONSONANTS: &[u8; 20] = b"bcdfghjklmnpqrstvwxz";
const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Identifier alphabet used to synthesize zero-semantics SQL table and column names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GibberishDialect {
    /// Vowel-free lowercase consonant codes (`t_qvxk`, `c_kpmz`) — zero English subwords.
    VowelFreeConsonant,
    /// Lowercase hexadecimal identifiers (`t_9f2a`, `c_77b1`).
    HexAlphaNumeric,
    /// Shared-prefix Hamming-1 swarm (`c_qvxk_b`, `c_qvxk_c`, `c_qvxk_d`, ...) within the child table.
    SharedPrefixSwarm,
}

/// SQLite foreign-key DDL surface syntax (`FOREIGN KEY (col)` table constraint vs inline `REFERENCES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForeignKeyDdlStyle {
    /// Canonical SQLite table-level constraint (`FOREIGN KEY (c_fk) REFERENCES t_p(c_pk)`).
    TableConstraint,
    /// Inline column-level constraint (`c_fk INTEGER NOT NULL REFERENCES t_p(c_pk)`) — causal lookahead trap.
    InlineColumnReferences,
}

/// Shrinkable specification for an opaque gibberish relational join + `<think>` decoy scenario.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueGibberishDungeonSpec {
    pub seed: u64,
    pub dialect: GibberishDialect,
    pub ddl_style: ForeignKeyDdlStyle,
    /// Total same-type `INTEGER NOT NULL` candidate columns on the child table (`4..=10`).
    pub child_candidate_columns: usize,
    /// Whether to include a wrong-column join guess inside `<think>...</think>` right before `<sql>`.
    pub include_think_scratchpad_decoy: bool,
}

/// Self-diagnosing execution verdict classifying exact failure cause against the SQLite mutant lattice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpaqueGibberishVerdict {
    ExactGoldMatch {
        sum: i64,
    },
    CopiedThinkScratchpadDecoy {
        got_sum: i64,
        expected_sum: i64,
    },
    CopiedUserProseDecoy {
        got_sum: i64,
        expected_sum: i64,
    },
    PickedSilentSchemaDecoy {
        decoy_column: String,
        got_sum: i64,
        expected_sum: i64,
    },
    OtherWrongResult {
        got_sum: Option<i64>,
        expected_sum: i64,
    },
    SqlExecutionError {
        message: String,
    },
}

/// Materialized gibberish SQLite scenario with seeded SQL, prompt blocks, and mutant-sum lookup table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueGibberishDungeonInstance {
    pub spec: OpaqueGibberishDungeonSpec,
    pub parent_table: String,
    pub parent_pk_col: String,
    pub parent_metric_col: String,
    pub child_table: String,
    pub child_columns_in_ddl_order: Vec<String>,
    pub true_fk_col: String,
    pub user_prose_decoy_col: String,
    pub think_scratchpad_decoy_col: String,
    pub silent_schema_decoy_cols: Vec<String>,
    pub setup_sql: Vec<String>,
    pub schema_block: String,
    pub user_block: String,
    pub think_block: String,
    pub sql_prefix_to_fk_cursor: String,
    pub gold_sql: String,
    pub gold_sum: i64,
    pub decoy_mutant_sums: BTreeMap<String, i64>,
}

struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() % (bound as u64)) as usize
        }
    }
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.next_usize(i + 1));
        }
    }
}

fn fresh_code(
    rng: &mut SplitMix64,
    dialect: GibberishDialect,
    pfx: &str,
    used: &mut BTreeSet<String>,
) -> String {
    let alpha: &[u8] = match dialect {
        GibberishDialect::HexAlphaNumeric => HEX_CHARS,
        GibberishDialect::VowelFreeConsonant | GibberishDialect::SharedPrefixSwarm => CONSONANTS,
    };
    loop {
        let b: [u8; 4] = std::array::from_fn(|_| alpha[rng.next_usize(alpha.len())]);
        let cand = format!("{pfx}_{}", std::str::from_utf8(&b).expect("ascii"));
        if used.insert(cand.clone()) {
            return cand;
        }
    }
}

impl OpaqueGibberishDungeonSpec {
    /// Materialize a deterministic gibberish SQLite scenario and verify strict mutant-sum separation.
    pub fn build(&self) -> Result<OpaqueGibberishDungeonInstance, EngineError> {
        let n_cols = self.child_candidate_columns.clamp(4, 10);
        let mut rng = SplitMix64::new(self.seed);
        let mut used = BTreeSet::new();

        let parent_table = fresh_code(&mut rng, self.dialect, "t", &mut used);
        let child_table = fresh_code(&mut rng, self.dialect, "t", &mut used);
        let parent_pk_col = fresh_code(&mut rng, self.dialect, "c", &mut used);
        let parent_metric_col = fresh_code(&mut rng, self.dialect, "c", &mut used);

        let mut raw_child_cols = Vec::with_capacity(n_cols);
        if self.dialect == GibberishDialect::SharedPrefixSwarm {
            let stem = fresh_code(
                &mut rng,
                GibberishDialect::VowelFreeConsonant,
                "c",
                &mut used,
            );
            for &ch in CONSONANTS.iter().take(n_cols) {
                let col = format!("{stem}_{}", char::from(ch));
                used.insert(col.clone());
                raw_child_cols.push(col);
            }
        } else {
            for _ in 0..n_cols {
                raw_child_cols.push(fresh_code(&mut rng, self.dialect, "c", &mut used));
            }
        }

        let true_fk_col = raw_child_cols[0].clone();
        let user_prose_decoy_col = raw_child_cols[1].clone();
        let think_scratchpad_decoy_col = raw_child_cols[2].clone();
        let silent_schema_decoy_cols: Vec<String> = raw_child_cols[3..].to_vec();

        let mut child_columns_in_ddl_order = raw_child_cols.clone();
        rng.shuffle(&mut child_columns_in_ddl_order);

        let parent_ddl = format!(
            "CREATE TABLE {parent_table} (\n  {parent_pk_col} INTEGER PRIMARY KEY,\n  {parent_metric_col} INTEGER NOT NULL\n);"
        );
        let mut child_lines: Vec<String> = child_columns_in_ddl_order
            .iter()
            .map(|c| {
                if *c == true_fk_col && self.ddl_style == ForeignKeyDdlStyle::InlineColumnReferences
                {
                    format!("  {c} INTEGER NOT NULL REFERENCES {parent_table}({parent_pk_col})")
                } else {
                    format!("  {c} INTEGER NOT NULL")
                }
            })
            .collect();
        if self.ddl_style == ForeignKeyDdlStyle::TableConstraint {
            child_lines.push(format!(
                "  FOREIGN KEY ({true_fk_col}) REFERENCES {parent_table}({parent_pk_col})"
            ));
        }
        let child_ddl = format!(
            "CREATE TABLE {child_table} (\n{}\n);",
            child_lines.join(",\n")
        );

        // Binary-spaced parent metrics `100 + 13 * 2^pk + jitter(0..3)` guarantee strictly distinct
        // weighted sums `2 * val[k] + val[(k+1) % n_pks]` across all `n_cols <= 10` for every seed!
        let n_pks = n_cols + 1;
        let mut setup_sql = vec![parent_ddl.clone(), child_ddl.clone()];
        for pk in 1..=n_pks {
            let val = 100 + 13 * (1_i64 << pk) + (rng.next_usize(4) as i64);
            setup_sql.push(format!(
                "INSERT INTO {parent_table} ({parent_pk_col}, {parent_metric_col}) VALUES ({pk}, {val});"
            ));
        }

        for r_step in [0usize, 0, 1] {
            let vals_by_col: BTreeMap<&str, i64> = raw_child_cols
                .iter()
                .enumerate()
                .map(|(c_idx, col)| (col.as_str(), (((c_idx + r_step) % n_pks) + 1) as i64))
                .collect();
            let ordered_vals: Vec<String> = child_columns_in_ddl_order
                .iter()
                .map(|c| vals_by_col[c.as_str()].to_string())
                .collect();
            setup_sql.push(format!(
                "INSERT INTO {child_table} ({}) VALUES ({});",
                child_columns_in_ddl_order.join(", "),
                ordered_vals.join(", ")
            ));
        }

        let conn = Connection::open_in_memory()?;
        configure_hardened_sqlite(&conn)?;
        for stmt in &setup_sql {
            conn.execute_batch(stmt)?;
        }

        let query_for_col = |col: &str| {
            format!(
                "SELECT SUM({parent_table}.{parent_metric_col}) FROM {parent_table} JOIN {child_table} ON {parent_table}.{parent_pk_col} = {child_table}.{col};"
            )
        };
        let gold_sql = query_for_col(&true_fk_col);
        let gold_sum: i64 = conn.query_row(&gold_sql, [], |r| r.get(0))?;
        let mut decoy_mutant_sums = BTreeMap::new();
        for decoy in &raw_child_cols[1..] {
            decoy_mutant_sums.insert(
                decoy.clone(),
                conn.query_row(&query_for_col(decoy), [], |r| r.get(0))?,
            );
        }

        let schema_block = format!("<schema>\n{parent_ddl}\n{child_ddl}\n</schema>\n");
        let user_block = format!(
            "<user>\nNote integer column {child_table}.{user_prose_decoy_col}; join {parent_table} with {child_table} on its foreign key and return SUM({parent_table}.{parent_metric_col}).\n</user>\n"
        );
        let think_block = if self.include_think_scratchpad_decoy {
            format!(
                "<think>\nWe need to join {parent_table} and {child_table}. Looking at {child_table}, maybe the join key is {child_table}.{think_scratchpad_decoy_col} = {parent_table}.{parent_pk_col}? Let's write the SQL query.\n</think>\n"
            )
        } else {
            format!("<think>\nJoin {parent_table} and {child_table} on the foreign key in <schema>.\n</think>\n")
        };
        let sql_prefix_to_fk_cursor = format!(
            "<sql>\nSELECT SUM({parent_table}.{parent_metric_col}) FROM {parent_table} JOIN {child_table} ON {parent_table}.{parent_pk_col} = {child_table}."
        );

        Ok(OpaqueGibberishDungeonInstance {
            spec: self.clone(),
            parent_table,
            parent_pk_col,
            parent_metric_col,
            child_table,
            child_columns_in_ddl_order,
            true_fk_col,
            user_prose_decoy_col,
            think_scratchpad_decoy_col,
            silent_schema_decoy_cols,
            setup_sql,
            schema_block,
            user_block,
            think_block,
            sql_prefix_to_fk_cursor,
            gold_sql,
            gold_sum,
            decoy_mutant_sums,
        })
    }
}
