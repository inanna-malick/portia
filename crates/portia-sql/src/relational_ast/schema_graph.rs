//! Multi-table relational schema graph, foreign-key edges, `CHECK` constraints, and coalgebraic
//! foreign-key dependency subtrees.

use std::collections::{BTreeMap, BTreeSet};

use portia_core::hylo::{CollapsibleExt, ExpandableExt, RoseFrame, RoseTree};
use portia_core::TableSchema;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::engine::EngineError;

use super::coercion::SqlRowDomainError;
use super::predicate::SqlPredicateAst;
use super::type_check::is_numeric_sql_type_str;

/// Observable diagnostic recorded when a `CREATE TABLE` `CHECK (...)` clause in `sqlite_master`
/// uses SQL syntax outside [`SqlPredicateAst::from_sqlparser_expr`] (or when DDL parsing fails).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedCheckConstraintDiagnostic {
    pub table_name: String,
    pub constraint_name: Option<String>,
    pub raw_expr_sql: String,
    pub reason: String,
}

/// Directed foreign-key edge (`child_table.child_col -> parent_table.parent_col`)
/// discovered live from SQLite `PRAGMA foreign_key_list(<child_table>)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ForeignKeyEdge {
    pub child_table: String,
    pub child_col: String,
    pub parent_table: String,
    pub parent_col: String,
}

impl ForeignKeyEdge {
    pub fn new(
        child_table: impl Into<String>,
        child_col: impl Into<String>,
        parent_table: impl Into<String>,
        parent_col: impl Into<String>,
    ) -> Self {
        Self {
            child_table: child_table.into(),
            child_col: child_col.into(),
            parent_table: parent_table.into(),
            parent_col: parent_col.into(),
        }
    }

    /// Canonical `ON <parent_table>.<parent_col> = <child_table>.<child_col>` clause for a
    /// `FROM <base_table> JOIN <join_table> ON ...` query.
    pub fn canonical_on_condition(&self, base_table: &str) -> String {
        portia_core::domain::ForeignKeyEdge::from(self).canonical_on_condition(base_table)
    }
}

impl From<&ForeignKeyEdge> for portia_core::domain::ForeignKeyEdge {
    fn from(edge: &ForeignKeyEdge) -> Self {
        Self {
            child_table: edge.child_table.clone(),
            child_fk_column: edge.child_col.clone(),
            parent_table: edge.parent_table.clone(),
            parent_pk_column: edge.parent_col.clone(),
            on_delete: portia_core::domain::ForeignKeyDeleteRule::Cascade,
        }
    }
}

impl From<portia_core::domain::ForeignKeyEdge> for ForeignKeyEdge {
    fn from(edge: portia_core::domain::ForeignKeyEdge) -> Self {
        Self {
            child_table: edge.child_table,
            child_col: edge.child_fk_column,
            parent_table: edge.parent_table,
            parent_col: edge.parent_pk_column,
        }
    }
}

/// Multi-table relational schema graph holding `PRAGMA table_info` schemas,
/// `PRAGMA foreign_key_list` edges, nullable column sets, and `sqlite_master` `CHECK (...)`
/// constraint ASTs across all user tables in a SQLite database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalSchemaGraph {
    pub tables: BTreeMap<String, TableSchema>,
    pub foreign_keys: Vec<ForeignKeyEdge>,
    /// Columns whose DDL omits `NOT NULL` (or that are explicitly nullable in `PRAGMA table_info`).
    pub nullable_columns: BTreeMap<String, BTreeSet<String>>,
    /// Live `CHECK (...)` constraints extracted from `sqlite_master` `CREATE TABLE` DDL into
    /// [`SqlPredicateAst`] via `sqlparser` and [`SqlPredicateAst::from_sqlparser_expr`].
    #[serde(default)]
    pub check_constraints: BTreeMap<String, Vec<SqlPredicateAst>>,
    /// Observable diagnostics for any `CHECK (...)` clauses in `sqlite_master` that used SQL
    /// syntax outside [`SqlPredicateAst::from_sqlparser_expr`] (or failed DDL parsing).
    #[serde(default)]
    pub skipped_check_diagnostics: Vec<SkippedCheckConstraintDiagnostic>,
}

impl RelationalSchemaGraph {
    pub fn new(
        tables: Vec<TableSchema>,
        foreign_keys: Vec<ForeignKeyEdge>,
        nullable_columns: BTreeMap<String, BTreeSet<String>>,
    ) -> Self {
        let mut map = BTreeMap::new();
        for t in tables {
            map.insert(t.table_name.clone(), t);
        }
        Self {
            tables: map,
            foreign_keys,
            nullable_columns,
            check_constraints: BTreeMap::new(),
            skipped_check_diagnostics: Vec::new(),
        }
    }

    /// Ergonomic alias for [`Self::inspect_from_connection`]: introspects all user tables,
    /// column types, nullability, foreign-key edges, and `sqlite_master` `CHECK (...)`
    /// constraint ASTs directly from a live SQLite [`Connection`].
    pub fn from_sqlite_connection(conn: &Connection) -> Result<Self, EngineError> {
        Self::inspect_from_connection(conn)
    }

    pub fn get_table(&self, table_name: &str) -> Option<&TableSchema> {
        self.tables
            .values()
            .find(|t| t.table_name.eq_ignore_ascii_case(table_name))
    }

    /// Return the parsed [`SqlPredicateAst`] `CHECK (...)` constraints for `table_name`.
    pub fn check_constraints_for(&self, table_name: &str) -> &[SqlPredicateAst] {
        self.check_constraints
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(table_name))
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    /// Return any [`SkippedCheckConstraintDiagnostic`] records for `table_name` whose `CHECK (...)`
    /// expression could not be imported into [`SqlPredicateAst`].
    pub fn skipped_checks_for(&self, table_name: &str) -> Vec<&SkippedCheckConstraintDiagnostic> {
        self.skipped_check_diagnostics
            .iter()
            .filter(|d| d.table_name.eq_ignore_ascii_case(table_name))
            .collect()
    }

    /// Fallibly evaluate all `CHECK (...)` constraints on `table_name` against a candidate row map,
    /// returning `Err(SqlRowDomainError)` on any incompatible non-null domain value.
    pub fn try_evaluate_row_check_constraints(
        &self,
        table_name: &str,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> Result<bool, SqlRowDomainError> {
        let mut all_passed = true;
        for pred in self.check_constraints_for(table_name) {
            if !pred.try_evaluate_check_constraint(row)? {
                all_passed = false;
            }
        }
        Ok(all_passed)
    }

    /// Evaluate all `CHECK (...)` constraints on `table_name` against a candidate row map,
    /// failing closed (`false`) on domain type violations while passing genuine SQL `NULL`
    /// (`Ok(SqlTruthValue::Unknown)` -> `true`).
    pub fn evaluate_row_check_constraints(
        &self,
        table_name: &str,
        row: &BTreeMap<String, serde_json::Value>,
    ) -> bool {
        self.try_evaluate_row_check_constraints(table_name, row)
            .unwrap_or(false)
    }

    /// Return an opt-in sanitized copy of this [`RelationalSchemaGraph`] with synthetic benchmark
    /// decoy columns quarantined from every table via [`TableSchema::without_quarantined_decoys`].
    pub fn without_quarantined_decoys(&self) -> Self {
        let tables = self
            .tables
            .iter()
            .map(|(k, v)| (k.clone(), v.without_quarantined_decoys()))
            .collect();
        let nullable_columns = self
            .nullable_columns
            .iter()
            .map(|(k, set)| {
                let filtered = set
                    .iter()
                    .filter(|c| !portia_core::domain::is_quarantined_decoy_column_name(c))
                    .cloned()
                    .collect();
                (k.clone(), filtered)
            })
            .collect();
        let check_constraints = self
            .check_constraints
            .iter()
            .map(|(k, preds)| {
                let filtered = preds
                    .iter()
                    .filter(|pred| {
                        pred.referenced_column_refs().iter().all(|c| {
                            !portia_core::domain::is_quarantined_decoy_column_name(&c.column.name)
                        })
                    })
                    .cloned()
                    .collect();
                (k.clone(), filtered)
            })
            .collect();
        Self {
            tables,
            foreign_keys: self.foreign_keys.clone(),
            nullable_columns,
            check_constraints,
            skipped_check_diagnostics: self.skipped_check_diagnostics.clone(),
        }
    }

    /// Return all tables connected to `base_table` via a foreign-key edge (either as child or parent).
    pub fn joinable_tables_for(&self, base_table: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for edge in &self.foreign_keys {
            if edge.parent_table.eq_ignore_ascii_case(base_table) {
                out.insert(edge.child_table.clone());
            } else if edge.child_table.eq_ignore_ascii_case(base_table) {
                out.insert(edge.parent_table.clone());
            }
        }
        out
    }

    /// Find the foreign-key edge connecting `table_a` and `table_b` (in either direction).
    pub fn fk_edge_between(&self, table_a: &str, table_b: &str) -> Option<&ForeignKeyEdge> {
        self.foreign_keys.iter().find(|e| {
            (e.parent_table.eq_ignore_ascii_case(table_a)
                && e.child_table.eq_ignore_ascii_case(table_b))
                || (e.child_table.eq_ignore_ascii_case(table_a)
                    && e.parent_table.eq_ignore_ascii_case(table_b))
        })
    }

    /// Return true if `col_name` in `table_name` has an `INTEGER`, `REAL`, or `NUMERIC` type in `PRAGMA table_info`.
    pub fn is_numeric_column(&self, table_name: &str, col_name: &str) -> bool {
        let Some(schema) = self.get_table(table_name) else {
            return false;
        };
        schema
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(col_name))
            .map(|c| is_numeric_sql_type_str(&c.col_type))
            .unwrap_or(false)
    }

    /// Return true if `col_name` in `table_name` is nullable according to `PRAGMA table_info`.
    pub fn is_nullable_column(&self, table_name: &str, col_name: &str) -> bool {
        self.nullable_columns
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(table_name))
            .map(|(_, cols)| cols.iter().any(|c| c.eq_ignore_ascii_case(col_name)))
            .unwrap_or(false)
    }

    /// Return the combined set of column names (both unqualified and `<table>.<col>` qualified)
    /// valid in a `WHERE` clause over `base_table` and optional `joined_table`.
    pub fn valid_where_columns(
        &self,
        base_table: &str,
        joined_table: Option<&str>,
    ) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        if let Some(base_sc) = self.get_table(base_table) {
            for c in &base_sc.columns {
                out.insert(c.name.clone());
                if joined_table.is_some() {
                    out.insert(format!("{}.{}", base_sc.table_name, c.name));
                }
            }
        }
        if let Some(j_tbl) = joined_table {
            if let Some(join_sc) = self.get_table(j_tbl) {
                for c in &join_sc.columns {
                    out.insert(c.name.clone());
                    out.insert(format!("{}.{}", join_sc.table_name, c.name));
                }
            }
        }
        out
    }

    /// Unfold the multi-hop foreign-key dependency subtree rooted at `root_table` via
    /// [`RoseTree::expand_frames`] over `self.foreign_keys`, carrying a visited-table set in the
    /// coalgebra seed so cycles are impossible.
    pub fn fk_reachable_subtree(&self, root_table: &str) -> RoseTree<String> {
        let canonical_root = self
            .get_table(root_table)
            .map(|t| t.table_name.clone())
            .unwrap_or_else(|| root_table.to_string());
        let initial_visited = BTreeSet::from([canonical_root.to_ascii_lowercase()]);

        RoseTree::expand_frames(
            (canonical_root, initial_visited),
            |(table_name, visited)| {
                let mut child_seeds = Vec::new();
                for edge in &self.foreign_keys {
                    if edge.parent_table.eq_ignore_ascii_case(&table_name) {
                        let child_low = edge.child_table.to_ascii_lowercase();
                        if !visited.contains(&child_low) {
                            let mut next_visited = visited.clone();
                            next_visited.insert(child_low);
                            child_seeds.push((edge.child_table.clone(), next_visited));
                        }
                    }
                }
                RoseFrame {
                    label: table_name,
                    children: child_seeds,
                }
            },
        )
    }

    /// Collapse the multi-hop foreign-key dependency subtree rooted at `root_table` into its
    /// maximum table-chain depth (`1` for a table with no child foreign keys, `d` for a
    /// `d`-table chain $T_1 \leftarrow T_2 \leftarrow \dots \leftarrow T_d$) via
    /// [`CollapsibleExt::collapse_frames`].
    pub fn fk_subtree_depth(&self, root_table: &str) -> usize {
        self.fk_reachable_subtree(root_table)
            .collapse_frames(|frame| 1 + frame.children.into_iter().max().unwrap_or(0))
    }
}
