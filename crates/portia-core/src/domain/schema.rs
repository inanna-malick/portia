//! Canonical relational schema descriptors (`ColumnSchema`, `TableSchema`, `ForeignKeyDeleteRule`, `ForeignKeyEdge`).

use super::benchmark_decoys::{is_quarantined_decoy_column_name, ColumnAliasClass};
use serde::{Deserialize, Serialize};

/// Canonical physical SQLite column schema descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ColumnSchema {
    #[serde(default)]
    pub cid: i64,
    pub name: String,
    #[serde(alias = "sql_type")]
    pub col_type: String,
    #[serde(default)]
    pub notnull: bool,
    #[serde(alias = "pk")]
    pub is_primary_key: bool,
}

impl ColumnSchema {
    #[must_use]
    pub fn new(name: impl Into<String>, col_type: impl Into<String>, is_primary_key: bool) -> Self {
        Self {
            cid: 0,
            name: name.into(),
            col_type: col_type.into(),
            notnull: false,
            is_primary_key,
        }
    }

    #[must_use]
    pub fn new_canonical(
        name: impl Into<String>,
        col_type: impl Into<String>,
        is_primary_key: bool,
        nullable: bool,
    ) -> Self {
        Self {
            cid: 0,
            name: name.into(),
            col_type: col_type.into(),
            notnull: !nullable,
            is_primary_key,
        }
    }

    #[must_use]
    pub fn with_sqlite_metadata(
        cid: i64,
        name: impl Into<String>,
        col_type: impl Into<String>,
        notnull: bool,
        is_primary_key: bool,
    ) -> Self {
        Self {
            cid,
            name: name.into(),
            col_type: col_type.into(),
            notnull,
            is_primary_key,
        }
    }

    #[must_use]
    pub fn nullable(&self) -> bool {
        !self.notnull
    }

    /// Synthetic nursery test-oracle decoy classification (`benchmark_decoys::ColumnAliasClass`).
    #[must_use]
    pub fn alias_class(&self) -> ColumnAliasClass {
        ColumnAliasClass::classify_column_name(&self.name)
    }
}

/// Canonical physical SQLite table schema descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_name: String,
    pub columns: Vec<ColumnSchema>,
    #[serde(default)]
    pub row_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub foreign_keys: Vec<ForeignKeyEdge>,
}

impl TableSchema {
    #[must_use]
    pub fn new(table_name: impl Into<String>, columns: Vec<ColumnSchema>) -> Self {
        Self {
            table_name: table_name.into(),
            columns,
            row_count: 0,
            foreign_keys: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_row_count(mut self, row_count: usize) -> Self {
        self.row_count = row_count;
        self
    }

    #[must_use]
    pub fn with_foreign_keys(mut self, foreign_keys: Vec<ForeignKeyEdge>) -> Self {
        self.foreign_keys = foreign_keys;
        self
    }

    #[must_use]
    pub fn column_name_set(&self) -> std::collections::BTreeSet<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    #[must_use]
    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    #[must_use]
    pub fn primary_key_column(&self) -> Option<&str> {
        self.columns
            .iter()
            .find(|c| c.is_primary_key)
            .map(|c| c.name.as_str())
    }

    /// Filter non-decoy columns for opt-in synthetic nursery test-oracle checks.
    #[must_use]
    pub fn live_columns(&self) -> Vec<&ColumnSchema> {
        self.columns
            .iter()
            .filter(|c| !is_quarantined_decoy_column_name(&c.name))
            .collect()
    }

    /// Filter synthetic decoy columns for opt-in nursery test-oracle checks.
    #[must_use]
    pub fn decoy_columns(&self) -> Vec<&ColumnSchema> {
        self.columns
            .iter()
            .filter(|c| is_quarantined_decoy_column_name(&c.name))
            .collect()
    }

    /// Return a copy omitting synthetic benchmark decoy columns (`benchmark_decoys`).
    #[must_use]
    pub fn without_quarantined_decoys(&self) -> Self {
        let columns = self
            .columns
            .iter()
            .filter(|c| c.is_primary_key || !is_quarantined_decoy_column_name(&c.name))
            .cloned()
            .collect();
        Self {
            table_name: self.table_name.clone(),
            columns,
            row_count: self.row_count,
            foreign_keys: self.foreign_keys.clone(),
        }
    }
}

/// Referential integrity action on a [`ForeignKeyEdge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ForeignKeyDeleteRule {
    Cascade,
    Restrict,
    SetNull,
    NoAction,
}

impl ForeignKeyDeleteRule {
    #[inline]
    pub const fn as_sql_str(self) -> &'static str {
        match self {
            ForeignKeyDeleteRule::Cascade => "ON DELETE CASCADE",
            ForeignKeyDeleteRule::Restrict => "ON DELETE RESTRICT",
            ForeignKeyDeleteRule::SetNull => "ON DELETE SET NULL",
            ForeignKeyDeleteRule::NoAction => "ON DELETE NO ACTION",
        }
    }
}

/// Directed foreign-key dependency edge between a child table and a parent table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ForeignKeyEdge {
    pub child_table: String,
    pub child_fk_column: String,
    pub parent_table: String,
    pub parent_pk_column: String,
    pub on_delete: ForeignKeyDeleteRule,
}

impl ForeignKeyEdge {
    /// Canonical `ON <parent_table>.<parent_col> = <child_table>.<child_col>` clause for a
    /// `FROM <base_table> JOIN <join_table> ON ...` query.
    pub fn canonical_on_condition(&self, base_table: &str) -> String {
        if base_table.eq_ignore_ascii_case(&self.parent_table) {
            format!(
                "{}.{} = {}.{}",
                self.parent_table, self.parent_pk_column, self.child_table, self.child_fk_column
            )
        } else {
            format!(
                "{}.{} = {}.{}",
                self.child_table, self.child_fk_column, self.parent_table, self.parent_pk_column
            )
        }
    }
}
