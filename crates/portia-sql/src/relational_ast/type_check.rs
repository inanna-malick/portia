//! Schema type-checking for [`SqlPredicateAst`] (`validate_against_schema`, `validate_against_graph`).

use std::collections::BTreeSet;

use portia_core::hylo::CollapsibleExt;
use portia_core::{ColumnSchema, TableSchema};
use serde::{Deserialize, Serialize};

use super::ident::SqlColumnRef;
use super::predicate::{SqlPredicateAst, SqlPredicateFrame};
use super::schema_graph::RelationalSchemaGraph;

pub(super) fn is_numeric_sql_type_str(col_type: &str) -> bool {
    let up = col_type.to_ascii_uppercase();
    up.contains("INT")
        || up.contains("REAL")
        || up.contains("FLOA")
        || up.contains("DOUB")
        || up.contains("NUM")
        || up.contains("DEC")
}

/// Schema type-checking error raised by [`SqlPredicateAst::validate_against_schema`]
/// and [`SqlPredicateAst::validate_against_graph`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum SqlTypeError {
    #[error("unknown column `{column}` in table `{table}`")]
    UnknownColumn { table: String, column: String },
    #[error(
        "numeric range operator `{op}` applied to non-numeric column `{column}` (`{col_type}`) in table `{table}`"
    )]
    NonNumericRangeComparison {
        table: String,
        column: String,
        col_type: String,
        op: String,
    },
    #[error(
        "`IS NULL` / `IS NOT NULL` applied to non-nullable column `{column}` in table `{table}`"
    )]
    NonNullableColumnNullCheck { table: String, column: String },
}

impl SqlPredicateAst {
    /// Verify that every referenced column exists in `schema`, numeric range operators (`<`, `>`,
    /// `<=`, `>=`) are only applied to numeric columns, and `IS NULL` / `IS NOT NULL` is only
    /// applied to nullable columns (when `nullable_columns` is provided).
    pub fn validate_against_schema(
        &self,
        schema: &TableSchema,
        nullable_columns: Option<&BTreeSet<String>>,
    ) -> Result<BTreeSet<String>, SqlTypeError> {
        let resolve_schema_col = |col_ref: &SqlColumnRef| -> Result<&ColumnSchema, SqlTypeError> {
            if let Some(tbl) = &col_ref.table_qualifier {
                if !schema.table_name.is_empty()
                    && !tbl.matches_ascii_case_insensitive(&schema.table_name)
                {
                    return Err(SqlTypeError::UnknownColumn {
                        table: schema.table_name.clone(),
                        column: col_ref.qualified_or_column_name(),
                    });
                }
            }
            schema
                .columns
                .iter()
                .find(|c| col_ref.column.matches_ascii_case_insensitive(&c.name))
                .ok_or_else(|| SqlTypeError::UnknownColumn {
                    table: schema.table_name.clone(),
                    column: col_ref.qualified_or_column_name(),
                })
        };

        self.try_collapse_frames(|frame: SqlPredicateFrame<BTreeSet<String>>| match frame {
            SqlPredicateFrame::Comparison { column, op, .. } => {
                let col_sc = resolve_schema_col(&column)?;
                if op.is_range_op() && !is_numeric_sql_type_str(&col_sc.col_type) {
                    return Err(SqlTypeError::NonNumericRangeComparison {
                        table: schema.table_name.clone(),
                        column: column.qualified_or_column_name(),
                        col_type: col_sc.col_type.clone(),
                        op: op.as_sql_str().to_string(),
                    });
                }
                Ok(BTreeSet::from([col_sc.name.clone()]))
            }
            SqlPredicateFrame::IsNull { column, .. } => {
                let col_sc = resolve_schema_col(&column)?;
                if let Some(null_set) = nullable_columns {
                    let is_nullable = !col_sc.is_primary_key
                        && null_set
                            .iter()
                            .any(|c| column.column.matches_ascii_case_insensitive(c));
                    if !is_nullable {
                        return Err(SqlTypeError::NonNullableColumnNullCheck {
                            table: schema.table_name.clone(),
                            column: column.qualified_or_column_name(),
                        });
                    }
                } else if col_sc.is_primary_key {
                    return Err(SqlTypeError::NonNullableColumnNullCheck {
                        table: schema.table_name.clone(),
                        column: column.qualified_or_column_name(),
                    });
                }
                Ok(BTreeSet::from([col_sc.name.clone()]))
            }
            SqlPredicateFrame::InList { column, .. } => {
                let col_sc = resolve_schema_col(&column)?;
                Ok(BTreeSet::from([col_sc.name.clone()]))
            }
            SqlPredicateFrame::And(mut left, right) | SqlPredicateFrame::Or(mut left, right) => {
                left.extend(right);
                Ok(left)
            }
            SqlPredicateFrame::Not(inner) => Ok(inner),
        })
    }

    /// Validate this predicate AST against `table_name` in a [`RelationalSchemaGraph`].
    pub fn validate_against_graph(
        &self,
        graph: &RelationalSchemaGraph,
        table_name: &str,
    ) -> Result<BTreeSet<String>, SqlTypeError> {
        let Some(schema) = graph.get_table(table_name) else {
            let first_col = self
                .referenced_columns()
                .into_iter()
                .next()
                .unwrap_or_default();
            return Err(SqlTypeError::UnknownColumn {
                table: table_name.to_string(),
                column: first_col,
            });
        };
        let null_set = graph
            .nullable_columns
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(&schema.table_name))
            .map(|(_, set)| set);
        self.validate_against_schema(schema, null_set)
    }
}
