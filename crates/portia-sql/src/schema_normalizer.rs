//! Canonical SQLite `<schema>` Normalizer & Progressive Table-Directory Renderer (`schema_normalizer`).
//!
//! Aligns three host-to-model seams deterministically from [`RelationalSchemaGraph`] (`PRAGMA table_info`
//! + `PRAGMA foreign_key_list`) without spending neural `SwiGLU` capacity on raw DDL quirks:
//! 1. **Inline-`REFERENCES` Causal Prefix Normalization:** Rewrites arbitrary SQLite `CREATE TABLE` DDL
//!    (including inline `col INT REFERENCES parent(pk)` and implicit-PK `REFERENCES parent`) into
//!    canonical table-level `FOREIGN KEY (<child_col>) REFERENCES <parent_table>(<parent_col>)` clauses
//!    so `FOREIGN KEY (` always precedes `<child_col>` for causal `L0..L18` key-stamping.
//! 2. **Deterministic `1:N` Fan-Out Cardinality Classification:** Classifies foreign keys where
//!    `<child_col>` is not the sole primary key of `<child_table>` (`1:N` fan-out) directly from
//!    `PRAGMA` metadata (`classify_fk_cardinality`), without injecting English coaching comments into SQL.
//! 3. **Progressive Wide-Schema Disclosure (`<schema_index>` -> `<schema>` slice):** Renders compact
//!    table + FK directories for `40+`-table databases and canonicalizes only the inspected table subset
//!    (plus optional 1-hop FK neighbors).

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::engine::EngineError;
use crate::relational_ast::{ForeignKeyEdge, RelationalSchemaGraph};
use portia_core::TableSchema;

/// Deterministic relational join cardinality for a [`ForeignKeyEdge`] (`child.fk -> parent.pk`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FkJoinCardinality {
    /// `child.fk` is the sole `PRIMARY KEY` of `child` (`<= 1` child row per parent row).
    OneToZeroOrOne,
    /// `child.fk` can repeat across multiple `child` rows (`JOIN child` multiplies `parent` aggregates!).
    OneToManyFanOut,
}

/// Classify whether joining `parent_table` to `child_table` along `edge` can multiply `parent_table` rows.
#[must_use]
pub fn classify_fk_cardinality(
    child_schema: &TableSchema,
    edge: &ForeignKeyEdge,
) -> FkJoinCardinality {
    let pk_cols: Vec<&str> = child_schema
        .columns
        .iter()
        .filter(|c| c.is_primary_key)
        .map(|c| c.name.as_str())
        .collect();
    if pk_cols.len() == 1 && pk_cols[0].eq_ignore_ascii_case(&edge.child_col) {
        FkJoinCardinality::OneToZeroOrOne
    } else {
        FkJoinCardinality::OneToManyFanOut
    }
}

/// Render a single table in canonical causal-prefix SQLite DDL (`FOREIGN KEY (...)` always explicit
/// at table level, pure standard SQL syntax with zero English prompt comments).
#[must_use]
pub fn render_canonical_table_ddl(schema: &TableSchema, outgoing_fks: &[ForeignKeyEdge]) -> String {
    let pk_cols: Vec<&str> = schema
        .columns
        .iter()
        .filter(|c| c.is_primary_key)
        .map(|c| c.name.as_str())
        .collect();
    let composite_pk = pk_cols.len() > 1;

    let mut lines: Vec<String> = Vec::with_capacity(schema.columns.len() + outgoing_fks.len() + 1);
    for col in &schema.columns {
        let col_type = if col.col_type.trim().is_empty() {
            "ANY"
        } else {
            col.col_type.trim()
        };
        let mut decl = format!("  {} {}", col.name, col_type);
        if col.is_primary_key && !composite_pk {
            decl.push_str(" PRIMARY KEY");
        } else if !col.nullable() {
            decl.push_str(" NOT NULL");
        }
        lines.push(decl);
    }

    if composite_pk {
        lines.push(format!("  PRIMARY KEY ({})", pk_cols.join(", ")));
    }

    for fk in outgoing_fks {
        lines.push(format!(
            "  FOREIGN KEY ({}) REFERENCES {}({})",
            fk.child_col, fk.parent_table, fk.parent_col
        ));
    }

    format!(
        "CREATE TABLE {} (\n{}\n);",
        schema.table_name,
        lines.join(",\n")
    )
}

/// Render a compact `<schema_index>` directory (`1` line per table + FK topology) for `40+`-table
/// databases so Turn 0 avoids flooding `DeltaNet` state and `GQA` softmax before table selection.
#[must_use]
pub fn render_compact_table_directory(graph: &RelationalSchemaGraph) -> String {
    let mut fk_by_child: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for fk in &graph.foreign_keys {
        fk_by_child
            .entry(fk.child_table.as_str())
            .or_default()
            .push(format!(
                "{}->{}.{}",
                fk.child_col, fk.parent_table, fk.parent_col
            ));
    }

    let mut out = format!("<schema_index tables=\"{}\">\n", graph.tables.len());
    for (tname, tschema) in &graph.tables {
        let pk_list: Vec<&str> = tschema
            .columns
            .iter()
            .filter(|c| c.is_primary_key)
            .map(|c| c.name.as_str())
            .collect();
        let pk_tag = if pk_list.is_empty() {
            String::new()
        } else {
            format!(" pk=({})", pk_list.join(","))
        };
        let fk_tag = match fk_by_child.get(tname.as_str()) {
            Some(edges) => format!(" fk=[{}]", edges.join(", ")),
            None => String::new(),
        };
        out.push_str(&format!(
            "  - {} ({} cols{}{})\n",
            tname,
            tschema.columns.len(),
            pk_tag,
            fk_tag
        ));
    }
    out.push_str("</schema_index>");
    out
}

/// Render canonical `<schema>...</schema>` DDL for a selected subset of tables (or all tables when
/// `selected_tables` is `None`), optionally expanding 1-hop FK neighbors so parent/child targets
/// are always co-present for joins.
#[must_use]
pub fn render_canonical_schema_slice(
    graph: &RelationalSchemaGraph,
    selected_tables: Option<&BTreeSet<String>>,
    expand_one_hop_fk_neighbors: bool,
) -> String {
    let mut keep: BTreeSet<String> = match selected_tables {
        None => graph.tables.keys().cloned().collect(),
        Some(sel) => graph
            .tables
            .keys()
            .filter(|t| sel.iter().any(|s| s.eq_ignore_ascii_case(t)))
            .cloned()
            .collect(),
    };

    if expand_one_hop_fk_neighbors && selected_tables.is_some() {
        let seed = keep.clone();
        for fk in &graph.foreign_keys {
            if seed.contains(&fk.child_table) && graph.tables.contains_key(&fk.parent_table) {
                keep.insert(fk.parent_table.clone());
            }
            if seed.contains(&fk.parent_table) && graph.tables.contains_key(&fk.child_table) {
                keep.insert(fk.child_table.clone());
            }
        }
    }

    let mut blocks = Vec::with_capacity(keep.len());
    for tname in &keep {
        if let Some(schema) = graph.tables.get(tname) {
            let mut outgoing: Vec<ForeignKeyEdge> = graph
                .foreign_keys
                .iter()
                .filter(|fk| fk.child_table == *tname && keep.contains(&fk.parent_table))
                .cloned()
                .collect();
            outgoing.sort();
            blocks.push(render_canonical_table_ddl(schema, &outgoing));
        }
    }

    format!("<schema>\n{}\n</schema>", blocks.join("\n"))
}

/// Convenience entry point: introspect `conn` via [`RelationalSchemaGraph::inspect_from_connection`]
/// and emit normalized causal-prefix `<schema>...</schema>` DDL across all tables.
pub fn normalize_connection_schema_ddl(conn: &Connection) -> Result<String, EngineError> {
    let graph = RelationalSchemaGraph::inspect_from_connection(conn)?;
    Ok(render_canonical_schema_slice(&graph, None, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_references_and_implicit_pk_normalize_to_explicit_table_foreign_key() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE t_parent (c_pk INTEGER PRIMARY KEY, c_fee INTEGER NOT NULL);
             CREATE TABLE t_child (
                 id INTEGER PRIMARY KEY,
                 c_inline_fk INTEGER NOT NULL REFERENCES t_parent,
                 c_decoy INTEGER NOT NULL
             );",
        )
        .unwrap();

        let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
        let ddl = normalize_connection_schema_ddl(&conn).unwrap();
        assert!(
            ddl.contains("FOREIGN KEY (c_inline_fk) REFERENCES t_parent(c_pk)"),
            "expected explicit normalized FK clause, got:\n{ddl}"
        );
        assert!(
            !ddl.contains("--"),
            "normalized DDL must be pure standard SQL with zero English coaching comments"
        );
        assert_eq!(
            classify_fk_cardinality(&graph.tables["t_child"], &graph.foreign_keys[0]),
            FkJoinCardinality::OneToManyFanOut
        );
    }

    #[test]
    fn progressive_directory_and_one_hop_slice_isolate_target_tables_from_50_distractors() {
        let conn = Connection::open_in_memory().unwrap();
        let mut batch = String::from(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, shipping_cents INTEGER NOT NULL);
             CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(order_id), sku TEXT NOT NULL);",
        );
        for idx in 0..48 {
            batch.push_str(&format!(
                "CREATE TABLE distractor_{idx:02} (id INTEGER PRIMARY KEY, v0 INT, v1 TEXT);"
            ));
        }
        conn.execute_batch(&batch).unwrap();

        let graph = RelationalSchemaGraph::inspect_from_connection(&conn).unwrap();
        assert_eq!(graph.tables.len(), 50);
        let dir = render_compact_table_directory(&graph);
        assert!(dir.contains("<schema_index tables=\"50\">"));
        assert!(dir.contains("order_items (3 cols pk=(item_id) fk=[order_id->orders.order_id])"));

        let mut selected = BTreeSet::new();
        selected.insert("order_items".to_string());
        let slice = render_canonical_schema_slice(&graph, Some(&selected), true);
        assert!(slice.contains("CREATE TABLE orders"));
        assert!(slice.contains("CREATE TABLE order_items"));
        assert!(!slice.contains("distractor_00"));
    }
}
