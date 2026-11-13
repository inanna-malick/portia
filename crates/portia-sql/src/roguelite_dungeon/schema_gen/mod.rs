//! Roguelite SQL Dungeon Schema & Preset Generators (`RowHaystackDungeon`, `WideTableDungeon`,
//! `MultiTableFkGraphDungeon`) running directly against `rusqlite::Connection`.

mod ddl_shared;
mod fk_graph;
mod haystack;
mod wide_table;

#[cfg(test)]
mod tests;

pub(super) use ddl_shared::{
    append_shared_fk_chain_and_distractors_ddl, build_shared_column_layout,
    open_seeded_in_memory_connection,
};
pub use ddl_shared::{DECEPTIVE_COLUMN_STEMS, HAYSTACK_CLIFF_POINTS};
pub use fk_graph::{
    arb_multi_table_count, arb_multi_table_fk_graph_spec, FkEdgeAction,
    MultiTableFkGraphDungeonInstance, MultiTableFkGraphDungeonSpec,
};
pub use haystack::{
    arb_row_haystack_count, arb_row_haystack_dungeon_spec, evaluate_raw_tanking_vs_fog_finesse,
    HaystackStrategyComparison, RowHaystackDungeonInstance, RowHaystackDungeonSpec,
};
pub use wide_table::{
    arb_wide_column_count, arb_wide_table_dungeon_spec, WideTableDungeonInstance,
    WideTableDungeonSpec, WideTableFinesseComparison,
};
