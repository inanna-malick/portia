//! Rust CST/AST `todo!("...")` Typed-Hole Lens, Splicer, Hylomorphic Octo-Merger,
//! and Interactive Scenario Session (`portia_nursery::rust_ast_lens`).
//!
//! Splits general-purpose `tree-sitter-rust` + `syn` hole extraction, window slicing,
//! and AST-verified splicing (`span`, `cst_extract`, `splice_merge`, `grammar`) from
//! the interactive `UnifiedRustAstScenarioSession` state machine (`scenario_session`).

pub mod cst_extract;
pub mod grammar;
mod scenario_affordances;
pub mod scenario_session;
pub mod span;
pub mod splice_merge;

#[cfg(test)]
mod tests;

pub use cst_extract::{
    extract_rust_todo_holes, format_rust_ast_outline, parse_and_validate_rust_cst_and_ast,
    slice_rust_fn_window, verify_balanced_rust_delimiters,
};
pub use grammar::RustAstHoleGrammarCartridge;
pub use scenario_session::{RustAstHyloSession, RustAstTurnOutcome, UnifiedRustAstScenarioSession};
pub use span::{ByteSpan, ExtractedRustTodoHole};
pub use splice_merge::{hylo_octo_merge_rust_holes, splice_rust_todo_hole_body};
