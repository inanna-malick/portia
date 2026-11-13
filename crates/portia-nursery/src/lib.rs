//! `portia-nursery` — Procedural Roguelite Stress Lattices, Multilingual Scenario Coordinates,
//! & Verified Rust AST Hylomorphism Lenses (`roguelite_lattice`, `nursery_scenario`, `rust_ast_lens`).
//!
//! Keeps procedural `proptest` shrinkers (`RogueliteStressVector`, `ScenarioCoordinate`) and
//! `tree-sitter-rust` / `syn` AST hole lenses cleanly decoupled from `portia-core`'s minimal
//! wire-protocol and single-pen relay kernel.

pub mod nursery_scenario;
pub mod roguelite_lattice;
pub mod rust_ast_lens;
pub mod v2_ternary_hylo;

pub use roguelite_lattice::{arb_biased_three_band_range, arb_needle_depth};
pub use rust_ast_lens::ByteSpan;
