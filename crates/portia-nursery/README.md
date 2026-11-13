# `portia-nursery` — Procedural Stress Lattices, Scenario Coordinates, & Rust AST Lenses

`portia-nursery` houses Portia's procedural environment generators, monotone `proptest` failure-cliff shrinkers, and verified `tree-sitter-rust` + `syn` AST hole lenses, keeping `portia-core` lean and free of parser/property-testing production dependencies.

## Modules

- **[`roguelite_lattice`](src/roguelite_lattice/mod.rs):** 7-dimensional continuous stress lattice (`RogueliteStressVector`), `65% / 25% / 10%` sweet-spot / deep-dungeon / boss-floor `proptest` strategies, monotone binary-search wipeout-cliff finder (`find_exact_wipeout_cliff`), and poset lower-cover shrinker (`shrink_stress_vector_to_irreducible_cliff`).
- **[`nursery_scenario`](src/nursery_scenario/mod.rs):** Tensor-product `ScenarioCoordinate` (`DomainSkin` × `ScenarioLanguage` across `EN`, `JA`, `RU`, `ES`, `DE`, `ID` × `SqlQueryShape` × `AppType`), Steward multi-turn agendas, and sealed `rustc --test` verification harnesses (`CleanroomRustHyloSpec`).
- **[`rust_ast_lens`](src/rust_ast_lens/mod.rs):** Dual `tree-sitter-rust` CST + `syn::parse_file` AST typed-hole extractor (`extract_rust_todo_holes`), `syn::Block` hole-splicer (`splice_rust_todo_hole_body`), and `RoseFrame` hylomorphism octo-merger (`hylo_octo_merge_rust_holes`).
- **[`v2_ternary_hylo`](src/v2_ternary_hylo.rs):** Convex-midpoint ternary `FORK / ASK / INLINE` seam classifier, `DeltaNet-1` rollback scrubber lowering via `portia_compiler`, and cleanroom Rust-AST `todo!("...")` world synthesis.
