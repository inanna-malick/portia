# Algebraic Rust eDSLs & Hylomorphic Compilation (`hylo.rs`)

> Back to **[`README.md`](../README.md)** · See also: **[`docs/HOW_IT_WORKS.md`](HOW_IT_WORKS.md)** · **[`docs/ARCHITECTURE_AND_EXPLOITS.md`](ARCHITECTURE_AND_EXPLOITS.md)** · **[`docs/VISION.md`](VISION.md)**

---

## 1. Commuting Homomorphism Squares

Under the hood, [`crates/portia-core/src/hylo.rs`](../crates/portia-core/src/hylo.rs) uses the [`recursion`](https://docs.rs/recursion/latest/recursion/) crate (`expand_and_collapse`, `try_expand_and_collapse`, and `collapse_frames` over `MappableFrame`) to compile symbolic specifications directly into verified weight circuits across commuting homomorphism squares:

1. **Epistemic Regimes to `Clarabel` Conic `IntentExpr` Programs ([`crates/portia-core/src/pocket_dsl/mod.rs`](../crates/portia-core/src/pocket_dsl/mod.rs), [`crates/portia-compiler/src/ir/mod.rs`](../crates/portia-compiler/src/ir/mod.rs), & [`crates/portia-compiler/src/lowering/mod.rs`](../crates/portia-compiler/src/lowering/mod.rs)):**
   $$\text{decode}_{\text{mode}}\!\big(\text{MLP}_{W_0 \oplus \Delta W}(\mathbf{x}_s)\big) \;=\; \text{select\_mode}(R, s) \quad \forall s \in \text{PocketObsState}$$
   Lowers Futamura-specialized epistemic regime formulas ([`RegimeExpr`](../crates/portia-core/src/pocket_dsl/mod.rs)) into [`ScopedIntentProgram<DynamicSpanScope>`](../crates/portia-compiler/src/ir/scope_types.rs) & [`IntentExpr`](../crates/portia-compiler/src/ir/mod.rs) conic programs (`compile_intent_program`), automatically allocating disjoint slot registers (`allocate_neural_registers`) and verifying forward `bf16` SwiGLU bisimulation ([`verify_wire_cartridge_bisimulation`](../crates/portia-cli/src/compile/bisimulation.rs)).
2. **Monotonic Observation Lattice & Trace Diffing ([`crates/portia-core/src/pocket_dsl/mod.rs`](../crates/portia-core/src/pocket_dsl/mod.rs) & [`crates/portia-core/src/steward_coalgebra/mod.rs`](../crates/portia-core/src/steward_coalgebra/mod.rs)):**
   Unfolds reference agent trees (`Free`) in lockstep against stateful environment observations (`Cofree`) to verify monotonic invariants across all reachable future transitions (`prop_specialize_monotonic_invariants_valid_across_all_future_transitions`) alongside Kleene 3-valued relational AST evaluation ([`SqlPredicateAst`](../crates/portia-sql/src/relational_ast/mod.rs)).

---

## 2. Authoring & Specializing Regimes, `IntentExpr` Circuits, and Relational ASTs in Rust

All embedded DSLs in `portia-core`, `portia-compiler`, and `portia-sql` implement `recursion::MappableFrame` and overload standard Rust operators (`!`, `&`, `|`, `>>`) so you can author, specialize, and compile new behaviors in pure Rust without manual slot-offset bookkeeping.

Run the standalone `faer` + submerged `ASID` dyno and compiler test suites with:
```bash
cargo run --release -p portia-compiler --example compiler_dyno
cargo test -p portia-compiler
```

### Declarative `RegimeExpr` & `IntentExpr` Composition ([`crates/portia-compiler/src/ir/mod.rs`](../crates/portia-compiler/src/ir/mod.rs))

```rust
use portia_compiler::ir::scope_types::{DynamicSpanScope, ScopedIntentProgram};
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::compile_intent_program;
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_core::pocket_dsl::*;
use portia_core::{GrammarCartridgeSpec, Mode, MountPolicy};
use portia_sql::relational_ast::col;

// 1. Epistemic Regime eDSL (`RegimeExpr`): build with !, &, |, >> and Futamura-specialize:
let regime = (!has_schema_probe() >> Mode::Probe)
    | (!candidates_observed() >> Mode::Read)
    | ((is_locked_or_denied() & !steward_decision_is(Some(true))) >> Mode::Ask)
    | (!post_write_verified() >> Mode::Write)
    | emit_mode(Mode::Result);

let ro_regime = regime.specialize_for_schema_invariants(true, false, false, true);
assert_eq!(ro_regime.active_modes(), vec![Mode::Read, Mode::Result]);

// 2. Declarative `IntentExpr`: `high | low` propagates asymmetric priority inhibition (`<= -8.0`),
//    while `allocate_neural_registers` assigns disjoint `[slot_lo .. slot_hi)` ranges automatically:
let read_rule = IntentExpr::rule(IntentRule {
    name: "read_before_result".into(),
    guard: !candidates_observed(),
    mount_policy: MountPolicy::OrientTrunk,
    grammar: GrammarCartridgeSpec::Unconstrained,
    kind: HardwareIntentKind::StanceRoute {
        layer_idx: 22,
        raw_target_dir: v_read,
        angle: ChordalAngleRad::new(0.45),
        expected_mode: Some(Mode::Read),
    },
});
let result_rule = IntentExpr::rule(IntentRule {
    name: "emit_result".into(),
    guard: RegimeExpr::Lit(true),
    mount_policy: MountPolicy::OrientTrunk,
    grammar: GrammarCartridgeSpec::Unconstrained,
    kind: HardwareIntentKind::StanceRoute {
        layer_idx: 22,
        raw_target_dir: v_result,
        angle: ChordalAngleRad::new(0.45),
        expected_mode: Some(Mode::Result),
    },
});
let program = ScopedIntentProgram::<DynamicSpanScope>::new_dynamic(read_rule | result_rule);
let compiled = compile_intent_program(&program.expr, &captures, &geom)?;

// 3. Compositional SQL Predicate AST (`SqlPredicateAst`) with Kleene 3-valued truth:
let where_clause = col("delay_minutes").ge(30) & !col("change_freeze").eq_str("LOCKED");
assert_eq!(
    where_clause.render_sql(),
    "delay_minutes >= 30 AND NOT (change_freeze = 'LOCKED')"
);
```

---

## 3. Type-Level Exoskeleton Making Invalid States Irrepresentable (`ir/scope_types`, `ir/regalloc`, `discovery`, `wire`)

Beyond runtime `Clarabel` conic guarantees, `portia-compiler` and `portia-core` enforce physical and protocol invariants directly in Rust's type system (backed by `cargo test --doc` `compile_fail` proofs in [`ir/scope_types.rs`](../crates/portia-compiler/src/ir/scope_types.rs)):

1. **Static vs. Dynamic Scope Separation ([`ScopedIntentProgram<S>`](../crates/portia-compiler/src/ir/scope_types.rs)):**
   `ScopedIntentProgram<StaticPermanentScope>` forbids attaching runtime `llguidance` `.lark` or JSON-Schema masks to harness-free `.safetensors` patches at compile time (`with_runtime_grammar` is implemented exclusively on `ScopedIntentProgram<DynamicSpanScope>`).
2. **Causal Intra-Token Depth Monotonicity ([`TokenDepthChain<Start, End>`](../crates/portia-compiler/src/ir/scope_types.rs) & [`regalloc.rs`](../crates/portia-compiler/src/ir/regalloc.rs)):**
   Sealed `CausalDepthAdvance<Next>` markers (`EarlyPreGqa -> MidDeltaNet -> LateStance`) make reverse layer chains (`LateStance -> EarlyPreGqa`) uncompilable, while `allocate_neural_registers` verifies `upstream_mlp_layer < target_layer` on every cross-layer intent (`SubmergedPointerRoute`, `AsidSegmentColor`, `DeltaNetRollbackScrub`).
3. **Static-First Disjoint Register Allocation ([`allocate_neural_registers`](../crates/portia-compiler/src/ir/regalloc.rs)):**
   Pass 1 reserves `[0 .. static_floor[layer])` for `MountPolicy::PermanentStatic` rules before Pass 2 bump-allocates dynamic phase banks starting at `static_floor[layer]`, guaranteeing `permanent_static ∩ dynamic_banks == ∅` across every layer in [`WireCartridgeManifestJson`](../crates/portia-compiler/src/wire.rs).
4. **Contextual Secant Anti-Echo & Submerged `W_V` Nullspaces ([`discovery/mod.rs`](../crates/portia-compiler/src/discovery/mod.rs) & [`discovery/gqa_bus.rs`](../crates/portia-compiler/src/discovery/gqa_bus.rs)):**
   `discover_and_project_anti_echo` probes Layer $\ell + 1$ with constant-energy $\operatorname{SO}(d)$ steps to project out true downstream suppressor neurons, while `GqaPointerLayer::build_submerged_forbidden_basis` locks GQA pointer/ASID writes inside $\ker(\mathbf{W}_V \operatorname{diag}(\gamma_{\text{in}})) \cap \ker(\mathbf{W}_{U,\text{top}})$.
5. **Type-Witnessed Wire Normalization ([`LotlNormalizedWireChunk`](../crates/portia-core/src/relay/mod.rs)):**
   `StdioQwenWorkerClient` requires `LotlNormalizedWireChunk` and explicit `KvPrefillDisposition` at compile time, keeping state-machine logic exclusively in Rust.

---

## 4. Crate-by-Crate Architecture (`0` Clippy Warnings)

| Component | Path | Core Responsibilities & Compile-Time Invariants |
| :--- | :--- | :--- |
| **`portia-core`** | [`crates/portia-core`](../crates/portia-core/src/lib.rs) | Canonical 6-mode vocabulary (`Mode`: natural BPE tokens `" PRO"`, `" READ"`, `" ASK"`, `" WRITE"`, `" RESULT"`, `" AB"`) and `RouteTag` (`<sql>`, `<to_user>`), phase-specialized `MountPolicy`, Single-Pen typestate [`RelayMachine`](../crates/portia-core/src/relay/mod.rs) & 3-phase [`SpanCartridge`](../crates/portia-core/src/relay/mod.rs) (`ModeLabel`, `PlanBody`, `PayloadBody`), canonical `TableSchema` & `ColumnSchema` ([`domain.rs`](../crates/portia-core/src/domain/mod.rs)), Qwen3.5 BPE `\n\n` (`271`) boundary healer ([`bpe.rs`](../crates/portia-core/src/bpe.rs)), stack-safe recursion engine & `RegimeExpr` eDSL ([`hylo.rs`](../crates/portia-core/src/hylo.rs), [`pocket_dsl.rs`](../crates/portia-core/src/pocket_dsl/mod.rs)), and lockstep Steward negotiation trace differ ([`steward_coalgebra.rs`](../crates/portia-core/src/steward_coalgebra/mod.rs)). |
| **`portia-compiler`** | [`crates/portia-compiler`](../crates/portia-compiler/src/lib.rs) | Unified `<= 350`-LOC Pointer-ISA compiler: `faer` thin-SVD subspaces ([`numerics/subspace.rs`](../crates/portia-compiler/src/numerics/subspace.rs)), constant-energy [`StiefelChordalFrame`](../crates/portia-compiler/src/numerics/sod.rs), pure-Rust `.safetensors` readers ([`numerics/safetensors_io.rs`](../crates/portia-compiler/src/numerics/safetensors_io.rs)), contextual secant anti-echo & submerged GQA `ASID` bus ([`discovery/mod.rs`](../crates/portia-compiler/src/discovery/mod.rs), [`discovery/gqa_bus.rs`](../crates/portia-compiler/src/discovery/gqa_bus.rs)), 7-bank/2-slot phase-multiplexed `ASID` MMU ([`phase_mmu.rs`](../crates/portia-compiler/src/phase_mmu.rs)), `AlphaEdit` / `AlphaEdit+` `Clarabel` SOCP solvers ([`solvers/conic_qp.rs`](../crates/portia-compiler/src/solvers/conic_qp.rs)), 2-slot convex-midpoint ternary forks ([`solvers/clarabel_ternary.rs`](../crates/portia-compiler/src/solvers/clarabel_ternary.rs)), `in_proj_a` `DeltaNet-1` scrubbers ([`solvers/deltanet_scrub.rs`](../crates/portia-compiler/src/solvers/deltanet_scrub.rs)), [`IntentExpr`](../crates/portia-compiler/src/ir/mod.rs), [`ScopedIntentProgram<S>`](../crates/portia-compiler/src/ir/scope_types.rs), [`regalloc.rs`](../crates/portia-compiler/src/ir/regalloc.rs), [`lowering/mod.rs`](../crates/portia-compiler/src/lowering/mod.rs), [`ConcolicDiodeLens`](../crates/portia-compiler/src/concolic/mod.rs), and [`WireCartridgeManifestJson`](../crates/portia-compiler/src/wire.rs). |
| **`portia-sql`** | [`crates/portia-sql`](../crates/portia-sql/src/lib.rs) | Hardened SQLite provenance & shadow-diff helpers ([`engine.rs`](../crates/portia-sql/src/engine/mod.rs): `authorizer` provenance, `PRAGMA query_only = ON`, `SAVEPOINT` rollback, shadow database diffs), live `PRAGMA table_info` / `PRAGMA foreign_key_list` [`RelationalSchemaGraph`](../crates/portia-sql/src/relational_ast/mod.rs) & Kleene 3-valued `SqlPredicateAst`, stratified [`grammars/upstream/sqlite.lark`](../grammars/upstream/sqlite.lark) + prompt-blind causal [`EpistemicSqlGrammarCartridge`](../crates/portia-sql/src/grammar_trie/mod.rs), submerged pointer & `ASID` bridge ([`v2_pointer_bridge.rs`](../crates/portia-sql/src/v2_pointer_bridge.rs)), DDL normalizer ([`schema_normalizer.rs`](../crates/portia-sql/src/schema_normalizer.rs)), and modular SQL + gibberish mutant-oracle suite ([`roguelite_dungeon/`](../crates/portia-sql/src/roguelite_dungeon/mod.rs)). |
| **`portia-nursery` & `portia-garak`** | [`crates/portia-nursery`](../crates/portia-nursery/README.md) & [`crates/portia-garak`](../crates/portia-garak/README.md) | 7-axis `proptest` stress lattice ([`roguelite_lattice.rs`](../crates/portia-nursery/src/roguelite_lattice/mod.rs)), multilingual `ScenarioCoordinate`s ([`nursery_scenario.rs`](../crates/portia-nursery/src/nursery_scenario/mod.rs)), 3-state `SignedTernaryFork` & `DeltaNet` rollback evaluators ([`v2_ternary_hylo.rs`](../crates/portia-nursery/src/v2_ternary_hylo.rs)), `tree-sitter-rust` + `syn` AST hole lens ([`rust_ast_lens.rs`](../crates/portia-nursery/src/rust_ast_lens/mod.rs)), and `Garak` `StrictZeroCone` static honesty specs + white-box concolic `MAP-Elites` explorer ([`v2_concolic.rs`](../crates/portia-garak/src/v2_concolic.rs)). |
| **`portia-cli` + Stdio Worker** | [`crates/portia-cli`](../crates/portia-cli/src/main.rs) & [`scripts/portia_qwen_stdio_worker.py`](../scripts/portia_qwen_stdio_worker.py) | Daily-driver `clap-derive` CLI (`tui`, `repl`, `run`, `batch`, `inspect`, `compile`, `compile-garak`, `export-safetensors`, `verify-schedule`, `nursery`; [`main.rs`](../crates/portia-cli/src/main.rs), [`batch_cmd.rs`](../crates/portia-cli/src/batch_cmd.rs), [`cli_tests.rs`](../crates/portia-cli/src/cli_tests.rs)), prompt-blind `EpistemicDomainEnv` backends ([`driver/`](../crates/portia-cli/src/driver/mod.rs), [`rust_driver.rs`](../crates/portia-cli/src/rust_driver.rs)), single OODA coroutine loop ([`worker/`](../crates/portia-cli/src/worker/mod.rs)), pure-Rust simulated-user coalgebra ([`synth_user/`](../crates/portia-cli/src/synth_user/mod.rs)), nursery & Rust hylo runners ([`nursery_cmd/`](../crates/portia-cli/src/nursery_cmd/mod.rs), [`rust_hylo_cmd.rs`](../crates/portia-cli/src/rust_hylo_cmd.rs)), cartridge compilers ([`compile.rs`](../crates/portia-cli/src/compile.rs), [`compile_v2.rs`](../crates/portia-cli/src/compile_v2.rs), [`compile_garak.rs`](../crates/portia-cli/src/compile_garak.rs)), streaming `.safetensors` exporter ([`export_safetensors.rs`](../crates/portia-cli/src/export_safetensors.rs)), and `iocraft` TUI ([`tui.rs`](../crates/portia-cli/src/tui.rs)). |
