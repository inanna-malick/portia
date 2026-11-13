# `portia-compiler` — Pointer-ISA, 4-Color `ASID` MMU, & `faer` + `Clarabel` Weight Compiler

`portia-compiler` compiles declarative weight-space programs into biasless `SwiGLU` neuron banks using SIMD/Rayon linear algebra (`faer 0.24`), [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.rs) interior-point second-order cone programming (`SOCP`), submerged address-bus pointer routing (`GCAD`), deterministic phase-multiplexed memory tagging (`ASID`), 3-state ternary `SwiGLU` forks, and white-box concolic `MAP-Elites` boundary exploration—keeping every `.rs` module strictly `<= 350` lines.

---

## 1. Module Map (`<= 350` LOC per file)

| Module | Role |
| :--- | :--- |
| **[`src/numerics/`](src/numerics/mod.rs)** | `StateMatrix`, `faer` thin-SVD [`OrthonormalSubspace`](src/numerics/subspace.rs), constant-energy [`StiefelChordalFrame`](src/numerics/sod.rs) & [`ChordalAngleRad`](src/numerics/sod.rs) $\operatorname{SO}(d)$ 2-frames, and pure-Rust `.safetensors` readers ([`safetensors_io.rs`](src/numerics/safetensors_io.rs)). |
| **[`src/discovery/`](src/discovery/mod.rs)** | Contextual chordal-secant suppressor discovery ([`DownstreamSwigluBlock`](src/discovery/mod.rs)) and submerged GQA pointer routing + `ASID` codebooks inside $\ker(\mathbf{W}_V) \cap \ker(\mathbf{W}_{U,\text{top}})$ ([`gqa_bus.rs`](src/discovery/gqa_bus.rs)). |
| **[`src/phase_mmu.rs`](src/phase_mmu.rs)**, **[`src/phase_mmu/dc_slots.rs`](src/phase_mmu/dc_slots.rs)**, & **[`src/phase_mmu/cursor_slots.rs`](src/phase_mmu/cursor_slots.rs)** | **Deterministic Phase-Multiplexed 4-Color `ASID` MMU:** Compiles `7` host-known phase banks (`Prefill:Schema`, `Prefill:User`, `Prefill:ToolObs`, `ModePlan:Read`, `ModePayload:Read`, `ModePlan:Result`, `ModePayload:Result`) into **just `2` physical `SwiGLU` neuron slots (`M -> M + 2`)** using either DC-carrier constant-on slots or `Clarabel` intra-phase 1-token Kronecker cursor diodes (`< 1e-13` `f64` `W_V` bleed, preserving non-cursor SQL syntax & `\n` tokens). |
| **[`src/solvers/`](src/solvers/mod.rs)** | `AlphaEdit` (`StrictZeroCone`) & `AlphaEdit+` (`BoundedSocSlack`) sparse diagonal-`P` `Clarabel` SOCP solvers ([`conic_qp.rs`](src/solvers/conic_qp.rs)) with exact orthonormal row-space reduction $\mathbf{w} = \mathbf{U}\mathbf{z}$ ([`conic_subspace.rs`](src/solvers/conic_subspace.rs)) and exact KKT active-set cutting-plane reduction ([`conic_active_set.rs`](src/solvers/conic_active_set.rs)), 2-slot 3-state `SignedTernaryFork` (`{+1, 0, -1}` on linear `w_up_tan`, solving convex-midpoint `ASK` routing in [`clarabel_ternary.rs`](src/solvers/clarabel_ternary.rs)), and `conv1d`-bypassing `in_proj_a` `DeltaNet-1` head-partitioned scrubbers ([`deltanet_scrub.rs`](src/solvers/deltanet_scrub.rs)). |
| **[`src/ir/`](src/ir/mod.rs)** & **[`src/lowering/`](src/lowering/mod.rs)** | Stack-safe `recursion` `MappableFrame` [`IntentExpr`](src/ir/mod.rs), scope & causal-depth type witnesses ([`scope_types.rs`](src/ir/scope_types.rs): `StaticPermanentScope`, `DynamicSpanScope`, `ScopedIntentProgram<S>`, `TokenDepthChain<Start, End>`), automatic static-first slot register allocator ([`regalloc.rs`](src/ir/regalloc.rs)), and deterministic `std::thread::scope` parallel lowering ([`rule_worker.rs`](src/lowering/rule_worker.rs)) to runtime-compatible [`WireCartridgeManifestJson`](src/wire.rs). |
| **[`src/concolic/`](src/concolic/mod.rs)** | White-box [`ConcolicDiodeLens`](src/concolic/mod.rs) (`cliff_margin = w_gate · x`) & 4-axis $\ker(w_{\text{gate}}) \cap \ker(\mathbf{W}_{U,\text{top}})$ `MAP-Elites` archive for discrete structured boundary fuzzing. |

---

## 2. Quick Verification & Dyno Commands

```bash
# Unit & invariant test suites (enforcing <= 350 LOC per file, wire roundtrips, and < 1e-12 W_V isolation):
cargo test -p portia-compiler

# Pure-Rust faer thin-SVD, Clarabel sparse SOCP, ASID codebook, & Qwen3.5 .safetensors compilation dyno:
cargo run --release -p portia-compiler --example compiler_dyno
```
