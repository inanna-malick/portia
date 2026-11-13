//! `portia-compiler` — Pointer-ISA & Subspace Weight Compiler (`faer` SIMD/Rayon + `clarabel`).
//!
//! Modular `< 350 LOC`-per-file architecture:
//! - [`numerics`]: Math-free `StateMatrix` boundary + `faer` thin-SVD subspaces + constant-energy `SO(d)` frames.
//! - [`discovery`]: Contextual-secant anti-echo suppressor discovery + submerged GQA pointer bus (`ker(W_V) ∩ ker(W_U_top)`) + paired `ASID` symbol-table coloring (`W_K` write ⋈ `W_Q` read).
//! - [`solvers`]: `AlphaEdit` / `AlphaEdit+` conic nullspace policies + convex-feasible ternary bilinear `SwiGLU` solver + `DeltaNet-1` `conv1d`-bypassing `in_proj_a` scrubber.
//! - [`ir`]: Declarative `IntentExpr` (`MappableFrame`) DSL + automatic neural slot register allocator.
//! - [`lowering`]: Lowering pass emitting runtime-compatible [`wire::WireCartridgeManifestJson`].
//! - [`concolic`]: White-box `MAP-Elites` cliff-margin & `ker(w_gate) ∩ Ω_dark` diversity explorer.

pub mod concolic;
pub mod discovery;
pub mod error;
pub mod ir;
pub mod lowering;
pub mod numerics;
pub mod phase_mmu;
pub mod solvers;
pub mod wire;

pub use error::V2CompilerError;
