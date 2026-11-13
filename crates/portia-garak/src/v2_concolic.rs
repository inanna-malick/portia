//! `portia-garak` Declarative `IntentExpr` Lowering & White-Box Concolic `MAP-Elites` Suite.
//!
//! Lowers [`GarakCompileInputSpec`] directly through unified [`portia_compiler`]:
//! - Composes the 4 [`crate::GarakProbeFamily`] obligations via parallel superposition (`&`) in
//!   [`portia_compiler::ir::IntentExpr`] wrapped in [`portia_compiler::ir::scope_types::ScopedIntentProgram<StaticPermanentScope>`],
//! - Enforces [`portia_compiler::solvers::NullspacePolicy::StrictZeroCone`] (`AlphaEdit`) preservation
//!   on benign/continuation subspaces extracted via `faer` thin SVD,
//! - Runs contextual chordal-secant [`DownstreamSwigluBlock`] anti-echo discovery whenever downstream
//!   (`L_total - 1`) MLP weights are supplied,
//! - Emits `(WireCartridgeManifestJson, GarakCompileReceipt, BTreeMap<String, ConcolicDiodeLens>)` and
//!   exposes white-box concolic `MAP-Elites` discrete structured probe exploration (`explorer`).

pub mod explorer;
mod explorer_blind;
pub mod lattice;
pub mod lowering;
pub(crate) mod readout_ds;
pub mod splits;
mod splits_classic;

pub use explorer::{
    run_blind_blackbox_probe_search, run_whitebox_concolic_map_elites_search,
    ConcolicProbeEvaluation, DiscreteGarakProbe, GarakSearchReport,
};
pub use lattice::{build_structured_garak_probe_lattice, StructuredGarakLattice};
pub use lowering::{build_garak_v2_intent_expr, GARAK_V2_OBLIGATION_FAMILIES};

use crate::specs::{GarakCompileInputSpec, GarakCompileReceipt, GarakSolverError};
use portia_compiler::concolic::ConcolicDiodeLens;
use portia_compiler::discovery::DownstreamSwigluBlock;
use portia_compiler::lowering::CompiledProgramV2;
use portia_compiler::numerics::subspace::OrthonormalSubspace;
use portia_compiler::wire::WireCartridgeManifestJson;
use std::collections::BTreeMap;

/// Optional model-weight geometry tables (loaded from `.safetensors` or downstream MLP payloads)
/// enriching `portia-garak` compilation with contextual secant anti-echo and `W_U` ridge bases.
#[derive(Debug, Clone, Default)]
pub struct GarakV2GeometryExtras {
    pub downstream_mlp: Option<DownstreamSwigluBlock>,
    pub w_u_top_basis: Option<OrthonormalSubspace>,
}

/// Complete `portia-garak` compilation bundle pairing the runtime wire manifest, scalar Cold Seam
/// Audit receipt, per-rule [`ConcolicDiodeLens`] map, and underlying [`CompiledProgramV2`].
#[derive(Debug, Clone)]
pub struct CompiledGarakV2Output {
    pub manifest: WireCartridgeManifestJson,
    pub receipt: GarakCompileReceipt,
    pub concolic_lenses: BTreeMap<String, ConcolicDiodeLens>,
    pub program: CompiledProgramV2,
}

/// Compile a [`GarakCompileInputSpec`] directly through unified [`portia_compiler`] (`IntentExpr`
/// `Parallel &`, `StaticPermanentScope`, `NullspacePolicy::StrictZeroCone` `AlphaEdit` preservation,
/// and `faer` thin SVD), failing closed when `guard_bg_max > spec.max_bg_rss_leakage`.
pub fn compile_garak_v2_cartridge(
    spec: &GarakCompileInputSpec,
) -> Result<
    (
        WireCartridgeManifestJson,
        GarakCompileReceipt,
        BTreeMap<String, ConcolicDiodeLens>,
    ),
    GarakSolverError,
> {
    let out = compile_garak_v2_with_geometry(spec, GarakV2GeometryExtras::default())?;
    Ok((out.manifest, out.receipt, out.concolic_lenses))
}

/// Compile a [`GarakCompileInputSpec`] through unified [`portia_compiler`] with optional explicit
/// [`GarakV2GeometryExtras`] (e.g. pure-Rust `.safetensors` Layer-`(L+1)` `DownstreamSwigluBlock`
/// and top-`k` `W_U` basis for 4-axis `ker(w_gate) ∩ ker(W_U_top)` concolic ridge cells).
pub fn compile_garak_v2_with_geometry(
    spec: &GarakCompileInputSpec,
    extras: GarakV2GeometryExtras,
) -> Result<CompiledGarakV2Output, GarakSolverError> {
    lowering::lower_and_compile_garak_v2(spec, extras)
}
