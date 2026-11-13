//! Model geometry & activation capture tables consumed by the `v2` lowering pass (`lowering::tables`).

use crate::concolic::ConcolicDiodeLens;
use crate::discovery::gqa_bus::GqaPointerLayer;
use crate::discovery::DownstreamSwigluBlock;
use crate::ir::regalloc::AllocatedIntentRule;
use crate::numerics::subspace::OrthonormalSubspace;
use crate::numerics::StateMatrix;
use crate::solvers::clarabel_ternary::CursorPolarity;
use crate::solvers::deltanet_scrub::DeltaNetGateLayer;
use crate::solvers::V2SolverConfig;
use crate::wire::WireCartridgeManifestJson;
use std::collections::BTreeMap;

/// Captured activations & polarities for a single declarative rule.
#[derive(Debug, Clone)]
pub struct RuleCaptureData {
    pub on_cursor_pre_mlp: StateMatrix,
    pub polarities: Vec<CursorPolarity>,
    pub pos_residuals: StateMatrix,
    pub off_cursor_pre_mlp: StateMatrix,
    pub q_preserve_gate: Option<OrthonormalSubspace>,
    pub solver_config: V2SolverConfig,
}

impl RuleCaptureData {
    /// Convenience constructor for standard `+1` positive rules (`StanceRoute`, `AsidSegmentColor`, etc.).
    #[must_use]
    pub fn standard_positive(
        on_cursor_pre_mlp: StateMatrix,
        pos_residuals: StateMatrix,
        off_cursor_pre_mlp: StateMatrix,
        solver_config: V2SolverConfig,
    ) -> Self {
        let n = on_cursor_pre_mlp.nrows();
        Self {
            on_cursor_pre_mlp,
            polarities: vec![CursorPolarity::PlusOne; n],
            pos_residuals,
            off_cursor_pre_mlp,
            q_preserve_gate: None,
            solver_config,
        }
    }
}

/// Model hardware geometry tables provided to the `v2` lowering pass.
#[derive(Debug, Clone, Default)]
pub struct ModelGeometryTables {
    pub model_id: String,
    pub base_intermediate_size: usize,
    pub widened_slot_capacity: usize,
    pub w_u_top_basis: Option<OrthonormalSubspace>,
    pub downstream_mlps: BTreeMap<usize, DownstreamSwigluBlock>,
    pub gqa_layers: BTreeMap<usize, GqaPointerLayer>,
    pub deltanet_layers: BTreeMap<usize, DeltaNetGateLayer>,
}

/// Complete compiled `v2` output carrying the wire manifest and white-box concolic `MAP-Elites` lenses.
#[derive(Debug, Clone)]
pub struct CompiledProgramV2 {
    pub allocated: Vec<AllocatedIntentRule>,
    pub manifest: WireCartridgeManifestJson,
    pub concolic_lenses: BTreeMap<String, ConcolicDiodeLens>,
}
