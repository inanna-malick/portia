//! Convex-midpoint ternary `FORK / ASK / INLINE` and `DeltaNet-1` rollback scrubber
//! (`portia_nursery::v2_ternary_hylo`).

pub mod ast_seam;
pub mod builders;
pub mod capture_bridge;
pub mod crowding_metrics;
pub mod deltanet_eval;
pub mod rollback_harness;
pub mod worlds;

pub use ast_seam::{
    classify_rust_todo_seam, AstSeamClassificationReceipt, HyloTernaryBranch, NeverTypeHoleSeam,
};
pub use builders::{
    build_deltanet_rollback_scrub_intent, build_signed_ternary_fork_intent,
    build_ternary_fork_capture_data, compose_hylo_scrub_then_ternary_fork,
    evaluate_bank_collateral_and_l23_drift, evaluate_compiled_ternary_slots,
    partition_deltanet_heads_by_retention, synthesize_convex_midpoint_states,
    DeltaNetRollbackScrubSpec, SignedTernaryForkSpec, TernarySlotEval,
};
pub use capture_bridge::{
    capture_real_qwen35_bundle, latest_qwen35_08b_snapshot, CapturedRollbackEpisode,
    Layer22GroupCapture, RealModelCaptureBundle, RollbackEpisodeSpec,
};
pub use crowding_metrics::{
    evaluate_bank_background_collateral, load_qwen35_readout_and_l23, wire_pair_to_bf16,
    BankCollateralSummary, Qwen35ReadoutAndL23Bundle,
};
pub use deltanet_eval::{DeltaNetRecurrentSnapshot, Qwen35GatedDeltaNetLayer};
pub use rollback_harness::{evaluate_rollback_arm_trajectory, RollbackArmMetrics};
pub use worlds::{
    build_rust_ast_ternary_world, generate_crowding_rule_prompt_families,
    generate_rust_ast_rollback_episodes, generate_rust_ast_ternary_corpus,
    generate_unrelated_background_prompts, RustAstTernaryWorld,
};
