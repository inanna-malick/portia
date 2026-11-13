//! JSON-over-stdio PyTorch Qwen3.5 Worker Bridge (`StdioQwenWorkerClient`, `WorkerRequest`,
//! `WorkerResponse`), Pretrain-LOTL prompt formatting, and live session runners.

pub mod protocol;
pub mod session_loop;
pub mod spawn;
pub mod summary;
#[cfg(test)]
mod tests;

pub use crate::driver::{
    classify_ask_prompt_kind, ensure_none_of_above_escape_choice, extract_ask_interaction_mode,
    extract_ask_select_choices, format_ask_payload_for_display, parse_ask_user_reply,
    parse_ask_user_reply_for_kind, AskDecision, AskInteractionMode, AskPromptKind, AskUserReply,
    AstOperation, CoroutineStepEffect, ObservationOperation, ReadPurpose,
    NONE_OF_ABOVE_ENTER_TEXT_CHOICE,
};
pub use crate::model_identity::normalize_model_identifier;

pub use protocol::{
    format_driver_observation_for_worker, format_live_schema_summary,
    format_observation_wire_chunk, is_unpatched_base_cartridge_sentinel,
    validate_explicit_cartridge_path, ActivationState, CaptureBoundary,
    OnPolicyCaptureWithBackground, PayloadActivationRow, WorkerRequest, WorkerResponse,
    WorkerWeightGeometryPayload,
};
pub use session_loop::{
    run_epistemic_env_turn_sequence_streaming_cancellable_into,
    run_epistemic_env_turn_sequence_streaming_into, run_live_worker_session,
    run_live_worker_session_into, run_live_worker_session_with_tui, run_live_worker_turn_sequence,
    run_live_worker_turn_sequence_into, run_live_worker_turn_sequence_streaming_cancellable_into,
    run_live_worker_turn_sequence_streaming_into,
};
pub use spawn::{
    resolve_default_cartridge_label, resolve_default_cartridge_label_for_model,
    resolve_default_worker_argv, resolve_default_worker_command, StdioQwenWorkerClient,
};
pub use summary::{
    attach_session_error, build_lotl_followup_prompt, build_lotl_initial_prompt,
    emit_session_outcome_stderr, format_lotl_user_prompt_envelope, summarize_session_outcome,
    SessionEvent, SessionOutcomeStatus, SessionRunError, SessionSummaryReceipt,
};
