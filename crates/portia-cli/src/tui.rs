//! Clean-Sheet `iocraft 0.9.5` Multi-Scale Agent Harness Terminal UI (`TUI`) for `portia-cli`:
//! - **Declarative Flexbox + Unicode Layout ([`app`])**: Native `iocraft` `View` + `ScrollView` +
//!   `MixedText` + `TextInput` + `Button` + `Checkbox` component tree supporting both `.fullscreen()`
//!   split-drawer mode and `--inline` native terminal scrollback mode.
//! - **Decoupled Background Worker & SQLite Actor ([`session_actor`])**: Runs stdio Qwen inference
//!   and `rusqlite` transactions on a dedicated OS thread with atomic token-boundary `Esc` / `Ctrl+C`
//!   cancellation, automatic post-write schema refresh, and SQLite session `/undo` savepoints.
//! - **Practical Result Inspection ([`result_table_view`], [`export_data`])**: Aligned Unicode box
//!   grids, `psql`-style `/x` expanded vertical record cards (`─[ RECORD k ]─`), `/rows <N|all>`,
//!   and `/export <path.csv|path.json>`.
//! - **Multi-Scale `Qwen3.5` Geometry ([`model_scale`], [`drawer_view`], [`trace_drawer`])**:
//!   `0.8B-Base`, `2B-Base`, `4B-Base`, `9B-Base`, and `27B` header badges, `SchemaInspector`
//!   (`Tab` / `Ctrl+B`), and `RawTokenTrace` (`Ctrl+T` / `/trace`).

pub mod app;
pub mod drawer_view;
pub mod export_data;
pub mod history;
pub mod model_scale;
pub mod result_table_view;
pub mod session_actor;
pub mod snapshot;
pub mod sql_highlight;
pub mod state;
mod state_recording;
pub mod steward_dock;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_harness;
#[cfg(test)]
mod tests_multiscale;
pub mod theme_chrome;
pub mod tokens;
pub mod trace_drawer;
pub mod transcript_view;

pub use app::{
    compute_pane_column_widths, run_iocraft_tui_session, PortiaTuiApp, PortiaWorkspaceSnapshot,
};
pub use drawer_view::{build_right_drawer_lines, build_schema_inspector_lines, RightDrawerPane};
pub use export_data::{
    export_observation_to_path, format_observation_csv, format_observation_json,
    parse_row_display_limit_arg, ExportFormat,
};
pub use history::{delete_prev_word, delete_to_line_start, ComposerBuffer, PromptHistory};
pub use model_scale::{
    format_cartridge_bank_badge, format_cartridge_bank_badge_for_scale, TuiModelScaleSpec,
};
pub use session_actor::{
    spawn_session_actor, SessionActorHandle, UiLocalMutation, UiToWorkerMsg, WorkerSpawner,
    WorkerToUiEvent,
};
pub use snapshot::{
    detect_terminal_size, format_surfaced_error_ansi, format_surfaced_error_note,
    render_two_pane_tui_frame,
};
pub use state::{
    AnnotatedContextSpan, ContextSpanKind, InteractionItem, PaneScrollViewport, SidebarMode,
    TwoPaneTuiState,
};
pub use tokens::{
    annotate_raw_text_with_control_tokens, extract_control_token_badges, ControlTokenBadge,
    ControlTokenRole,
};
pub use trace_drawer::{
    build_right_context_window_lines, coalesce_cartridge_epochs,
    coalesce_cartridge_epochs_for_scale, summarize_loaded_cartridge_rack, CartridgeEpoch,
    LoadedCartridgeBankCard,
};
pub use transcript_view::{build_left_interaction_lines, LeftTranscriptPane, TranscriptLine};
