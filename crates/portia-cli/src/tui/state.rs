//! Core TUI session state (`TwoPaneTuiState`, `PaneScrollViewport`, `InteractionItem`,
//! `AnnotatedContextSpan`) and model/sidebar transitions.

use super::model_scale::TuiModelScaleSpec;
use super::tokens::ControlTokenBadge;
use crate::{DriverObservation, TableSchemaInfo};
use portia_core::relay::RelayPhase;
use portia_core::{CartridgeBank, Mode, PenOwner, RouteTag};
use serde::{Deserialize, Serialize};

/// Practical 3-mode right-hand drawer configuration for `portia-cli tui`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidebarMode {
    /// Clean 100% full-width conversation & SQL workspace.
    #[default]
    Hidden,
    /// Narrow (`30%` width) Schema & Steward Governance Inspector drawer.
    SchemaInspector,
    /// Power-user BPE token & `CartridgeBank` boundary inspector (`Ctrl+T` / `/trace`).
    RawTokenTrace,
}

const fn default_row_display_limit() -> Option<usize> {
    Some(12)
}

/// Explicit scroll viewport state for the Two-Pane TUI (replacing the overloaded
/// `0 == auto-bottom` sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaneScrollViewport {
    /// When `true`, viewport automatically tracks `total_rows.saturating_sub(visible_rows)`.
    pub follow_tail: bool,
    /// Explicit top row offset when `follow_tail == false`.
    pub top_offset: usize,
}

impl Default for PaneScrollViewport {
    fn default() -> Self {
        Self::new_follow_tail()
    }
}

impl PaneScrollViewport {
    #[must_use]
    pub const fn new_follow_tail() -> Self {
        Self {
            follow_tail: true,
            top_offset: 0,
        }
    }

    #[must_use]
    pub fn resolve_top_offset(&self, total_rows: usize, visible_rows: usize) -> usize {
        let max_offset = total_rows.saturating_sub(visible_rows);
        if self.follow_tail {
            max_offset
        } else {
            self.top_offset.min(max_offset)
        }
    }

    pub fn scroll_up(&mut self, step: usize, total_rows: usize, visible_rows: usize) {
        let current = self.resolve_top_offset(total_rows, visible_rows);
        self.follow_tail = false;
        self.top_offset = current.saturating_sub(step);
    }

    pub fn scroll_down(&mut self, step: usize, total_rows: usize, visible_rows: usize) {
        let max_offset = total_rows.saturating_sub(visible_rows);
        let current = self.resolve_top_offset(total_rows, visible_rows);
        let next = (current + step).min(max_offset);
        self.top_offset = next;
        self.follow_tail = next >= max_offset;
    }
}

/// Origin kind of a contiguous span inside the annotated context window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextSpanKind {
    /// Initial `<|im_start|>user\n...<|im_end|>\n` prefill.
    UserPromptEnvelope { turn_index: usize },
    /// Researcher turn sub-span (`TurnOpenPrefill`, `ModeLabel`, `PostModePrefill`,
    /// `PlanBody`, `PostPlanPrefill`, `PayloadBody`, `TurnComplete`).
    RelayPenSpan {
        turn_index: usize,
        span_index: usize,
        phase: RelayPhase,
    },
    /// Environment or steward `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n` prefill.
    ToolResponseEnvelope {
        turn_index: usize,
        event_id: String,
        route: RouteTag,
    },
}

/// A single cartridge- and pen-homogeneous span in the right-hand Context Window pane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnotatedContextSpan {
    pub global_span_index: usize,
    pub kind: ContextSpanKind,
    pub pen: PenOwner,
    pub producer_bank: Option<CartridgeBank>,
    pub consumer_bank: CartridgeBank,
    /// Present whenever `consumer_bank` changed from the immediately preceding span's `consumer_bank`.
    pub cartridge_transition: Option<(CartridgeBank, CartridgeBank)>,
    pub token_start: usize,
    pub token_end: usize,
    pub token_ids: Vec<u32>,
    pub raw_text: String,
    pub control_badges: Vec<ControlTokenBadge>,
}

/// A single turn entry in the left-hand Interaction pane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractionItem {
    UserPrompt {
        turn_index: usize,
        schema_header: String,
        user_text: String,
    },
    AssistantTurn {
        turn_index: usize,
        mode: Mode,
        route: RouteTag,
        plan: String,
        payload: String,
        single_pen_verified: bool,
    },
    Observation {
        turn_index: usize,
        observation: DriverObservation,
    },
    HarnessNote {
        title: String,
        detail: String,
    },
}

/// Complete state of the Daily-Driver TUI session (`InteractionPane` ⋈ `SidebarMode`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TwoPaneTuiState {
    pub db_path: String,
    pub model_name: String,
    pub cartridge_label: String,
    pub tables: Vec<TableSchemaInfo>,
    pub interactions: Vec<InteractionItem>,
    pub context_spans: Vec<AnnotatedContextSpan>,
    pub total_context_tokens: usize,
    pub cartridge_switch_count: usize,
    pub last_mounted_bank: Option<CartridgeBank>,
    #[serde(default)]
    pub left_viewport: PaneScrollViewport,
    #[serde(default)]
    pub right_viewport: PaneScrollViewport,
    pub show_inline_token_ids: bool,
    /// Whether `RawTokenTrace` (`PANE 2: CONTEXT WINDOW & CARTRIDGE BOUNDARIES`) is active.
    #[serde(default)]
    pub show_debug_sidebar: bool,
    /// Active right-drawer mode (`Hidden`, `SchemaInspector`, or `RawTokenTrace`).
    #[serde(default)]
    pub sidebar_mode: SidebarMode,
    /// Whether the underlying database session is opened in read-only governance mode.
    #[serde(default)]
    pub read_only: bool,
    /// Optional session-level candidate rows seeded from prior `Mode::Read` turns.
    #[serde(default)]
    pub session_preview_rows: Vec<Vec<serde_json::Value>>,
    /// Explicitly configured steward policy column bindings (`"col_name"` or `"table.col_name"`).
    #[serde(default)]
    pub steward_policy_columns: Vec<String>,
    /// Whether automatic multi-row candidate `MultiSelect` disambiguation synthesis is enabled.
    #[serde(default)]
    pub candidate_disambiguation_enabled: bool,
    /// Whether `/x` (`psql`-style expanded vertical record cards) is active for SQL results.
    #[serde(default)]
    pub expanded_display: bool,
    /// Maximum rows displayed per SQL observation (`Some(12)` default, `None` for `/rows all`).
    #[serde(default = "default_row_display_limit")]
    pub row_display_limit: Option<usize>,
    /// Number of committed SQLite session savepoints currently available for `/undo`.
    #[serde(default)]
    pub undo_depth: usize,
    /// Whether the background worker is actively streaming a turn.
    #[serde(default)]
    pub is_streaming: bool,
}

impl TwoPaneTuiState {
    pub fn new(
        db_path: impl Into<String>,
        model_name: impl Into<String>,
        cartridge_label: impl Into<String>,
        tables: Vec<TableSchemaInfo>,
    ) -> Self {
        let sidebar_mode = if tables.is_empty() {
            SidebarMode::Hidden
        } else {
            SidebarMode::SchemaInspector
        };
        Self {
            db_path: db_path.into(),
            model_name: model_name.into(),
            cartridge_label: cartridge_label.into(),
            tables,
            interactions: Vec::new(),
            context_spans: Vec::new(),
            total_context_tokens: 0,
            cartridge_switch_count: 0,
            last_mounted_bank: None,
            left_viewport: PaneScrollViewport::new_follow_tail(),
            right_viewport: PaneScrollViewport::new_follow_tail(),
            show_inline_token_ids: true,
            show_debug_sidebar: false,
            sidebar_mode,
            read_only: false,
            session_preview_rows: Vec::new(),
            steward_policy_columns: Vec::new(),
            candidate_disambiguation_enabled: false,
            expanded_display: false,
            row_display_limit: default_row_display_limit(),
            undo_depth: 0,
            is_streaming: false,
        }
    }

    /// Resolve active `Qwen3.5` scale & widened SwiGLU rack geometry for this session.
    #[must_use]
    pub fn model_scale_spec(&self) -> TuiModelScaleSpec {
        TuiModelScaleSpec::resolve(&self.model_name, &self.cartridge_label)
    }

    #[must_use]
    pub fn with_debug_sidebar(mut self, enabled: bool) -> Self {
        self.set_debug_sidebar(enabled);
        self
    }

    #[must_use]
    pub fn with_sidebar_mode(mut self, mode: SidebarMode) -> Self {
        self.set_sidebar_mode(mode);
        self
    }

    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    #[must_use]
    pub fn default_workspace_sidebar_mode(&self) -> SidebarMode {
        if self.tables.is_empty() {
            SidebarMode::Hidden
        } else {
            SidebarMode::SchemaInspector
        }
    }

    #[must_use]
    pub fn effective_sidebar_mode(&self) -> SidebarMode {
        if self.show_debug_sidebar {
            SidebarMode::RawTokenTrace
        } else if self.sidebar_mode == SidebarMode::RawTokenTrace {
            self.default_workspace_sidebar_mode()
        } else {
            self.sidebar_mode
        }
    }

    pub fn set_sidebar_mode(&mut self, mode: SidebarMode) {
        self.sidebar_mode = mode;
        self.show_debug_sidebar = mode == SidebarMode::RawTokenTrace;
    }

    pub fn set_debug_sidebar(&mut self, enabled: bool) {
        if enabled {
            self.sidebar_mode = SidebarMode::RawTokenTrace;
            self.show_debug_sidebar = true;
        } else {
            self.show_debug_sidebar = false;
            if self.sidebar_mode == SidebarMode::RawTokenTrace {
                self.sidebar_mode = self.default_workspace_sidebar_mode();
            }
        }
    }

    pub fn toggle_schema_drawer(&mut self) {
        let next = match self.effective_sidebar_mode() {
            SidebarMode::SchemaInspector => SidebarMode::Hidden,
            SidebarMode::Hidden | SidebarMode::RawTokenTrace => SidebarMode::SchemaInspector,
        };
        self.set_sidebar_mode(next);
    }

    pub fn toggle_raw_token_trace(&mut self) {
        if self.effective_sidebar_mode() == SidebarMode::RawTokenTrace {
            let fallback = self.default_workspace_sidebar_mode();
            self.set_sidebar_mode(fallback);
        } else {
            self.set_sidebar_mode(SidebarMode::RawTokenTrace);
        }
    }

    #[must_use]
    pub fn with_session_preview_rows(mut self, rows: Vec<Vec<serde_json::Value>>) -> Self {
        self.session_preview_rows = rows;
        self
    }

    #[must_use]
    pub fn with_steward_policy_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.steward_policy_columns = columns
            .into_iter()
            .map(|s| s.into().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        self
    }

    #[must_use]
    pub fn with_candidate_disambiguation(mut self, enabled: bool) -> Self {
        self.candidate_disambiguation_enabled = enabled;
        self
    }

    /// Clear accumulated interaction turns and context-window spans (`/clear` or `/reset`).
    pub fn reset_conversation_context(&mut self) {
        self.interactions.clear();
        self.reset_kv_window_tracking();
        self.left_viewport = PaneScrollViewport::new_follow_tail();
    }

    /// Reset KV/DeltaNet context-span counters when hot-swapping worker processes (`/model`),
    /// preserving the human-visible `interactions` scrollback in `LeftInteractionPane`.
    pub fn reset_kv_window_tracking(&mut self) {
        self.context_spans.clear();
        self.total_context_tokens = 0;
        self.cartridge_switch_count = 0;
        self.last_mounted_bank = None;
        self.right_viewport = PaneScrollViewport::new_follow_tail();
    }

    /// Update active `(model_name, cartridge_label)` after a successful `/model` worker hot-swap.
    pub fn apply_hot_swapped_model(
        &mut self,
        new_model: impl Into<String>,
        new_cartridge_label: impl Into<String>,
    ) -> TuiModelScaleSpec {
        self.model_name = new_model.into();
        self.cartridge_label = new_cartridge_label.into();
        self.reset_kv_window_tracking();
        self.left_viewport = PaneScrollViewport::new_follow_tail();
        self.model_scale_spec()
    }

    /// Return the most recent successful SQL/tabular [`DriverObservation`] for `/export`.
    #[must_use]
    pub fn latest_tabular_observation(&self) -> Option<&DriverObservation> {
        self.interactions.iter().rev().find_map(|item| match item {
            InteractionItem::Observation { observation, .. }
                if observation.ok
                    && (!observation.columns.is_empty() || !observation.rows.is_empty()) =>
            {
                Some(observation)
            }
            _ => None,
        })
    }

    /// Clean up an aborted in-flight `AssistantTurn` when the user cancels mid-stream (`Esc` / `Ctrl+C`).
    pub fn prune_cancelled_streaming_turn(&mut self) {
        self.is_streaming = false;
        if let Some(InteractionItem::AssistantTurn { plan, payload, .. }) = self.interactions.last()
        {
            let is_placeholder =
                (plan.trim().is_empty() || plan == "streaming...") && payload.trim().is_empty();
            if is_placeholder {
                self.interactions.pop();
            }
        }
    }
}
