//! Headless `iocraft` [`Canvas`] snapshot serialization, ANSI error helpers, and terminal geometry detection.

use super::app::PortiaWorkspaceSnapshot;
use super::TwoPaneTuiState;
use iocraft::prelude::*;

/// Shared ANSI red error formatter used across REPL and CLI error surfaces,
/// downcasting to [`portia_sql::engine::EngineError`] when available.
#[must_use]
pub fn format_surfaced_error_ansi(err: &anyhow::Error) -> String {
    if let Some(eng_err) = err.downcast_ref::<portia_sql::engine::EngineError>() {
        eng_err.render_ansi()
    } else {
        format!("\x1b[1;31m✗ [SessionError]:\x1b[0m \x1b[31m{err}\x1b[0m")
    }
}

/// Extract `(title, detail)` for an `InteractionItem::HarnessNote` from a surfaced error.
#[must_use]
pub fn format_surfaced_error_note(err: &anyhow::Error) -> (String, String) {
    if let Some(eng_err) = err.downcast_ref::<portia_sql::engine::EngineError>() {
        (eng_err.code().to_string(), eng_err.to_string())
    } else {
        ("SessionError".to_string(), err.to_string())
    }
}

/// Render a deterministic Two-Pane TUI snapshot (`width` columns × `height` rows) via `iocraft`'s
/// Flexbox/Unicode [`Canvas`] engine for `--tui` CLI output and headless unit tests.
#[must_use]
pub fn render_two_pane_tui_frame(
    state: &TwoPaneTuiState,
    total_width: usize,
    total_height: usize,
    use_ansi_colors: bool,
) -> String {
    let width = total_width.max(72);
    let height = total_height.max(16);

    let mut el = element! {
        PortiaWorkspaceSnapshot(
            state: state.clone(),
            total_width: width,
            total_height: height,
            composer_has_focus: false,
        )
    };
    let canvas = el.render(Some(width));

    if use_ansi_colors {
        let mut buf = Vec::with_capacity(width * height * 2);
        canvas
            .write_ansi(&mut buf)
            .expect("in-memory iocraft Canvas::write_ansi");
        String::from_utf8_lossy(&buf)
            .replace("\r\n", "\n")
            .trim_end_matches('\n')
            .to_string()
    } else {
        canvas.to_string().trim_end_matches('\n').to_string()
    }
}

/// Detect terminal `(columns, rows)` via environment or [`crossterm::terminal::size`], falling back to `(140, 38)`.
#[must_use]
pub fn detect_terminal_size() -> (usize, usize) {
    if let (Ok(cols), Ok(lines)) = (std::env::var("COLUMNS"), std::env::var("LINES")) {
        if let (Ok(c), Ok(l)) = (cols.parse::<usize>(), lines.parse::<usize>()) {
            if c >= 72 && l >= 16 {
                return (c, l);
            }
        }
    }
    if let Ok((cols, rows)) = crossterm::terminal::size() {
        let c = cols as usize;
        let r = rows as usize;
        if c >= 72 && r >= 16 {
            return (c, r);
        }
    }
    (140, 38)
}
