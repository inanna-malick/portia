//! `iocraft` SQL result renderer supporting both aligned Unicode box-drawing tables (default) and
//! `psql`-style `/x` expanded vertical record cards (`─[ RECORD k ]─`) with untruncated values,
//! plus configurable `/rows <N|all>` row display limits.

use crate::DriverObservation;
use iocraft::prelude::{Color, MixedTextContent, Weight};
use portia_core::StewardAuthState;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const MAX_NATURAL_COL_WIDTH: usize = 40;
const MIN_COL_WIDTH: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellKind {
    Null,
    Numeric,
    Boolean,
    Text,
}

#[derive(Debug, Clone)]
struct FormattedCell {
    text: String,
    kind: CellKind,
}

fn sanitize_single_line(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_ws = false;
    for ch in raw.chars() {
        if matches!(ch, '\n' | '\r' | '\t') {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            prev_ws = false;
            out.push(ch);
        }
    }
    out
}

fn format_sql_value(val: &serde_json::Value, single_line: bool) -> FormattedCell {
    match val {
        serde_json::Value::Null => FormattedCell {
            text: "NULL".to_string(),
            kind: CellKind::Null,
        },
        serde_json::Value::Number(n) => FormattedCell {
            text: n.to_string(),
            kind: CellKind::Numeric,
        },
        serde_json::Value::Bool(b) => FormattedCell {
            text: b.to_string(),
            kind: CellKind::Boolean,
        },
        serde_json::Value::String(s) => FormattedCell {
            text: if single_line {
                sanitize_single_line(s)
            } else {
                s.clone()
            },
            kind: CellKind::Text,
        },
        other => {
            let raw = other.to_string();
            FormattedCell {
                text: if single_line {
                    sanitize_single_line(&raw)
                } else {
                    raw
                },
                kind: CellKind::Text,
            }
        }
    }
}

/// Truncate `text` to at most `max_width` terminal columns, appending `…` when clipped.
#[must_use]
pub fn truncate_display_width(text: &str, max_width: usize) -> (String, usize) {
    let full_w = UnicodeWidthStr::width(text);
    if full_w <= max_width {
        return (text.to_string(), full_w);
    }
    if max_width == 0 {
        return (String::new(), 0);
    }
    let ell_w = UnicodeWidthChar::width('…').unwrap_or(1);
    if max_width <= ell_w {
        return ("…".to_string(), ell_w);
    }
    let target = max_width - ell_w;
    let (mut out, mut used) = (String::new(), 0usize);
    for ch in text.chars() {
        let ch_w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + ch_w > target {
            break;
        }
        out.push(ch);
        used += ch_w;
    }
    out.push('…');
    (out, used + ell_w)
}

fn pad_cell(text: &str, col_width: usize, right_align: bool) -> String {
    let (clipped, used_w) = truncate_display_width(text, col_width);
    let pad = col_width.saturating_sub(used_w);
    if pad == 0 {
        clipped
    } else if right_align {
        format!("{}{clipped}", " ".repeat(pad))
    } else {
        format!("{clipped}{}", " ".repeat(pad))
    }
}

fn is_governance_col(col_name: &str, steward_cols: &[String]) -> bool {
    let lower = col_name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "policy_decision" | "steward_policy" | "lock_state" | "change_freeze"
    ) || steward_cols.iter().any(|spec| {
        spec.rsplit('.')
            .next()
            .unwrap_or(spec)
            .eq_ignore_ascii_case(col_name)
    })
}

fn style_cell_span(content: String, cell: &FormattedCell, is_gov: bool) -> MixedTextContent {
    if is_gov {
        match cell.text.trim().to_ascii_uppercase().as_str() {
            "ASK" | "PENDING" => {
                return MixedTextContent::new(content)
                    .color(Color::Yellow)
                    .weight(Weight::Bold)
            }
            "DENY" | "LOCKED" | "FROZEN" | "BLOCKED" => {
                return MixedTextContent::new(content)
                    .color(Color::Red)
                    .weight(Weight::Bold)
            }
            "ALLOW" | "CLEAR" | "OPEN" => {
                return MixedTextContent::new(content).color(Color::Green)
            }
            _ => {}
        }
    }
    match cell.kind {
        CellKind::Null => MixedTextContent::new(content)
            .color(Color::DarkGrey)
            .italic(),
        CellKind::Numeric => MixedTextContent::new(content).color(Color::Cyan),
        CellKind::Boolean => MixedTextContent::new(content).color(Color::Magenta),
        CellKind::Text => MixedTextContent::new(content).color(Color::White),
    }
}

fn compute_column_widths(
    headers: &[String],
    grid: &[Vec<FormattedCell>],
    max_width: usize,
) -> (usize, Vec<usize>) {
    let total_cols = headers.len();
    if total_cols == 0 {
        return (0, Vec::new());
    }
    let mut vis_cols = total_cols;
    while vis_cols > 1 && (3 + 3 * vis_cols + vis_cols * MIN_COL_WIDTH) > max_width {
        vis_cols -= 1;
    }

    let mut widths: Vec<usize> = (0..vis_cols)
        .map(|c| {
            let hdr_w = UnicodeWidthStr::width(headers[c].as_str());
            let max_cell = grid
                .iter()
                .filter_map(|row| row.get(c))
                .map(|cell| UnicodeWidthStr::width(cell.text.as_str()))
                .max()
                .unwrap_or(0);
            hdr_w
                .max(max_cell)
                .clamp(MIN_COL_WIDTH, MAX_NATURAL_COL_WIDTH)
        })
        .collect();

    let budget = max_width.saturating_sub(3 + 3 * vis_cols).max(vis_cols);
    let mut sum_w: usize = widths.iter().sum();
    let floor = MIN_COL_WIDTH.min(budget / vis_cols).max(1);

    while sum_w > budget {
        let Some((max_idx, &max_val)) = widths.iter().enumerate().max_by_key(|(_, &w)| w) else {
            break;
        };
        if max_val <= floor {
            break;
        }
        widths[max_idx] -= 1;
        sum_w -= 1;
    }
    (vis_cols, widths)
}

fn build_border_line(
    widths: &[usize],
    left: &str,
    mid: &str,
    right: &str,
) -> Vec<MixedTextContent> {
    let segs: Vec<String> = widths.iter().map(|&w| "─".repeat(w + 2)).collect();
    vec![MixedTextContent::new(format!("  {left}{}{right}", segs.join(mid))).color(Color::Grey)]
}

fn render_expanded_record_cards(
    obs: &DriverObservation,
    headers: &[String],
    preview_slice: &[Vec<serde_json::Value>],
    steward_cols: &[String],
    max_width: usize,
) -> Vec<Vec<MixedTextContent>> {
    let mut out = Vec::new();
    let row_total = obs.rows.len();
    let row_noun = if row_total == 1 { "row" } else { "rows" };
    let mut badge = vec![MixedTextContent::new(format!(
        "  ✓ {row_total} {row_noun} returned (expanded /x · {} cols)",
        headers.len()
    ))
    .color(Color::Green)];
    if obs.steward_auth.requires_ask() {
        badge.push(
            MixedTextContent::new(" · STEWARD: ASK REQUIRED")
                .color(Color::Yellow)
                .weight(Weight::Bold),
        );
    }
    out.push(badge);

    let max_hdr_w = headers
        .iter()
        .map(|h| UnicodeWidthStr::width(h.as_str()))
        .max()
        .unwrap_or(4)
        .min(28);
    let rule_w = max_width.saturating_sub(4).clamp(20, 72);

    for (r_idx, raw_row) in preview_slice.iter().enumerate() {
        let title = format!("─[ RECORD {} ]", r_idx + 1);
        let fill = rule_w.saturating_sub(UnicodeWidthStr::width(title.as_str()));
        out.push(vec![MixedTextContent::new(format!(
            "  {title}{}",
            "─".repeat(fill)
        ))
        .color(Color::Cyan)]);
        for (c_idx, hdr) in headers.iter().enumerate() {
            let cell = format_sql_value(
                raw_row.get(c_idx).unwrap_or(&serde_json::Value::Null),
                false,
            );
            let is_gov = is_governance_col(hdr, steward_cols);
            let hdr_pad = max_hdr_w.saturating_sub(UnicodeWidthStr::width(hdr.as_str()));
            let prefix = format!("  {hdr}{} │ ", " ".repeat(hdr_pad));
            out.push(vec![
                MixedTextContent::new(prefix)
                    .color(if is_gov { Color::Yellow } else { Color::Grey })
                    .weight(Weight::Bold),
                style_cell_span(cell.text.clone(), &cell, is_gov),
            ]);
        }
    }

    if row_total > preview_slice.len() {
        let hidden = row_total - preview_slice.len();
        out.push(vec![MixedTextContent::new(format!(
            "  … {hidden} additional row(s) hidden (use /rows all or /export out.csv)"
        ))
        .color(Color::Grey)
        .italic()]);
    }
    out
}

/// Render a non-empty [`DriverObservation`] row set into `(styled_lines, wrap_lines)` where
/// box-grid mode sets `wrap_lines = false` (already column-fitted) and `/x` expanded mode sets
/// `wrap_lines = true` so arbitrarily wide values wrap without truncation!
#[must_use]
pub fn render_observation_result_lines(
    obs: &DriverObservation,
    steward_cols: &[String],
    max_width: usize,
    expanded_display: bool,
    row_display_limit: Option<usize>,
) -> (Vec<Vec<MixedTextContent>>, bool) {
    let row_total = obs.rows.len();
    let col_count = if !obs.columns.is_empty() {
        obs.columns.len()
    } else {
        obs.rows.first().map_or(0, Vec::len)
    };
    if row_total == 0 || col_count == 0 {
        return (Vec::new(), false);
    }

    let headers: Vec<String> = if !obs.columns.is_empty() {
        obs.columns
            .iter()
            .map(|c| sanitize_single_line(c))
            .collect()
    } else {
        (1..=col_count).map(|i| format!("col_{i}")).collect()
    };

    let visible_cap = row_display_limit.map_or(row_total, |cap| row_total.min(cap));
    let preview_slice = &obs.rows[..visible_cap];

    if expanded_display {
        return (
            render_expanded_record_cards(obs, &headers, preview_slice, steward_cols, max_width),
            true,
        );
    }

    let grid: Vec<Vec<FormattedCell>> = preview_slice
        .iter()
        .map(|row| {
            (0..col_count)
                .map(|c| format_sql_value(row.get(c).unwrap_or(&serde_json::Value::Null), true))
                .collect()
        })
        .collect();

    let (vis_cols, widths) = compute_column_widths(&headers, &grid, max_width.max(24));
    let omitted_cols = col_count.saturating_sub(vis_cols);

    let numeric_cols: Vec<bool> = (0..vis_cols)
        .map(|c| {
            grid.iter().any(|r| r[c].kind == CellKind::Numeric)
                && grid
                    .iter()
                    .all(|r| matches!(r[c].kind, CellKind::Numeric | CellKind::Null))
        })
        .collect();
    let gov_cols: Vec<bool> = (0..vis_cols)
        .map(|c| is_governance_col(&headers[c], steward_cols))
        .collect();

    let mut out = Vec::new();
    let row_noun = if row_total == 1 { "row" } else { "rows" };
    let mut badge = vec![
        MixedTextContent::new(format!("  ✓ {row_total} {row_noun} returned")).color(Color::Green),
    ];
    if obs.steward_auth.requires_ask() {
        badge.push(
            MixedTextContent::new(" · STEWARD: ASK REQUIRED")
                .color(Color::Yellow)
                .weight(Weight::Bold),
        );
    } else if obs.steward_auth == StewardAuthState::Granted {
        badge.push(MixedTextContent::new(" · STEWARD: GRANTED").color(Color::Green));
    }
    if omitted_cols > 0 {
        badge.push(
            MixedTextContent::new(format!(" · +{omitted_cols} cols (type /x to expand)"))
                .color(Color::Grey),
        );
    }
    out.push(badge);

    out.push(build_border_line(&widths, "┌", "┬", "┐"));

    let mut hdr_spans = vec![MixedTextContent::new("  │").color(Color::Grey)];
    for c in 0..vis_cols {
        let padded = pad_cell(&headers[c], widths[c], numeric_cols[c]);
        let col_color = if gov_cols[c] {
            Color::Yellow
        } else {
            Color::Cyan
        };
        hdr_spans.push(
            MixedTextContent::new(format!(" {padded} "))
                .color(col_color)
                .weight(Weight::Bold),
        );
        hdr_spans.push(MixedTextContent::new("│").color(Color::Grey));
    }
    out.push(hdr_spans);

    out.push(build_border_line(&widths, "├", "┼", "┤"));

    for row in &grid {
        let mut r_spans = vec![MixedTextContent::new("  │").color(Color::Grey)];
        for c in 0..vis_cols {
            let cell = &row[c];
            let padded = pad_cell(&cell.text, widths[c], numeric_cols[c]);
            r_spans.push(style_cell_span(format!(" {padded} "), cell, gov_cols[c]));
            r_spans.push(MixedTextContent::new("│").color(Color::Grey));
        }
        out.push(r_spans);
    }

    out.push(build_border_line(&widths, "└", "┴", "┘"));

    if row_total > visible_cap {
        let hidden = row_total - visible_cap;
        out.push(vec![MixedTextContent::new(format!(
            "  … {hidden} more row(s) omitted (showing {visible_cap}/{row_total}; /rows all or /export out.csv)"
        ))
        .color(Color::Grey)
        .italic()]);
    }

    (out, false)
}
