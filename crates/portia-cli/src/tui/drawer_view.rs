//! `iocraft` Right Drawer component (`SidebarMode::SchemaInspector` ⋈ `SidebarMode::RawTokenTrace`).

use super::theme_chrome::{build_right_pane_title, db_basename};
use super::trace_drawer::build_right_context_window_lines;
use super::{SidebarMode, TwoPaneTuiState};
use crate::TableSchemaInfo;
use iocraft::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnGovernanceTag {
    None,
    StewardPolicy,
    LockGuard,
}

fn classify_column_governance(
    table_name: &str,
    col_name: &str,
    explicit_steward_cols: &[String],
) -> ColumnGovernanceTag {
    let qualified = format!("{table_name}.{col_name}");
    for spec in explicit_steward_cols {
        if spec.eq_ignore_ascii_case(col_name) || spec.eq_ignore_ascii_case(&qualified) {
            return ColumnGovernanceTag::StewardPolicy;
        }
    }
    let lower = col_name.to_ascii_lowercase();
    if matches!(lower.as_str(), "policy_decision" | "steward_policy") {
        ColumnGovernanceTag::StewardPolicy
    } else if matches!(lower.as_str(), "lock_state" | "change_freeze") {
        ColumnGovernanceTag::LockGuard
    } else {
        ColumnGovernanceTag::None
    }
}

fn lookup_fk_target<'a>(table: &'a TableSchemaInfo, col_name: &str) -> Option<(&'a str, &'a str)> {
    table
        .foreign_keys
        .iter()
        .find(|fk| fk.child_fk_column.eq_ignore_ascii_case(col_name))
        .map(|fk| (fk.parent_table.as_str(), fk.parent_pk_column.as_str()))
}

fn build_table_card_lines(
    table: &TableSchemaInfo,
    steward_cols: &[String],
    inner_width: usize,
    out: &mut Vec<Vec<MixedTextContent>>,
) {
    let row_noun = if table.row_count == 1 { "row" } else { "rows" };
    out.push(vec![
        MixedTextContent::new(format!("▾ {}", table.table_name))
            .color(Color::Cyan)
            .weight(Weight::Bold),
        MixedTextContent::new(format!(" ({} {row_noun})", table.row_count)).color(Color::Grey),
    ]);

    if table.columns.is_empty() {
        out.push(vec![MixedTextContent::new("    (no columns disclosed)")
            .color(Color::Grey)
            .italic()]);
        return;
    }

    for col in &table.columns {
        let mut spans = vec![MixedTextContent::new("  ")];
        let gov = classify_column_governance(&table.table_name, &col.name, steward_cols);

        if col.is_primary_key {
            spans.push(
                MixedTextContent::new("[PK] ")
                    .color(Color::Yellow)
                    .weight(Weight::Bold),
            );
        } else {
            spans.push(MixedTextContent::new(" ·   ").color(Color::Grey));
        }

        let name_span = if col.is_primary_key || gov != ColumnGovernanceTag::None {
            MixedTextContent::new(&col.name)
                .color(Color::White)
                .weight(Weight::Bold)
        } else {
            MixedTextContent::new(&col.name).color(Color::White)
        };
        spans.push(name_span);

        let col_type = if col.col_type.trim().is_empty() {
            "ANY"
        } else {
            col.col_type.trim()
        };
        let nn_suffix = if col.notnull && !col.is_primary_key {
            if inner_width >= 52 {
                " NOT NULL"
            } else {
                " NN"
            }
        } else {
            ""
        };
        spans.push(MixedTextContent::new(format!(" {col_type}{nn_suffix}")).color(Color::Grey));

        let gov_badge = match gov {
            ColumnGovernanceTag::StewardPolicy => Some(("[STEWARD]", Color::Yellow)),
            ColumnGovernanceTag::LockGuard => Some(("[LOCK]", Color::Red)),
            ColumnGovernanceTag::None => None,
        };

        let base_len = 7 + col.name.len() + 1 + col_type.len() + nn_suffix.len();
        let mut deferred_gov: Option<(&'static str, Color)> = None;
        if let Some((label, color)) = gov_badge {
            if base_len + 1 + label.len() <= inner_width {
                spans.push(
                    MixedTextContent::new(format!(" {label}"))
                        .color(color)
                        .weight(Weight::Bold),
                );
            } else {
                deferred_gov = Some((label, color));
            }
        }
        out.push(spans);

        if let Some((label, color)) = deferred_gov {
            out.push(vec![
                MixedTextContent::new("       ↳ ").color(Color::Grey),
                MixedTextContent::new(label)
                    .color(color)
                    .weight(Weight::Bold),
            ]);
        }

        if let Some((parent_tbl, parent_col)) = lookup_fk_target(table, &col.name) {
            out.push(vec![MixedTextContent::new(format!(
                "       ↳ -> {parent_tbl}.{parent_col}"
            ))
            .color(Color::Cyan)]);
        }
    }
}

/// Build styled [`MixedTextContent`] lines for `SidebarMode::SchemaInspector`.
#[must_use]
pub fn build_schema_inspector_lines(
    state: &TwoPaneTuiState,
    inner_width: usize,
) -> Vec<Vec<MixedTextContent>> {
    let mut lines = Vec::new();
    let sep_w = inner_width.saturating_sub(2).max(16);
    let base = db_basename(&state.db_path);
    let total_rows: usize = state.tables.iter().map(|t| t.row_count).sum();
    let tbl_count = state.tables.len();
    let tbl_noun = if tbl_count == 1 { "table" } else { "tables" };

    lines.push(vec![
        MixedTextContent::new(base)
            .color(Color::White)
            .weight(Weight::Bold),
        MixedTextContent::new(format!(" · {tbl_count} {tbl_noun} ({total_rows} rows)"))
            .color(Color::Grey),
    ]);

    if state.read_only {
        lines.push(vec![
            MixedTextContent::new("[READ-ONLY]")
                .color(Color::Cyan)
                .weight(Weight::Bold),
            MixedTextContent::new(" Mutations blocked").color(Color::Grey),
        ]);
    } else if !state.steward_policy_columns.is_empty() {
        lines.push(vec![
            MixedTextContent::new("STEWARD: ")
                .color(Color::Yellow)
                .weight(Weight::Bold),
            MixedTextContent::new(state.steward_policy_columns.join(", ")).color(Color::White),
        ]);
    } else {
        lines.push(vec![
            MixedTextContent::new("ACCESS: Read / Write").color(Color::Green)
        ]);
    }

    lines.push(vec![
        MixedTextContent::new("─".repeat(sep_w)).color(Color::Grey)
    ]);

    if state.tables.is_empty() {
        lines.push(vec![MixedTextContent::new("No user tables loaded.")
            .color(Color::Grey)
            .italic()]);
        return lines;
    }

    for (idx, tbl) in state.tables.iter().enumerate() {
        build_table_card_lines(tbl, &state.steward_policy_columns, inner_width, &mut lines);
        if idx + 1 < state.tables.len() {
            lines.push(vec![MixedTextContent::new("")]);
        }
    }

    lines
}

/// Build styled [`MixedTextContent`] lines for whichever right-hand drawer mode is active.
#[must_use]
pub fn build_right_drawer_lines(
    state: &TwoPaneTuiState,
    inner_width: usize,
) -> Vec<Vec<MixedTextContent>> {
    match state.effective_sidebar_mode() {
        SidebarMode::RawTokenTrace => build_right_context_window_lines(state, inner_width),
        SidebarMode::SchemaInspector | SidebarMode::Hidden => {
            build_schema_inspector_lines(state, inner_width)
        }
    }
}

/// Props for [`RightDrawerPane`].
#[derive(Default, Props)]
pub struct RightDrawerPaneProps {
    pub state: TwoPaneTuiState,
    pub inner_width: usize,
    pub scroll_handle: Option<Ref<ScrollViewHandle>>,
}

/// Right-hand `iocraft` drawer component (`SchemaInspector` or `RawTokenTrace`).
#[component]
pub fn RightDrawerPane(props: &RightDrawerPaneProps) -> impl Into<AnyElement<'static>> {
    let (title, border_color) = build_right_pane_title(&props.state);
    let lines = build_right_drawer_lines(&props.state, props.inner_width);
    let auto_scroll = props.state.effective_sidebar_mode() == SidebarMode::RawTokenTrace;

    element! {
        View(
            width: 100pct,
            height: 100pct,
            flex_direction: FlexDirection::Column,
            border_style: BorderStyle::Round,
            border_color: border_color,
        ) {
            View(width: 100pct, padding_left: 1, padding_right: 1) {
                Text(content: title, color: border_color, weight: Weight::Bold, wrap: TextWrap::NoWrap)
            }
            View(flex_grow: 1.0_f32, width: 100pct, overflow: Overflow::Hidden, padding_left: 1, padding_right: 1) {
                ScrollView(
                    auto_scroll: auto_scroll,
                    keyboard_scroll: Some(false),
                    scrollbar: Some(true),
                    handle: props.scroll_handle,
                ) {
                    View(width: 100pct, flex_direction: FlexDirection::Column) {
                        #(lines.into_iter().map(|contents| {
                            if contents.len() == 1 && contents[0].text.is_empty() {
                                element! { View(height: 1) }.into_any()
                            } else {
                                element! {
                                    MixedText(wrap: TextWrap::Wrap, contents: contents)
                                }.into_any()
                            }
                        }))
                    }
                }
            }
        }
    }
}
