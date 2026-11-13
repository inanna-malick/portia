//! Interactive `iocraft` Steward authorization (`Button`) & multi-row candidate disambiguation
//! (`Checkbox`) dock for `[MODE: ASK]`.

use crate::AskInteractionMode;
use iocraft::prelude::*;
use smol::channel::Sender;

/// Active interactive `[MODE: ASK]` mode inside [`StewardDock`].
#[derive(Debug, Clone)]
pub enum ActiveAskKind {
    Permission {
        choices: Vec<String>,
        cursor: usize,
        custom_text_mode: bool,
    },
    MultiSelect {
        choices: Vec<String>,
        checked: Vec<bool>,
        cursor: usize,
        custom_text_mode: bool,
    },
    Clarify,
}

/// Live UI state for an in-flight `[MODE: ASK]` suspension from the background worker.
#[derive(Clone)]
pub struct ActiveAskSession {
    pub question: String,
    pub context_hint: Option<String>,
    pub kind: ActiveAskKind,
    pub reply_tx: Sender<Option<String>>,
}

impl ActiveAskSession {
    #[must_use]
    pub fn from_request(
        question: String,
        ask_mode: AskInteractionMode,
        context_hint: Option<String>,
        reply_tx: Sender<Option<String>>,
    ) -> Self {
        let kind = match ask_mode {
            AskInteractionMode::StewardPermissionChoice {
                question: q,
                choices,
            } => {
                return Self {
                    question: q,
                    context_hint,
                    kind: ActiveAskKind::Permission {
                        choices,
                        cursor: 0,
                        custom_text_mode: false,
                    },
                    reply_tx,
                };
            }
            AskInteractionMode::MultiSelect {
                question: q,
                choices,
            } => {
                let n = choices.len();
                return Self {
                    question: q,
                    context_hint,
                    kind: ActiveAskKind::MultiSelect {
                        choices,
                        checked: vec![false; n],
                        cursor: 0,
                        custom_text_mode: false,
                    },
                    reply_tx,
                };
            }
            AskInteractionMode::ConversationalClarify => ActiveAskKind::Clarify,
        };
        Self {
            question,
            context_hint,
            kind,
            reply_tx,
        }
    }

    /// Whether keyboard typing should go into the free-text `TextInput` composer.
    #[must_use]
    pub const fn uses_composer_input(&self) -> bool {
        match &self.kind {
            ActiveAskKind::Clarify => true,
            ActiveAskKind::Permission {
                custom_text_mode, ..
            }
            | ActiveAskKind::MultiSelect {
                custom_text_mode, ..
            } => *custom_text_mode,
        }
    }

    pub fn move_up(&mut self) {
        match &mut self.kind {
            ActiveAskKind::Permission {
                choices, cursor, ..
            }
            | ActiveAskKind::MultiSelect {
                choices, cursor, ..
            } if !choices.is_empty() => {
                *cursor = if *cursor == 0 {
                    choices.len() - 1
                } else {
                    *cursor - 1
                };
            }
            _ => {}
        }
    }

    pub fn move_down(&mut self) {
        match &mut self.kind {
            ActiveAskKind::Permission {
                choices, cursor, ..
            }
            | ActiveAskKind::MultiSelect {
                choices, cursor, ..
            } if !choices.is_empty() => {
                *cursor = (*cursor + 1) % choices.len();
            }
            _ => {}
        }
    }

    pub fn toggle_current_checkbox(&mut self) {
        if let ActiveAskKind::MultiSelect {
            choices,
            checked,
            cursor,
            custom_text_mode,
        } = &mut self.kind
        {
            if let Some(opt) = choices.get(*cursor) {
                if is_escape_choice(opt) {
                    *custom_text_mode = true;
                    return;
                }
            }
            if let Some(slot) = checked.get_mut(*cursor) {
                *slot = !*slot;
            }
        }
    }

    /// Resolve `Enter` on the structured choice menu: returns `Some(wire_reply)` when complete,
    /// or `None` if switching into `custom_text_mode` so the user can type a custom reply.
    pub fn resolve_enter_choice(&mut self) -> Option<String> {
        match &mut self.kind {
            ActiveAskKind::Permission {
                choices,
                cursor,
                custom_text_mode,
            } => {
                let selected = choices.get(*cursor)?;
                resolve_permission_option(selected, custom_text_mode)
            }
            ActiveAskKind::MultiSelect {
                choices,
                checked,
                cursor,
                custom_text_mode,
            } => {
                let picked: Vec<String> = choices
                    .iter()
                    .zip(checked.iter())
                    .filter(|&(opt, &is_on)| is_on && !is_escape_choice(opt))
                    .map(|(opt, _)| opt.clone())
                    .collect();
                if !picked.is_empty() {
                    return Some(picked.join("; "));
                }
                let current = choices.get(*cursor)?;
                if is_escape_choice(current) {
                    *custom_text_mode = true;
                    None
                } else {
                    Some(current.clone())
                }
            }
            ActiveAskKind::Clarify => None,
        }
    }
}

#[must_use]
pub fn is_escape_choice(opt: &str) -> bool {
    let lower = opt.trim().to_ascii_lowercase();
    lower.starts_with("none of the above") || lower.starts_with("other")
}

fn resolve_permission_option(opt: &str, custom_text_mode: &mut bool) -> Option<String> {
    if is_escape_choice(opt) {
        *custom_text_mode = true;
        return None;
    }
    let upper = opt.trim().to_ascii_uppercase();
    if upper.starts_with("ALLOW") {
        Some("ALLOW".to_string())
    } else if upper.starts_with("DENY") {
        Some("DENY".to_string())
    } else {
        Some(opt.trim().to_string())
    }
}

/// Props for [`StewardDock`].
#[derive(Default, Props)]
pub struct StewardDockProps {
    pub ask: Option<ActiveAskSession>,
    pub on_select_permission: Handler<usize>,
    pub on_toggle_multi: Handler<usize>,
    pub on_submit_multi: Handler<()>,
}

/// Interactive `iocraft` Steward / Candidate Disambiguation dock (`Button` & `Checkbox`).
#[component]
pub fn StewardDock(props: &StewardDockProps) -> impl Into<AnyElement<'static>> {
    let Some(ask) = props.ask.clone() else {
        return element! { View() }.into_any();
    };

    let banner = match &ask.kind {
        ActiveAskKind::Permission { .. } => {
            "⛨ Steward Authorization Required (Up/Down + Enter, click Button, or Esc to cancel)"
        }
        ActiveAskKind::MultiSelect { .. } => {
            "▸ Candidate Row Disambiguation (Up/Down + Space to toggle, Enter to submit, Esc to cancel)"
        }
        ActiveAskKind::Clarify => {
            "▸ Assistant Question (type reply below and press Enter, or Esc to cancel)"
        }
    };

    let body = match &ask.kind {
        ActiveAskKind::Permission {
            choices,
            cursor,
            custom_text_mode,
        } if !*custom_text_mode => {
            let cur = *cursor;
            let items: Vec<_> = choices.iter().cloned().enumerate().collect();
            let base_cb = props.on_select_permission.clone();
            element! {
                View(flex_direction: FlexDirection::Column, margin_top: 1) {
                    #(items.into_iter().map(|(idx, choice)| {
                        let is_sel = idx == cur;
                        let cb = base_cb.clone();
                        let prefix = if is_sel { "❯ " } else { "  " };
                        let fg = if is_sel {
                            Color::Yellow
                        } else if is_escape_choice(&choice) {
                            Color::Grey
                        } else {
                            Color::Cyan
                        };
                        element! {
                            Button(
                                has_focus: false,
                                handler: move |_| cb(idx),
                            ) {
                                View(width: 100pct) {
                                    Text(
                                        content: format!("{prefix}{choice}"),
                                        color: fg,
                                        weight: if is_sel { Weight::Bold } else { Weight::Normal },
                                    )
                                }
                            }
                        }
                    }))
                }
            }
            .into_any()
        }
        ActiveAskKind::MultiSelect {
            choices,
            checked,
            cursor,
            custom_text_mode,
        } if !*custom_text_mode => {
            let cur = *cursor;
            let rows: Vec<_> = choices
                .iter()
                .cloned()
                .zip(checked.iter().copied())
                .enumerate()
                .collect();
            let toggle_base = props.on_toggle_multi.clone();
            let submit_cb = props.on_submit_multi.clone();
            element! {
                View(flex_direction: FlexDirection::Column, margin_top: 1) {
                    #(rows.into_iter().map(|(idx, (choice, is_on))| {
                        let is_sel = idx == cur;
                        let toggle_cb = toggle_base.clone();
                        let fg = if is_sel { Color::Yellow } else { Color::Cyan };
                        element! {
                            View(flex_direction: FlexDirection::Row) {
                                Text(
                                    content: if is_sel { "❯ " } else { "  " },
                                    color: Color::Yellow,
                                    weight: Weight::Bold,
                                )
                                Checkbox(
                                    checked: is_on,
                                    has_focus: false,
                                    on_change: move |_| toggle_cb(idx),
                                ) {
                                    Text(
                                        content: choice,
                                        color: fg,
                                        weight: if is_sel { Weight::Bold } else { Weight::Normal },
                                    )
                                }
                            }
                        }
                    }))
                    Button(has_focus: false, handler: move |_| submit_cb(())) {
                        View(margin_top: 1) {
                            Text(
                                content: "  [Submit Selected Row(s) ↵]",
                                color: Color::Green,
                                weight: Weight::Bold,
                            )
                        }
                    }
                }
            }
            .into_any()
        }
        _ => element! {
            Text(
                content: "  Type your custom answer in the input box below and press Enter.",
                color: Color::Grey,
                italic: true,
            )
        }
        .into_any(),
    };

    element! {
        View(
            width: 100pct,
            flex_direction: FlexDirection::Column,
            border_style: BorderStyle::Round,
            border_color: Color::Yellow,
            padding_left: 1,
            padding_right: 1,
        ) {
            Text(content: banner, color: Color::Yellow, weight: Weight::Bold)
            #(ask.context_hint.map(|ctx| element! {
                Text(content: format!("  Context: {ctx}"), color: Color::Grey)
            }))
            Text(content: format!("  {}", ask.question), color: Color::White, weight: Weight::Bold)
            #(body)
        }
    }
    .into_any()
}
