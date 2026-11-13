//! Declarative `iocraft 0.9.5` root application (`PortiaTuiApp` & `PortiaWorkspaceSnapshot`),
//! supporting both `.fullscreen()` split-drawer mode and `--inline` native terminal scrollback mode.

use super::drawer_view::RightDrawerPane;
use super::history::{ComposerBuffer, PromptHistory};
use super::session_actor::{SessionActorHandle, UiLocalMutation, UiToWorkerMsg, WorkerToUiEvent};
use super::steward_dock::{is_escape_choice, ActiveAskKind, ActiveAskSession, StewardDock};
use super::theme_chrome::{
    build_prompt_bar_title, sidebar_split_percentages, HeaderStatusBar, COMPOSER_PLACEHOLDER,
};
use super::transcript_view::LeftTranscriptPane;
use super::{SidebarMode, TwoPaneTuiState};
use crate::{parse_repl_command, ReplCommand};
use anyhow::Result;
use iocraft::prelude::*;
use smol::channel::{Receiver as SmolReceiver, Sender as SmolSender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Compute exact `(left_cols, Option<right_cols>)` integer widths for `total_width`.
#[must_use]
pub fn compute_pane_column_widths(mode: SidebarMode, total_width: usize) -> (usize, Option<usize>) {
    let w = total_width.max(72);
    match sidebar_split_percentages(mode) {
        None => (w, None),
        Some((_left_pct, right_pct)) => {
            let right_w = ((w as f32 * (right_pct / 100.0)).round() as usize).clamp(24, w - 36);
            let left_w = w - right_w;
            (left_w, Some(right_w))
        }
    }
}

fn build_composer_spans(buf: &ComposerBuffer, has_focus: bool) -> Vec<MixedTextContent> {
    if buf.text.is_empty() {
        if has_focus {
            let mut chars = COMPOSER_PLACEHOLDER.chars();
            let first = chars.next().unwrap_or(' ');
            let rest: String = chars.collect();
            return vec![
                MixedTextContent::new(first.to_string())
                    .color(Color::Cyan)
                    .invert(),
                MixedTextContent::new(rest).color(Color::DarkGrey),
            ];
        }
        return vec![MixedTextContent::new(COMPOSER_PLACEHOLDER).color(Color::DarkGrey)];
    }

    if !has_focus {
        return vec![MixedTextContent::new(buf.text.clone()).color(Color::White)];
    }

    let pos = buf.cursor.min(buf.text.len());
    let head = &buf.text[..pos];
    let mut out = Vec::with_capacity(3);
    if !head.is_empty() {
        out.push(MixedTextContent::new(head.to_string()).color(Color::White));
    }
    if pos < buf.text.len() {
        let mut tail_chars = buf.text[pos..].chars();
        let cur_ch = tail_chars.next().unwrap_or(' ');
        let rest: String = tail_chars.collect();
        out.push(
            MixedTextContent::new(cur_ch.to_string())
                .color(Color::Cyan)
                .invert()
                .weight(Weight::Bold),
        );
        if !rest.is_empty() {
            out.push(MixedTextContent::new(rest).color(Color::White));
        }
    } else {
        out.push(MixedTextContent::new(" ").color(Color::Cyan).invert());
    }
    out
}

/// Props for [`PortiaWorkspaceSnapshot`].
#[derive(Default, Props)]
pub struct PortiaWorkspaceSnapshotProps {
    pub state: TwoPaneTuiState,
    pub total_width: usize,
    pub total_height: usize,
    pub composer: ComposerBuffer,
    pub composer_has_focus: bool,
    pub left_scroll_handle: Option<Ref<ScrollViewHandle>>,
    pub right_scroll_handle: Option<Ref<ScrollViewHandle>>,
    pub active_ask: Option<ActiveAskSession>,
    pub on_select_permission: Handler<usize>,
    pub on_toggle_multi: Handler<usize>,
    pub on_submit_multi: Handler<()>,
    pub status_hint: Option<String>,
}

/// Fullscreen / headless two-pane workspace view (`HeaderStatusBar` + `LeftTranscriptPane` +
/// optional `RightDrawerPane` + optional `StewardDock` + `ComposerDock`).
#[component]
pub fn PortiaWorkspaceSnapshot(
    props: &mut PortiaWorkspaceSnapshotProps,
) -> impl Into<AnyElement<'static>> {
    let total_w = props.total_width.max(72);
    let total_h = props.total_height.max(16);
    let mode = props.state.effective_sidebar_mode();
    let (left_cols, right_cols_opt) = compute_pane_column_widths(mode, total_w);
    let left_inner_w = left_cols.saturating_sub(4).max(20);

    let prompt_title = build_prompt_bar_title(&props.state, props.status_hint.as_deref());
    let composer_border_color = if props.active_ask.is_some() {
        Color::Yellow
    } else if props.state.is_streaming {
        Color::Magenta
    } else {
        Color::Cyan
    };
    let composer_spans = build_composer_spans(&props.composer, props.composer_has_focus);

    element! {
        View(
            width: total_w as u32,
            height: total_h as u32,
            flex_direction: FlexDirection::Column,
        ) {
            HeaderStatusBar(state: props.state.clone())

            View(
                flex_grow: 1.0_f32,
                width: 100pct,
                flex_direction: FlexDirection::Row,
                overflow: Overflow::Hidden,
            ) {
                View(flex_grow: 1.0_f32, height: 100pct) {
                    LeftTranscriptPane(
                        state: props.state.clone(),
                        inner_width: left_inner_w,
                        scroll_handle: props.left_scroll_handle,
                    )
                }
                #(right_cols_opt.map(|right_cols| {
                    let right_inner_w = right_cols.saturating_sub(4).max(16);
                    element! {
                        View(width: right_cols as u32, height: 100pct) {
                            RightDrawerPane(
                                state: props.state.clone(),
                                inner_width: right_inner_w,
                                scroll_handle: props.right_scroll_handle,
                            )
                        }
                    }
                }))
            }

            StewardDock(
                ask: props.active_ask.clone(),
                on_select_permission: props.on_select_permission.clone(),
                on_toggle_multi: props.on_toggle_multi.clone(),
                on_submit_multi: props.on_submit_multi.clone(),
            )

            View(
                width: 100pct,
                flex_direction: FlexDirection::Column,
                border_style: BorderStyle::Round,
                border_color: composer_border_color,
                padding_left: 1,
                padding_right: 1,
            ) {
                Text(
                    content: prompt_title,
                    color: composer_border_color,
                    weight: Weight::Bold,
                    wrap: TextWrap::NoWrap,
                )
                View(width: 100pct, flex_direction: FlexDirection::Row, overflow: Overflow::Hidden) {
                    Text(content: "❯ ", color: Color::Cyan, weight: Weight::Bold, wrap: TextWrap::NoWrap)
                    View(flex_grow: 1.0_f32, overflow: Overflow::Hidden) {
                        MixedText(
                            wrap: if props.composer.is_empty() { TextWrap::NoWrap } else { TextWrap::Wrap },
                            contents: composer_spans,
                        )
                    }
                }
            }
        }
    }
}

/// Props for [`PortiaTuiApp`].
#[derive(Default, Props)]
pub struct PortiaTuiAppProps {
    pub initial_state: TwoPaneTuiState,
    pub cmd_tx: Option<SmolSender<UiToWorkerMsg>>,
    pub event_rx: Option<SmolReceiver<WorkerToUiEvent>>,
    pub cancel_requested: Option<Arc<AtomicBool>>,
    pub inline_mode: bool,
    pub initial_prompt: Option<String>,
    pub fixed_size: Option<(u16, u16)>,
}

/// Live interactive `iocraft` agent harness root component.
#[component]
pub fn PortiaTuiApp(
    mut hooks: Hooks,
    props: &mut PortiaTuiAppProps,
) -> impl Into<AnyElement<'static>> {
    let mut system = hooks.use_context_mut::<SystemContext>();
    let (stdout, _stderr) = hooks.use_output();
    let (live_w, live_h) = hooks.use_terminal_size();
    let (term_w, term_h) = props.fixed_size.unwrap_or((live_w.max(80), live_h.max(24)));

    let mut state = hooks.use_state(|| props.initial_state.clone());
    let mut composer = hooks.use_state(ComposerBuffer::new);
    let mut history = hooks.use_state(PromptHistory::new);
    let mut active_ask = hooks.use_state(|| None::<ActiveAskSession>);
    let mut status_hint = hooks.use_state(|| None::<String>);
    let should_exit = hooks.use_state(|| false);

    let mut left_scroll = hooks.use_ref(ScrollViewHandle::default);
    let right_scroll = hooks.use_ref(ScrollViewHandle::default);

    let cmd_tx = props.cmd_tx.clone();
    let cancel_flag = props.cancel_requested.clone();
    let inline_mode = props.inline_mode;

    let rx_opt = props.event_rx.take();
    let init_prompt = props.initial_prompt.take();
    let boot_tx = cmd_tx.clone();
    let out_handle = stdout.clone();
    let mut st_bg = state;
    let mut ask_bg = active_ask;
    let mut hint_bg = status_hint;
    let mut exit_bg = should_exit;
    hooks.use_future(async move {
        if let (Some(p), Some(tx)) = (init_prompt, boot_tx) {
            let _ = tx.send(UiToWorkerMsg::SubmitInput(p)).await;
        }
        let Some(rx) = rx_opt else {
            return;
        };
        while let Ok(ev) = rx.recv().await {
            match ev {
                WorkerToUiEvent::StateSnapshot(snap) => {
                    let still_streaming = snap.is_streaming;
                    st_bg.set(*snap);
                    if !still_streaming {
                        hint_bg.set(None);
                    }
                }
                WorkerToUiEvent::AskRequested {
                    question,
                    ask_mode,
                    context_hint,
                    reply_tx,
                } => {
                    ask_bg.set(Some(ActiveAskSession::from_request(
                        question,
                        ask_mode,
                        context_hint,
                        reply_tx,
                    )));
                }
                WorkerToUiEvent::InlineTurnCompleted(block) => {
                    if inline_mode {
                        out_handle.println(block);
                    }
                }
                WorkerToUiEvent::ExitRequested => {
                    exit_bg.set(true);
                    break;
                }
            }
        }
    });

    let on_select_permission: Handler<usize> = Handler::from(move |idx: usize| {
        let mut ask_state = active_ask;
        let mut guard = ask_state.write();
        if let Some(session) = guard.as_mut() {
            if let ActiveAskKind::Permission { cursor, .. } = &mut session.kind {
                *cursor = idx;
            }
            if let Some(ans) = session.resolve_enter_choice() {
                let _ = session.reply_tx.try_send(Some(ans));
                *guard = None;
            }
        }
    });

    let on_toggle_multi: Handler<usize> = Handler::from(move |idx: usize| {
        let mut ask_state = active_ask;
        let mut guard = ask_state.write();
        if let Some(session) = guard.as_mut() {
            if let ActiveAskKind::MultiSelect {
                choices,
                checked,
                cursor,
                custom_text_mode,
            } = &mut session.kind
            {
                *cursor = idx;
                if choices.get(idx).is_some_and(|c| is_escape_choice(c)) {
                    *custom_text_mode = true;
                } else if let Some(slot) = checked.get_mut(idx) {
                    *slot = !*slot;
                }
            }
        }
    });

    let on_submit_multi: Handler<()> = Handler::from(move |()| {
        let mut ask_state = active_ask;
        let mut guard = ask_state.write();
        if let Some(session) = guard.as_mut() {
            if let Some(ans) = session.resolve_enter_choice() {
                let _ = session.reply_tx.try_send(Some(ans));
                *guard = None;
            }
        }
    });

    hooks.use_terminal_events({
        let cmd_tx = cmd_tx.clone();
        let cancel_flag = cancel_flag.clone();
        let mut exit_ev = should_exit;
        move |event| {
            match event {
                TerminalEvent::BracketedPaste(pasted) => {
                    let composer_focused = active_ask
                        .read()
                        .as_ref()
                        .is_none_or(ActiveAskSession::uses_composer_input);
                    if composer_focused && !pasted.is_empty() {
                        composer.write().insert_pasted(&pasted);
                        history.write().reset_navigation();
                    }
                }
                TerminalEvent::Key(KeyEvent {
                    code,
                    modifiers,
                    kind,
                    ..
                }) => {
                    if kind == KeyEventKind::Release {
                        return;
                    }

                    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
                    let alt = modifiers.contains(KeyModifiers::ALT);
                    let shift = modifiers.contains(KeyModifiers::SHIFT);

                    // 1. Clean exit keys (`Ctrl+Q`, or `Ctrl+D` on empty idle composer):
                    if ctrl && matches!(code, KeyCode::Char('q' | 'Q')) {
                        if let Some(f) = &cancel_flag {
                            f.store(true, Ordering::SeqCst);
                        }
                        if let Some(tx) = &cmd_tx {
                            let _ = tx.try_send(UiToWorkerMsg::Shutdown);
                        }
                        exit_ev.set(true);
                        return;
                    }
                    if ctrl
                        && matches!(code, KeyCode::Char('d' | 'D'))
                        && composer.read().is_empty()
                        && !state.read().is_streaming
                        && active_ask.read().is_none()
                    {
                        if let Some(tx) = &cmd_tx {
                            let _ = tx.try_send(UiToWorkerMsg::Shutdown);
                        }
                        exit_ev.set(true);
                        return;
                    }

                    // 2. Non-lethal `Ctrl+C` (cancels active turn/ask, or clears draft):
                    if ctrl && matches!(code, KeyCode::Char('c' | 'C')) {
                        if let Some(ask) = active_ask.write().take() {
                            let _ = ask.reply_tx.try_send(None);
                            if let Some(f) = &cancel_flag {
                                f.store(true, Ordering::SeqCst);
                            }
                            status_hint.set(Some("Cancelled Steward prompt".to_string()));
                        } else if state.read().is_streaming {
                            if let Some(f) = &cancel_flag {
                                f.store(true, Ordering::SeqCst);
                            }
                            status_hint.set(Some("Cancelling turn...".to_string()));
                        } else if !composer.read().is_empty() {
                            composer.write().clear();
                            history.write().reset_navigation();
                            status_hint
                                .set(Some("Cleared input (Ctrl+D or /quit to exit)".to_string()));
                        } else {
                            status_hint
                                .set(Some("Use Ctrl+D, Ctrl+Q, or /quit to exit".to_string()));
                        }
                        return;
                    }

                    // 3. Strictly non-lethal `Esc` hierarchy:
                    if code == KeyCode::Esc {
                        let mut ask_guard = active_ask.write();
                        if let Some(session) = ask_guard.as_mut() {
                            match &mut session.kind {
                                ActiveAskKind::Permission {
                                    custom_text_mode, ..
                                }
                                | ActiveAskKind::MultiSelect {
                                    custom_text_mode, ..
                                } if *custom_text_mode => {
                                    *custom_text_mode = false;
                                    return;
                                }
                                _ => {}
                            }
                            let taken = ask_guard.take().expect("checked Some");
                            let _ = taken.reply_tx.try_send(None);
                            if let Some(f) = &cancel_flag {
                                f.store(true, Ordering::SeqCst);
                            }
                            status_hint.set(Some("Cancelled Steward prompt".to_string()));
                            return;
                        }
                        drop(ask_guard);

                        if state.read().is_streaming {
                            if let Some(f) = &cancel_flag {
                                f.store(true, Ordering::SeqCst);
                            }
                            status_hint.set(Some("Cancelling turn...".to_string()));
                        } else if !composer.read().is_empty() {
                            composer.write().clear();
                            history.write().reset_navigation();
                            status_hint.set(None);
                        } else if state.read().effective_sidebar_mode() != SidebarMode::Hidden {
                            state.write().set_sidebar_mode(SidebarMode::Hidden);
                            if let Some(tx) = &cmd_tx {
                                let _ = tx.try_send(UiToWorkerMsg::UiMutation(
                                    UiLocalMutation::SetSidebarMode(SidebarMode::Hidden),
                                ));
                            }
                        } else {
                            status_hint.set(Some(
                                "Esc never exits — use /quit or Ctrl+D on empty input".to_string(),
                            ));
                        }
                        return;
                    }

                    // 4. Interactive Steward button/checkbox keyboard navigation when modal is active:
                    let structured_ask_active = active_ask
                        .read()
                        .as_ref()
                        .is_some_and(|a| !a.uses_composer_input());
                    if structured_ask_active {
                        match code {
                            KeyCode::Up => active_ask.write().as_mut().unwrap().move_up(),
                            KeyCode::Down => active_ask.write().as_mut().unwrap().move_down(),
                            KeyCode::Char(' ') => active_ask
                                .write()
                                .as_mut()
                                .unwrap()
                                .toggle_current_checkbox(),
                            KeyCode::Enter => {
                                let mut guard = active_ask.write();
                                if let Some(session) = guard.as_mut() {
                                    if let Some(ans) = session.resolve_enter_choice() {
                                        let _ = session.reply_tx.try_send(Some(ans));
                                        *guard = None;
                                    }
                                }
                            }
                            _ => {}
                        }
                        return;
                    }

                    // 5. Drawer & scroll shortcuts (`Tab`, `Ctrl+B`, `Ctrl+T`, `PageUp`/`PageDown`):
                    if code == KeyCode::Tab || (ctrl && matches!(code, KeyCode::Char('b' | 'B'))) {
                        state.write().toggle_schema_drawer();
                        if let Some(tx) = &cmd_tx {
                            let _ = tx.try_send(UiToWorkerMsg::UiMutation(
                                UiLocalMutation::ToggleSchemaDrawer,
                            ));
                        }
                        return;
                    }
                    if ctrl && matches!(code, KeyCode::Char('t' | 'T')) {
                        state.write().toggle_raw_token_trace();
                        if let Some(tx) = &cmd_tx {
                            let _ = tx.try_send(UiToWorkerMsg::UiMutation(
                                UiLocalMutation::ToggleRawTokenTrace,
                            ));
                        }
                        return;
                    }
                    match code {
                        KeyCode::PageUp => {
                            left_scroll.write().scroll_by(-10);
                            return;
                        }
                        KeyCode::PageDown => {
                            left_scroll.write().scroll_by(10);
                            return;
                        }
                        KeyCode::Up if shift => {
                            left_scroll.write().scroll_by(-3);
                            return;
                        }
                        KeyCode::Down if shift => {
                            left_scroll.write().scroll_by(3);
                            return;
                        }
                        _ => {}
                    }

                    // 6. Sequential UTF-8 composer editing, history (`Up`/`Down`), & submit (`Enter`):
                    if ctrl {
                        match code {
                            KeyCode::Char('u' | 'U') => {
                                composer.write().delete_to_start();
                                history.write().reset_navigation();
                            }
                            KeyCode::Char('w' | 'W') => {
                                composer.write().delete_word_back();
                                history.write().reset_navigation();
                            }
                            KeyCode::Char('a' | 'A') => composer.write().move_home(),
                            KeyCode::Char('e' | 'E') => composer.write().move_end(),
                            _ => {}
                        }
                        return;
                    }
                    if alt {
                        return;
                    }

                    match code {
                        KeyCode::Char(ch) => {
                            composer.write().insert_char(ch);
                            history.write().reset_navigation();
                            status_hint.set(None);
                        }
                        KeyCode::Backspace => {
                            composer.write().backspace();
                            history.write().reset_navigation();
                        }
                        KeyCode::Delete => {
                            composer.write().delete_forward();
                            history.write().reset_navigation();
                        }
                        KeyCode::Left => composer.write().move_left(),
                        KeyCode::Right => composer.write().move_right(),
                        KeyCode::Home => composer.write().move_home(),
                        KeyCode::End => composer.write().move_end(),
                        KeyCode::Up => {
                            let current = composer.read().text.clone();
                            if let Some(recalled) = history.write().prev(&current) {
                                composer.write().set_text_at_end(recalled);
                            }
                        }
                        KeyCode::Down => {
                            if let Some(recalled) = history.write().next_entry() {
                                composer.write().set_text_at_end(recalled);
                            }
                        }
                        KeyCode::Enter => {
                            let trimmed = composer.read().text.trim().to_string();
                            if trimmed.is_empty() {
                                return;
                            }
                            if let Some(ask) = active_ask.write().take() {
                                composer.write().clear();
                                let _ = ask.reply_tx.try_send(Some(trimmed));
                                return;
                            }
                            if state.read().is_streaming {
                                status_hint.set(Some(
                                    "Turn in progress — press Esc or Ctrl+C to cancel first"
                                        .to_string(),
                                ));
                                return;
                            }
                            let is_quit = matches!(parse_repl_command(&trimmed), ReplCommand::Quit);
                            history.write().push(&trimmed);
                            composer.write().clear();
                            status_hint.set(None);
                            left_scroll.write().scroll_to_bottom();
                            if let Some(tx) = &cmd_tx {
                                let _ = tx.try_send(UiToWorkerMsg::SubmitInput(trimmed));
                            } else if is_quit {
                                exit_ev.set(true);
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    });

    if should_exit.get() {
        system.exit();
    }

    let composer_has_focus = active_ask
        .read()
        .as_ref()
        .is_none_or(ActiveAskSession::uses_composer_input);
    let cur_state = state.read().clone();
    let cur_composer = composer.read().clone();
    let cur_ask = active_ask.read().clone();
    let cur_hint = status_hint.read().clone();

    if inline_mode {
        let prompt_title = build_prompt_bar_title(&cur_state, cur_hint.as_deref());
        let inline_spans = build_composer_spans(&cur_composer, composer_has_focus);
        return element! {
            View(width: term_w as u32, flex_direction: FlexDirection::Column) {
                HeaderStatusBar(state: cur_state)
                StewardDock(
                    ask: cur_ask,
                    on_select_permission: on_select_permission,
                    on_toggle_multi: on_toggle_multi,
                    on_submit_multi: on_submit_multi,
                )
                View(
                    width: 100pct,
                    flex_direction: FlexDirection::Column,
                    border_style: BorderStyle::Round,
                    border_color: Color::Cyan,
                    padding_left: 1,
                    padding_right: 1,
                ) {
                    Text(content: prompt_title, color: Color::Cyan, weight: Weight::Bold)
                    View(width: 100pct, flex_direction: FlexDirection::Row, overflow: Overflow::Hidden) {
                        Text(content: "❯ ", color: Color::Cyan, weight: Weight::Bold)
                        View(flex_grow: 1.0_f32, overflow: Overflow::Hidden) {
                            MixedText(wrap: TextWrap::Wrap, contents: inline_spans)
                        }
                    }
                }
            }
        }
        .into_any();
    }

    element! {
        PortiaWorkspaceSnapshot(
            state: cur_state,
            total_width: term_w as usize,
            total_height: term_h as usize,
            composer: cur_composer,
            composer_has_focus: composer_has_focus,
            left_scroll_handle: Some(left_scroll),
            right_scroll_handle: Some(right_scroll),
            active_ask: cur_ask,
            on_select_permission: on_select_permission,
            on_toggle_multi: on_toggle_multi,
            on_submit_multi: on_submit_multi,
            status_hint: cur_hint,
        )
    }
    .into_any()
}

/// Launch the interactive `iocraft` TUI (`fullscreen` split-drawer by default, or native terminal
/// scrollback `.render_loop()` when `inline_mode == true`).
pub fn run_iocraft_tui_session(
    mut actor: SessionActorHandle,
    inline_mode: bool,
    initial_prompt: Option<String>,
) -> Result<()> {
    let initial_state = actor.initial_state.clone();
    let cmd_tx = actor.cmd_tx.clone();
    let event_rx = actor.event_rx.clone();
    let cancel_requested = Arc::clone(&actor.cancel_requested);

    let run_res = smol::block_on(async move {
        let mut el = element! {
            PortiaTuiApp(
                initial_state: initial_state,
                cmd_tx: Some(cmd_tx),
                event_rx: Some(event_rx),
                cancel_requested: Some(cancel_requested),
                inline_mode: inline_mode,
                initial_prompt: initial_prompt,
            )
        };
        if inline_mode {
            el.render_loop()
                .ignore_ctrl_c()
                .enable_bracketed_paste()
                .await
        } else {
            el.render_loop()
                .fullscreen()
                .ignore_ctrl_c()
                .enable_bracketed_paste()
                .await
        }
    });

    actor.shutdown();
    Ok(run_res?)
}
