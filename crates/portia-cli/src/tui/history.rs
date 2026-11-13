//! Interactive composer buffer, prompt history (`Up` / `Down` with live draft preservation),
//! and UTF-8 char-boundary line-editing operations (`Ctrl+A`, `Ctrl+E`, `Ctrl+U`, `Ctrl+W`,
//! `Left`, `Right`, `Backspace`, `Delete`, bracketed paste).

/// Single-line UTF-8 text + byte-cursor state for the TUI natural-language composer.
///
/// Keeping character insertion, cursor motion, history recall, `Esc`, and `Enter` in one
/// sequential state machine prevents split-queue ordering races when fast typing or SSH/paste
/// bursts deliver printable characters and `Enter`/`Esc` in the same frame tick.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComposerBuffer {
    pub text: String,
    pub cursor: usize,
}

impl ComposerBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn set_text_at_end(&mut self, next: String) {
        self.cursor = next.len();
        self.text = next;
    }

    pub fn insert_char(&mut self, ch: char) {
        let pos = clamp_char_boundary(&self.text, self.cursor);
        self.text.insert(pos, ch);
        self.cursor = pos + ch.len_utf8();
    }

    pub fn insert_pasted(&mut self, pasted: &str) {
        let normalized: String = pasted
            .chars()
            .map(|c| {
                if matches!(c, '\r' | '\n' | '\t') {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        let trimmed = normalized.replace("  ", " ");
        let pos = clamp_char_boundary(&self.text, self.cursor);
        self.text.insert_str(pos, &trimmed);
        self.cursor = pos + trimmed.len();
    }

    pub fn backspace(&mut self) {
        let pos = clamp_char_boundary(&self.text, self.cursor);
        if pos == 0 {
            return;
        }
        let prev = self.text[..pos]
            .char_indices()
            .next_back()
            .map_or(0, |(idx, _)| idx);
        self.text.replace_range(prev..pos, "");
        self.cursor = prev;
    }

    pub fn delete_forward(&mut self) {
        let pos = clamp_char_boundary(&self.text, self.cursor);
        if pos >= self.text.len() {
            return;
        }
        let next = self.text[pos..]
            .chars()
            .next()
            .map_or(self.text.len(), |c| pos + c.len_utf8());
        self.text.replace_range(pos..next, "");
    }

    pub fn move_left(&mut self) {
        let pos = clamp_char_boundary(&self.text, self.cursor);
        if pos > 0 {
            self.cursor = self.text[..pos]
                .char_indices()
                .next_back()
                .map_or(0, |(idx, _)| idx);
        }
    }

    pub fn move_right(&mut self) {
        let pos = clamp_char_boundary(&self.text, self.cursor);
        if pos < self.text.len() {
            if let Some(ch) = self.text[pos..].chars().next() {
                self.cursor = pos + ch.len_utf8();
            }
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn delete_to_start(&mut self) {
        let (next_txt, next_cur) = delete_to_line_start(&self.text, self.cursor);
        self.text = next_txt;
        self.cursor = next_cur;
    }

    pub fn delete_word_back(&mut self) {
        let (next_txt, next_cur) = delete_prev_word(&self.text, self.cursor);
        self.text = next_txt;
        self.cursor = next_cur;
    }
}

/// Stateful prompt/command history ring for the TUI composer (`Up` / `Down`).
#[derive(Debug, Clone, Default)]
pub struct PromptHistory {
    entries: Vec<String>,
    cursor: Option<usize>,
    saved_draft: String,
}

impl PromptHistory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record a submitted non-empty line, deduplicating consecutive identical submissions and
    /// resetting navigation cursor + saved draft.
    pub fn push(&mut self, raw: &str) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            self.cursor = None;
            self.saved_draft.clear();
            return;
        }
        if self.entries.last().map(String::as_str) != Some(trimmed) {
            self.entries.push(trimmed.to_string());
        }
        self.cursor = None;
        self.saved_draft.clear();
    }

    /// Step backward (`Up` arrow) into older history entries, saving `current_input` as the live
    /// bottom draft when entering history navigation. Returns `None` if already at the oldest entry.
    pub fn prev(&mut self, current_input: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let next_idx = match self.cursor {
            None => {
                self.saved_draft = current_input.to_string();
                self.entries.len() - 1
            }
            Some(0) => return None,
            Some(idx) => idx - 1,
        };
        self.cursor = Some(next_idx);
        Some(self.entries[next_idx].clone())
    }

    /// Step forward (`Down` arrow) toward newer entries, restoring the saved in-progress draft
    /// when stepping past the newest history entry.
    pub fn next_entry(&mut self) -> Option<String> {
        let idx = self.cursor?;
        if idx + 1 < self.entries.len() {
            let next_idx = idx + 1;
            self.cursor = Some(next_idx);
            Some(self.entries[next_idx].clone())
        } else {
            self.cursor = None;
            Some(self.saved_draft.clone())
        }
    }

    /// Reset active history navigation when the user edits text directly.
    pub fn reset_navigation(&mut self) {
        self.cursor = None;
    }
}

fn clamp_char_boundary(text: &str, byte_offset: usize) -> usize {
    let mut idx = byte_offset.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Delete from `cursor_byte_offset` back to byte `0` (`Ctrl+U`), returning `(new_text, new_cursor)`.
#[must_use]
pub fn delete_to_line_start(text: &str, cursor_byte_offset: usize) -> (String, usize) {
    let cut = clamp_char_boundary(text, cursor_byte_offset);
    (text[cut..].to_string(), 0)
}

/// Delete the word immediately preceding `cursor_byte_offset` (`Ctrl+W`), returning `(new_text, new_cursor)`.
#[must_use]
pub fn delete_prev_word(text: &str, cursor_byte_offset: usize) -> (String, usize) {
    let end = clamp_char_boundary(text, cursor_byte_offset);
    if end == 0 {
        return (text.to_string(), 0);
    }
    let head = &text[..end];
    let trimmed_ws = head.trim_end_matches(char::is_whitespace);
    let cut = trimmed_ws
        .char_indices()
        .rev()
        .find(|&(_, ch)| ch.is_whitespace())
        .map_or(0, |(idx, ch)| idx + ch.len_utf8());
    let mut out = String::with_capacity(text.len() - (end - cut));
    out.push_str(&text[..cut]);
    out.push_str(&text[end..]);
    (out, cut)
}
