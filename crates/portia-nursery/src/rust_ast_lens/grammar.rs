//! Context-specific `H = 0` Rust AST typed-hole grammar cartridge.

use std::collections::BTreeSet;

use portia_core::relay::{GrammarCartridgeSpec, PayloadGrammarAutomaton};
use portia_core::{CoreProtocolError, Mode};
use serde::{Deserialize, Serialize};

use super::cst_extract::extract_rust_todo_holes;
use super::scenario_affordances::format_write_body_lark_rule;

/// Context-specific `H = 0` Rust AST typed-hole grammar cartridge implementing
/// [`PayloadGrammarAutomaton`].
///
/// - In `Mode::Write`, constrains the hole splice to start with `<hole_fn_name> => ` for one of
///   `self.hole_fn_names` (`GrammarCartridgeSpec::LarkGrammar(...)`).
/// - In `Mode::Probe | Mode::Read`, constrains the payload to valid outline/hole inspection commands.
/// - In `Mode::Result | Mode::Ask | Mode::Abstain` (or when `hole_fn_names` is empty), detaches
///   to `GrammarCartridgeSpec::Unconstrained`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustAstHoleGrammarCartridge {
    pub hole_fn_names: Vec<String>,
    pub approved_crate_name: Option<String>,
}

impl RustAstHoleGrammarCartridge {
    pub fn new(hole_fn_names: Vec<String>, approved_crate_name: Option<String>) -> Self {
        Self {
            hole_fn_names,
            approved_crate_name,
        }
    }

    /// Build a [`RustAstHoleGrammarCartridge`] directly from the `todo!("...")` holes extracted
    /// via `tree-sitter-rust` from `rust_source`.
    pub fn from_rust_source(
        rust_source: &str,
        approved_crate_name: Option<String>,
    ) -> Result<Self, CoreProtocolError> {
        let holes = extract_rust_todo_holes(rust_source)?;
        Ok(Self {
            hole_fn_names: holes.into_iter().map(|h| h.fn_name).collect(),
            approved_crate_name,
        })
    }

    fn clean_hole_names(&self) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for name in &self.hole_fn_names {
            let trimmed = name.trim().trim_matches('`').trim();
            if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
                out.push(trimmed.to_string());
            }
        }
        out
    }

    fn format_hole_alternatives(holes: &[String]) -> String {
        holes
            .iter()
            .map(|h| {
                let escaped = h.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"{escaped}\"")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Emit a wire-closed `llguidance` [`GrammarCartridgeSpec`] terminating `Probe | Read | Write`
    /// spans cleanly at `\n</sql>` and `Ask | Result | Abstain` spans at `\n</to_user>`.
    pub fn span_grammar_cartridge_spec(&self, mode: Mode) -> GrammarCartridgeSpec {
        let holes = self.clean_hole_names();
        match mode {
            Mode::Probe => GrammarCartridgeSpec::LarkGrammar(
                "start: (\"inspect rust ast outline\"i | \"outline\"i) \"\\n</sql>\"".to_string(),
            ),
            Mode::Read => {
                if holes.is_empty() {
                    return GrammarCartridgeSpec::Unconstrained;
                }
                let hole_alts = Self::format_hole_alternatives(&holes);
                GrammarCartridgeSpec::LarkGrammar(format!(
                    "start: (\"fn \"? hole_fn_name) \"\\n</sql>\"\nhole_fn_name: {hole_alts}"
                ))
            }
            Mode::Write => {
                if holes.is_empty() {
                    return GrammarCartridgeSpec::Unconstrained;
                }
                let hole_alts = Self::format_hole_alternatives(&holes);
                let body_rule = format_write_body_lark_rule(self.approved_crate_name.as_deref());
                GrammarCartridgeSpec::LarkGrammar(format!(
                    "start: (hole_fn_name \" => \" hole_body) \"\\n</sql>\"\nhole_fn_name: {hole_alts}\nhole_body: {body_rule}"
                ))
            }
            Mode::Ask => GrammarCartridgeSpec::LarkGrammar(
                "start: /[^\\r\\n<>]{2,320}/ (\"\\n\" /[^\\r\\n<>]{1,320}/){0,3} \"\\n</to_user>\""
                    .to_string(),
            ),
            Mode::Result | Mode::Abstain => GrammarCartridgeSpec::LarkGrammar(
                "start: /[^\\r\\n<>]{2,400}/ (\"\\n\" /[^\\r\\n<>]{1,400}/){0,5} \"\\n</to_user>\""
                    .to_string(),
            ),
        }
    }
}

impl PayloadGrammarAutomaton for RustAstHoleGrammarCartridge {
    fn grammar_cartridge_spec(&self, mode: Mode) -> GrammarCartridgeSpec {
        let holes = self.clean_hole_names();
        if holes.is_empty() {
            return GrammarCartridgeSpec::Unconstrained;
        }
        let hole_alts = Self::format_hole_alternatives(&holes);

        match mode {
            Mode::Probe => GrammarCartridgeSpec::LarkGrammar(
                "start: \"inspect rust ast outline\"i | \"outline\"i\n%ignore /[ \\t\\r\\n]+/\n"
                    .to_string(),
            ),
            Mode::Read => GrammarCartridgeSpec::LarkGrammar(format!(
                "start: \"fn \"? hole_fn_name\nhole_fn_name: {hole_alts}\n%ignore /[ \\t\\r\\n]+/\n"
            )),
            Mode::Write => {
                let body_rule = format_write_body_lark_rule(self.approved_crate_name.as_deref());
                GrammarCartridgeSpec::LarkGrammar(format!(
                    "start: hole_fn_name \" => \" hole_body\nhole_fn_name: {hole_alts}\nhole_body: {body_rule}\n"
                ))
            }
            Mode::Result | Mode::Ask | Mode::Abstain => GrammarCartridgeSpec::Unconstrained,
        }
    }
}
