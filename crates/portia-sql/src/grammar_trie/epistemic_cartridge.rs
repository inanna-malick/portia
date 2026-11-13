//! Causal, prompt-blind (`∀prompt`-parametric) [`EpistemicSqlGrammarCartridge`] for progressive SQLite discovery.

use std::collections::BTreeSet;

use portia_core::TableSchema;

use super::lark_specialize::{
    compile_schema_ident_lark_rule_and_terminal, specialize_sqlite_lark_for_schemas,
};

/// Causal, prompt-blind (`∀prompt`-parametric) Epistemic SQLite Grammar Cartridge governing
/// 10–100 table SQLite databases without upfront schema dumps or `_decoy` column filters:
/// - Before any table schema is probed (`probed_schemas.is_empty()`), `Read` and `Write` are
///   epistemically unavailable (`legal_modes` = `[Probe, Ask, Abstain]`).
/// - `Mode::Probe` emits `SELECT name FROM sqlite_master WHERE type = 'table';\n</sql>` (catalog
///   discovery) and `PRAGMA table_info(<table>);\n</sql>` (preferring unprobed tables once catalog is known).
/// - `Mode::Read` and `Mode::Write` bind stratified `sqlite.lark` (`";\n</sql>"`) strictly to `probed_schemas`.
/// - `Mode::Ask` mounts structured `%json` GUI schemas when `pending_steward_ask` or
///   `ambiguous_candidates.len() >= 2`, and bounded stop-tag prose (`"\n</to_user>"`) on Chitchat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicSqlGrammarCartridge<'a> {
    pub all_table_names: &'a [String],
    pub catalog_discovered: bool,
    pub probed_schemas: &'a [TableSchema],
    pub pending_steward_ask: bool,
    pub steward_grant_approved: bool,
    pub ambiguous_candidates: &'a [String],
    pub read_only: bool,
}

impl portia_core::relay::PayloadGrammarAutomaton for EpistemicSqlGrammarCartridge<'_> {
    fn legal_modes(&self) -> Vec<portia_core::Mode> {
        use portia_core::Mode;
        if self.probed_schemas.is_empty() {
            return vec![Mode::Probe, Mode::Ask, Mode::Abstain];
        }
        let mut modes = vec![Mode::Probe, Mode::Read, Mode::Ask];
        if !self.read_only && (!self.pending_steward_ask || self.steward_grant_approved) {
            modes.push(Mode::Write);
        }
        modes.push(Mode::Result);
        modes.push(Mode::Abstain);
        modes
    }

    fn grammar_cartridge_spec(&self, mode: portia_core::Mode) -> portia_core::GrammarCartridgeSpec {
        compile_epistemic_span_grammar(self, mode)
    }
}

/// Compile the `llguidance` [`portia_core::GrammarCartridgeSpec`] (including terminal `\n</sql>`
/// or `\n</to_user>` span closure) for `cart` under `mode`.
pub fn compile_epistemic_span_grammar(
    cart: &EpistemicSqlGrammarCartridge<'_>,
    mode: portia_core::Mode,
) -> portia_core::GrammarCartridgeSpec {
    use portia_core::{GrammarCartridgeSpec, Mode};
    match mode {
        Mode::Probe => {
            let probed_lower: BTreeSet<String> = cart
                .probed_schemas
                .iter()
                .map(|s| s.table_name.trim().to_ascii_lowercase())
                .collect();
            let unprobed: Vec<&String> = cart
                .all_table_names
                .iter()
                .filter(|t| !probed_lower.contains(&t.trim().to_ascii_lowercase()))
                .collect();
            let candidate_tables: Vec<&String> = if !unprobed.is_empty() {
                unprobed
            } else {
                cart.all_table_names.iter().collect()
            };
            let compiled_probe_tables = compile_schema_ident_lark_rule_and_terminal(
                candidate_tables.iter().map(|s| s.as_str()),
                "PROBE_TABLE_IDENT",
            );
            // SQL keywords/identifiers are case-insensitive; SQLite string values are not.
            let catalog_rule = "\"SELECT name FROM sqlite_master WHERE type = \"i \"'table'\"";
            let (start_rhs, term_suffix) = match (cart.catalog_discovered, compiled_probe_tables) {
                (_, None) => (catalog_rule.to_string(), String::new()),
                (false, Some((table_rhs, term_def))) => (
                    format!("({catalog_rule} | \"PRAGMA table_info(\"i ({table_rhs}) \")\")"),
                    term_def.map(|d| format!("\n{d}")).unwrap_or_default(),
                ),
                (true, Some((table_rhs, term_def))) => (
                    format!("\"PRAGMA table_info(\"i ({table_rhs}) \")\""),
                    term_def.map(|d| format!("\n{d}")).unwrap_or_default(),
                ),
            };
            GrammarCartridgeSpec::LarkGrammar(format!(
                "start: {start_rhs} \";\\n</sql>\"{term_suffix}"
            ))
        }
        Mode::Read | Mode::Write => {
            if cart.probed_schemas.is_empty() {
                return GrammarCartridgeSpec::Unconstrained;
            }
            let start_rule = match mode {
                Mode::Read => {
                    "start: (select_stmt | compound_select_stmt | simple_select_stmt) \";\\n</sql>\""
                }
                Mode::Write => {
                    "start: (update_stmt | insert_stmt | delete_stmt) \";\\n</sql>\""
                }
                _ => unreachable!(),
            };
            specialize_sqlite_lark_for_schemas(cart.probed_schemas, start_rule)
        }
        Mode::Ask => {
            if cart.pending_steward_ask && !cart.steward_grant_approved {
                let schema = r#"{"type":"object","properties":{"select":{"type":"string","minLength":4,"maxLength":160},"options":{"type":"array","items":{"type":"string","enum":["ALLOW","DENY"]},"minItems":2,"maxItems":2}},"required":["select","options"],"additionalProperties":false}"#;
                GrammarCartridgeSpec::LarkGrammar(format!(
                    "start: ask_obj \"\\n</to_user>\"\nask_obj: %json {schema}\n"
                ))
            } else if cart.ambiguous_candidates.len() >= 2 {
                let enum_json = serde_json::to_string(cart.ambiguous_candidates)
                    .unwrap_or_else(|_| "[]".into());
                let schema = format!(
                    r#"{{"type":"object","properties":{{"multi":{{"type":"string","minLength":4,"maxLength":160}},"options":{{"type":"array","items":{{"type":"string","enum":{enum_json}}},"minItems":2}}}},"required":["multi","options"],"additionalProperties":false}}"#
                );
                GrammarCartridgeSpec::LarkGrammar(format!(
                    "start: multi_obj \"\\n</to_user>\"\nmulti_obj: %json {schema}\n"
                ))
            } else {
                GrammarCartridgeSpec::LarkGrammar(
                    "start: /[^\\r\\n<>]+/ (\"\\n\" /[^\\r\\n<>]+/)* \"\\n</to_user>\"".to_string(),
                )
            }
        }
        Mode::Result | Mode::Abstain => {
            // Terminal wire bodies end naturally on `\n</to_user>` without artificial line-count
            // or 320-byte training wheels; the outer 512-token turn loop bounds runaway failures.
            GrammarCartridgeSpec::LarkGrammar(
                concat!(
                    "start: FIRST (\"\\n\" LINE)* \"\\n</to_user>\"\n",
                    "FIRST: /[^\\r\\n<>]+/ & /[^\\S\\r\\n]*[^\\s{<>][^\\r\\n<>]*/\n",
                    "LINE: /[^\\r\\n<>]+/ & /[^\\S\\r\\n]*[^\\s<>][^\\r\\n<>]*/\n",
                )
                .to_string(),
            )
        }
    }
}
