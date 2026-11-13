//! JSON-over-stdio wire protocol (`WorkerRequest`, `WorkerResponse`, `CaptureBoundary`) and
//! `<tool_response>` observation wire formatting (`format_observation_wire_chunk`).

use crate::driver::{
    parse_ask_user_reply, parse_ask_user_reply_for_kind, AskPromptKind, AskUserReply,
    DriverObservation, LiveSqliteDailyDriver, ObservationOperation, ReadPurpose,
};
use crate::model_identity::RuntimeProvenance;
use anyhow::{bail, Context, Result};
use portia_core::relay::RelayPhase;
use portia_core::{
    CartridgeBank, ColumnSchema, LotlNormalizedWireChunk, Mode, PenOwner, StewardAuthState,
    TableSchema,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Return `true` when `cartridge` explicitly requests un-augmented `Base` trunk weights (`"none"`).
pub fn is_unpatched_base_cartridge_sentinel(cartridge: &Path) -> bool {
    cartridge
        .to_str()
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("none"))
}

fn default_worker_mean_natural_token_norm() -> f64 {
    1.0
}

/// Validate an explicitly supplied `--cartridge` JSON path before spawning the worker so missing,
/// unreadable, or malformed JSON files fail immediately instead of silently running unpatched.
pub fn validate_explicit_cartridge_path(cartridge: &Path) -> Result<()> {
    if is_unpatched_base_cartridge_sentinel(cartridge) {
        return Ok(());
    }
    if !cartridge.exists() {
        bail!("Explicit cartridge file not found: {:?}", cartridge);
    }
    if !cartridge.is_file() {
        bail!(
            "Explicit cartridge path is not a regular file: {:?}",
            cartridge
        );
    }
    let raw = std::fs::read_to_string(cartridge)
        .with_context(|| format!("Failed to read explicit cartridge file {:?}", cartridge))?;
    portia_compiler::wire::WireCartridgeManifestJson::from_json_str(&raw).with_context(|| {
        format!(
            "Explicit cartridge file {:?} failed WireCartridgeManifestJson schema validation",
            cartridge
        )
    })?;
    Ok(())
}

/// JSON-over-stdio request sent from `portia-cli` to the thin PyTorch Qwen3.5 worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkerRequest {
    RuntimeMetadata,
    /// Reset the worker's incremental hybrid 18-DeltaNet + 6-GQA `past_key_values` cache.
    ResetSession {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        table_name: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        columns: Vec<ColumnSchema>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tables: Vec<TableSchema>,
    },
    /// Reset per-task execution state (`db_observations_seen`, `last_tool_response_json`,
    /// `last_observed_row`, `write_applied_in_session`) at the start of a follow-up user prompt
    /// in a multi-task REPL or TUI session while preserving `past_key_values`.
    BeginTask {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        table_name: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        columns: Vec<ColumnSchema>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tables: Vec<TableSchema>,
    },
    /// Mount `bank` (`Base`, `OrientTrunk`, or `ModePayload(mode)`) and consume `text` into the
    /// hybrid 18-DeltaNet + 6-GQA cache (`use_cache = True`).
    Prefill {
        bank: CartridgeBank,
        pen: PenOwner,
        text: LotlNormalizedWireChunk,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        table_name: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        columns: Vec<ColumnSchema>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tables: Vec<TableSchema>,
        #[serde(default)]
        reset_kv: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        capture_layer: Option<usize>,
    },
    /// Sample 1 token from `last_logits` under `bank` and `grammar_spec` without advancing
    /// `past_key_values` yet (Phase 1 of the two-phase `Sample -> Plan -> Consume` handoff).
    SampleToken {
        bank: CartridgeBank,
        pen: PenOwner,
        phase: RelayPhase,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        table_name: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        columns: Vec<ColumnSchema>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tables: Vec<TableSchema>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_token_ids: Option<Vec<u32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grammar_spec: Option<portia_core::GrammarCartridgeSpec>,
    },
    /// Mount `bank` (`planned.consume_bank`, e.g. `CartridgeBank::Base` on closing `</sql>` /
    /// `</to_user>`) and forward `token_id` into `past_key_values` (Phase 2 of two-phase handoff).
    ConsumeToken {
        bank: CartridgeBank,
        pen: PenOwner,
        token_id: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        capture_layer: Option<usize>,
    },
    /// Extract weight-space geometry (`norm_weight`, `named_token_unembeddings`, `downstream_mlp_anti_echo`,
    /// `base_intermediate_size`, etc.) directly from the loaded trunk weights.
    ExtractCompilerWeightGeometry {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_layer: Option<usize>,
        #[serde(default)]
        token_ids: Vec<u32>,
    },
    CapturePayloadActivations {
        wire_prefix: String,
        payload: String,
        closing_tag: String,
        target_layer: usize,
    },
    /// Capture `(pre_mlp, residual)` at `<|im_start|>assistant\n<think>\n[MODE:` on `target_layer`,
    /// force `" <ORACLE_MODE>]\nPlan:"`, and generate genuine on-policy `Plan:` tokens under `Base`.
    CaptureAndGenerateOnPolicyPlan {
        wire_prefix: String,
        oracle_mode: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_layer: Option<usize>,
        max_plan_tokens: usize,
    },
}

/// Payload returned by `extract_compiler_weight_geometry` (`"status": "weight_geometry"`), boxed
/// inside [`WorkerResponse::WeightGeometry`] so per-token RPC variants keep a small stack footprint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerWeightGeometryPayload {
    #[serde(default)]
    pub target_layer: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    pub base_intermediate_size: usize,
    #[serde(default)]
    pub vocab_top_k_eigvecs: Vec<Vec<f64>>,
    #[serde(default)]
    pub protected_sql_embeddings: Vec<Vec<f64>>,
    #[serde(alias = "final_norm_weight", alias = "model_norm_weight")]
    pub norm_weight: Vec<f64>,
    #[serde(default = "default_worker_mean_natural_token_norm")]
    pub mean_natural_token_norm: f64,
    #[serde(default)]
    pub qk_layer_vectors: BTreeMap<String, crate::compile::QkLayerVectorsSpec>,
    #[serde(default)]
    pub deltanet_layer_weights: BTreeMap<String, crate::compile::DeltaNetLayerWeightSpec>,
    #[serde(default)]
    pub named_token_unembeddings: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    pub downstream_mlp_anti_echo: Option<crate::compile::DownstreamMlpAntiEchoSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationState {
    pub pre_mlp: Vec<f64>,
    pub residual: Vec<f64>,
}

pub type OnPolicyCaptureWithBackground = (Vec<f64>, Vec<f64>, String, Vec<ActivationState>);

/// Strongly-typed activation capture boundary tag (`"router_decision"`, `"router_background"`,
/// `"opener"`, `"field"`, `"value"`, `"background"`, `"close_0"`, `"close_1"`, ...).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CaptureBoundary {
    RouterDecision,
    RouterBackground,
    Opener,
    Field,
    Value,
    Background,
    CloseTag { subtoken_index: usize },
}

impl CaptureBoundary {
    #[must_use]
    pub fn as_wire_str(&self) -> std::borrow::Cow<'static, str> {
        match self {
            Self::RouterDecision => std::borrow::Cow::Borrowed("router_decision"),
            Self::RouterBackground => std::borrow::Cow::Borrowed("router_background"),
            Self::Opener => std::borrow::Cow::Borrowed("opener"),
            Self::Field => std::borrow::Cow::Borrowed("field"),
            Self::Value => std::borrow::Cow::Borrowed("value"),
            Self::Background => std::borrow::Cow::Borrowed("background"),
            Self::CloseTag { subtoken_index: 0 } => std::borrow::Cow::Borrowed("close_0"),
            Self::CloseTag { subtoken_index } => {
                std::borrow::Cow::Owned(format!("close_{subtoken_index}"))
            }
        }
    }

    pub fn from_wire_str(raw: &str) -> Option<Self> {
        match raw.trim() {
            "router_decision" => Some(Self::RouterDecision),
            "router_background" => Some(Self::RouterBackground),
            "opener" => Some(Self::Opener),
            "field" => Some(Self::Field),
            "value" => Some(Self::Value),
            "background" => Some(Self::Background),
            other => {
                let idx = other.strip_prefix("close_")?.parse::<usize>().ok()?;
                Some(Self::CloseTag {
                    subtoken_index: idx,
                })
            }
        }
    }
}

impl std::fmt::Display for CaptureBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_wire_str())
    }
}

impl Serialize for CaptureBoundary {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.as_wire_str())
    }
}

impl<'de> Deserialize<'de> for CaptureBoundary {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::from_wire_str(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!("unknown CaptureBoundary wire tag {raw:?}"))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayloadActivationRow {
    pub boundary: CaptureBoundary,
    pub pre_mlp: Vec<f64>,
    pub residual: Vec<f64>,
    pub target_token_id: u32,
    pub competitor_token_id: u32,
}

/// JSON-over-stdio response returned by the PyTorch Qwen3.5 worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WorkerResponse {
    RuntimeMetadata {
        provenance: RuntimeProvenance,
    },
    Prefilled {
        token_ids: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        captured_pre_mlp: Option<Vec<f32>>,
    },
    SampledToken {
        token_id: u32,
        token_piece: String,
        #[serde(default)]
        token_ids: Vec<u32>,
    },
    ConsumedToken {
        #[serde(default)]
        token_ids: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        captured_pre_mlp: Option<Vec<f32>>,
    },
    WeightGeometry(Box<WorkerWeightGeometryPayload>),
    PayloadActivations {
        rows: Vec<PayloadActivationRow>,
    },
    OnPolicyPlanCaptured {
        target_layer: usize,
        pre_mlp: Vec<f64>,
        residual: Vec<f64>,
        generated_plan_line: String,
        #[serde(default)]
        router_background: Vec<ActivationState>,
    },
    Error {
        message: String,
    },
}

/// Format an environment or inquirer observation into the exact `<|im_start|>user\n<tool_response>...`
/// wire chunk consumed by the worker under `CartridgeBank::Base`.
///
/// If a query returns duplicate column names (e.g., `SELECT 1 AS x, 2 AS x` or an unaliased `JOIN`),
/// the second and subsequent occurrences in each row object are disambiguated with their 1-based
/// column position (`"x#2"`, `"x#3"`, ...), and the raw positional `"column_names"` and `"row_cells"`
/// are included whenever duplicate column names are present so zero cells are ever overwritten.
pub fn format_observation_wire_chunk(obs: &DriverObservation) -> Result<String> {
    let mut has_duplicate_columns = false;
    let mut seen_cols = std::collections::BTreeSet::new();
    for col in &obs.columns {
        if !seen_cols.insert(col.as_str()) {
            has_duplicate_columns = true;
            break;
        }
    }

    let row_maps: Vec<BTreeMap<String, serde_json::Value>> = obs
        .rows
        .iter()
        .map(|r| {
            let mut map = BTreeMap::new();
            for (col_idx, (col_name, val)) in obs.columns.iter().zip(r.iter()).enumerate() {
                if !map.contains_key(col_name) {
                    map.insert(col_name.clone(), val.clone());
                } else {
                    map.insert(format!("{}#{}", col_name, col_idx + 1), val.clone());
                }
            }
            map
        })
        .collect();
    let mut env_payload = if !obs.ok {
        serde_json::json!({
            "error": obs.error,
            "event_id": obs.event_id,
            "mode": obs.mode.as_str(),
            "status": "ERROR",
        })
    } else {
        match obs.mode {
            Mode::Probe => {
                if has_duplicate_columns {
                    serde_json::json!({
                        "column_names": obs.columns,
                        "columns": row_maps,
                        "purpose": "schema_probe",
                        "row_cells": obs.rows,
                        "status": "OK",
                    })
                } else {
                    serde_json::json!({
                        "columns": row_maps,
                        "purpose": "schema_probe",
                        "status": "OK",
                    })
                }
            }
            Mode::Read => {
                let purpose = match obs.operation {
                    Some(ObservationOperation::SqlRead(read_purpose)) => read_purpose.as_str(),
                    Some(ObservationOperation::Ast(ast_op)) => ast_op.as_str(),
                    Some(ObservationOperation::SqlWrite(write_op)) => write_op.as_str(),
                    None => {
                        if obs.db_version == 0 {
                            ReadPurpose::Candidate.as_str()
                        } else {
                            ReadPurpose::Verification.as_str()
                        }
                    }
                };
                if has_duplicate_columns {
                    serde_json::json!({
                        "column_names": obs.columns,
                        "purpose": purpose,
                        "row_cells": obs.rows,
                        "row_count": row_maps.len(),
                        "rows": row_maps,
                        "status": "OK",
                        "steward_auth": obs.steward_auth.as_wire_str(),
                    })
                } else {
                    serde_json::json!({
                        "purpose": purpose,
                        "row_count": row_maps.len(),
                        "rows": row_maps,
                        "status": "OK",
                        "steward_auth": obs.steward_auth.as_wire_str(),
                    })
                }
            }
            Mode::Write => {
                let status_str = if obs.rows_affected == 0 {
                    "ZERO_ROWS_AFFECTED"
                } else {
                    "APPLIED"
                };
                serde_json::json!({
                    "operation": obs.operation,
                    "rows_affected": obs.rows_affected,
                    "status": status_str,
                })
            }
            Mode::Ask => {
                let resolved_reply = match &obs.ask_reply {
                    Some(parsed) => Some(parsed.clone()),
                    None => obs
                        .message
                        .as_deref()
                        .and_then(|msg| parse_ask_user_reply(msg).ok()),
                };
                match resolved_reply {
                    Some(AskUserReply::Decision {
                        decision,
                        raw_reply,
                    }) => serde_json::json!({
                        "decision": decision.as_str(),
                        "reply": raw_reply,
                        "status": decision.status_str(),
                        "steward_auth": obs.steward_auth.as_wire_str(),
                    }),
                    Some(AskUserReply::Clarification { clarification }) => serde_json::json!({
                        "clarification": clarification,
                        "status": "CLARIFIED",
                        "steward_auth": obs.steward_auth.as_wire_str(),
                    }),
                    None => {
                        let msg = obs.message.as_deref().unwrap_or("");
                        if let Ok(parsed_clar) =
                            parse_ask_user_reply_for_kind(AskPromptKind::Clarification, msg)
                        {
                            match parsed_clar {
                                AskUserReply::Decision {
                                    decision,
                                    raw_reply,
                                } => serde_json::json!({
                                    "decision": decision.as_str(),
                                    "reply": raw_reply,
                                    "status": decision.status_str(),
                                    "steward_auth": obs.steward_auth.as_wire_str(),
                                }),
                                AskUserReply::Clarification { clarification } => {
                                    if msg.trim().starts_with('{') {
                                        if let Ok(parsed_json) =
                                            serde_json::from_str::<serde_json::Value>(msg)
                                        {
                                            let has_clar_key =
                                                parsed_json.as_object().is_some_and(|o| {
                                                    o.contains_key("clarification")
                                                        || o.contains_key("selected")
                                                        || o.contains_key("choices")
                                                        || o.contains_key("options")
                                                });
                                            if !has_clar_key {
                                                parsed_json
                                            } else {
                                                serde_json::json!({
                                                    "clarification": clarification,
                                                    "status": "CLARIFIED",
                                                    "steward_auth": obs.steward_auth.as_wire_str(),
                                                })
                                            }
                                        } else {
                                            serde_json::json!({
                                                "clarification": clarification,
                                                "status": "CLARIFIED",
                                                "steward_auth": obs.steward_auth.as_wire_str(),
                                            })
                                        }
                                    } else {
                                        serde_json::json!({
                                            "clarification": clarification,
                                            "status": "CLARIFIED",
                                            "steward_auth": obs.steward_auth.as_wire_str(),
                                        })
                                    }
                                }
                            }
                        } else {
                            serde_json::json!({
                                "error": format!("Unrecognized [MODE: ASK] reply: {msg:?}"),
                                "status": "ERROR",
                                "steward_auth": obs.steward_auth.as_wire_str(),
                            })
                        }
                    }
                }
            }
            Mode::Result | Mode::Abstain => serde_json::json!({
                "status": "OK",
            }),
        }
    };
    if obs.steward_auth != StewardAuthState::NotRequired {
        if let Some(obj) = env_payload.as_object_mut() {
            obj.entry("steward_auth".to_string()).or_insert_with(|| {
                serde_json::Value::String(obs.steward_auth.as_wire_str().to_string())
            });
        }
    }
    let obs_json = serde_json::to_string(&env_payload)?;
    Ok(format!(
        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
        obs_json
    ))
}

/// Alias for [`format_observation_wire_chunk`].
pub fn format_driver_observation_for_worker(obs: &DriverObservation) -> Result<String> {
    format_observation_wire_chunk(obs)
}

/// Format a compact, rule-free schema signature of all tables and columns in the connected
/// SQLite database (e.g., ``[Table `wallets` (`wallet_id`, `holder`, `credits`, `locked`)]``)
/// using metadata-only introspection (`inspect_schema_metadata`) without running `SELECT COUNT(*)`.
pub fn format_live_schema_summary(driver: &LiveSqliteDailyDriver) -> Result<String> {
    let tables = driver.inspect_schema_metadata()?;
    if tables.is_empty() {
        return Ok("[Tables: (none)]".to_string());
    }
    let mut lines = Vec::with_capacity(tables.len());
    for tbl in tables {
        let col_specs: Vec<String> = tbl
            .columns
            .iter()
            .map(|c| format!("`{}`", c.name))
            .collect();
        lines.push(format!(
            "[Table `{}` ({})]",
            tbl.table_name,
            col_specs.join(", ")
        ));
    }
    Ok(lines.join("\n"))
}
