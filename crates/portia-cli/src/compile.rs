//! Pure-Rust Clarabel SOCP + `faer` weight-cartridge compilation (`portia-cli compile`),
//! `RegimeExpr` schema specialization, and `half::bf16` SwiGLU bisimulation verification.

pub mod bisimulation;
pub mod input_specs;

pub use bisimulation::verify_wire_cartridge_bisimulation;
pub use input_specs::{
    BackgroundActivationsSpec, CompilerWeightGeometrySpec, DeltaNetLayerWeightSpec,
    DownstreamMlpAntiEchoSpec, PipeSlotObligationSpec, QkLayerVectorsSpec,
    TensorPipeCaptureResponse, TokenReadoutContrastSpec, DEFAULT_QWEN35_08B_BASE_INTERMEDIATE_SIZE,
    DEFAULT_RESIDUAL_RADIUS_SCALE,
};

use crate::driver::LiveSqliteDailyDriver;
use anyhow::{Context, Result};
use portia_compiler::wire::WireCartridgeManifestJson;
use portia_core::pocket_dsl::{
    canonical_pocket_regime_dsl, GuardedModeRule, GuardedRuleStateBinding, PocketObsState,
    RegimeExpr,
};
use portia_core::{Mode, WireBankKey};
use portia_sql::SqlColumnRef;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCartridgeInputSpec {
    #[serde(default = "default_capture_status")]
    pub status: String,
    pub activations: BTreeMap<String, Vec<Vec<f64>>>,
    #[serde(default)]
    pub residuals: Option<BTreeMap<String, Vec<Vec<f64>>>>,
    #[serde(default)]
    pub row_states: Option<Vec<Option<PocketObsState>>>,
    #[serde(default)]
    pub obligations: Option<Vec<PipeSlotObligationSpec>>,
    #[serde(default)]
    pub mode_target_vectors: Option<BTreeMap<String, Vec<f64>>>,
    #[serde(default)]
    pub payload_target_vectors: Option<BTreeMap<String, Vec<f64>>>,
    /// Production unified captures require verified natural-token obligations for these banks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_payload_modes: Vec<Mode>,
    /// Per-row dynamic mount provenance for oracle captures; validates all obligation indices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capture_row_banks: Vec<WireBankKey>,
    #[serde(default)]
    pub model_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_revision: Option<String>,
    #[serde(default)]
    pub base_intermediate_size: Option<usize>,
    #[serde(default)]
    pub widened_slots: Option<usize>,
    #[serde(default)]
    pub compiler_weight_geometry: Option<CompilerWeightGeometrySpec>,
    #[serde(default)]
    pub background_activations: Option<BackgroundActivationsSpec>,
    #[serde(default)]
    pub steward_columns: Vec<String>,
    #[serde(default)]
    pub lock_columns: Vec<String>,
    #[serde(default)]
    pub schema_disclosed: bool,
    #[serde(default)]
    pub allow_synthetic_fixture_targets: bool,
}

fn default_capture_status() -> String {
    "activations".to_string()
}

/// Explicit governance column bindings (`--steward-col` / `--lock-col` / `--schema-disclosed` or
/// `CompileCartridgeInputSpec` fields) replacing English-word substring guessing and preventing
/// `--db` table inspection from implicitly pruning `Mode::Probe`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaGovernanceColumns {
    pub steward_columns: Vec<String>,
    pub lock_columns: Vec<String>,
    #[serde(default)]
    pub schema_disclosed: bool,
}

impl SchemaGovernanceColumns {
    #[must_use]
    pub fn new(steward_columns: Vec<String>, lock_columns: Vec<String>) -> Self {
        Self {
            steward_columns,
            lock_columns,
            schema_disclosed: false,
        }
    }

    #[must_use]
    pub const fn with_schema_disclosed(mut self, schema_disclosed: bool) -> Self {
        self.schema_disclosed = schema_disclosed;
        self
    }

    /// Match an explicit column binding (`"col_name"`, `"\"a.b\""`, `"table_name.col_name"`, or
    /// `"\"my.tbl\".\"a.b\""`, case-insensitive) against `(table_name, col_name)` via
    /// [`SqlColumnRef::parse_sql_binding`] with zero English-word guessing.
    #[must_use]
    pub fn matches_binding(binding: &str, table_name: &str, col_name: &str) -> bool {
        SqlColumnRef::parse_sql_binding(binding)
            .is_some_and(|col_ref| col_ref.matches_table_and_column(Some(table_name), col_name))
    }

    #[must_use]
    pub fn is_steward_column(&self, table_name: &str, col_name: &str) -> bool {
        self.steward_columns
            .iter()
            .any(|b| Self::matches_binding(b, table_name, col_name))
    }

    #[must_use]
    pub fn is_lock_column(&self, table_name: &str, col_name: &str) -> bool {
        self.lock_columns
            .iter()
            .any(|b| Self::matches_binding(b, table_name, col_name))
    }
}

/// Detailed `bf16` SwiGLU forward-pass bisimulation receipt over a compiled [`WireCartridgeManifestJson`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CartridgeBisimulationSummary {
    pub bisimulation_verified: bool,
    pub min_positive_gate_pre_silu: f64,
    pub max_negative_gate_pre_silu: f64,
    pub min_positive_chordal_cosine: f64,
    pub max_negative_swiglu_leak_l2: f64,
    pub decoded_mode_matches: usize,
    pub residual_source: String,
}

/// Provenance & schema-specialization receipt produced by `portia-cli compile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompileProvenanceReceipt {
    pub regime_source: String,
    pub capture_source: String,
    pub obligation_source: String,
    pub hidden_dim: usize,
    pub matched_steward_columns: Vec<String>,
    pub matched_lock_columns: Vec<String>,
    pub read_only: bool,
    pub active_modes: Vec<Mode>,
    pub guarded_rules: Vec<GuardedModeRule>,
    pub rule_state_bindings: Vec<GuardedRuleStateBinding>,
    pub bisimulation_verified: bool,
    pub min_positive_gate_pre_silu: f64,
    pub max_negative_gate_pre_silu: f64,
    pub min_positive_chordal_cosine: f64,
    pub max_negative_swiglu_leak_l2: f64,
    pub decoded_mode_matches: usize,
    pub residual_source: String,
}

/// Verify pure-Rust `bf16` SwiGLU forward-pass bisimulation on a compiled [`WireCartridgeManifestJson`]
/// against the captured activation batch `capture` and obligation specs `obligations`.
pub fn verify_compiled_cartridge_bisimulation(
    cartridge: &WireCartridgeManifestJson,
    capture: &TensorPipeCaptureResponse,
    obligations: &[PipeSlotObligationSpec],
    _base_intermediate_size: usize,
) -> Result<CartridgeBisimulationSummary> {
    let receipt = verify_wire_cartridge_bisimulation(cartridge, capture, obligations)?;
    Ok(CartridgeBisimulationSummary {
        bisimulation_verified: receipt.bisimulation_holds,
        min_positive_gate_pre_silu: receipt.min_positive_gate_pre_silu,
        max_negative_gate_pre_silu: receipt.max_negative_gate_pre_silu,
        min_positive_chordal_cosine: receipt.min_positive_chordal_cosine,
        max_negative_swiglu_leak_l2: receipt.max_negative_swiglu_leak_l2,
        decoded_mode_matches: receipt.decoded_mode_matches,
        residual_source: receipt.residual_source,
    })
}

/// Intermediate summary of Futamura-1 regime specialization over a target database schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecializedRegimeSummary {
    pub specialized_regime: RegimeExpr,
    pub active_modes: Vec<Mode>,
    pub guarded_rules: Vec<GuardedModeRule>,
    pub matched_steward_columns: Vec<String>,
    pub matched_lock_columns: Vec<String>,
    pub regime_source: String,
}

/// Inspect schema invariants (`has_steward_col`, `has_lock_col`) from an optional `.sqlite`
/// database file and specialize [`canonical_pocket_regime_dsl`].
pub fn specialize_regime_for_compile_target(
    db: Option<&Path>,
    read_only: bool,
) -> Result<Vec<Mode>> {
    let summary = specialize_regime_and_columns_for_compile_target(db, None, read_only)?;
    Ok(summary.active_modes)
}

pub fn specialize_regime_and_columns_for_compile_target(
    db: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
) -> Result<SpecializedRegimeSummary> {
    specialize_regime_and_columns_with_governance(
        db,
        regime_json,
        read_only,
        &SchemaGovernanceColumns::default(),
    )
}

pub fn specialize_regime_and_columns_with_governance(
    db: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
    governance: &SchemaGovernanceColumns,
) -> Result<SpecializedRegimeSummary> {
    let (base_regime, regime_source) = if let Some(reg_path) = regime_json {
        let raw = std::fs::read_to_string(reg_path)
            .with_context(|| format!("failed to read --regime-json {reg_path:?}"))?;
        let parsed: RegimeExpr = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse RegimeExpr from {reg_path:?}"))?;
        (parsed, "custom_regime_json".to_string())
    } else {
        (
            canonical_pocket_regime_dsl(),
            "canonical_pocket_regime_dsl".to_string(),
        )
    };

    let mut matched_steward_columns = Vec::new();
    let mut matched_lock_columns = Vec::new();
    let is_schema_disclosed = if let Some(db_path) = db {
        let driver = LiveSqliteDailyDriver::open_read_only(db_path)?;
        let tables = driver.inspect_schema_metadata()?;
        for t in &tables {
            for c in &t.columns {
                let qualified = format!("{}.{}", t.table_name, c.name);
                if governance.is_steward_column(&t.table_name, &c.name) {
                    matched_steward_columns.push(qualified.clone());
                }
                if governance.is_lock_column(&t.table_name, &c.name) {
                    matched_lock_columns.push(qualified);
                }
            }
        }
        governance.schema_disclosed && !tables.is_empty()
    } else {
        governance.schema_disclosed
    };

    let has_steward_col = if db.is_some() && !governance.steward_columns.is_empty() {
        !matched_steward_columns.is_empty()
    } else {
        true
    };
    let has_lock_col = if db.is_some() && !governance.lock_columns.is_empty() {
        !matched_lock_columns.is_empty()
    } else {
        true
    };

    let specialized = base_regime.specialize_for_schema_invariants(
        read_only,
        has_steward_col,
        has_lock_col,
        is_schema_disclosed,
    );
    Ok(SpecializedRegimeSummary {
        active_modes: specialized.active_modes(),
        guarded_rules: specialized.extract_guarded_rules(),
        specialized_regime: specialized,
        matched_steward_columns,
        matched_lock_columns,
        regime_source,
    })
}

/// Compile a [`WireCartridgeManifestJson`] in pure Rust (`portia-compiler` `faer` + Clarabel SOCP).
pub fn compile_cartridge_command(
    input_json: &Path,
    out_json: &Path,
) -> Result<WireCartridgeManifestJson> {
    let (cartridge, _provenance) =
        compile_cartridge_for_target(input_json, None, None, false, false, out_json)?;
    Ok(cartridge)
}

/// Unified `portia-cli compile` entry point returning both the verified
/// [`WireCartridgeManifestJson`] and [`CompileProvenanceReceipt`].
pub fn compile_cartridge_for_target(
    input_json: &Path,
    db: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
    allow_synthetic_fixture_targets: bool,
    out_json: &Path,
) -> Result<(WireCartridgeManifestJson, CompileProvenanceReceipt)> {
    compile_cartridge_for_target_with_governance(
        input_json,
        db,
        regime_json,
        read_only,
        &SchemaGovernanceColumns::default(),
        allow_synthetic_fixture_targets,
        out_json,
    )
}

pub fn compile_cartridge_for_target_with_governance(
    input_json: &Path,
    db: Option<&Path>,
    regime_json: Option<&Path>,
    read_only: bool,
    governance: &SchemaGovernanceColumns,
    allow_synthetic_fixture_targets: bool,
    out_json: &Path,
) -> Result<(WireCartridgeManifestJson, CompileProvenanceReceipt)> {
    crate::compile_v2::compile_cartridge_v2_with_governance(
        input_json,
        db,
        regime_json,
        read_only,
        governance,
        allow_synthetic_fixture_targets,
        out_json,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_v2::derive_bindings::derive_obligations_from_bindings;
    use portia_core::pocket_dsl::{
        candidate_count_eq, canonical_compile_fixture_row_states, emit_mode,
    };

    #[test]
    fn external_capture_rejects_missing_row_states_and_uncovered_custom_regimes() {
        let tmp = tempfile::tempdir().unwrap();
        let in_path = tmp.path().join("external_no_states.json");
        let in_states_path = tmp.path().join("external_with_states.json");
        let out_path = tmp.path().join("out.json");
        let reg_path = tmp.path().join("uncovered_regime.json");

        let no_states_json = serde_json::json!({
            "status": "ok",
            "activations": {
                "22": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            }
        });
        std::fs::write(&in_path, serde_json::to_string(&no_states_json).unwrap()).unwrap();
        let err_g1 = compile_cartridge_for_target(&in_path, None, None, false, false, &out_path)
            .expect_err("external capture without obligations or row_states must fail");
        assert!(err_g1
            .to_string()
            .contains("omits both `obligations` and `row_states`"));

        let states = canonical_compile_fixture_row_states();
        let rows: Vec<Vec<f64>> = (0..states.len())
            .map(|i| vec![1.0, (i as f64) * 0.1, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0])
            .collect();
        let with_states_json = serde_json::json!({
            "status": "activations",
            "activations": { "22": rows },
            "row_states": states,
        });
        std::fs::write(
            &in_states_path,
            serde_json::to_string(&with_states_json).unwrap(),
        )
        .unwrap();
        let err_geom =
            compile_cartridge_for_target(&in_states_path, None, None, false, false, &out_path)
                .expect_err("production compile without model geometry must fail");
        assert!(err_geom.to_string().contains(
            "Production compile requires `compiler_weight_geometry` or explicit `mode_target_vectors`"
        ));

        let uncovered_regime = candidate_count_eq(999) >> emit_mode(Mode::Read);
        std::fs::write(&reg_path, serde_json::to_string(&uncovered_regime).unwrap()).unwrap();
        let err_g2 = compile_cartridge_for_target(
            &in_states_path,
            None,
            Some(&reg_path),
            false,
            true,
            &out_path,
        )
        .expect_err("uncovered custom regime must fail");
        assert!(err_g2
            .to_string()
            .contains("No guarded regime rule had non-empty positive"));
    }

    #[test]
    fn explicit_schema_governance_bindings_preserve_rules_without_english_word_guessing() {
        let false_pos_db = tempfile::NamedTempFile::new().unwrap();
        {
            let conn = rusqlite::Connection::open(false_pos_db.path()).unwrap();
            conn.execute_batch(
                "CREATE TABLE accounts (
                    id INTEGER PRIMARY KEY,
                    cardholder_name TEXT NOT NULL,
                    approval_notes TEXT NOT NULL,
                    permission_label TEXT NOT NULL,
                    lock_comment TEXT NOT NULL,
                    freeze_tag TEXT NOT NULL
                );",
            )
            .unwrap();
        }
        let summary_unconfigured = specialize_regime_and_columns_for_compile_target(
            Some(false_pos_db.path()),
            None,
            false,
        )
        .unwrap();
        assert!(summary_unconfigured.matched_steward_columns.is_empty());
        assert!(summary_unconfigured.matched_lock_columns.is_empty());
        assert!(summary_unconfigured.active_modes.contains(&Mode::Ask));
        assert!(summary_unconfigured.active_modes.contains(&Mode::Abstain));

        let absent_gov = SchemaGovernanceColumns::new(
            vec!["steward_verdict".to_string()],
            vec!["interlock_flag".to_string()],
        );
        let summary_absent = specialize_regime_and_columns_with_governance(
            Some(false_pos_db.path()),
            None,
            false,
            &absent_gov,
        )
        .unwrap();
        assert!(summary_absent.guarded_rules.len() < summary_unconfigured.guarded_rules.len());
        let single_ask_obs = PocketObsState {
            world_id: "gov-check".to_string(),
            turn_index: 1,
            table_name: "accounts".to_string(),
            entity_name: "acct-1".to_string(),
            operation: "UPDATE".to_string(),
            has_schema_probe: true,
            candidate_count: Some(1),
            policy_requires_ask: true,
            is_locked_or_denied: false,
            steward_grant_approved: None,
            applied_write_count: 0,
            has_post_write_verification: false,
            zero_row_toctou_race: false,
        };
        assert_eq!(
            summary_unconfigured
                .specialized_regime
                .evaluate(&single_ask_obs)
                .selected_mode,
            Some(Mode::Ask)
        );
        assert_eq!(
            summary_absent
                .specialized_regime
                .evaluate(&single_ask_obs)
                .selected_mode,
            Some(Mode::Write)
        );

        let true_pos_db = tempfile::NamedTempFile::new().unwrap();
        {
            let conn = rusqlite::Connection::open(true_pos_db.path()).unwrap();
            conn.execute_batch(
                "CREATE TABLE incidents (
                    id INTEGER PRIMARY KEY,
                    policy_decision TEXT NOT NULL,
                    change_freeze TEXT NOT NULL
                );",
            )
            .unwrap();
        }
        let explicit_gov = SchemaGovernanceColumns::new(
            vec!["policy_decision".to_string()],
            vec!["incidents.change_freeze".to_string()],
        );
        let summary_tp = specialize_regime_and_columns_with_governance(
            Some(true_pos_db.path()),
            None,
            false,
            &explicit_gov,
        )
        .unwrap();
        assert_eq!(
            summary_tp.matched_steward_columns,
            vec!["incidents.policy_decision".to_string()]
        );
        assert_eq!(
            summary_tp.matched_lock_columns,
            vec!["incidents.change_freeze".to_string()]
        );
        assert!(SchemaGovernanceColumns::matches_binding(
            "\"a.b\"", "accounts", "a.b"
        ));
        assert!(!SchemaGovernanceColumns::matches_binding(
            "\"a.b\"", "a", "b"
        ));

        let states = canonical_compile_fixture_row_states();
        let bindings = summary_tp
            .specialized_regime
            .bind_capture_row_states(&states);
        let mut payload_map = BTreeMap::new();
        payload_map.insert("read".to_string(), vec![0.5; 16]);
        let obs_with_payload = derive_obligations_from_bindings(
            16,
            22,
            &summary_tp.active_modes,
            &bindings,
            None,
            Some(&payload_map),
            true,
        )
        .unwrap();
        assert!(obs_with_payload
            .iter()
            .any(|ob| ob.bank_name == "ModePayload:Read"));
    }

    #[test]
    fn qk_resolve_directions_applies_qwen35_weight_offset_and_geometry_orthogonality() {
        let d = 16;
        let spec = QkLayerVectorsSpec {
            w_q: Some(vec![unit_vec(d, 0), unit_vec(d, 1)]),
            w_k: Some(vec![vec![0.0; d], vec![0.0; d]]),
            ln_gamma: Some({
                let mut g = vec![0.6; d];
                g[0] = 2.0;
                g[1] = 0.0;
                g
            }),
            k_set: Some(vec![1.0, 1.0]),
            k_where: Some(vec![1.0, -1.0]),
            ..QkLayerVectorsSpec::default()
        };
        let (v_set, _) = spec.resolve_directions(d).unwrap();
        assert!((v_set[0] - 3.0 / 10.0_f64.sqrt()).abs() < 1e-10);
        assert!((v_set[1] - 1.0 / 10.0_f64.sqrt()).abs() < 1e-10);

        let mut named = BTreeMap::new();
        for (idx, m) in Mode::ALL.into_iter().enumerate() {
            named.insert(m.qwen35_first_token_id().to_string(), unit_vec(d, 4 + idx));
        }
        let sql_emb = vec![unit_vec(d, 2), unit_vec(d, 3)];
        let geom = CompilerWeightGeometrySpec {
            vocab_top_k_eigvecs: Vec::new(),
            protected_sql_embeddings: sql_emb.clone(),
            norm_weight: vec![0.0; d],
            mean_natural_token_norm: 1.0,
            qk_layer_vectors: BTreeMap::new(),
            deltanet_layer_weights: BTreeMap::new(),
            named_token_unembeddings: named,
            downstream_mlp_anti_echo: None,
        };
        let targets = geom.derive_mode_readout_targets().unwrap();
        assert_eq!(targets.len(), 6);
        for v in targets.values() {
            for s in &sql_emb {
                let dot: f64 = v.iter().zip(s).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-10);
            }
        }
    }

    fn unit_vec(d: usize, idx: usize) -> Vec<f64> {
        let mut v = vec![0.0; d];
        v[idx] = 1.0;
        v
    }
}
