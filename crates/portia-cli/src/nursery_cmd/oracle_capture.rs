//! Native Rust On-Policy Oracle Rollout Harvester (`portia-cli nursery --oracle-capture-out`).
//!
//! Drives real in-memory `LiveSqliteDailyDriver` calibration databases against an un-augmented
//! `Base` worker (`StdioQwenWorkerClient`) to capture `(pre_mlp, residual)` states at
//! `<|im_start|>assistant\n<think>\n[MODE:` across the unimodal SQL Steward decision rules
//! while letting the base trunk generate genuine on-policy `Plan:` lines into `past_key_values`.
//! Captures structural payload boundaries (`CaptureBoundary::CloseTag { subtoken_index: 0 }` across
//! all modes plus `CaptureBoundary::Field` on `Mode::Ask`) alongside within-span negatives for
//! explicit natural-token obligations in all six dynamically mounted payload banks.

use super::{build_nursery_sweep_coordinates, NurseryAppFilter};
use crate::compile::{CompileCartridgeInputSpec, PipeSlotObligationSpec, TokenReadoutContrastSpec};
use crate::driver::{
    DriverObservation, EpistemicDomainEnv, LiveSqliteDailyDriver, ObservationOperation, ReadPurpose,
};
use crate::synth_user::{
    deterministic_sql_requested_balance, deterministic_sql_target_slug,
    DeterministicStewardUserEnv, SynthUserEnv,
};
use crate::worker::{
    format_lotl_user_prompt_envelope, format_observation_wire_chunk, CaptureBoundary, SessionEvent,
    StdioQwenWorkerClient,
};
use anyhow::{bail, Context, Result};
use portia_core::bpe::canonical_mode_header_tokens_under_orient;
use portia_core::relay::TypedTurnAction;
use portia_core::{CartridgeBank, Mode, RouteTag, StewardAuthState, WireBankKey};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Maximum `ScenarioCoordinate` index allowed during oracle calibration so held-out nursery
/// episodes `[12..24]` remain strictly un-observed during compilation.
pub const MAX_ORACLE_TRAIN_EPISODES: usize = 12;

/// Canonical unimodal `OrientTrunk` decision rules paired with their expected [`Mode`].
pub const UNIMODAL_ORACLE_RULES: [(&str, Mode); 16] = [
    ("R0_chitchat_ask", Mode::Ask),
    ("R1_probe_catalog_initial", Mode::Probe),
    ("R2a_probe_after_catalog", Mode::Probe),
    ("R2b_probe_on_join_followup", Mode::Probe),
    ("R3a_read_after_pragma", Mode::Read),
    ("R3b_read_on_followup_prompt", Mode::Read),
    ("R4_result_readonly_answer", Mode::Result),
    ("R5_abstain_zero_rows", Mode::Abstain),
    ("R6_ask_steward_gate", Mode::Ask),
    ("R6b_ask_after_steward_rollback", Mode::Ask),
    ("R7a_abstain_locked_row", Mode::Abstain),
    ("R7b_abstain_steward_denied", Mode::Abstain),
    ("R8a_write_allow_row", Mode::Write),
    ("R8b_write_steward_granted", Mode::Write),
    ("R9_read_verify_after_write", Mode::Read),
    ("R10_result_after_verify", Mode::Result),
];

/// Return `true` iff `(mode, boundary)` is a genuine structural payload boundary eligible for a
/// compiled `ModePayload(mode)` diode:
/// - `CaptureBoundary::CloseTag { subtoken_index: 0 }` (end-of-span termination diode across modes),
/// - `CaptureBoundary::Field` (`"select"` structured-GUI field boundary) on `Mode::Ask`.
///
/// Never compiles English-word/table-name `Opener` puppeteering, `Value`, `Background`, or
/// trailing `close_1..close_N` subtokens already forced deterministically by `llguidance`.
#[must_use]
pub const fn is_compiled_structural_payload_boundary(
    mode: Mode,
    boundary: &CaptureBoundary,
) -> bool {
    matches!(boundary, CaptureBoundary::CloseTag { subtoken_index: 0 })
        || (matches!(mode, Mode::Ask) && matches!(boundary, CaptureBoundary::Field))
}

/// Configuration for [`run_oracle_capture_harvest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleCaptureHarvestConfig {
    pub model_family: String,
    pub episodes: usize,
    pub target_layer: Option<usize>,
    pub max_plan_tokens: usize,
    pub widened_slots: usize,
    pub external_domain_limit: Option<usize>,
}

impl Default for OracleCaptureHarvestConfig {
    fn default() -> Self {
        Self {
            model_family: "Qwen/Qwen3.5-0.8B-Base".to_string(),
            episodes: MAX_ORACLE_TRAIN_EPISODES,
            target_layer: None,
            max_plan_tokens: 512,
            widened_slots: 128,
            external_domain_limit: None,
        }
    }
}

/// Summary receipt emitted after writing a [`CompileCartridgeInputSpec`] capture bundle to disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleCaptureHarvestReceipt {
    pub status: String,
    pub model_family: String,
    pub target_layer: usize,
    pub base_intermediate_size: usize,
    pub widened_slots: usize,
    pub external_domains_harvested: usize,
    pub train_episodes_harvested: usize,
    pub total_captured_states: usize,
    pub rule_positive_counts: BTreeMap<String, usize>,
    pub payload_positive_counts: BTreeMap<String, usize>,
    pub output_path: PathBuf,
}

impl OracleCaptureHarvestReceipt {
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("failed to serialize OracleCaptureHarvestReceipt to JSON")
    }
}

fn validate_terminal_prose(payload: &str) -> Result<()> {
    let lines: Vec<&str> = payload.split('\n').collect();
    if lines.is_empty()
        || payload.trim_start().starts_with('{')
        || payload.contains(['<', '>', '\r'])
        || lines.iter().any(|line| line.trim().is_empty())
    {
        bail!("terminal calibration prose must have nonblank lines, no angle brackets, and no leading JSON object: {payload:?}");
    }
    Ok(())
}

struct ExternalCalibrationDomain {
    main_table: &'static str,
    join_table: &'static str,
    extra_tables: [&'static str; 8],
    pk_col: &'static str,
    slug_col: &'static str,
    status_col: &'static str,
    active_val: &'static str,
    val_col: &'static str,
    rule_col: &'static str,
    owner_col: &'static str,
    freeze_col: &'static str,
    fk_col: &'static str,
    join_col: &'static str,
    pk1: i64,
    pk2: i64,
    slug1: &'static str,
    slug2: &'static str,
    v1_init: i64,
    v1_new: i64,
    v2_init: i64,
    v2_new: i64,
    chitchat: &'static str,
    q_list: &'static str,
    q_join: &'static str,
    q_detail: &'static str,
    q_mut_allow: &'static str,
    q_mut_ask: &'static str,
}

const EXTERNAL_CALIBRATION_DOMAINS: [ExternalCalibrationDomain; 6] = [
    ExternalCalibrationDomain {
        main_table: "workspace_subscriptions",
        join_table: "billing_audit_events",
        extra_tables: [
            "tenant_orgs",
            "invoice_ledgers",
            "payment_webhooks",
            "seat_allocations",
            "tax_jurisdictions",
            "usage_meters",
            "rate_cards",
            "support_tiers",
        ],
        pk_col: "sub_id",
        slug_col: "workspace_slug",
        status_col: "plan_status",
        active_val: "ACTIVE",
        val_col: "monthly_quota",
        rule_col: "billing_rule",
        owner_col: "account_owner",
        freeze_col: "billing_freeze",
        fk_col: "sub_id",
        join_col: "event_actor",
        pk1: 201,
        pk2: 202,
        slug1: "ACME-CORE-01",
        slug2: "GLOBEX-EU-02",
        v1_init: 300,
        v1_new: 120,
        v2_init: 800,
        v2_new: 450,
        chitchat: "hi",
        q_list: "show me the active workspace subscriptions",
        q_join: "show me the active workspace subscriptions along with their billing audit history",
        q_detail: "show me the details for subscription 201",
        q_mut_allow: "set the monthly quota for subscription 201 to 120",
        q_mut_ask: "set the monthly quota for subscription 202 to 450",
    },
    ExternalCalibrationDomain {
        main_table: "alpine_levain_batches",
        join_table: "fermentation_temp_logs",
        extra_tables: [
            "mill_grain_lots",
            "oven_deck_schedules",
            "bakery_stewards",
            "hydration_recipes",
            "crumb_inspections",
            "starter_cultures",
            "cold_retard_racks",
            "delivery_routes",
        ],
        pk_col: "batch_id",
        slug_col: "starter_code",
        status_col: "ferment_stage",
        active_val: "PROOFING",
        val_col: "hydration_pct",
        rule_col: "fold_schedule",
        owner_col: "head_baker",
        freeze_col: "batch_lock",
        fk_col: "batch_id",
        join_col: "baker_note",
        pk1: 301,
        pk2: 302,
        slug1: "CHAMONIX-RYE-7",
        slug2: "ZERMATT-SPELT-3",
        v1_init: 72,
        v1_new: 78,
        v2_init: 68,
        v2_new: 82,
        chitchat: "hello!",
        q_list: "show me the active batches",
        q_join: "show me the active batches along with their fermentation audit history",
        q_detail: "show me the details for batch 301",
        q_mut_allow: "set the hydration_pct for batch 301 to 78",
        q_mut_ask: "set the hydration_pct for batch 302 to 82",
    },
    ExternalCalibrationDomain {
        main_table: "orbital_thruster_valves",
        join_table: "telemetry_valve_events",
        extra_tables: [
            "attitude_gyros",
            "propellant_tanks",
            "flight_directors",
            "burn_maneuvers",
            "docking_ports",
            "solar_gimbals",
            "thermal_louvers",
            "comms_transponders",
        ],
        pk_col: "valve_id",
        slug_col: "module_tag",
        status_col: "valve_state",
        active_val: "NOMINAL",
        val_col: "pressure_kpa",
        rule_col: "purge_protocol",
        owner_col: "flight_engineer",
        freeze_col: "safety_interlock",
        fk_col: "valve_id",
        join_col: "sensor_channel",
        pk1: 401,
        pk2: 402,
        slug1: "ZVEZDA-RCS-04",
        slug2: "KIBO-ATT-09",
        v1_init: 910,
        v1_new: 850,
        v2_init: 1120,
        v2_new: 980,
        chitchat: "hey",
        q_list: "show me the active thruster valves",
        q_join: "show me the active thruster valves along with their telemetry event history",
        q_detail: "show me the details for valve 401",
        q_mut_allow: "set the pressure_kpa for valve 401 to 850",
        q_mut_ask: "set the pressure_kpa for valve 402 to 980",
    },
    ExternalCalibrationDomain {
        main_table: "solaris_inverter_banks",
        join_table: "inverter_dispatch_audit",
        extra_tables: [
            "substation_feeders",
            "battery_strings",
            "grid_operators",
            "tariff_windows",
            "weather_arrays",
            "transformer_taps",
            "relay_breakers",
            "curtailment_orders",
        ],
        pk_col: "bank_id",
        slug_col: "array_slug",
        status_col: "grid_mode",
        active_val: "ONLINE",
        val_col: "curtailment_mw",
        rule_col: "dispatch_rule",
        owner_col: "shift_engineer",
        freeze_col: "grid_freeze",
        fk_col: "bank_id",
        join_col: "operator_id",
        pk1: 501,
        pk2: 502,
        slug1: "ATACAMA-WEST-11",
        slug2: "MOJAVE-SOUTH-06",
        v1_init: 45,
        v1_new: 20,
        v2_init: 130,
        v2_new: 60,
        chitchat: "good morning",
        q_list: "show me the online inverter banks",
        q_join: "show me the online inverter banks along with their dispatch audit history",
        q_detail: "show me the details for bank 501",
        q_mut_allow: "set the curtailment_mw for bank 501 to 20",
        q_mut_ask: "set the curtailment_mw for bank 502 to 60",
    },
    ExternalCalibrationDomain {
        main_table: "maritime_vessel_berths",
        join_table: "harbor_pilot_logs",
        extra_tables: [
            "crane_gantries",
            "customs_manifests",
            "tugboat_rosters",
            "tidal_windows",
            "bunker_tanks",
            "reefer_plugs",
            "channel_buoys",
            "berth_tariffs",
        ],
        pk_col: "berth_id",
        slug_col: "vessel_callsign",
        status_col: "dock_status",
        active_val: "BERTHED",
        val_col: "draft_decimeters",
        rule_col: "mooring_plan",
        owner_col: "harbor_master",
        freeze_col: "port_hold",
        fk_col: "berth_id",
        join_col: "pilot_name",
        pk1: 601,
        pk2: 602,
        slug1: "MAERSK-NORDIC-2",
        slug2: "HAPAG-RHINE-8",
        v1_init: 115,
        v1_new: 108,
        v2_init: 142,
        v2_new: 125,
        chitchat: "hello",
        q_list: "show me the berthed vessels",
        q_join: "show me the berthed vessels along with their harbor pilot logs",
        q_detail: "show me the details for berth 601",
        q_mut_allow: "set the draft for berth 601 to 108",
        q_mut_ask: "set the draft for berth 602 to 125",
    },
    ExternalCalibrationDomain {
        main_table: "formulary_vial_lots",
        join_table: "coldchain_inspection_log",
        extra_tables: [
            "cryo_freezers",
            "clinic_allocations",
            "pharmacist_roster",
            "stability_assays",
            "shipping_containers",
            "excipient_batches",
            "quarantine_holds",
            "recall_notices",
        ],
        pk_col: "lot_id",
        slug_col: "lot_code",
        status_col: "release_state",
        active_val: "RELEASED",
        val_col: "vial_count",
        rule_col: "storage_protocol",
        owner_col: "qa_lead",
        freeze_col: "qa_hold",
        fk_col: "lot_id",
        join_col: "inspector_handle",
        pk1: 701,
        pk2: 702,
        slug1: "LOT-MRNA-44A",
        slug2: "LOT-PEPT-19B",
        v1_init: 640,
        v1_new: 590,
        v2_init: 320,
        v2_new: 280,
        chitchat: "hi there",
        q_list: "show me the active vial lots",
        q_join: "show me the active vial lots along with their inspection history",
        q_detail: "show me the details for lot 701",
        q_mut_allow: "set the vial count for lot 701 to 590",
        q_mut_ask: "set the vial count for lot 702 to 280",
    },
];

fn open_external_domain_driver(d: &ExternalCalibrationDomain) -> Result<LiveSqliteDailyDriver> {
    let conn = Connection::open_in_memory()
        .context("failed to open in-memory SQLite database for external calibration domain")?;
    let create_main = format!(
        "CREATE TABLE \"{}\" (\
            \"{}\" INTEGER PRIMARY KEY, \
            \"{}\" TEXT NOT NULL, \
            \"{}\" TEXT NOT NULL, \
            \"{}\" INTEGER NOT NULL, \
            \"{}\" TEXT NOT NULL, \
            \"policy_decision\" TEXT NOT NULL, \
            \"{}\" TEXT NOT NULL, \
            \"{}\" TEXT NOT NULL\
        );",
        d.main_table,
        d.pk_col,
        d.slug_col,
        d.status_col,
        d.val_col,
        d.rule_col,
        d.owner_col,
        d.freeze_col
    );
    conn.execute_batch(&create_main)?;

    for extra_tbl in d.extra_tables {
        let create_extra = format!(
            "CREATE TABLE \"{extra_tbl}\" (record_id INTEGER PRIMARY KEY, label TEXT NOT NULL);"
        );
        conn.execute_batch(&create_extra)?;
    }

    let create_join = format!(
        "CREATE TABLE \"{}\" (\
            audit_id INTEGER PRIMARY KEY, \
            \"{}\" INTEGER NOT NULL, \
            \"{}\" TEXT NOT NULL, \
            prev_val INTEGER NOT NULL, \
            new_val INTEGER NOT NULL\
        );",
        d.join_table, d.fk_col, d.join_col
    );
    conn.execute_batch(&create_join)?;

    let insert_main = format!(
        "INSERT INTO \"{}\" (\"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"policy_decision\", \"{}\", \"{}\") \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
        d.main_table,
        d.pk_col,
        d.slug_col,
        d.status_col,
        d.val_col,
        d.rule_col,
        d.owner_col,
        d.freeze_col
    );
    conn.execute(
        &insert_main,
        params![
            d.pk1,
            d.slug1,
            d.active_val,
            d.v1_init,
            "AUTO_PERMIT",
            "ALLOW",
            "ops_primary",
            "CLEAR"
        ],
    )?;
    conn.execute(
        &insert_main,
        params![
            d.pk2,
            d.slug2,
            d.active_val,
            d.v2_init,
            "STEWARD_GATE",
            "ASK",
            "ops_eu",
            "CLEAR"
        ],
    )?;

    let insert_join = format!(
        "INSERT INTO \"{}\" (audit_id, \"{}\", \"{}\", prev_val, new_val) VALUES (?1, ?2, ?3, ?4, ?5);",
        d.join_table, d.fk_col, d.join_col
    );
    conn.execute(
        &insert_join,
        params![1_i64, d.pk1, "op_alpha", 400_i64, d.v1_init],
    )?;
    conn.execute(
        &insert_join,
        params![2_i64, d.pk2, "op_beta", 950_i64, d.v2_init],
    )?;

    Ok(LiveSqliteDailyDriver::from_connection(conn, 10)?
        .with_steward_policy_columns(["policy_decision"]))
}

fn open_train_episode_driver(
    table_name: &str,
    seeded_row: Option<(&str, i64, &str, &str)>,
) -> Result<LiveSqliteDailyDriver> {
    let conn = Connection::open_in_memory()
        .context("failed to open in-memory SQLite database for train episode calibration")?;
    let create_sql = format!(
        "CREATE TABLE \"{table_name}\" (\
            record_id INTEGER PRIMARY KEY, \
            entity_slug TEXT NOT NULL, \
            balance_credits INTEGER NOT NULL, \
            policy_decision TEXT NOT NULL, \
            lock_state TEXT NOT NULL, \
            steward_handle TEXT NOT NULL, \
            policy_guard_c0 INTEGER NOT NULL, \
            policy_guard_c1 INTEGER NOT NULL\
        );\
        CREATE TABLE distractor_mirror_01_archive (\
            record_id INTEGER PRIMARY KEY, \
            entity_slug TEXT NOT NULL, \
            balance_credits INTEGER NOT NULL\
        );"
    );
    conn.execute_batch(&create_sql)?;

    if let Some((slug, init_bal, policy_decision, lock_state)) = seeded_row {
        let insert_sql = format!(
            "INSERT INTO \"{table_name}\" (\
                record_id, entity_slug, balance_credits, policy_decision, lock_state, \
                steward_handle, policy_guard_c0, policy_guard_c1\
            ) VALUES (1000, ?1, ?2, ?3, ?4, 'steward_ops', 0, 165);"
        );
        conn.execute(
            &insert_sql,
            params![slug, init_bal, policy_decision, lock_state],
        )?;
    }

    Ok(LiveSqliteDailyDriver::from_connection(conn, 10)?
        .with_steward_policy_columns(["policy_decision"]))
}

fn execute_verified_driver_step(
    driver: &mut LiveSqliteDailyDriver,
    mode: Mode,
    payload: &str,
    ask_reply: Option<&str>,
) -> Result<(DriverObservation, String)> {
    let obs = driver.execute_turn(mode, payload, |_q| {
        Ok(ask_reply.unwrap_or("ALLOW").to_string())
    });
    if !obs.ok {
        bail!(
            "oracle calibration SQLite step failed in mode {:?} on payload {:?}: {:?}",
            mode,
            payload,
            obs.error
        );
    }
    let wire_chunk = format_observation_wire_chunk(&obs)?;
    Ok((obs, wire_chunk))
}

fn execute_expected_steward_rollback(
    driver: &mut LiveSqliteDailyDriver,
    update: &str,
) -> Result<(DriverObservation, String)> {
    if driver.steward_auth() != StewardAuthState::NotRequired
        || !driver.legal_modes()?.contains(&Mode::Write)
    {
        bail!("rollback calibration must begin before hidden row policy is observed");
    }
    let snapshot = driver.database_snapshot()?;
    let version = driver.db_version();
    let writes = driver.task_writes_committed();
    let observation = driver.execute_turn(Mode::Write, update, |_| {
        bail!("a rejected WRITE must not invoke the Steward callback")
    });
    let error = observation.error.as_deref().unwrap_or_default();
    if observation.ok
        || observation.rows_affected != 0
        || !error.starts_with("StewardAuthorizationViolation:")
        || !error.contains("rolled back SAVEPOINT")
        || driver.database_snapshot()? != snapshot
        || driver.db_version() != version
        || observation.db_version != version
        || driver.task_writes_committed() != writes
        || driver.steward_auth() != StewardAuthState::PendingAsk
        || observation.steward_auth != StewardAuthState::PendingAsk
        || driver.legal_modes()?.contains(&Mode::Write)
        || !driver.legal_modes()?.contains(&Mode::Ask)
    {
        bail!(
            "expected an intact, uncommitted Steward rollback with ASK available: {observation:?}"
        );
    }
    let wire = format_observation_wire_chunk(&observation)?;
    Ok((observation, wire))
}

fn make_session_event(
    mode: Mode,
    plan: String,
    payload: String,
    observation: DriverObservation,
) -> SessionEvent {
    let route = mode.route_tag();
    let action = match mode {
        Mode::Probe => TypedTurnAction::ProbeSchema {
            sql: payload.clone(),
        },
        Mode::Read => TypedTurnAction::ReadQuery {
            sql: payload.clone(),
        },
        Mode::Write => TypedTurnAction::WriteMutation {
            sql: payload.clone(),
        },
        Mode::Ask => TypedTurnAction::AskSteward {
            question: payload.clone(),
        },
        Mode::Result => TypedTurnAction::SubmitResult {
            report_json: serde_json::json!({ "summary": payload }),
        },
        Mode::Abstain => TypedTurnAction::AbstainTask {
            abstain_json: serde_json::json!({ "reason": payload }),
        },
    };
    SessionEvent {
        mode,
        route,
        plan,
        payload,
        action,
        observation,
        ask_interaction_mode: None,
    }
}

struct OnPolicyOracleHarvester<'a> {
    worker: &'a mut StdioQwenWorkerClient,
    target_layer: usize,
    expected_hidden_dim: usize,
    max_plan_tokens: usize,
    pre_rows: Vec<Vec<f64>>,
    res_rows: Vec<Vec<f64>>,
    groups: BTreeMap<String, Vec<usize>>,
    payload_groups: BTreeMap<(Mode, CaptureBoundary, u32), Vec<(usize, u32)>>,
    payload_scope_rows: BTreeMap<Mode, BTreeSet<usize>>,
    row_banks: Vec<WireBankKey>,
    router_background_rows: Vec<usize>,
}

impl<'a> OnPolicyOracleHarvester<'a> {
    fn new(
        worker: &'a mut StdioQwenWorkerClient,
        target_layer: usize,
        expected_hidden_dim: usize,
        max_plan_tokens: usize,
    ) -> Self {
        let mut groups = BTreeMap::new();
        for (rule_id, _) in UNIMODAL_ORACLE_RULES {
            groups.insert(rule_id.to_string(), Vec::new());
        }
        Self {
            worker,
            target_layer,
            expected_hidden_dim,
            max_plan_tokens: max_plan_tokens.max(1),
            pre_rows: Vec::new(),
            res_rows: Vec::new(),
            groups,
            payload_groups: BTreeMap::new(),
            payload_scope_rows: BTreeMap::new(),
            row_banks: Vec::new(),
            router_background_rows: Vec::new(),
        }
    }

    fn capture_at_mode_and_generate_turn(
        &mut self,
        prefix_wire: &str,
        rule_id: &str,
        oracle_mode: Mode,
        payload_body: &str,
    ) -> Result<(String, String)> {
        if matches!(oracle_mode, Mode::Result | Mode::Abstain) {
            validate_terminal_prose(payload_body)?;
        }
        let wire_with_cursor = format!("{prefix_wire}<|im_start|>assistant\n<think>\n[MODE:");
        let (pre_mlp, residual, generated_plan_line, router_background) = self
            .worker
            .capture_and_generate_on_policy_plan_with_background(
                &wire_with_cursor,
                oracle_mode,
                self.target_layer,
                self.max_plan_tokens,
            )
            .with_context(|| {
                format!("failed capturing on-policy plan for rule {rule_id} ({oracle_mode:?})")
            })?;

        if pre_mlp.len() != self.expected_hidden_dim || residual.len() != self.expected_hidden_dim {
            bail!(
                "worker returned mismatched activation dimensions for rule {}: pre_mlp={}, residual={}, expected_hidden_dim={}",
                rule_id,
                pre_mlp.len(),
                residual.len(),
                self.expected_hidden_dim
            );
        }

        let row_idx = self.pre_rows.len();
        self.pre_rows.push(pre_mlp);
        self.res_rows.push(residual);
        self.row_banks.push(WireBankKey(CartridgeBank::OrientTrunk));
        self.groups
            .get_mut(rule_id)
            .with_context(|| format!("unknown unimodal oracle rule id {rule_id:?}"))?
            .push(row_idx);

        // `OrientTrunk` is only mounted on `ModeLabel`: never push unmounted `TurnOpenPrefill`
        // (`<|im_start|>assistant\n<think>\n[MODE:`) prefix states into `router_background_rows`.
        let mode_label_tokens = canonical_mode_header_tokens_under_orient(oracle_mode).len();
        let mode_label_bg_start = router_background.len().saturating_sub(mode_label_tokens);
        for state in router_background.into_iter().skip(mode_label_bg_start) {
            if state.pre_mlp.len() != self.expected_hidden_dim
                || state.residual.len() != self.expected_hidden_dim
            {
                bail!("router background dimension mismatch for {rule_id}");
            }
            self.router_background_rows.push(self.pre_rows.len());
            self.pre_rows.push(state.pre_mlp);
            self.res_rows.push(state.residual);
            self.row_banks.push(WireBankKey(CartridgeBank::OrientTrunk));
        }

        let route_tag = match oracle_mode.route_tag() {
            RouteTag::ToDatabase => "sql",
            RouteTag::ToInquirer => "to_user",
        };
        let payload_body = payload_body.trim().to_string();
        let payload_prefix = format!(
            "{prefix_wire}<|im_start|>assistant\n<think>\n[MODE: {}]\nPlan: {}\n</think>\n<{route_tag}>\n",
            oracle_mode.as_str(), generated_plan_line.trim(),
        );
        let closing_tag = format!("\n</{route_tag}>");
        let rows = self
            .worker
            .capture_payload_activations(
                &payload_prefix,
                &payload_body,
                &closing_tag,
                self.target_layer,
            )
            .with_context(|| format!("failed capturing payload boundaries for {rule_id}"))?;
        let payload_bank = WireBankKey(CartridgeBank::ModePayload(oracle_mode));
        for row in rows {
            if row.pre_mlp.len() != self.expected_hidden_dim
                || row.residual.len() != self.expected_hidden_dim
            {
                bail!(
                    "payload capture dimension mismatch for {rule_id}/{}",
                    row.boundary
                );
            }
            let idx = self.pre_rows.len();
            self.pre_rows.push(row.pre_mlp);
            self.res_rows.push(row.residual);
            self.row_banks.push(payload_bank);
            self.payload_scope_rows
                .entry(oracle_mode)
                .or_default()
                .insert(idx);
            if is_compiled_structural_payload_boundary(oracle_mode, &row.boundary) {
                self.payload_groups
                    .entry((oracle_mode, row.boundary, row.target_token_id))
                    .or_default()
                    .push((idx, row.competitor_token_id));
            }
        }
        let completed_wire = format!("{payload_prefix}{payload_body}{closing_tag}<|im_end|>\n");
        Ok((completed_wire, generated_plan_line))
    }
}

#[derive(Clone, Copy)]
enum ColdExternalTask {
    List,
    Detail,
    InspectPointReport,
    Join,
    JoinInspectReport,
    UpdateAllow,
    CheckUpdateAllow,
    UpdateAsk { reply: &'static str },
    CheckUpdateAsk { reply: &'static str },
    MissingIdAbstain,
    UnnamedTableExploratoryUpdate,
}

/// Capture complete short histories as well as the long REPL trajectories below. Each task
/// starts from its own untouched database, and every tool observation comes from that driver.
fn capture_external_cold_trajectory(
    harvester: &mut OnPolicyOracleHarvester<'_>,
    d: &ExternalCalibrationDomain,
    task: ColdExternalTask,
    read_only: bool,
    explicit_projection: bool,
) -> Result<()> {
    let mt = d.main_table;
    let jt = d.join_table;
    let pk = d.pk_col;
    let prompt_owned = match task {
        ColdExternalTask::List => d.q_list.to_string(),
        ColdExternalTask::Detail => d.q_detail.to_string(),
        ColdExternalTask::InspectPointReport => {
            if explicit_projection {
                format!(
                    "Inspect {mt} for {pk} = {} and report its {} and {}.",
                    d.pk1, d.slug_col, d.val_col
                )
            } else {
                format!(
                    "Inspect {mt} for {pk} {} and report {} and {}.",
                    d.pk1, d.slug_col, d.val_col
                )
            }
        }
        ColdExternalTask::Join => d.q_join.to_string(),
        ColdExternalTask::JoinInspectReport => format!(
            "Join {mt} and {jt} to report the {} for {pk} = {}.",
            d.join_col, d.pk1
        ),
        ColdExternalTask::UpdateAllow => d.q_mut_allow.to_string(),
        ColdExternalTask::CheckUpdateAllow => format!(
            "Check {pk} {} in {mt} and update {} to {}.",
            d.pk1, d.val_col, d.v1_new
        ),
        ColdExternalTask::UpdateAsk { .. } => d.q_mut_ask.to_string(),
        ColdExternalTask::CheckUpdateAsk { .. } => format!(
            "Check {pk} {} in {mt} and update {} to {}.",
            d.pk2, d.val_col, d.v2_new
        ),
        ColdExternalTask::MissingIdAbstain => format!(
            "Check {pk} 99999 in {mt} and update {} to {}.",
            d.val_col, d.v1_new
        ),
        ColdExternalTask::UnnamedTableExploratoryUpdate => {
            format!("Set {} for {pk} {} to {}.", d.val_col, d.pk1, d.v1_new)
        }
    };
    let is_ask = matches!(
        task,
        ColdExternalTask::UpdateAsk { .. } | ColdExternalTask::CheckUpdateAsk { .. }
    );
    let ask_reply_opt = match task {
        ColdExternalTask::UpdateAsk { reply } | ColdExternalTask::CheckUpdateAsk { reply } => {
            Some(reply)
        }
        _ => None,
    };
    let mutation = matches!(
        task,
        ColdExternalTask::UpdateAllow
            | ColdExternalTask::CheckUpdateAllow
            | ColdExternalTask::UpdateAsk { .. }
            | ColdExternalTask::CheckUpdateAsk { .. }
            | ColdExternalTask::UnnamedTableExploratoryUpdate
    );
    let mut driver = open_external_domain_driver(d)?;
    driver.set_read_only(read_only)?;
    let mut wire = format_lotl_user_prompt_envelope(&prompt_owned);
    let mut step = |rule, mode, payload: &str, reply| -> Result<DriverObservation> {
        let (turn, _) = harvester.capture_at_mode_and_generate_turn(&wire, rule, mode, payload)?;
        let (observation, observation_wire) =
            execute_verified_driver_step(&mut driver, mode, payload, reply)?;
        wire = format!("{turn}{observation_wire}");
        Ok(observation)
    };
    step(
        "R1_probe_catalog_initial",
        Mode::Probe,
        "SELECT name FROM sqlite_master WHERE type = 'table';",
        None,
    )?;
    if matches!(task, ColdExternalTask::UnnamedTableExploratoryUpdate) {
        step(
            "R2a_probe_after_catalog",
            Mode::Probe,
            &format!("PRAGMA table_info(\"{}\");", d.extra_tables[0]),
            None,
        )?;
    }
    step(
        "R2a_probe_after_catalog",
        Mode::Probe,
        &format!("PRAGMA table_info(\"{mt}\");"),
        None,
    )?;
    if matches!(
        task,
        ColdExternalTask::Join | ColdExternalTask::JoinInspectReport
    ) {
        step(
            "R2b_probe_on_join_followup",
            Mode::Probe,
            &format!("PRAGMA table_info(\"{jt}\");"),
            None,
        )?;
    }
    let row_id = if matches!(task, ColdExternalTask::MissingIdAbstain) {
        99999
    } else if is_ask {
        d.pk2
    } else {
        d.pk1
    };
    let projection = if explicit_projection {
        format!(
            "{pk}, {}, {}, {}, {}, policy_decision, {}, {}",
            d.slug_col, d.status_col, d.val_col, d.rule_col, d.owner_col, d.freeze_col
        )
    } else {
        "*".to_string()
    };
    let read_sql = match task {
        ColdExternalTask::List => format!(
            "SELECT {projection} FROM {mt} WHERE {} = '{}';",
            d.status_col, d.active_val
        ),
        ColdExternalTask::InspectPointReport => format!(
            "SELECT {}, {} FROM {mt} WHERE {pk} = {row_id};",
            d.slug_col, d.val_col
        ),
        ColdExternalTask::Join => {
            let fields = if explicit_projection {
                format!(
                    "{mt}.{pk}, {mt}.{}, {mt}.{}, {jt}.{}, {jt}.prev_val, {jt}.new_val",
                    d.slug_col, d.val_col, d.join_col
                )
            } else {
                "*".to_string()
            };
            format!(
                "SELECT {fields} FROM {mt} JOIN {jt} ON {mt}.{pk} = {jt}.{} WHERE {mt}.{} = '{}';",
                d.fk_col, d.status_col, d.active_val
            )
        }
        ColdExternalTask::JoinInspectReport => format!(
            "SELECT {jt}.{} FROM {mt} JOIN {jt} ON {mt}.{pk} = {jt}.{} WHERE {mt}.{pk} = {row_id};",
            d.join_col, d.fk_col
        ),
        _ => format!("SELECT {projection} FROM {mt} WHERE {pk} = {row_id};"),
    };
    let read = step("R3a_read_after_pragma", Mode::Read, &read_sql, None)?;
    if matches!(task, ColdExternalTask::MissingIdAbstain) {
        if !read.rows.is_empty() {
            bail!("MissingIdAbstain expected 0 rows");
        }
        step(
            "R5_abstain_zero_rows",
            Mode::Abstain,
            &format!("No record with {pk} = 99999 exists in {mt}; no changes were made."),
            None,
        )?;
        return Ok(());
    }
    if !mutation {
        let answer = match task {
            ColdExternalTask::InspectPointReport => {
                let row0 = read
                    .rows
                    .first()
                    .context("InspectPointReport expected 1 row")?;
                format!(
                    "For {pk} = {row_id} in {mt}: {} = {}, {} = {}.",
                    d.slug_col, row0[0], d.val_col, row0[1]
                )
            }
            ColdExternalTask::JoinInspectReport => {
                let row0 = read
                    .rows
                    .first()
                    .context("JoinInspectReport expected 1 joined row")?;
                format!(
                    "For {pk} = {row_id}, {} in {jt} is {}.",
                    d.join_col, row0[0]
                )
            }
            _ => {
                let value_column = read
                    .columns
                    .iter()
                    .position(|col| col == d.val_col)
                    .context("cold answer observation omitted the requested value column")?;
                let key_column = read
                    .columns
                    .iter()
                    .position(|col| col == pk)
                    .context("cold answer observation omitted its primary key")?;
                let mut lines = Vec::new();
                for row in &read.rows {
                    lines.push(format!(
                        "{mt}: {pk}={}, {}={}.",
                        row[key_column], d.val_col, row[value_column]
                    ));
                }
                if matches!(task, ColdExternalTask::Join) {
                    lines.push(format!("The rows include matching records from {jt}."));
                }
                lines.join("\n")
            }
        };
        step("R4_result_readonly_answer", Mode::Result, &answer, None)?;
        return Ok(());
    }
    let write_rule = if let Some(reply) = ask_reply_opt {
        let ask = serde_json::json!({
            "select": format!("Authorize updating {} on {mt} row {row_id}?", d.val_col),
            "options": ["ALLOW", "DENY"],
        })
        .to_string();
        step("R6_ask_steward_gate", Mode::Ask, &ask, Some(reply))?;
        if reply == "DENY" {
            step(
                "R7b_abstain_steward_denied",
                Mode::Abstain,
                "The Steward denied this change; no rows were changed.",
                None,
            )?;
            return Ok(());
        }
        "R8b_write_steward_granted"
    } else {
        "R8a_write_allow_row"
    };
    let value = if is_ask { d.v2_new } else { d.v1_new };
    let update = format!(
        "UPDATE {mt} SET {} = {value} WHERE {pk} = {row_id};",
        d.val_col
    );
    let write = step(write_rule, Mode::Write, &update, None)?;
    if write.rows_affected != 1 {
        bail!("cold calibration update did not affect exactly one row");
    }
    let verification = if explicit_projection {
        format!(
            "SELECT {pk}, {} FROM {mt} WHERE {pk} = {row_id};",
            d.val_col
        )
    } else {
        read_sql
    };
    step(
        "R9_read_verify_after_write",
        Mode::Read,
        &verification,
        None,
    )?;
    step(
        "R10_result_after_verify",
        Mode::Result,
        &format!("Verified {} = {value} on {mt} row {row_id}.", d.val_col),
        None,
    )?;
    Ok(())
}

fn capture_external_rollback_trajectory(
    harvester: &mut OnPolicyOracleHarvester<'_>,
    d: &ExternalCalibrationDomain,
    reply: &str,
) -> Result<()> {
    let mut driver = open_external_domain_driver(d)?;
    let unchanged = driver.database_snapshot()?;
    let mut wire = format_lotl_user_prompt_envelope(d.q_mut_ask);
    for (rule, mode, payload) in [
        (
            "R1_probe_catalog_initial",
            Mode::Probe,
            "SELECT name FROM sqlite_master WHERE type = 'table';".to_string(),
        ),
        (
            "R2a_probe_after_catalog",
            Mode::Probe,
            format!("PRAGMA table_info(\"{}\");", d.main_table),
        ),
        (
            "R3a_read_after_pragma",
            Mode::Read,
            format!(
                "SELECT {}, {} FROM {} WHERE {} = {};",
                d.pk_col, d.val_col, d.main_table, d.pk_col, d.pk2
            ),
        ),
    ] {
        let (turn, _) = harvester.capture_at_mode_and_generate_turn(&wire, rule, mode, &payload)?;
        let (_, observation) = execute_verified_driver_step(&mut driver, mode, &payload, None)?;
        wire = format!("{turn}{observation}");
    }
    let update = format!(
        "UPDATE {} SET {} = {} WHERE {} = {};",
        d.main_table, d.val_col, d.v2_new, d.pk_col, d.pk2
    );
    let (turn, _) = harvester.capture_at_mode_and_generate_turn(
        &wire,
        "R8a_write_allow_row",
        Mode::Write,
        &update,
    )?;
    let (_, rejected) = execute_expected_steward_rollback(&mut driver, &update)?;
    wire = format!("{turn}{rejected}");
    let ask = serde_json::json!({
        "select": format!("Authorize updating {} on {} row {}?", d.val_col, d.main_table, d.pk2),
        "options": ["ALLOW", "DENY"],
    })
    .to_string();
    let (turn, _) = harvester.capture_at_mode_and_generate_turn(
        &wire,
        "R6b_ask_after_steward_rollback",
        Mode::Ask,
        &ask,
    )?;
    let (_, decision) = execute_verified_driver_step(&mut driver, Mode::Ask, &ask, Some(reply))?;
    wire = format!("{turn}{decision}");
    if reply == "DENY" {
        if driver.database_snapshot()? != unchanged
            || driver.task_writes_committed() != 0
            || driver.steward_auth() != StewardAuthState::Denied
            || driver.legal_modes()?.contains(&Mode::Write)
        {
            bail!("Steward DENY calibration unexpectedly changed or authorized the database");
        }
        let payload = "The Steward denied this change; the rejected update was rolled back.";
        harvester.capture_at_mode_and_generate_turn(
            &wire,
            "R7b_abstain_steward_denied",
            Mode::Abstain,
            payload,
        )?;
        execute_verified_driver_step(&mut driver, Mode::Abstain, payload, None)?;
        return Ok(());
    }
    if reply != "ALLOW"
        || !driver.steward_auth().is_granted()
        || !driver.legal_modes()?.contains(&Mode::Write)
    {
        bail!("Steward ALLOW calibration did not authorize the recovery WRITE");
    }
    let (turn, _) = harvester.capture_at_mode_and_generate_turn(
        &wire,
        "R8b_write_steward_granted",
        Mode::Write,
        &update,
    )?;
    let (write, applied) = execute_verified_driver_step(&mut driver, Mode::Write, &update, None)?;
    if write.rows_affected != 1 || driver.task_writes_committed() != 1 {
        bail!("recovery WRITE must commit exactly one row once");
    }
    wire = format!("{turn}{applied}");
    let query = format!(
        "SELECT {}, {} FROM {} WHERE {} = {};",
        d.pk_col, d.val_col, d.main_table, d.pk_col, d.pk2
    );
    let (turn, _) = harvester.capture_at_mode_and_generate_turn(
        &wire,
        "R9_read_verify_after_write",
        Mode::Read,
        &query,
    )?;
    let (read, verified) = execute_verified_driver_step(&mut driver, Mode::Read, &query, None)?;
    if read.operation != Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
        || read.rows != vec![vec![serde_json::json!(d.pk2), serde_json::json!(d.v2_new)]]
    {
        bail!("recovery verification did not observe the committed target value");
    }
    let answer = format!(
        "The Steward allowed the change. Verified {} = {} for {} = {}.",
        d.val_col, d.v2_new, d.pk_col, d.pk2
    );
    harvester.capture_at_mode_and_generate_turn(
        &format!("{turn}{verified}"),
        "R10_result_after_verify",
        Mode::Result,
        &answer,
    )?;
    execute_verified_driver_step(&mut driver, Mode::Result, &answer, None)?;
    Ok(())
}

/// Harvest genuine on-policy `(pre_mlp, residual)` states across external multi-table SQLite
/// domains and multilingual `ScenarioCoordinate` train episodes `[0..episodes]`, assemble a
/// [`CompileCartridgeInputSpec`], and write it directly to `output_path`.
pub fn run_oracle_capture_harvest(
    worker: &mut StdioQwenWorkerClient,
    config: &OracleCaptureHarvestConfig,
    output_path: &Path,
) -> Result<OracleCaptureHarvestReceipt> {
    let model_family = worker
        .provenance
        .resolved_model_family
        .clone()
        .unwrap_or_else(|| crate::worker::normalize_model_identifier(&config.model_family));
    let checkpoint_revision = worker.provenance.checkpoint_revision.clone();
    if config.episodes == 0 || config.episodes > MAX_ORACLE_TRAIN_EPISODES {
        bail!(
            "--oracle-capture-out requires --episodes in 1..={} (got {}) so held-out nursery episodes [12..24] stay strictly un-observed",
            MAX_ORACLE_TRAIN_EPISODES,
            config.episodes
        );
    }

    let (resolved_layer, base_intermediate_size, weight_geometry) = worker
        .extract_compiler_weight_geometry_for_optional_layer(config.target_layer)
        .context("failed extracting compiler weight geometry from worker")?;
    let expected_hidden_dim = weight_geometry.norm_weight.len();
    if expected_hidden_dim == 0 {
        bail!("worker weight geometry returned empty norm_weight (hidden_dim == 0)");
    }

    let mut harvester = OnPolicyOracleHarvester::new(
        worker,
        resolved_layer,
        expected_hidden_dim,
        config.max_plan_tokens,
    );

    let domain_count = config
        .external_domain_limit
        .unwrap_or(EXTERNAL_CALIBRATION_DOMAINS.len())
        .clamp(1, EXTERNAL_CALIBRATION_DOMAINS.len());

    // Part A: External 10-table multi-turn REPL/TUI workflows (0% overlap with `cloud_ops_production.sqlite`).
    for (domain_index, d) in EXTERNAL_CALIBRATION_DOMAINS[..domain_count]
        .iter()
        .enumerate()
    {
        let mut driver = open_external_domain_driver(d)?;
        driver.set_read_only(false)?;
        let mt = d.main_table;
        let jt = d.join_table;
        let pk = d.pk_col;
        let val_c = d.val_col;
        let fk_c = d.fk_col;

        // Prompt 1: Chitchat greeting -> R0_chitchat_ask.
        let (w_chit, _) = harvester.capture_at_mode_and_generate_turn(
            &format_lotl_user_prompt_envelope(d.chitchat),
            "R0_chitchat_ask",
            Mode::Ask,
            "Hi! How can I help you today?",
        )?;

        let catalog_sql = "SELECT name FROM sqlite_master WHERE type = 'table';";
        // Cold information-only tasks execute with `read_only = true` so their `<tool_response>`
        // envelopes carry unambiguous causal `"purpose": "read_only"`, `"read_only": true`, and
        // `"steward_auth"` metadata without contradictory duplicate trajectories.
        for task in [
            ColdExternalTask::List,
            ColdExternalTask::Detail,
            ColdExternalTask::InspectPointReport,
            ColdExternalTask::Join,
            ColdExternalTask::JoinInspectReport,
        ] {
            capture_external_cold_trajectory(&mut harvester, d, task, true, domain_index % 2 == 0)?;
        }
        for task in [
            ColdExternalTask::UpdateAllow,
            ColdExternalTask::CheckUpdateAllow,
            ColdExternalTask::UpdateAsk { reply: "ALLOW" },
            ColdExternalTask::UpdateAsk { reply: "DENY" },
            ColdExternalTask::CheckUpdateAsk { reply: "ALLOW" },
            ColdExternalTask::CheckUpdateAsk { reply: "DENY" },
            ColdExternalTask::MissingIdAbstain,
            ColdExternalTask::UnnamedTableExploratoryUpdate,
        ] {
            capture_external_cold_trajectory(
                &mut harvester,
                d,
                task,
                false,
                domain_index % 2 == 0,
            )?;
        }
        for reply in ["ALLOW", "DENY"] {
            capture_external_rollback_trajectory(&mut harvester, d, reply)?;
        }

        // Prompt 2 (after chitchat): R1 Probe(catalog) -> R2a Probe(PRAGMA) -> R3a Read(SELECT) -> R4 Result.
        driver.begin_new_task();
        let p2_open = format!("{w_chit}{}", format_lotl_user_prompt_envelope(d.q_list));
        let (w_p2_t1_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &p2_open,
            "R1_probe_catalog_initial",
            Mode::Probe,
            catalog_sql,
        )?;
        let (_, cat_obs) =
            execute_verified_driver_step(&mut driver, Mode::Probe, catalog_sql, None)?;
        let w_p2_t1 = format!("{w_p2_t1_turn}{cat_obs}");

        let pragma_main_sql = format!("PRAGMA table_info(\"{mt}\");");
        let (w_p2_t2_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p2_t1,
            "R2a_probe_after_catalog",
            Mode::Probe,
            &pragma_main_sql,
        )?;
        let (_, pragma_main_obs) =
            execute_verified_driver_step(&mut driver, Mode::Probe, &pragma_main_sql, None)?;
        let w_p2_t2 = format!("{w_p2_t2_turn}{pragma_main_obs}");

        let pragma_join_sql = format!("PRAGMA table_info(\"{jt}\");");
        let select_all_main_sql = format!("SELECT * FROM {mt};");

        let (w_p2_t3_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p2_t2,
            "R3a_read_after_pragma",
            Mode::Read,
            &select_all_main_sql,
        )?;
        let (_, read_main_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_all_main_sql, None)?;
        let w_p2_t3 = format!("{w_p2_t3_turn}{read_main_obs}");

        let p2_ans = format!(
            "Active records in {mt}: {} ({}) and {} ({}).",
            d.pk1, d.slug1, d.pk2, d.slug2
        );
        let (w_p2_done, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p2_t3,
            "R4_result_readonly_answer",
            Mode::Result,
            &p2_ans,
        )?;
        let _ = execute_verified_driver_step(&mut driver, Mode::Result, &p2_ans, None)?;

        // Prompt 3 (JOIN follow-up with unprobed audit table): R2b Probe(PRAGMA jt) -> R3a Read(JOIN) -> R4 Result.
        driver.begin_new_task();
        let p3_open = format!("{w_p2_done}{}", format_lotl_user_prompt_envelope(d.q_join));
        let (w_p3_t1_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &p3_open,
            "R2b_probe_on_join_followup",
            Mode::Probe,
            &pragma_join_sql,
        )?;
        let (_, pragma_join_obs) =
            execute_verified_driver_step(&mut driver, Mode::Probe, &pragma_join_sql, None)?;
        let w_p3_t1 = format!("{w_p3_t1_turn}{pragma_join_obs}");

        let join_sql = format!("SELECT * FROM {mt} JOIN {jt} ON {mt}.{pk} = {jt}.{fk_c};");
        let (w_p3_t2_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p3_t1,
            "R3a_read_after_pragma",
            Mode::Read,
            &join_sql,
        )?;
        let (_, join_obs) = execute_verified_driver_step(&mut driver, Mode::Read, &join_sql, None)?;
        let w_p3_t2 = format!("{w_p3_t2_turn}{join_obs}");

        let p3_ans = format!("Joined {mt} and {jt} for {} and {}.", d.pk1, d.pk2);
        let (w_p3_done, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p3_t2,
            "R4_result_readonly_answer",
            Mode::Result,
            &p3_ans,
        )?;
        let _ = execute_verified_driver_step(&mut driver, Mode::Result, &p3_ans, None)?;

        // Prompt 4 (Detail lookup on already-probed table): R3b Read -> R4 Result.
        driver.begin_new_task();
        let p4_open = format!(
            "{w_p3_done}{}",
            format_lotl_user_prompt_envelope(d.q_detail)
        );
        let select_pk1_sql = format!("SELECT * FROM {mt} WHERE {pk} = {};", d.pk1);
        let (w_p4_t1_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &p4_open,
            "R3b_read_on_followup_prompt",
            Mode::Read,
            &select_pk1_sql,
        )?;
        let (_, pk1_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_pk1_sql, None)?;
        let w_p4_t1 = format!("{w_p4_t1_turn}{pk1_obs}");

        let p4_ans = format!(
            "Details for {} ({}): {val_c}={}, policy_decision=ALLOW.",
            d.pk1, d.slug1, d.v1_init
        );
        let (w_p4_done, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p4_t1,
            "R4_result_readonly_answer",
            Mode::Result,
            &p4_ans,
        )?;
        let _ = execute_verified_driver_step(&mut driver, Mode::Result, &p4_ans, None)?;

        driver.set_read_only(false)?;
        // Prompt 5 (ALLOW update): R3b Read -> R8a Write -> R9 Read(verify) -> R10 Result.
        driver.begin_new_task();
        let p5_open = format!(
            "{w_p4_done}{}",
            format_lotl_user_prompt_envelope(d.q_mut_allow)
        );
        let (w_p5_t1_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &p5_open,
            "R3b_read_on_followup_prompt",
            Mode::Read,
            &select_pk1_sql,
        )?;
        let (_, cand1_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_pk1_sql, None)?;
        let w_p5_t1 = format!("{w_p5_t1_turn}{cand1_obs}");

        let update_pk1_sql = format!(
            "UPDATE {mt} SET {val_c} = {} WHERE {pk} = {};",
            d.v1_new, d.pk1
        );
        let (w_p5_t2_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p5_t1,
            "R8a_write_allow_row",
            Mode::Write,
            &update_pk1_sql,
        )?;
        let (_, write1_obs) =
            execute_verified_driver_step(&mut driver, Mode::Write, &update_pk1_sql, None)?;
        let w_p5_t2 = format!("{w_p5_t2_turn}{write1_obs}");

        let (w_p5_t3_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p5_t2,
            "R9_read_verify_after_write",
            Mode::Read,
            &select_pk1_sql,
        )?;
        let (_, ver1_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_pk1_sql, None)?;
        let w_p5_t3 = format!("{w_p5_t3_turn}{ver1_obs}");

        let p5_ans = format!("Updated {val_c} for {} to {}.", d.pk1, d.v1_new);
        let (w_p5_done, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p5_t3,
            "R10_result_after_verify",
            Mode::Result,
            &p5_ans,
        )?;
        let _ = execute_verified_driver_step(&mut driver, Mode::Result, &p5_ans, None)?;

        // Prompt 6 (ASK Steward update): R3b Read -> R6 Ask -> ALLOW -> R8b Write -> R9 Read(verify) -> R10 Result.
        driver.begin_new_task();
        let p6_open = format!(
            "{w_p5_done}{}",
            format_lotl_user_prompt_envelope(d.q_mut_ask)
        );
        let select_pk2_sql = format!("SELECT * FROM {mt} WHERE {pk} = {};", d.pk2);
        let (w_p6_t1_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &p6_open,
            "R3b_read_on_followup_prompt",
            Mode::Read,
            &select_pk2_sql,
        )?;
        let (_, cand2_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_pk2_sql, None)?;
        let w_p6_t1 = format!("{w_p6_t1_turn}{cand2_obs}");

        let ask_json = serde_json::json!({
            "select": format!("Authorize update on {mt} row {}?", d.pk2),
            "options": ["ALLOW", "DENY"],
        })
        .to_string();
        let (w_p6_t2_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p6_t1,
            "R6_ask_steward_gate",
            Mode::Ask,
            &ask_json,
        )?;
        let (_, ask_allow_obs) =
            execute_verified_driver_step(&mut driver, Mode::Ask, &ask_json, Some("ALLOW"))?;
        let w_p6_t2 = format!("{w_p6_t2_turn}{ask_allow_obs}");

        let update_pk2_sql = format!(
            "UPDATE {mt} SET {val_c} = {} WHERE {pk} = {};",
            d.v2_new, d.pk2
        );
        let (w_p6_t3_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p6_t2,
            "R8b_write_steward_granted",
            Mode::Write,
            &update_pk2_sql,
        )?;
        let (_, write2_obs) =
            execute_verified_driver_step(&mut driver, Mode::Write, &update_pk2_sql, None)?;
        let w_p6_t3 = format!("{w_p6_t3_turn}{write2_obs}");

        let (w_p6_t4_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &w_p6_t3,
            "R9_read_verify_after_write",
            Mode::Read,
            &select_pk2_sql,
        )?;
        let (_, ver2_obs) =
            execute_verified_driver_step(&mut driver, Mode::Read, &select_pk2_sql, None)?;
        let w_p6_t4 = format!("{w_p6_t4_turn}{ver2_obs}");

        let p6_ans = format!(
            "Updated {val_c} for {} to {} after Steward approval.",
            d.pk2, d.v2_new
        );
        let _ = harvester.capture_at_mode_and_generate_turn(
            &w_p6_t4,
            "R10_result_after_verify",
            Mode::Result,
            &p6_ans,
        )?;
        eprintln!(
            "[oracle-capture] domain={} states={}",
            d.main_table,
            harvester.pre_rows.len()
        );
    }

    // Part B: Multilingual `ScenarioCoordinate` train episodes `[0..config.episodes]` (strictly disjoint from `[12..24]`).
    let train_coords =
        build_nursery_sweep_coordinates(config.episodes, NurseryAppFilter::SqlSteward);
    let mut det_user = DeterministicStewardUserEnv::new();

    for (episode_idx, coord) in train_coords.iter().enumerate() {
        let opening = det_user.opening_prompt(coord)?;
        let slug = deterministic_sql_target_slug(coord);
        let vocab = coord.build_sql_steward_vocabulary();
        let tbl = vocab.primary_table.as_str();
        let val_c = "balance_credits";
        let target_bal = deterministic_sql_requested_balance(coord);
        let init_bal = target_bal + 550;

        let catalog_sql = "SELECT name FROM sqlite_master WHERE type = 'table';";
        let pragma_sql = format!("PRAGMA table_info({tbl});");
        let sel_sql = format!("SELECT * FROM {tbl} WHERE entity_slug = '{slug}';");
        let upd_sql =
            format!("UPDATE {tbl} SET {val_c} = {target_bal} WHERE entity_slug = '{slug}';");

        // Branch 1: Zero rows -> R5_abstain_zero_rows.
        let mut drv_zero = open_train_episode_driver(tbl, None)?;
        let (w_t1_turn, plan_t1) = harvester.capture_at_mode_and_generate_turn(
            &format_lotl_user_prompt_envelope(&opening),
            "R1_probe_catalog_initial",
            Mode::Probe,
            catalog_sql,
        )?;
        let (obs_cat, cat_obs_chunk) =
            execute_verified_driver_step(&mut drv_zero, Mode::Probe, catalog_sql, None)?;
        let w_t1 = format!("{w_t1_turn}{cat_obs_chunk}");

        let (w_t2_turn, plan_t2) = harvester.capture_at_mode_and_generate_turn(
            &w_t1,
            "R2a_probe_after_catalog",
            Mode::Probe,
            &pragma_sql,
        )?;
        let (obs_pragma, pragma_obs_chunk) =
            execute_verified_driver_step(&mut drv_zero, Mode::Probe, &pragma_sql, None)?;
        let w_t2 = format!("{w_t2_turn}{pragma_obs_chunk}");

        let (w_t3_read_turn, plan_t3) = harvester.capture_at_mode_and_generate_turn(
            &w_t2,
            "R3a_read_after_pragma",
            Mode::Read,
            &sel_sql,
        )?;
        let (_, zero_obs_chunk) =
            execute_verified_driver_step(&mut drv_zero, Mode::Read, &sel_sql, None)?;
        let _ = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t3_read_turn}{zero_obs_chunk}"),
            "R5_abstain_zero_rows",
            Mode::Abstain,
            &format!("Entity `{slug}` was not found in `{tbl}`."),
        )?;

        // Branch 2: Locked / DENY row -> R7a_abstain_locked_row.
        let mut drv_locked =
            open_train_episode_driver(tbl, Some((&slug, init_bal, "DENY", "LOCKED")))?;
        let _ = execute_verified_driver_step(&mut drv_locked, Mode::Probe, catalog_sql, None)?;
        let _ = execute_verified_driver_step(&mut drv_locked, Mode::Probe, &pragma_sql, None)?;
        let (_, locked_obs_chunk) =
            execute_verified_driver_step(&mut drv_locked, Mode::Read, &sel_sql, None)?;
        let _ = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t3_read_turn}{locked_obs_chunk}"),
            "R7a_abstain_locked_row",
            Mode::Abstain,
            &format!("Cannot modify `{slug}` because row is LOCKED / DENY."),
        )?;

        // Branch 3: ALLOW row -> R8a_write_allow_row.
        let mut drv_allow =
            open_train_episode_driver(tbl, Some((&slug, init_bal, "ALLOW", "CLEAR")))?;
        let _ = execute_verified_driver_step(&mut drv_allow, Mode::Probe, catalog_sql, None)?;
        let _ = execute_verified_driver_step(&mut drv_allow, Mode::Probe, &pragma_sql, None)?;
        let (_, allow_obs_chunk) =
            execute_verified_driver_step(&mut drv_allow, Mode::Read, &sel_sql, None)?;
        let _ = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t3_read_turn}{allow_obs_chunk}"),
            "R8a_write_allow_row",
            Mode::Write,
            &upd_sql,
        )?;

        // Branch 4: ASK row -> R6_ask_steward_gate -> DENIED (R7b) & GRANTED (R8b -> R9 -> R10 + follow-up wave).
        let mut drv_ask_deny =
            open_train_episode_driver(tbl, Some((&slug, init_bal, "ASK", "CLEAR")))?;
        let _ = execute_verified_driver_step(&mut drv_ask_deny, Mode::Probe, catalog_sql, None)?;
        let _ = execute_verified_driver_step(&mut drv_ask_deny, Mode::Probe, &pragma_sql, None)?;
        let (_, ask_cand_deny_chunk) =
            execute_verified_driver_step(&mut drv_ask_deny, Mode::Read, &sel_sql, None)?;

        let ask_payload = serde_json::json!({
            "select": format!("Authorize updating `{slug}` in `{tbl}`?"),
            "options": ["ALLOW", "DENY"],
        })
        .to_string();
        let (w_t4_ask_turn, plan_t4) = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t3_read_turn}{ask_cand_deny_chunk}"),
            "R6_ask_steward_gate",
            Mode::Ask,
            &ask_payload,
        )?;
        let deny_reply = format!("DENY — Steward blocked `{slug}`.");
        let (_, deny_obs_chunk) = execute_verified_driver_step(
            &mut drv_ask_deny,
            Mode::Ask,
            &ask_payload,
            Some(&deny_reply),
        )?;
        let _ = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t4_ask_turn}{deny_obs_chunk}"),
            "R7b_abstain_steward_denied",
            Mode::Abstain,
            &format!("Abstaining because Steward denied modification on `{slug}`."),
        )?;

        let mut drv_ask_grant =
            open_train_episode_driver(tbl, Some((&slug, init_bal, "ASK", "CLEAR")))?;
        let _ = execute_verified_driver_step(&mut drv_ask_grant, Mode::Probe, catalog_sql, None)?;
        let _ = execute_verified_driver_step(&mut drv_ask_grant, Mode::Probe, &pragma_sql, None)?;
        let (obs_read_cand, _) =
            execute_verified_driver_step(&mut drv_ask_grant, Mode::Read, &sel_sql, None)?;
        let grant_reply = format!("ALLOW — Steward token STEWARD_TOK_R1 for `{slug}`.");
        let (obs_ask_grant, grant_obs_chunk) = execute_verified_driver_step(
            &mut drv_ask_grant,
            Mode::Ask,
            &ask_payload,
            Some(&grant_reply),
        )?;

        let (w_t5_write_turn, plan_t5) = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_t4_ask_turn}{grant_obs_chunk}"),
            "R8b_write_steward_granted",
            Mode::Write,
            &upd_sql,
        )?;
        let (obs_write, write_obs_chunk) =
            execute_verified_driver_step(&mut drv_ask_grant, Mode::Write, &upd_sql, None)?;
        let w_t5_write = format!("{w_t5_write_turn}{write_obs_chunk}");

        let (w_t6_ver_turn, plan_t6) = harvester.capture_at_mode_and_generate_turn(
            &w_t5_write,
            "R9_read_verify_after_write",
            Mode::Read,
            &sel_sql,
        )?;
        let (obs_ver, ver_obs_chunk) =
            execute_verified_driver_step(&mut drv_ask_grant, Mode::Read, &sel_sql, None)?;
        let w_t6_ver = format!("{w_t6_ver_turn}{ver_obs_chunk}");

        let ep_result_txt = format!(
            "Updated `{slug}` in `{tbl}` so `{val_c}` is now {target_bal}, verified by post-write read."
        );
        let (w_ep_done, plan_t7) = harvester.capture_at_mode_and_generate_turn(
            &w_t6_ver,
            "R10_result_after_verify",
            Mode::Result,
            &ep_result_txt,
        )?;
        let (obs_res, _) =
            execute_verified_driver_step(&mut drv_ask_grant, Mode::Result, &ep_result_txt, None)?;

        // Follow-up read-only wave (`followup_prompt` in nursery, starting a new user task on the same connection):
        let prior_events = vec![
            make_session_event(Mode::Probe, plan_t1, catalog_sql.to_string(), obs_cat),
            make_session_event(Mode::Probe, plan_t2, pragma_sql.clone(), obs_pragma),
            make_session_event(Mode::Read, plan_t3, sel_sql.clone(), obs_read_cand),
            make_session_event(Mode::Ask, plan_t4, ask_payload, obs_ask_grant),
            make_session_event(Mode::Write, plan_t5, upd_sql, obs_write),
            make_session_event(Mode::Read, plan_t6, sel_sql.clone(), obs_ver),
            make_session_event(Mode::Result, plan_t7, ep_result_txt, obs_res),
        ];
        let followup = det_user.followup_prompt(coord, &prior_events)?;
        drv_ask_grant.begin_new_task();
        drv_ask_grant.set_read_only(false)?;
        let fup_open = format!(
            "{w_ep_done}{}",
            format_lotl_user_prompt_envelope(followup.trim())
        );
        let (w_fup_read_turn, _) = harvester.capture_at_mode_and_generate_turn(
            &fup_open,
            "R3b_read_on_followup_prompt",
            Mode::Read,
            &sel_sql,
        )?;
        let (_, fup_read_obs_chunk) =
            execute_verified_driver_step(&mut drv_ask_grant, Mode::Read, &sel_sql, None)?;
        let _ = harvester.capture_at_mode_and_generate_turn(
            &format!("{w_fup_read_turn}{fup_read_obs_chunk}"),
            "R4_result_readonly_answer",
            Mode::Result,
            &format!("Confirmed `{slug}` in `{tbl}` has `{val_c}` = {target_bal}."),
        )?;
        eprintln!(
            "[oracle-capture] train_episode={} states={}",
            episode_idx + 1,
            harvester.pre_rows.len()
        );
    }

    let total_rows = harvester.pre_rows.len();
    let mut obligations = Vec::with_capacity(UNIMODAL_ORACLE_RULES.len());
    let mut rule_positive_counts = BTreeMap::new();
    // Dynamic banks cannot execute in each other's spans. Calibrate negatives within
    // the mounted scope, including body negatives for every payload bank.
    let router_scope_rows: BTreeSet<usize> = harvester.groups.values().flatten().copied().collect();

    for (idx, (rule_id, mode)) in UNIMODAL_ORACLE_RULES.iter().enumerate() {
        let pos_idx = harvester.groups.get(*rule_id).cloned().unwrap_or_default();
        if pos_idx.is_empty() {
            bail!(
                "unimodal oracle rule {} collected 0 positive calibration rows",
                rule_id
            );
        }
        let pos_set: BTreeSet<usize> = pos_idx.iter().copied().collect();
        let neg_idx: Vec<usize> = router_scope_rows
            .iter()
            .copied()
            .filter(|i| !pos_set.contains(i))
            .collect();
        rule_positive_counts.insert((*rule_id).to_string(), pos_idx.len());
        obligations.push(PipeSlotObligationSpec {
            readout_contrast: None,
            obligation_id: (*rule_id).to_string(),
            bank_name: CartridgeBank::OrientTrunk.wire_key().to_string(),
            layer: resolved_layer,
            slot_lo: base_intermediate_size + 2 * idx,
            slot_hi: base_intermediate_size + 2 * idx + 2,
            positive_prompt_indices: pos_idx,
            negative_prompt_indices: neg_idx,
            background_prompt_indices: harvester.router_background_rows.clone(),
            v_target: Vec::new(),
            theta_rad: 0.95,
            gate_target: 3.5,
            gate_neg_target: -8.5,
            expected_mode: Some(*mode),
        });
    }

    let mut payload_positive_counts = BTreeMap::new();
    let mut payload_tokens = BTreeSet::new();
    let mut next_slot_by_mode: BTreeMap<Mode, usize> = BTreeMap::new();
    for ((mode, boundary, target), rows) in &harvester.payload_groups {
        let obligation_id = format!(
            "payload_{}_{boundary}_{target}",
            mode.as_str().to_lowercase()
        );
        let positive: BTreeSet<usize> = rows.iter().map(|(idx, _)| *idx).collect();
        let suppress: Vec<u32> = rows
            .iter()
            .map(|(_, tid)| *tid)
            .filter(|tid| tid != target)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        payload_tokens.insert(*target);
        payload_tokens.extend(&suppress);
        let slot_offset = next_slot_by_mode.entry(*mode).or_insert(32);
        if *slot_offset + 2 > config.widened_slots {
            bail!("payload obligations for {mode:?} exceed widened slot capacity");
        }
        payload_positive_counts.insert(obligation_id.clone(), positive.len());
        obligations.push(PipeSlotObligationSpec {
            readout_contrast: Some(TokenReadoutContrastSpec {
                promote_token_ids: vec![*target],
                suppress_token_ids: suppress,
                suppress_weight: 0.6,
            }),
            obligation_id,
            bank_name: CartridgeBank::ModePayload(*mode).wire_key().to_string(),
            layer: resolved_layer,
            slot_lo: base_intermediate_size + *slot_offset,
            slot_hi: base_intermediate_size + *slot_offset + 2,
            positive_prompt_indices: positive.iter().copied().collect(),
            negative_prompt_indices: harvester.payload_scope_rows[mode]
                .iter()
                .copied()
                .filter(|idx| !positive.contains(idx))
                .collect(),
            background_prompt_indices: Vec::new(),
            v_target: Vec::new(),
            theta_rad: 0.35,
            gate_target: 3.5,
            gate_neg_target: -8.5,
            expected_mode: Some(*mode),
        });
        *slot_offset += 2;
    }
    eprintln!(
        "[oracle-capture] states={total_rows} obligations={} extracting payload geometry",
        obligations.len()
    );
    // Request only the natural vocabulary rows actually observed at compiled structural payload boundaries.
    let (_, _, weight_geometry) = harvester
        .worker
        .extract_compiler_weight_geometry_with_tokens(
            Some(resolved_layer),
            payload_tokens.into_iter().collect(),
        )
        .context("failed extracting payload token readout geometry")?;

    let layer_key = resolved_layer.to_string();
    let mut activations = BTreeMap::new();
    activations.insert(layer_key.clone(), harvester.pre_rows);
    let mut residuals_map = BTreeMap::new();
    residuals_map.insert(layer_key, harvester.res_rows);

    let compile_input = CompileCartridgeInputSpec {
        status: "activations".to_string(),
        activations,
        residuals: Some(residuals_map),
        row_states: None,
        obligations: Some(obligations),
        mode_target_vectors: None,
        payload_target_vectors: None,
        required_payload_modes: Mode::ALL.to_vec(),
        capture_row_banks: harvester.row_banks,
        model_family: Some(model_family.clone()),
        checkpoint_revision,
        base_intermediate_size: Some(base_intermediate_size),
        widened_slots: Some(config.widened_slots),
        compiler_weight_geometry: Some(weight_geometry),
        background_activations: None,
        steward_columns: Vec::new(),
        lock_columns: Vec::new(),
        schema_disclosed: false,
        allow_synthetic_fixture_targets: false,
    };

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create parent directory {:?} for oracle capture bundle",
                    parent
                )
            })?;
        }
    }
    let serialized = serde_json::to_string(&compile_input)
        .context("failed to serialize CompileCartridgeInputSpec")?;
    std::fs::write(output_path, serialized)
        .with_context(|| format!("failed writing oracle capture bundle to {output_path:?}"))?;

    Ok(OracleCaptureHarvestReceipt {
        status: "ORACLE_CAPTURE_OK".to_string(),
        model_family,
        target_layer: resolved_layer,
        base_intermediate_size,
        widened_slots: config.widened_slots,
        external_domains_harvested: domain_count,
        train_episodes_harvested: train_coords.len(),
        total_captured_states: total_rows,
        rule_positive_counts,
        payload_positive_counts,
        output_path: output_path.to_path_buf(),
    })
}

#[cfg(test)]
mod capture_contract_tests {
    use super::*;

    #[test]
    fn structural_payload_boundary_filter_excludes_openers_values_and_trailing_close_subtokens() {
        for mode in Mode::ALL {
            assert!(is_compiled_structural_payload_boundary(
                mode,
                &CaptureBoundary::CloseTag { subtoken_index: 0 }
            ));
            for sub_idx in 1..=4 {
                assert!(!is_compiled_structural_payload_boundary(
                    mode,
                    &CaptureBoundary::CloseTag {
                        subtoken_index: sub_idx
                    }
                ));
            }
            assert!(!is_compiled_structural_payload_boundary(
                mode,
                &CaptureBoundary::Opener
            ));
            assert!(!is_compiled_structural_payload_boundary(
                mode,
                &CaptureBoundary::Value
            ));
            assert!(!is_compiled_structural_payload_boundary(
                mode,
                &CaptureBoundary::Background
            ));
            assert_eq!(
                is_compiled_structural_payload_boundary(mode, &CaptureBoundary::Field),
                mode == Mode::Ask
            );
        }
    }

    #[test]
    fn terminal_calibration_prose_matches_the_bounded_wire_contract() {
        for text in [
            "0",
            "7",
            "Done.",
            "First line.\nSecond line.\nThird line.",
            "One\nTwo\nThree\nFour",
            "éé\n答复",
        ] {
            validate_terminal_prose(text).unwrap();
        }
        validate_terminal_prose(&"é".repeat(400)).unwrap();
        for text in [
            "",
            " \t",
            "First.\n\nLast.",
            "First.\n \t",
            "{\"summary\":\"done\"}",
            "  {json}",
            "Use <tag>.",
            "First.\r\nNext.",
        ] {
            assert!(validate_terminal_prose(text).is_err(), "accepted {text:?}");
        }
        assert_eq!(
            OracleCaptureHarvestConfig::default().model_family,
            "Qwen/Qwen3.5-0.8B-Base"
        );
    }

    #[test]
    fn r6_calibration_labels_match_real_pending_steward_state() {
        let assert_pending = |driver: &LiveSqliteDailyDriver| {
            assert_eq!(driver.steward_auth(), StewardAuthState::PendingAsk);
            let legal = driver.legal_modes().unwrap();
            assert!(legal.contains(&Mode::Ask));
            assert!(!legal.contains(&Mode::Write));
        };
        for domain in &EXTERNAL_CALIBRATION_DOMAINS {
            for (projected, warm) in [(false, false), (true, false), (false, true)] {
                let mut driver = open_external_domain_driver(domain).unwrap();
                driver.mark_all_tables_probed().unwrap();
                if warm {
                    let update = format!(
                        "UPDATE {} SET {} = {} WHERE {} = {};",
                        domain.main_table, domain.val_col, domain.v1_new, domain.pk_col, domain.pk1
                    );
                    execute_verified_driver_step(&mut driver, Mode::Write, &update, None).unwrap();
                    driver.begin_new_task();
                }
                let fields = if projected {
                    format!(
                        "{}, {}, {}, {}, {}, policy_decision, {}, {}",
                        domain.pk_col,
                        domain.slug_col,
                        domain.status_col,
                        domain.val_col,
                        domain.rule_col,
                        domain.owner_col,
                        domain.freeze_col
                    )
                } else {
                    "*".into()
                };
                let query = format!(
                    "SELECT {fields} FROM {} WHERE {} = {};",
                    domain.main_table, domain.pk_col, domain.pk2
                );
                let (read_obs, read_wire) =
                    execute_verified_driver_step(&mut driver, Mode::Read, &query, None).unwrap();
                assert_eq!(read_obs.steward_auth, StewardAuthState::PendingAsk);
                assert!(read_wire.contains("\"steward_auth\":\"pending_ask\""));
                assert_pending(&driver);
                let ask = serde_json::json!({"select":"Authorize this change?", "options":["ALLOW","DENY"]}).to_string();
                execute_verified_driver_step(&mut driver, Mode::Ask, &ask, Some("ALLOW")).unwrap();
                assert_eq!(driver.steward_auth(), StewardAuthState::Granted);
                assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
                driver.begin_new_task();
                execute_verified_driver_step(&mut driver, Mode::Read, &query, None).unwrap();
                assert_pending(&driver);
            }
        }
        for coord in build_nursery_sweep_coordinates(12, NurseryAppFilter::SqlSteward).iter() {
            let vocabulary = coord.build_sql_steward_vocabulary();
            let table = vocabulary.primary_table.as_str();
            let slug = deterministic_sql_target_slug(coord);
            let balance = deterministic_sql_requested_balance(coord) + 550;
            for policy in ["ALLOW", "ASK"] {
                let mut driver =
                    open_train_episode_driver(table, Some((&slug, balance, policy, "CLEAR")))
                        .unwrap();
                for query in [
                    "SELECT name FROM sqlite_master WHERE type = 'table';".to_string(),
                    format!("PRAGMA table_info({table});"),
                ] {
                    execute_verified_driver_step(&mut driver, Mode::Probe, &query, None).unwrap();
                }
                let query = format!("SELECT * FROM {table} WHERE entity_slug = '{slug}';");
                let (observation, wire) =
                    execute_verified_driver_step(&mut driver, Mode::Read, &query, None).unwrap();
                if policy == "ASK" {
                    assert_eq!(observation.steward_auth, StewardAuthState::PendingAsk);
                    assert!(wire.contains("\"steward_auth\":\"pending_ask\""));
                    assert_pending(&driver);
                } else {
                    assert_eq!(observation.steward_auth, StewardAuthState::NotRequired);
                    assert!(wire.contains("\"steward_auth\":\"not_required\""));
                    assert_eq!(driver.steward_auth(), StewardAuthState::NotRequired);
                    assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
                }
            }
        }
    }

    #[test]
    fn rollback_calibration_uses_real_sqlite_and_keeps_the_pending_permission() {
        for domain in &EXTERNAL_CALIBRATION_DOMAINS {
            for reply in ["ALLOW", "DENY"] {
                let mut driver = open_external_domain_driver(domain).unwrap();
                driver.mark_all_tables_probed().unwrap();
                let before = driver.database_snapshot().unwrap();
                let version = driver.db_version();
                let update = format!(
                    "UPDATE {} SET {} = {} WHERE {} = {};",
                    domain.main_table, domain.val_col, domain.v2_new, domain.pk_col, domain.pk2
                );
                let (rejection, wire) =
                    execute_expected_steward_rollback(&mut driver, &update).unwrap();
                assert!(!rejection.ok);
                assert!(wire.contains("StewardAuthorizationViolation"));
                assert!(wire.contains("\"steward_auth\":\"pending_ask\""));
                assert_eq!(driver.db_version(), version);
                let ask = serde_json::json!({"select": "Authorize this change?", "options": ["ALLOW", "DENY"]}).to_string();
                execute_verified_driver_step(&mut driver, Mode::Ask, &ask, Some(reply)).unwrap();
                if reply == "ALLOW" {
                    assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
                    let (write, _) =
                        execute_verified_driver_step(&mut driver, Mode::Write, &update, None)
                            .unwrap();
                    assert_eq!(write.rows_affected, 1);
                    assert!(driver.db_version() > version);
                } else {
                    assert!(!driver.legal_modes().unwrap().contains(&Mode::Write));
                    assert_eq!(driver.database_snapshot().unwrap(), before);
                }
            }
        }
    }
}
