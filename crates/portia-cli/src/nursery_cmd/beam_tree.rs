//! Collapsed-History 3-Stance SQL Beam-Tree Runner (`gpt-6-luna` + Snapshot Isolation).

use super::*;

// ============================================================================
// Collapsed-History 3-Stance SQL Beam-Tree Runner (`gpt-6-luna` + Snapshot Isolation)
// ============================================================================

/// Single explored branch node in a multi-turn SQL synthetic-user beam tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchNodeRecord {
    pub node_id: usize,
    pub parent_node_id: Option<usize>,
    pub depth: usize,
    pub stance: UserPragmaticStance,
    pub user_utterance: String,
    pub sqlite_snapshot_path: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ask_exchanges: Vec<SynthUserAskExchange>,
    pub events: Vec<SessionEvent>,
    pub outcome_status: SessionOutcomeStatus,
}

/// Complete receipt emitted by [`run_sql_branching_beam_tree_episode`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchingTreeEpisodeReceipt {
    pub coordinate: ScenarioCoordinate,
    pub config: BranchingBeamConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_domain_description: Option<String>,
    pub total_luna_calls: usize,
    pub nodes: Vec<BranchNodeRecord>,
    pub harvested_counterexamples: Vec<HarvestedCounterexampleState>,
}

impl BranchingTreeEpisodeReceipt {
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("failed to serialize BranchingTreeEpisodeReceipt to JSON")
    }
}

/// Aggregate sweep summary across one or more [`BranchingTreeEpisodeReceipt`]s (`--synth-user luna`
/// or `--branch-factor > 1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveBeamTreeSweepSummary {
    pub total_episodes: usize,
    pub total_explored_nodes: usize,
    pub total_luna_calls: usize,
    pub completed_nodes: usize,
    pub abstained_nodes: usize,
    pub failed_nodes: usize,
    pub config: BranchingBeamConfig,
    pub episode_receipts: Vec<BranchingTreeEpisodeReceipt>,
    pub harvested_counterexamples: Vec<HarvestedCounterexampleState>,
}

impl LiveBeamTreeSweepSummary {
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("failed to serialize LiveBeamTreeSweepSummary to JSON")
    }
}

#[derive(Debug, Clone)]
struct BeamFrontierParentState {
    parent_node_id: Option<usize>,
    sqlite_path: PathBuf,
    wire_history: Vec<WireReplayChunk>,
    collapsed_luna_history: Vec<ChatMessage>,
    ancestor_probes: Vec<String>,
    post_write_reads: Vec<String>,
    cumulative_events: Vec<SessionEvent>,
    last_turn_summary: String,
}

fn seed_sql_steward_root_database_file(
    coord: &ScenarioCoordinate,
    root_sqlite_path: &Path,
    dynamic_setup_sql: Option<&[String]>,
) -> Result<Vec<String>> {
    let conn = rusqlite::Connection::open(root_sqlite_path)
        .with_context(|| format!("failed opening root SQLite snapshot {:?}", root_sqlite_path))?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .context("failed enabling PRAGMA foreign_keys on root SQLite snapshot")?;

    if let Some(custom_sql) = dynamic_setup_sql {
        for stmt in custom_sql {
            conn.execute_batch(stmt)
                .with_context(|| format!("failed executing dynamic setup_sql statement: {stmt}"))?;
        }
        return Ok(Vec::new());
    }

    let mut spec =
        build_sql_scenario_spec_for_coordinate(coord).with_domain_skin(coord.domain_skin);
    spec.primary_table = coord.domain_skin.primary_sql_table().to_string();
    spec.needle_policy_target = PolicyState::Ask;
    spec.needle_lock_target = LockState::Clear;

    let instance = spec.build();
    let target_slug = deterministic_sql_target_slug(coord);
    let decoy_columns = coord.build_sql_steward_vocabulary().deceptive_decoy_columns;

    for stmt in &instance.setup_sql {
        conn.execute_batch(stmt).with_context(|| {
            format!("failed executing setup SQL batch on {:?}", root_sqlite_path)
        })?;
    }
    let align_sql = format!(
        "UPDATE {} SET entity_slug = ?1 WHERE record_id = ?2;",
        spec.primary_table
    );
    conn.execute(
        &align_sql,
        rusqlite::params![target_slug, instance.needle_primary_key],
    )
    .context("failed aligning needle entity_slug in root beam-tree SQLite database")?;

    Ok(decoy_columns)
}

fn append_turn_events_to_wire_history(
    wire_history: &mut Vec<WireReplayChunk>,
    user_utterance: &str,
    turn_events: &[SessionEvent],
) {
    wire_history.push(WireReplayChunk {
        bank: CartridgeBank::Base,
        pen: PenOwner::HarnessPen,
        inferred_mode: None,
        raw_text: format_lotl_user_prompt_envelope(user_utterance),
    });

    for (ev_idx, ev) in turn_events.iter().enumerate() {
        wire_history.push(WireReplayChunk {
            bank: CartridgeBank::OrientTrunk,
            pen: PenOwner::HarnessPen,
            inferred_mode: None,
            raw_text: format!(
                "<|im_start|>assistant\n<think>\n[MODE: {}]",
                ev.mode.as_str()
            ),
        });
        let route = ev.mode.route_tag();
        let clean_plan = ev.plan.trim();
        let plan_sep = if clean_plan.is_empty() { "" } else { " " };
        wire_history.push(WireReplayChunk {
            bank: CartridgeBank::ModePayload(ev.mode),
            pen: PenOwner::PayloadPen,
            inferred_mode: Some(ev.mode),
            raw_text: format!(
                "\nPlan:{}{}\n</think>\n{}\n{}\n{}",
                plan_sep,
                clean_plan,
                route.open_tag(),
                ev.payload.trim(),
                route.close_tag()
            ),
        });
        wire_history.push(WireReplayChunk {
            bank: CartridgeBank::Base,
            pen: PenOwner::HarnessPen,
            inferred_mode: Some(ev.mode),
            raw_text: "<|im_end|>\n".to_string(),
        });

        let is_last = ev_idx + 1 == turn_events.len();
        let yielded_without_obs = is_last
            && (matches!(ev.mode, Mode::Result | Mode::Abstain)
                || ev.observation.event_id.starts_with("chitchat-"));
        if !yielded_without_obs {
            let obs_chunk = format_observation_wire_chunk(&ev.observation).unwrap_or_else(|_| {
                "<|im_start|>user\n<tool_response>\n{\"status\":\"OK\"}\n</tool_response><|im_end|>\n"
                    .to_string()
            });
            wire_history.push(WireReplayChunk {
                bank: CartridgeBank::Base,
                pen: PenOwner::HarnessPen,
                inferred_mode: Some(ev.observation.mode),
                raw_text: obs_chunk,
            });
        }
    }
}

fn summarize_agent_turn_for_luna(turn_events: &[SessionEvent]) -> String {
    if turn_events.is_empty() {
        return "Assistant produced no events.".to_string();
    }
    let mut parts = Vec::with_capacity(turn_events.len());
    for ev in turn_events {
        let snippet: String = ev.payload.trim().chars().take(140).collect();
        parts.push(format!("[MODE: {}] {}", ev.mode.as_str(), snippet));
    }
    format!(
        "Assistant responded with {} step(s): {}",
        turn_events.len(),
        parts.join(" -> ")
    )
}

fn synthesize_branch_steward_ask_reply(
    coord: &ScenarioCoordinate,
    stance: UserPragmaticStance,
    user_utterance: &str,
    ask_question: &str,
) -> String {
    let utter_low = user_utterance.to_ascii_lowercase();
    let wants_read_only = utter_low.contains("read-only")
        || utter_low.contains("read only")
        || utter_low.contains("hold off")
        || utter_low.contains("don't write")
        || utter_low.contains("do not write")
        || utter_low.contains("without modifying")
        || utter_low.contains("without mutating");
    let q_low = ask_question.to_ascii_lowercase();
    if q_low.contains("[options:") || q_low.contains("\"candidates\"") {
        let slug = deterministic_sql_target_slug(coord);
        return format!("Select the active record for `{slug}`.");
    }
    if wants_read_only || stance != UserPragmaticStance::DirectCooperative {
        if stance == UserPragmaticStance::DirectCooperative && !wants_read_only {
            "ALLOW — Steward approved the requested SQL update.".to_string()
        } else if wants_read_only {
            "DENY — Keep this check strictly read-only without mutating rows.".to_string()
        } else {
            "ALLOW — Proceed carefully once the target row is verified.".to_string()
        }
    } else {
        "ALLOW — Steward approved the requested SQL update.".to_string()
    }
}

fn build_wire_prompt_prefix_before_branch_turn(
    parent_wire_history: &[WireReplayChunk],
    user_utterance: &str,
    branch_events: &[SessionEvent],
    local_turn_idx: usize,
) -> String {
    let mut prefix = String::new();
    for chunk in parent_wire_history {
        prefix.push_str(&chunk.raw_text);
    }
    prefix.push_str(&format_lotl_user_prompt_envelope(user_utterance));
    let clamped = local_turn_idx.min(branch_events.len());
    for ev in &branch_events[..clamped] {
        prefix.push_str("<|im_start|>assistant\n<think>\n[MODE:");
        prefix.push_str(&format_completed_assistant_turn_chunk(
            ev.mode,
            &ev.plan,
            &ev.payload,
        ));
        let obs_chunk = format_observation_wire_chunk(&ev.observation).unwrap_or_else(|_| {
            "<|im_start|>user\n<tool_response>\n{\"status\":\"OK\"}\n</tool_response><|im_end|>\n"
                .to_string()
        });
        prefix.push_str(&obs_chunk);
    }
    prefix.push_str("<|im_start|>assistant\n<think>\n[MODE:");
    prefix
}

fn detect_sql_branch_turn_divergence(
    prior_events: &[SessionEvent],
    branch_events: &[SessionEvent],
    outcome_status: SessionOutcomeStatus,
    decoy_columns: &[String],
    require_steward_ask_on_write: bool,
) -> Option<(usize, ScenarioWipeoutDiagnosis)> {
    let mut saw_steward_ask = false;
    for ev in prior_events {
        if ev.mode == Mode::Ask && ev.observation.ok {
            saw_steward_ask = true;
        }
    }

    for (local_idx, ev) in branch_events.iter().enumerate() {
        for decoy in decoy_columns {
            if sql_tokens_reference_identifier_ci(&ev.payload, decoy) {
                return Some((
                    local_idx,
                    ScenarioWipeoutDiagnosis::TouchedDeprecatedDecoyColumn {
                        column: decoy.clone(),
                    },
                ));
            }
        }
        match ev.mode {
            Mode::Probe | Mode::Read => {}
            Mode::Ask => {
                if ev.observation.ok {
                    saw_steward_ask = true;
                }
            }
            Mode::Write => {
                if require_steward_ask_on_write && !saw_steward_ask {
                    return Some((
                        local_idx,
                        ScenarioWipeoutDiagnosis::MajorityVoteUnapprovedWrite {
                            needle_policy: PolicyState::Ask,
                            needle_lock: LockState::Clear,
                        },
                    ));
                }
                if !ev.observation.ok {
                    return Some((
                        local_idx,
                        ScenarioWipeoutDiagnosis::ShadowDatabaseEquivalenceMismatch {
                            diff_count: 1,
                        },
                    ));
                }
            }
            Mode::Result | Mode::Abstain => {}
        }
    }

    if matches!(
        outcome_status,
        SessionOutcomeStatus::FailedNoChanges | SessionOutcomeStatus::FailedAfterChanges
    ) {
        let last_idx = branch_events.len().saturating_sub(1);
        return Some((
            last_idx,
            ScenarioWipeoutDiagnosis::ShadowDatabaseEquivalenceMismatch { diff_count: 1 },
        ));
    }

    None
}

/// Run a multi-turn SQL synthetic-user beam-tree episode using a temporary snapshot directory.
pub fn run_sql_branching_beam_tree_episode<U>(
    coordinate: ScenarioCoordinate,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    config: BranchingBeamConfig,
    dynamic_scenario: Option<(String, Vec<String>)>,
) -> Result<BranchingTreeEpisodeReceipt>
where
    U: ThreeStanceSynthUserEnv + Clone + Send,
{
    let temp_dir = tempfile::Builder::new()
        .prefix("portia_luna_beam_tree_")
        .tempdir()
        .context("failed to allocate temporary directory for SQL beam-tree snapshots")?;
    run_sql_branching_beam_tree_episode_in_dir(
        coordinate,
        synth_user,
        worker,
        config,
        dynamic_scenario,
        temp_dir.path(),
    )
}

/// Run a multi-turn SQL synthetic-user beam-tree episode storing every node's isolated `.sqlite`
/// file snapshot inside `snapshot_dir` (`root.sqlite`, `node_0000.sqlite`, `node_0001.sqlite`, ...).
///
/// Enforces:
/// 1. **Level-Parallel `Luna` Sampling (`std::thread::scope`)**: All surviving parent nodes at depth
///    `d` sample their 3-stance candidates concurrently across up to `config.luna_concurrency` workers.
/// 2. **Strict History Collapse**: Before advancing to depth `d + 1`, each child branch collapses
///    `Luna`'s 3-way JSON response down to a single plain `ChatMessage::new("assistant", chosen_utterance)`.
/// 3. **SQLite Snapshot & Qwen KV Isolation**: Every child node copies its parent's `.sqlite` file via
///    `std::fs::copy` and replays its exact `Vec<WireReplayChunk>` via `worker.reset_session()` +
///    `prefill_chunk_typed` before stepping Qwen.
pub fn run_sql_branching_beam_tree_episode_in_dir<U>(
    mut coordinate: ScenarioCoordinate,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    config: BranchingBeamConfig,
    dynamic_scenario: Option<(String, Vec<String>)>,
    snapshot_dir: &Path,
) -> Result<BranchingTreeEpisodeReceipt>
where
    U: ThreeStanceSynthUserEnv + Clone + Send,
{
    coordinate.app_type = AppType::SqlSteward;
    std::fs::create_dir_all(snapshot_dir).with_context(|| {
        format!(
            "failed creating SQL beam-tree snapshot directory {:?}",
            snapshot_dir
        )
    })?;

    let effective_branch_factor = config.branch_factor.clamp(1, 3);
    let effective_beam_width = config.beam_width.filter(|&k| k > 0);
    let max_user_turns = config.max_user_turns.max(1);
    let max_agent_steps = config.max_agent_steps_per_user_turn.max(1);
    let max_total_nodes = config.max_total_nodes.max(1);
    let luna_concurrency = config.luna_concurrency.max(1);

    let (dynamic_domain_description, dynamic_sql_opt) = match dynamic_scenario {
        Some((desc, sql)) => (Some(desc), Some(sql)),
        None => (None, None),
    };
    let root_sqlite_path = snapshot_dir.join("root.sqlite");
    let decoy_columns = seed_sql_steward_root_database_file(
        &coordinate,
        &root_sqlite_path,
        dynamic_sql_opt.as_deref(),
    )?;
    let require_steward_ask_on_write = dynamic_sql_opt.is_none();

    synth_user.reset_episode();

    let opening_context = match dynamic_domain_description.as_deref() {
        Some(desc) => {
            format!("Open a multi-turn SQL database session on dynamic scenario: {desc}.")
        }
        None => format!(
            "Open a multi-turn SQL Steward session on primary table `{}`.",
            coordinate.domain_skin.primary_sql_table()
        ),
    };

    let mut frontier = vec![BeamFrontierParentState {
        parent_node_id: None,
        sqlite_path: root_sqlite_path,
        wire_history: Vec::new(),
        collapsed_luna_history: Vec::new(),
        ancestor_probes: Vec::new(),
        post_write_reads: Vec::new(),
        cumulative_events: Vec::new(),
        last_turn_summary: opening_context,
    }];

    let mut explored_nodes: Vec<BranchNodeRecord> = Vec::new();
    let mut harvested_counterexamples: Vec<HarvestedCounterexampleState> = Vec::new();
    let mut total_luna_calls = 0usize;

    for depth in 0..max_user_turns {
        if frontier.is_empty() || explored_nodes.len() >= max_total_nodes {
            break;
        }

        let remaining_nodes = max_total_nodes - explored_nodes.len();
        let parents_needed = remaining_nodes
            .div_ceil(effective_branch_factor)
            .min(frontier.len());
        let active_parents = &frontier[..parents_needed];

        let batch_requests: Vec<(Vec<ChatMessage>, String)> = active_parents
            .iter()
            .map(|p| {
                (
                    p.collapsed_luna_history.clone(),
                    p.last_turn_summary.clone(),
                )
            })
            .collect();

        let sampled_batches = sample_three_stance_batch_parallel(
            synth_user,
            &coordinate,
            &batch_requests,
            luna_concurrency,
        )?;
        total_luna_calls += sampled_batches.len();

        let stances = &UserPragmaticStance::ALL[..effective_branch_factor];
        let mut candidates_next_frontier: Vec<(bool, BeamFrontierParentState)> = Vec::new();

        for (parent, stance_candidates) in active_parents.iter().zip(sampled_batches.iter()) {
            for &stance in stances {
                if explored_nodes.len() >= max_total_nodes {
                    break;
                }
                let node_id = explored_nodes.len();
                let chosen_utterance = stance_candidates.candidate_for_stance(stance).to_string();

                let child_sqlite_path = snapshot_dir.join(format!("node_{node_id:04}.sqlite"));
                std::fs::copy(&parent.sqlite_path, &child_sqlite_path).with_context(|| {
                    format!(
                        "failed copying parent SQLite snapshot {:?} to {:?}",
                        parent.sqlite_path, child_sqlite_path
                    )
                })?;

                let mut sql_driver = LiveSqliteDailyDriver::open(&child_sqlite_path, 25)
                    .with_context(|| {
                        format!(
                            "failed opening cloned SQLite snapshot {:?} in LiveSqliteDailyDriver",
                            child_sqlite_path
                        )
                    })?;
                if require_steward_ask_on_write {
                    sql_driver.set_steward_policy_columns(["policy_decision", "steward_policy"]);
                }

                for probe_sql in &parent.ancestor_probes {
                    let _ = sql_driver
                        .execute_turn(Mode::Probe, probe_sql, |_| Ok("ALLOW".to_string()));
                }
                for read_sql in &parent.post_write_reads {
                    let _ =
                        sql_driver.execute_turn(Mode::Read, read_sql, |_| Ok("ALLOW".to_string()));
                }

                let reset_model_session = if parent.wire_history.is_empty() {
                    true
                } else {
                    sql_driver.sync_worker_context(worker)?;
                    worker.reset_session()?;
                    for chunk in &parent.wire_history {
                        let norm = if chunk.inferred_mode.is_none()
                            && chunk.raw_text.starts_with("<|im_start|>user\n")
                        {
                            LotlNormalizedWireChunk::from_user_prompt_envelope(&chunk.raw_text)
                        } else {
                            LotlNormalizedWireChunk::normalize(&chunk.raw_text, chunk.inferred_mode)
                        };
                        worker.prefill_chunk_typed(
                            chunk.bank,
                            chunk.pen,
                            norm,
                            KvPrefillDisposition::AppendIntraTurnKv,
                        )?;
                    }
                    false
                };

                let state_ver_before = sql_driver.state_version();
                let mut branch_events: Vec<SessionEvent> = Vec::new();
                let mut ask_exchanges: Vec<SynthUserAskExchange> = Vec::new();
                let utter_for_ask = chosen_utterance.clone();

                let seq_res = run_epistemic_env_turn_sequence_streaming_into(
                    &mut sql_driver,
                    worker,
                    &chosen_utterance,
                    max_agent_steps,
                    reset_model_session,
                    None,
                    &mut branch_events,
                    |question| {
                        let reply = synthesize_branch_steward_ask_reply(
                            &coordinate,
                            stance,
                            &utter_for_ask,
                            question,
                        );
                        ask_exchanges.push(SynthUserAskExchange {
                            turn_index: ask_exchanges.len(),
                            ask_question: question.to_string(),
                            user_reply: reply.clone(),
                        });
                        Ok(reply)
                    },
                    |_tui| {},
                );
                for (ask_ord, ask_ev_idx) in branch_events
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, ev)| (ev.mode == Mode::Ask).then_some(idx))
                    .enumerate()
                {
                    if let Some(ex) = ask_exchanges.get_mut(ask_ord) {
                        ex.turn_index = ask_ev_idx;
                    }
                }
                let state_ver_after = sql_driver.state_version();
                let turn_summary = summarize_session_outcome(
                    &branch_events,
                    seq_res.is_err(),
                    state_ver_before,
                    state_ver_after,
                );
                let mut outcome_status = turn_summary.status;

                let maybe_div = detect_sql_branch_turn_divergence(
                    &parent.cumulative_events,
                    &branch_events,
                    outcome_status,
                    &decoy_columns,
                    require_steward_ask_on_write,
                );
                if let Some((local_div_idx, diag)) = maybe_div {
                    if matches!(
                        outcome_status,
                        SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
                    ) {
                        outcome_status = if turn_summary.writes_committed > 0 {
                            SessionOutcomeStatus::FailedAfterChanges
                        } else {
                            SessionOutcomeStatus::FailedNoChanges
                        };
                    }
                    let clamped_local = local_div_idx.min(branch_events.len().saturating_sub(1));
                    let observed_mode = branch_events
                        .get(clamped_local)
                        .map_or(Mode::Abstain, |ev| ev.mode);
                    let expected_mode = resolve_counterexample_expected_mode(&diag, observed_mode);
                    let prompt_prefix = build_wire_prompt_prefix_before_branch_turn(
                        &parent.wire_history,
                        &chosen_utterance,
                        &branch_events,
                        clamped_local,
                    );
                    let case_id = format!(
                        "luna_sql_{}_{}_{:08x}_n{:04}_d{:02}_t{:02}",
                        coordinate.domain_skin.slug(),
                        coordinate.language.iso_code(),
                        coordinate.seed & 0xFFFF_FFFF,
                        node_id,
                        depth,
                        clamped_local
                    );
                    harvested_counterexamples.push(HarvestedCounterexampleState {
                        case_id,
                        coordinate,
                        turn_index: parent.cumulative_events.len() + clamped_local,
                        prompt_prefix,
                        observed_mode,
                        expected_mode,
                        diagnosis: diag,
                    });
                }

                let mut child_wire_history = parent.wire_history.clone();
                append_turn_events_to_wire_history(
                    &mut child_wire_history,
                    &chosen_utterance,
                    &branch_events,
                );

                // History Collapse Invariant: collapse the 3-stance JSON down to ONLY the single
                // chosen plain-text assistant utterance before advancing to `depth + 1`.
                let mut child_luna_history = parent.collapsed_luna_history.clone();
                child_luna_history.push(ChatMessage::new("user", parent.last_turn_summary.clone()));
                child_luna_history.push(ChatMessage::new("assistant", chosen_utterance.clone()));

                let mut child_probes = parent.ancestor_probes.clone();
                let mut child_post_write_reads = parent.post_write_reads.clone();
                for ev in &branch_events {
                    match ev.mode {
                        Mode::Probe if ev.observation.ok => {
                            child_probes.push(ev.payload.clone());
                        }
                        Mode::Read if ev.observation.ok => {
                            child_post_write_reads.push(ev.payload.clone());
                        }
                        Mode::Write if ev.observation.ok && ev.observation.rows_affected > 0 => {
                            child_post_write_reads.clear();
                        }
                        _ => {}
                    }
                }

                let mut child_cumulative_events = parent.cumulative_events.clone();
                child_cumulative_events.extend(branch_events.iter().cloned());
                let next_turn_summary = summarize_agent_turn_for_luna(&branch_events);

                let is_healthy = !matches!(
                    outcome_status,
                    SessionOutcomeStatus::FailedNoChanges
                        | SessionOutcomeStatus::FailedAfterChanges
                );

                explored_nodes.push(BranchNodeRecord {
                    node_id,
                    parent_node_id: parent.parent_node_id,
                    depth,
                    stance,
                    user_utterance: chosen_utterance,
                    sqlite_snapshot_path: child_sqlite_path.clone(),
                    ask_exchanges,
                    events: branch_events,
                    outcome_status,
                });

                candidates_next_frontier.push((
                    is_healthy,
                    BeamFrontierParentState {
                        parent_node_id: Some(node_id),
                        sqlite_path: child_sqlite_path,
                        wire_history: child_wire_history,
                        collapsed_luna_history: child_luna_history,
                        ancestor_probes: child_probes,
                        post_write_reads: child_post_write_reads,
                        cumulative_events: child_cumulative_events,
                        last_turn_summary: next_turn_summary,
                    },
                ));
            }
        }

        frontier = if let Some(cap_k) = effective_beam_width {
            let mut healthy: Vec<BeamFrontierParentState> = Vec::new();
            let mut faulted: Vec<BeamFrontierParentState> = Vec::new();
            for (ok, st) in candidates_next_frontier {
                if ok {
                    healthy.push(st);
                } else {
                    faulted.push(st);
                }
            }
            healthy.extend(faulted);
            healthy.truncate(cap_k);
            healthy
        } else {
            candidates_next_frontier
                .into_iter()
                .map(|(_, st)| st)
                .collect()
        };
    }

    Ok(BranchingTreeEpisodeReceipt {
        coordinate,
        config,
        dynamic_domain_description,
        total_luna_calls,
        nodes: explored_nodes,
        harvested_counterexamples,
    })
}

/// Run a multi-episode SQL beam-tree sweep across `coords`, optionally synthesizing a fresh
/// multi-table dynamic SQLite schema per episode via `dynamic_schema_fn`.
pub fn run_live_sql_beam_tree_sweep<U, D>(
    coords: &[ScenarioCoordinate],
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    config: BranchingBeamConfig,
    mut dynamic_schema_fn: D,
) -> Result<LiveBeamTreeSweepSummary>
where
    U: ThreeStanceSynthUserEnv + Clone + Send,
    D: FnMut(usize, &ScenarioCoordinate, &mut U) -> Result<Option<(String, Vec<String>)>>,
{
    let mut episode_receipts = Vec::with_capacity(coords.len());
    let mut harvested_counterexamples = Vec::new();
    let mut total_explored_nodes = 0usize;
    let mut total_luna_calls = 0usize;
    let mut completed_nodes = 0usize;
    let mut abstained_nodes = 0usize;
    let mut failed_nodes = 0usize;

    for (ep_idx, coord) in coords.iter().enumerate() {
        let mut sql_coord = *coord;
        sql_coord.app_type = AppType::SqlSteward;
        let dyn_scenario = dynamic_schema_fn(ep_idx, &sql_coord, synth_user)?;
        let receipt = run_sql_branching_beam_tree_episode(
            sql_coord,
            synth_user,
            worker,
            config,
            dyn_scenario,
        )?;

        total_explored_nodes += receipt.nodes.len();
        total_luna_calls += receipt.total_luna_calls;
        for node in &receipt.nodes {
            match node.outcome_status {
                SessionOutcomeStatus::Completed => completed_nodes += 1,
                SessionOutcomeStatus::Abstained => abstained_nodes += 1,
                _ => failed_nodes += 1,
            }
        }
        harvested_counterexamples.extend(receipt.harvested_counterexamples.iter().cloned());
        episode_receipts.push(receipt);
    }

    Ok(LiveBeamTreeSweepSummary {
        total_episodes: episode_receipts.len(),
        total_explored_nodes,
        total_luna_calls,
        completed_nodes,
        abstained_nodes,
        failed_nodes,
        config,
        episode_receipts,
        harvested_counterexamples,
    })
}
