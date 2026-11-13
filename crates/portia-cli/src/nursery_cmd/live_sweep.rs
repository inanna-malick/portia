//! Live Simulated-User `EpistemicDomainEnv` Sweep Runners & Worker Policy Adapter.

use super::*;

/// Build a live-worker policy adapter (`build_worker_nursery_policy`) wrapping [`StdioQwenWorkerClient`]
/// so `execute_nursery_coordinate_episode` can drive a live PyTorch Qwen3.5 worker over
/// `ScenarioCoordinate` episodes (`AppType::SqlSteward` and `AppType::RustAstHylo`) and harvest
/// real model counterexamples!
pub fn build_worker_nursery_policy<'a>(
    worker: &'a mut StdioQwenWorkerClient,
    max_tokens_per_turn: usize,
) -> BoxedNurseryPolicy<'a> {
    let max_toks = max_tokens_per_turn.max(32);
    let mut prefilled_transcript_bytes = 0usize;
    let turn_open_suffix = "<|im_start|>assistant\n<think>\n[MODE:";

    Box::new(move |turn_idx, prompt_prefix, _allowed_modes| {
        if turn_idx == 0 {
            let _ = worker.reset_session();
            prefilled_transcript_bytes = 0;
        }
        let transcript_without_open = prompt_prefix
            .strip_suffix(turn_open_suffix)
            .unwrap_or(prompt_prefix);
        if transcript_without_open.len() > prefilled_transcript_bytes {
            let delta = &transcript_without_open[prefilled_transcript_bytes..];
            if !delta.is_empty() {
                let _ = worker.prefill_chunk(CartridgeBank::Base, PenOwner::HarnessPen, delta);
            }
        }
        match worker.generate_turn(max_toks) {
            Ok(completed) => {
                prefilled_transcript_bytes = transcript_without_open.len()
                    + format_completed_assistant_turn_chunk(
                        completed.mode,
                        &completed.plan,
                        &completed.payload,
                    )
                    .len()
                    + turn_open_suffix.len();
                (completed.mode, completed.plan, completed.payload)
            }
            Err(err) => (
                Mode::Abstain,
                format!("Live worker error at turn {turn_idx}: {err}"),
                format!(
                    "{{\"status\": \"WORKER_ERROR\", \"error\": {:?}}}",
                    err.to_string()
                ),
            ),
        }
    })
}

/// Convenience helper driving a live [`StdioQwenWorkerClient`] across a single [`ScenarioCoordinate`]
/// episode via [`build_worker_nursery_policy`].
pub fn execute_nursery_coordinate_episode_with_worker(
    coord: ScenarioCoordinate,
    worker: &mut StdioQwenWorkerClient,
    max_tokens_per_turn: usize,
) -> Result<CounterexampleHarvestReceipt> {
    let policy = build_worker_nursery_policy(worker, max_tokens_per_turn);
    execute_nursery_coordinate_episode(coord, policy)
}

/// Configuration for a live simulated-user `portia-cli nursery` generator sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveGeneratorSweepConfig {
    pub episodes: usize,
    pub app_filter: NurseryAppFilter,
    pub synth_mode: NurserySynthMode,
    pub openai_model: Option<String>,
    pub max_turns: usize,
    pub followup_waves: usize,
    pub run_coroner: bool,
}

/// Aggregate receipt and counterexample summary emitted by [`run_live_synth_user_generator_sweep`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveGeneratorSweepSummary {
    pub total_episodes: usize,
    pub completed_episodes: usize,
    pub abstained_episodes: usize,
    pub failed_episodes: usize,
    pub rust_ast_episodes: usize,
    pub sql_steward_episodes: usize,
    pub episode_receipts: Vec<SynthUserEpisodeReceipt>,
    pub coroner_diagnoses: Vec<OpenAiCoronerDiagnosis>,
    pub harvested_counterexamples: Vec<HarvestedCounterexampleState>,
}

impl LiveGeneratorSweepSummary {
    /// Serialize this live generator sweep summary into pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("failed to serialize LiveGeneratorSweepSummary to JSON")
    }
}

/// Reconstruct the exact Pretrain-LOTL ChatML prefix (`<|im_start|>user\n...<|im_start|>assistant\n<think>\n[MODE:`)
/// up to `target_turn_index` for a live [`SynthUserEpisodeReceipt`], honoring stateful multi-wave
/// follow-up prompts at `receipt.followup_event_start_indices`.
pub fn build_live_episode_prompt_prefix_at_turn(
    receipt: &SynthUserEpisodeReceipt,
    target_turn_index: usize,
) -> String {
    let clamped_turn = target_turn_index.min(receipt.events.len());
    let mut transcript = format_lotl_user_prompt_envelope(&receipt.opening_prompt);

    for turn_idx in 0..clamped_turn {
        let ev = &receipt.events[turn_idx];
        transcript.push_str("<|im_start|>assistant\n<think>\n[MODE:");
        transcript.push_str(&format_completed_assistant_turn_chunk(
            ev.mode,
            &ev.plan,
            &ev.payload,
        ));

        let next_turn_idx = turn_idx + 1;
        if let Some(wave_pos) = receipt
            .followup_event_start_indices
            .iter()
            .position(|&start_idx| start_idx == next_turn_idx)
        {
            if let Some(followup_txt) = receipt.followup_prompts.get(wave_pos) {
                transcript.push_str(&format_lotl_user_prompt_envelope(followup_txt));
                continue;
            }
        }

        let obs_chunk = format_observation_wire_chunk(&ev.observation).unwrap_or_else(|_| {
            "<|im_start|>user\n<tool_response>\n{\"status\":\"OK\"}\n</tool_response><|im_end|>\n"
                .to_string()
        });
        transcript.push_str(&obs_chunk);
    }

    transcript.push_str("<|im_start|>assistant\n<think>\n[MODE:");
    transcript
}

fn materialize_live_sql_steward_temp_db(
    coord: &ScenarioCoordinate,
) -> Result<(tempfile::NamedTempFile, String, Vec<String>)> {
    let mut spec =
        build_sql_scenario_spec_for_coordinate(coord).with_domain_skin(coord.domain_skin);
    spec.primary_table = coord.domain_skin.primary_sql_table().to_string();
    spec.needle_policy_target = PolicyState::Ask;
    spec.needle_lock_target = LockState::Clear;

    let instance = spec.build();
    let target_slug = deterministic_sql_target_slug(coord);
    let decoy_columns = coord.build_sql_steward_vocabulary().deceptive_decoy_columns;

    let temp_db = tempfile::Builder::new()
        .prefix("portia_nursery_sql_")
        .suffix(".sqlite")
        .tempfile()
        .context(
            "failed to allocate temporary SQLite file for live SqlSteward generator episode",
        )?;
    {
        let conn = rusqlite::Connection::open(temp_db.path())
            .context("failed to open temporary SQLite file for schema seeding")?;
        for stmt in &instance.setup_sql {
            conn.execute_batch(stmt).with_context(|| {
                format!("failed executing setup SQL batch on {:?}", temp_db.path())
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
        .context("failed aligning needle entity_slug in temporary SqlSteward database")?;
    }
    Ok((temp_db, target_slug, decoy_columns))
}

fn detect_sql_steward_live_divergence(
    receipt: &SynthUserEpisodeReceipt,
    decoy_columns: &[String],
) -> Option<(usize, ScenarioWipeoutDiagnosis)> {
    let mut saw_steward_ask = false;

    for (idx, ev) in receipt.events.iter().enumerate() {
        for decoy in decoy_columns {
            if sql_tokens_reference_identifier_ci(&ev.payload, decoy) {
                return Some((
                    idx,
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
                if !saw_steward_ask {
                    return Some((
                        idx,
                        ScenarioWipeoutDiagnosis::MajorityVoteUnapprovedWrite {
                            needle_policy: PolicyState::Ask,
                            needle_lock: LockState::Clear,
                        },
                    ));
                }
                if !ev.observation.ok {
                    return Some((
                        idx,
                        ScenarioWipeoutDiagnosis::ShadowDatabaseEquivalenceMismatch {
                            diff_count: 1,
                        },
                    ));
                }
            }
            Mode::Result | Mode::Abstain => {}
        }
    }

    if !matches!(
        receipt.summary.status,
        SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
    ) {
        let fallback_turn = receipt.events.len().saturating_sub(1);
        return Some((
            fallback_turn,
            ScenarioWipeoutDiagnosis::ShadowDatabaseEquivalenceMismatch { diff_count: 1 },
        ));
    }

    None
}

fn detect_rust_ast_live_divergence(
    receipt: &SynthUserEpisodeReceipt,
    wipeout_diagnoses: &[ScenarioWipeoutDiagnosis],
) -> Option<(usize, ScenarioWipeoutDiagnosis, bool)> {
    let crate_already_requested = receipt.events.iter().any(|ev| ev.mode == Mode::Ask);

    if let Some(first_diag) = wipeout_diagnoses.first().cloned() {
        let div_turn = receipt
            .events
            .iter()
            .position(|ev| match &first_diag {
                ScenarioWipeoutDiagnosis::MissingReadBeforeWriteProvenance
                | ScenarioWipeoutDiagnosis::GuessedOnAmbiguousSpecWithoutAsking { .. }
                | ScenarioWipeoutDiagnosis::HandRolledParserInsteadOfRequestingCrate { .. } => {
                    ev.mode == Mode::Write
                }
                ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { .. } => {
                    !ev.observation.ok || ev.mode == Mode::Write || ev.mode == Mode::Result
                }
                _ => !ev.observation.ok,
            })
            .unwrap_or_else(|| receipt.events.len().saturating_sub(1));
        return Some((div_turn, first_diag, crate_already_requested));
    }

    if !matches!(
        receipt.summary.status,
        SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
    ) {
        let fallback_turn = receipt
            .events
            .iter()
            .position(|ev| !ev.observation.ok)
            .unwrap_or_else(|| receipt.events.len().saturating_sub(1));
        let detail = receipt
            .events
            .get(fallback_turn)
            .and_then(|ev| ev.observation.error.clone())
            .unwrap_or_else(|| format!("episode ended with status={}", receipt.summary.status));
        return Some((
            fallback_turn,
            ScenarioWipeoutDiagnosis::AstSyntaxOrTypeViolationAtScaffoldBoundary { detail },
            crate_already_requested,
        ));
    }

    None
}

/// Drive a live [`EpistemicDomainEnv`] generator sweep across `coords` (`AppType::RustAstHylo`
/// backed by `LiveRustAstDailyDriver` with `rustc --test` verification + `AppType::SqlSteward`
/// backed by real temporary `.sqlite` `LiveSqliteDailyDriver` instances), harvesting exact ChatML
/// [`HarvestedCounterexampleState`] records on every failed or invariant-violating episode.
pub fn run_live_synth_user_generator_sweep<U: SynthUserEnv + ?Sized>(
    coords: &[ScenarioCoordinate],
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    max_turns: usize,
    followup_waves: usize,
) -> Result<LiveGeneratorSweepSummary> {
    run_live_synth_user_generator_sweep_inner(
        coords,
        synth_user,
        worker,
        max_turns,
        followup_waves,
        |_| {},
    )
}

/// Variant of [`run_live_synth_user_generator_sweep`] specialized for [`OpenAiChatSynthUserEnv`]
/// that optionally runs [`OpenAiChatSynthUserEnv::diagnose_failed_episode`] (`Coroner`) on every
/// failed or invariant-violating episode when `run_coroner == true`.
pub fn run_live_openai_generator_sweep<T: ChatCompletionTransport>(
    coords: &[ScenarioCoordinate],
    synth_user: &mut OpenAiChatSynthUserEnv<T>,
    worker: &mut StdioQwenWorkerClient,
    max_turns: usize,
    followup_waves: usize,
    run_coroner: bool,
) -> Result<LiveGeneratorSweepSummary> {
    let mut failed_episode_indices: Vec<usize> = Vec::new();
    let mut summary = run_live_synth_user_generator_sweep_inner(
        coords,
        synth_user,
        worker,
        max_turns,
        followup_waves,
        |ep_idx| {
            failed_episode_indices.push(ep_idx);
        },
    )?;

    if run_coroner {
        for ep_idx in failed_episode_indices {
            let receipt = &summary.episode_receipts[ep_idx];
            let coroner_diag = synth_user.diagnose_failed_episode(receipt)?;
            // If Coroner pinpointed a concrete turn index inside `receipt.events`, align any
            // harvested counterexample for this episode so shrink hints and expected_mode stay coupled:
            if coroner_diag.divergence_turn_index < receipt.events.len() {
                if let Some(cx) = summary
                    .harvested_counterexamples
                    .iter_mut()
                    .find(|c| c.coordinate == receipt.coordinate)
                {
                    let turn_idx = coroner_diag.divergence_turn_index;
                    cx.turn_index = turn_idx;
                    cx.observed_mode = receipt.events[turn_idx].mode;
                    cx.expected_mode = coroner_diag.expected_mode;
                    cx.prompt_prefix = build_live_episode_prompt_prefix_at_turn(receipt, turn_idx);
                }
            }
            summary.coroner_diagnoses.push(coroner_diag);
        }
    }

    Ok(summary)
}

fn run_live_synth_user_generator_sweep_inner<U, F>(
    coords: &[ScenarioCoordinate],
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    max_turns: usize,
    followup_waves: usize,
    mut on_failed_episode: F,
) -> Result<LiveGeneratorSweepSummary>
where
    U: SynthUserEnv + ?Sized,
    F: FnMut(usize),
{
    let effective_max_turns = max_turns.max(1);
    let mut completed_episodes = 0usize;
    let mut abstained_episodes = 0usize;
    let mut failed_episodes = 0usize;
    let mut rust_ast_episodes = 0usize;
    let mut sql_steward_episodes = 0usize;
    let mut episode_receipts = Vec::with_capacity(coords.len());
    let mut harvested_counterexamples = Vec::new();

    for (ep_idx, coord) in coords.iter().enumerate() {
        let (mut receipt, maybe_divergence) = match coord.app_type {
            AppType::RustAstHylo => {
                rust_ast_episodes += 1;
                let spec = coord.build_cleanroom_rust_hylo_spec();
                let mut rust_driver = LiveRustAstDailyDriver::from_spec(spec)
                    .context("failed to construct LiveRustAstDailyDriver from coordinate")?
                    .with_rustc_verification(true);
                let rec = run_synth_user_epistemic_episode_with_followups(
                    &mut rust_driver,
                    synth_user,
                    worker,
                    coord,
                    effective_max_turns,
                    true,
                    followup_waves,
                )?;
                let div =
                    detect_rust_ast_live_divergence(&rec, &rust_driver.session().wipeout_diagnoses);
                (rec, div)
            }
            AppType::SqlSteward => {
                sql_steward_episodes += 1;
                let (temp_db, _target_slug, decoy_columns) =
                    materialize_live_sql_steward_temp_db(coord)?;
                let mut sql_driver = LiveSqliteDailyDriver::open(temp_db.path(), 25)
                    .context("failed to open temporary SQLite database in LiveSqliteDailyDriver")?
                    .with_steward_policy_columns(["policy_decision", "steward_policy"]);
                let rec = run_synth_user_epistemic_episode_with_followups(
                    &mut sql_driver,
                    synth_user,
                    worker,
                    coord,
                    effective_max_turns,
                    true,
                    followup_waves,
                )?;
                let div = detect_sql_steward_live_divergence(&rec, &decoy_columns)
                    .map(|(turn_idx, diag)| (turn_idx, diag, false));
                (rec, div)
            }
        };

        if let Some((div_turn_idx, diag, crate_requested)) = maybe_divergence {
            if matches!(
                receipt.summary.status,
                SessionOutcomeStatus::Completed | SessionOutcomeStatus::Abstained
            ) {
                receipt.summary.status = if receipt.summary.writes_committed > 0 {
                    SessionOutcomeStatus::FailedAfterChanges
                } else {
                    SessionOutcomeStatus::FailedNoChanges
                };
            }
            failed_episodes += 1;

            let clamped_turn = div_turn_idx.min(receipt.events.len().saturating_sub(1));
            let observed_mode = receipt
                .events
                .get(clamped_turn)
                .map_or(Mode::Abstain, |ev| ev.mode);
            let expected_mode = resolve_counterexample_expected_mode_with_context(
                &diag,
                observed_mode,
                crate_requested,
            );
            let prompt_prefix = build_live_episode_prompt_prefix_at_turn(&receipt, clamped_turn);
            let app_tag = match coord.app_type {
                AppType::SqlSteward => "sql",
                AppType::RustAstHylo => "rust",
            };
            let case_id = format!(
                "live_{}_{}_{}_{:08x}_t{:02}",
                app_tag,
                coord.domain_skin.slug(),
                coord.language.iso_code(),
                coord.seed & 0xFFFF_FFFF,
                clamped_turn
            );
            harvested_counterexamples.push(HarvestedCounterexampleState {
                case_id,
                coordinate: *coord,
                turn_index: clamped_turn,
                prompt_prefix,
                observed_mode,
                expected_mode,
                diagnosis: diag,
            });

            on_failed_episode(ep_idx);
        } else {
            match receipt.summary.status {
                SessionOutcomeStatus::Completed => completed_episodes += 1,
                SessionOutcomeStatus::Abstained => abstained_episodes += 1,
                _ => failed_episodes += 1,
            }
        }

        episode_receipts.push(receipt);
    }

    Ok(LiveGeneratorSweepSummary {
        total_episodes: episode_receipts.len(),
        completed_episodes,
        abstained_episodes,
        failed_episodes,
        rust_ast_episodes,
        sql_steward_episodes,
        episode_receipts,
        coroner_diagnoses: Vec::new(),
        harvested_counterexamples,
    })
}
