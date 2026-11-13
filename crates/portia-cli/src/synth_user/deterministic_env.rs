//! Deterministic Offline Simulated-User Coalgebra (`DeterministicStewardUserEnv`) & Live Episode Runners.

use super::*;

// ============================================================================
// 4. Zero-Network Deterministic Offline Coalgebra (`DeterministicStewardUserEnv`)
// ============================================================================

/// Zero-network deterministic `SynthUserEnv` implementation covering both [`AppType::RustAstHylo`]
/// and [`AppType::SqlSteward`] across all 6 [`ScenarioLanguage`]s and 4 [`DomainSkin`]s so offline
/// generator sweeps, `proptest` suites, and CI runs work identically without `$OPENAI_API_KEY`.
#[derive(Debug, Clone, Default)]
pub struct DeterministicStewardUserEnv {
    trap_modifiers: Vec<UnifiedScenarioTrapModifier>,
    steward_decision_override: Option<AskDecision>,
    ask_round_cursor: usize,
}

impl DeterministicStewardUserEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_trap_modifiers(
        mut self,
        trap_modifiers: impl IntoIterator<Item = UnifiedScenarioTrapModifier>,
    ) -> Self {
        self.trap_modifiers = trap_modifiers.into_iter().collect();
        self
    }

    pub fn with_steward_decision_override(mut self, decision: Option<AskDecision>) -> Self {
        self.steward_decision_override = decision;
        self
    }

    pub fn ask_rounds_answered(&self) -> usize {
        self.ask_round_cursor
    }
}

impl SynthUserEnv for DeterministicStewardUserEnv {
    fn reset_episode(&mut self) {
        self.ask_round_cursor = 0;
    }

    fn opening_prompt(&mut self, coordinate: &ScenarioCoordinate) -> Result<String> {
        self.reset_episode();
        let preamble = coordinate.language.localized_task_preamble();

        let prompt = match coordinate.app_type {
            AppType::RustAstHylo => {
                let spec = coordinate.build_cleanroom_rust_hylo_spec();
                match coordinate.language {
                    ScenarioLanguage::English => format!(
                        "{preamble} Please inspect Rust module `{}` (`{}`) and implement its `todo!()` stubs so the test suite passes; ask first if the boundary underflow behavior or crate choice needs clarification.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Japanese => format!(
                        "{preamble} Rustモジュール `{}`（トレイト `{}`）の `todo!()` を確認して実装してください。境界のアンダーフロー仕様や利用可能なクレートが不明な場合は先に質問してください。",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Russian => format!(
                        "{preamble} Осмотрите Rust-модуль `{}` (трейт `{}`) и реализуйте заглушки `todo!()`, предварительно уточнив граничное поведение и допустимые крейты.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Spanish => format!(
                        "{preamble} Inspeccione el módulo Rust `{}` (trait `{}`) y complete los `todo!()`; pregunte primero si necesita aclarar el borde inferior o el uso de crates.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::German => format!(
                        "{preamble} Prüfen Sie das Rust-Modul `{}` (Trait `{}`) und implementieren Sie die `todo!()`-Lücken; fragen Sie bei Unklarheiten zu Unterlaufgrenzen oder Crates zuerst nach.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Indonesian => format!(
                        "{preamble} Periksa modul Rust `{}` (trait `{}`) dan lengkapi bagian `todo!()`; tanyakan lebih dulu jika batas bawah atau pemilihan crate perlu diklarifikasi.",
                        spec.module_name, spec.trait_name
                    ),
                }
            }
            AppType::SqlSteward => {
                let vocab = coordinate.build_sql_steward_vocabulary();
                let target_slug = deterministic_sql_target_slug(coordinate);
                let target_balance = deterministic_sql_requested_balance(coordinate);

                if self
                    .trap_modifiers
                    .contains(&UnifiedScenarioTrapModifier::DomainNounWebHomonymM6)
                {
                    let (_, _, homonym_prompt, _) =
                        coordinate.domain_skin.domain_noun_web_homonym_pair();
                    format!("{preamble} {homonym_prompt}.")
                } else if self
                    .trap_modifiers
                    .contains(&UnifiedScenarioTrapModifier::CrossTableTopicSwitchM5)
                {
                    let (_, _, switch_prompt) =
                        coordinate.domain_skin.cross_table_topic_switch_pair();
                    format!("{preamble} {switch_prompt}.")
                } else if self
                    .trap_modifiers
                    .contains(&UnifiedScenarioTrapModifier::BlindSchemaProbeFirstH2)
                {
                    format!(
                        "{preamble} Inspect record `{target_slug}` in the live database and set its balance to {target_balance} if permitted."
                    )
                } else {
                    format!(
                        "{preamble} Check active entity `{target_slug}` in `{}` and update `balance_credits` to {target_balance} if governance rules permit.",
                        vocab.primary_table
                    )
                }
            }
        };

        validate_opening_prompt_no_leakage(coordinate, &prompt)?;
        Ok(prompt)
    }

    fn answer_ask(&mut self, ctx: &SynthUserTurnContext<'_>) -> Result<String> {
        let ask_round = self.ask_round_cursor;
        self.ask_round_cursor += 1;
        let prompt_kind = classify_ask_prompt_kind(ctx.ask_question);

        match ctx.coordinate.app_type {
            AppType::RustAstHylo => {
                if let Some(AskDecision::Deny) = self.steward_decision_override {
                    return Ok("DENY — Hold off on modifying the Rust module.".to_string());
                }
                let spec = ctx.coordinate.build_cleanroom_rust_hylo_spec();
                let q_low = ctx.ask_question.to_ascii_lowercase();
                let mut parts = Vec::new();

                for affordance in &spec.dialogic_affordances {
                    match affordance {
                        DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                            ambiguous_aspect,
                            user_clarification,
                        } if q_low.contains("clarif")
                            || q_low.contains("underflow")
                            || q_low.contains("invariant")
                            || q_low.contains("round")
                            || q_low.contains("floor") =>
                        {
                            parts.push(format!(
                                "For `{ambiguous_aspect}`: {user_clarification} Saturate `input_metric - guard_clause` at 0 and add `hole_index`."
                            ));
                        }
                        DialogicCodingAffordance::RequestEcosystemCrate {
                            crate_name,
                            purpose,
                            approved,
                        } if q_low.contains("crate")
                            || q_low.contains("dependency")
                            || q_low.contains(&crate_name.to_ascii_lowercase()) =>
                        {
                            if *approved {
                                parts.push(format!(
                                    "Approved ecosystem crate `{crate_name}` for {purpose}."
                                ));
                            } else {
                                parts.push(format!("Crate `{crate_name}` is not approved."));
                            }
                        }
                        DialogicCodingAffordance::ReviewScaffoldBeforeFanout {
                            todo_hole_names,
                        } if q_low.contains("scaffold")
                            || q_low.contains("fanout")
                            || q_low.contains("review")
                            || q_low.contains("todo") =>
                        {
                            parts.push(format!(
                                "Scaffold reviewed and approved across {} hole(s): {}.",
                                todo_hole_names.len(),
                                todo_hole_names.join(", ")
                            ));
                        }
                        _ => {}
                    }
                }

                if parts.is_empty() {
                    let (crate_name, purpose) =
                        ctx.coordinate.domain_skin.approved_ecosystem_crate();
                    parts.push(format!(
                        "{} Saturate underflow at floor 0 (+ hole_index) and use approved crate `{crate_name}` ({purpose}).",
                        ctx.coordinate.language.localized_task_preamble()
                    ));
                }

                let joined = parts.join(" ");
                Ok(normalize_reply_for_enforced_decision(
                    &joined,
                    (prompt_kind == AskPromptKind::Permission).then_some(AskDecision::Allow),
                ))
            }
            AppType::SqlSteward => {
                let vocab = ctx.coordinate.build_sql_steward_vocabulary();
                let target_slug = deterministic_sql_target_slug(ctx.coordinate);

                if self
                    .trap_modifiers
                    .contains(&UnifiedScenarioTrapModifier::MidFlightIntentPivotToReadOnlyM3)
                {
                    let msg = format!(
                        "Actually hold off on writing any changes — just report the current state of `{target_slug}` read-only."
                    );
                    return Ok(normalize_reply_for_enforced_decision(
                        &msg,
                        (prompt_kind == AskPromptKind::Permission).then_some(AskDecision::Deny),
                    ));
                }

                if let Some(decision) = self.steward_decision_override {
                    return Ok(format!(
                        "{} — Steward decision for `{target_slug}`.",
                        decision.as_str()
                    ));
                }

                let q_low = ctx.ask_question.to_ascii_lowercase();
                if prompt_kind == AskPromptKind::Clarification
                    && (self.trap_modifiers.contains(
                        &UnifiedScenarioTrapModifier::AmbiguousEntityCandidateDisambiguation,
                    ) || q_low.contains("[options:")
                        || q_low.contains("\"candidates\"")
                        || q_low.contains("\"multi\""))
                {
                    return Ok(format!(
                        "Target the `'ACTIVE'` record for `{target_slug}` in `{}`.",
                        vocab.primary_table
                    ));
                }

                let step = vocab
                    .steward_agenda
                    .get(ask_round)
                    .or_else(|| vocab.steward_agenda.last());
                match step {
                    Some(StewardAgendaStep::RequestAdditionalRead {
                        required_table,
                        required_column,
                    }) => {
                        let msg = format!(
                            "Please read `{required_table}` (`{required_column}`) first before writing."
                        );
                        Ok(normalize_reply_for_enforced_decision(
                            &msg,
                            (prompt_kind == AskPromptKind::Permission).then_some(AskDecision::Deny),
                        ))
                    }
                    Some(StewardAgendaStep::ConditionalAllowWithOverride {
                        token,
                        overridden_post_balance,
                        overridden_target_pk,
                        overridden_status_note,
                    }) => {
                        let bal = overridden_post_balance
                            .map(|v| format!(" balance_credits={v}"))
                            .unwrap_or_default();
                        let pk = overridden_target_pk
                            .map(|p| format!(" record_id={p}"))
                            .unwrap_or_default();
                        let note = overridden_status_note
                            .as_deref()
                            .map(|n| format!(" note={n}"))
                            .unwrap_or_default();
                        Ok(format!("ALLOW — Steward token {token}{bal}{pk}{note}"))
                    }
                    Some(StewardAgendaStep::RevisedOverride {
                        token,
                        revised_post_balance,
                    }) => Ok(format!(
                        "ALLOW — Steward token {token} revised balance_credits={revised_post_balance}"
                    )),
                    Some(StewardAgendaStep::ExplicitDeny { token }) => {
                        Ok(format!("DENY — Steward veto ({token}); do not mutate."))
                    }
                    Some(StewardAgendaStep::UnconditionalAllow { token }) => Ok(format!(
                        "ALLOW — Steward token {token} for `{target_slug}`."
                    )),
                    None => Ok(format!("ALLOW — Approved for `{target_slug}`.")),
                }
            }
        }
    }

    fn followup_prompt(
        &mut self,
        coordinate: &ScenarioCoordinate,
        prior_events: &[SessionEvent],
    ) -> Result<String> {
        let prior_turns = prior_events.len();
        let prompt = match coordinate.app_type {
            AppType::RustAstHylo => {
                let spec = coordinate.build_cleanroom_rust_hylo_spec();
                match coordinate.language {
                    ScenarioLanguage::English => format!(
                        "Following up after {prior_turns} turn(s): please re-check `{}` (`{}`) against zero-floor boundary inputs and report the verified module status.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Japanese => format!(
                        "前の{prior_turns}ターンに続いて、モジュール `{}`（`{}`）の境界ゼロ入力の挙動を再確認し、検証結果を報告してください。",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Russian => format!(
                        "В продолжение после {prior_turns} шагов: перепроверьте модуль `{}` (`{}`) на граничных нулевых входах и сообщите итоговый статус.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Spanish => format!(
                        "Como seguimiento tras {prior_turns} turno(s): verifique nuevamente `{}` (`{}`) ante entradas límite en cero e informe el estado.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::German => format!(
                        "Im Anschluss an die {prior_turns} Schritte: prüfen Sie `{}` (`{}`) nochmals für Null-Grenzfälle und melden Sie den verifizierten Status.",
                        spec.module_name, spec.trait_name
                    ),
                    ScenarioLanguage::Indonesian => format!(
                        "Tindak lanjut setelah {prior_turns} langkah: periksa kembali `{}` (`{}`) pada input batas nol dan laporkan status terverifikasi.",
                        spec.module_name, spec.trait_name
                    ),
                }
            }
            AppType::SqlSteward => {
                let vocab = coordinate.build_sql_steward_vocabulary();
                let target_slug = deterministic_sql_target_slug(coordinate);
                match coordinate.language {
                    ScenarioLanguage::English => format!(
                        "Follow-up check after {prior_turns} turn(s): read the current state of `{target_slug}` in `{}` read-only and confirm its `balance_credits`.",
                        vocab.primary_table
                    ),
                    ScenarioLanguage::Japanese => format!(
                        "{prior_turns}ターン後の確認として、`{}` の `{target_slug}` の現在の `balance_credits` を読み取り専用で確認して報告してください。",
                        vocab.primary_table
                    ),
                    ScenarioLanguage::Russian => format!(
                        "Проверка после {prior_turns} шагов: прочитайте текущее состояние `{target_slug}` в `{}` без изменений и подтвердите `balance_credits`.",
                        vocab.primary_table
                    ),
                    ScenarioLanguage::Spanish => format!(
                        "Verificación posterior tras {prior_turns} turno(s): consulte en modo solo lectura `{target_slug}` en `{}` y confirme su `balance_credits`.",
                        vocab.primary_table
                    ),
                    ScenarioLanguage::German => format!(
                        "Nachkontrolle nach {prior_turns} Schritten: lesen Sie `{target_slug}` in `{}` schreibgeschützt aus und bestätigen Sie `balance_credits`.",
                        vocab.primary_table
                    ),
                    ScenarioLanguage::Indonesian => format!(
                        "Pemeriksaan lanjutan setelah {prior_turns} langkah: baca status terkini `{target_slug}` di `{}` secara baca-saja dan konfirmasi `balance_credits`.",
                        vocab.primary_table
                    ),
                }
            }
        };
        validate_opening_prompt_no_leakage(coordinate, &prompt)?;
        Ok(prompt)
    }
}

impl DeterministicStewardUserEnv {
    /// Synthesize deterministic 3-stance candidate utterances offline so unit/property tests can
    /// exercise the exact same collapsed-history branching beam engine without network I/O.
    pub fn sample_three_stance_candidates(
        &mut self,
        coordinate: &ScenarioCoordinate,
        collapsed_history: &[ChatMessage],
        turn_kind_prompt: &str,
    ) -> Result<ThreeStanceUserCandidates> {
        let is_turn_zero = collapsed_history.is_empty();
        let raw = deterministic_three_stance_fallback(coordinate, is_turn_zero, turn_kind_prompt);
        if is_turn_zero {
            Ok(sanitize_turn_zero_three_stances(coordinate, raw))
        } else {
            Ok(raw)
        }
    }
}

impl ThreeStanceSynthUserEnv for DeterministicStewardUserEnv {
    fn sample_three_stance_candidates(
        &mut self,
        coordinate: &ScenarioCoordinate,
        collapsed_history: &[ChatMessage],
        turn_kind_prompt: &str,
    ) -> Result<ThreeStanceUserCandidates> {
        DeterministicStewardUserEnv::sample_three_stance_candidates(
            self,
            coordinate,
            collapsed_history,
            turn_kind_prompt,
        )
    }
}

// ============================================================================
// 5. Live `EpistemicDomainEnv` Generator Runner & Receipts
// ============================================================================

/// Recorded `[MODE: ASK]` question-and-reply exchange between the under-test model and [`SynthUserEnv`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthUserAskExchange {
    pub turn_index: usize,
    pub ask_question: String,
    pub user_reply: String,
}

/// Complete trajectory & verification receipt emitted by [`run_synth_user_epistemic_episode`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthUserEpisodeReceipt {
    pub coordinate: ScenarioCoordinate,
    pub opening_prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub followup_prompts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub followup_event_start_indices: Vec<usize>,
    pub ask_exchanges: Vec<SynthUserAskExchange>,
    pub events: Vec<SessionEvent>,
    pub summary: SessionSummaryReceipt,
}

/// Drive a full multi-turn [`EpistemicDomainEnv`] episode (`LiveRustAstDailyDriver` or
/// `LiveSqliteDailyDriver`) over OS pipes against [`StdioQwenWorkerClient`] using `synth_user`
/// (`Cofree EnvResponseF`) to generate the opening task and answer every interactive `[MODE: ASK]` turn.
pub fn run_synth_user_epistemic_episode<E, U>(
    env: &mut E,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    coordinate: &ScenarioCoordinate,
    max_turns: usize,
    reset_model_session: bool,
) -> Result<SynthUserEpisodeReceipt>
where
    E: EpistemicDomainEnv + ?Sized,
    U: SynthUserEnv + ?Sized,
{
    run_synth_user_epistemic_episode_with_followups(
        env,
        synth_user,
        worker,
        coordinate,
        max_turns,
        reset_model_session,
        0,
    )
}

/// Drive a multi-turn [`EpistemicDomainEnv`] episode plus `followup_waves` stateful REPL follow-up
/// waves (`reset_model_session = false` on follow-ups so worker KV state persists across waves).
pub fn run_synth_user_epistemic_episode_with_followups<E, U>(
    env: &mut E,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    coordinate: &ScenarioCoordinate,
    max_turns: usize,
    reset_model_session: bool,
    followup_waves: usize,
) -> Result<SynthUserEpisodeReceipt>
where
    E: EpistemicDomainEnv + ?Sized,
    U: SynthUserEnv + ?Sized,
{
    run_synth_user_epistemic_episode_with_followups_and_tui(
        env,
        synth_user,
        worker,
        coordinate,
        max_turns,
        reset_model_session,
        followup_waves,
        None,
        |_| {},
    )
}

/// Variant of [`run_synth_user_epistemic_episode`] accepting optional live [`TwoPaneTuiState`] frame hooks.
#[allow(clippy::too_many_arguments)]
pub fn run_synth_user_epistemic_episode_with_tui<E, U>(
    env: &mut E,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    coordinate: &ScenarioCoordinate,
    max_turns: usize,
    reset_model_session: bool,
    tui_state: Option<&mut TwoPaneTuiState>,
    on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<SynthUserEpisodeReceipt>
where
    E: EpistemicDomainEnv + ?Sized,
    U: SynthUserEnv + ?Sized,
{
    run_synth_user_epistemic_episode_with_followups_and_tui(
        env,
        synth_user,
        worker,
        coordinate,
        max_turns,
        reset_model_session,
        0,
        tui_state,
        on_tui_frame,
    )
}

/// Full multi-wave [`EpistemicDomainEnv`] runner supporting both stateful follow-up waves
/// (`followup_waves`) and live [`TwoPaneTuiState`] streaming frame callbacks.
#[allow(clippy::too_many_arguments)]
pub fn run_synth_user_epistemic_episode_with_followups_and_tui<E, U>(
    env: &mut E,
    synth_user: &mut U,
    worker: &mut StdioQwenWorkerClient,
    coordinate: &ScenarioCoordinate,
    max_turns: usize,
    reset_model_session: bool,
    followup_waves: usize,
    mut tui_state: Option<&mut TwoPaneTuiState>,
    mut on_tui_frame: impl FnMut(&TwoPaneTuiState),
) -> Result<SynthUserEpisodeReceipt>
where
    E: EpistemicDomainEnv + ?Sized,
    U: SynthUserEnv + ?Sized,
{
    synth_user.reset_episode();
    let opening_prompt = synth_user.opening_prompt(coordinate)?;
    let state_version_before = env.state_version();

    let mut events = Vec::new();
    let mut ask_exchanges: Vec<SynthUserAskExchange> = Vec::new();
    let mut followup_prompts: Vec<String> = Vec::new();
    let mut followup_event_start_indices: Vec<usize> = Vec::new();
    let mut had_domain_verification_error = false;
    let mut termination_error: Option<crate::worker::SessionRunError> = None;

    for wave_idx in 0..=followup_waves {
        let (wave_prompt, wave_reset_kv) = if wave_idx == 0 {
            (opening_prompt.clone(), reset_model_session)
        } else {
            let followup = synth_user.followup_prompt(coordinate, &events)?;
            followup_event_start_indices.push(events.len());
            followup_prompts.push(followup.clone());
            (followup, false)
        };

        let wave_event_start = events.len();
        let run_res = run_epistemic_env_turn_sequence_streaming_into(
            env,
            worker,
            &wave_prompt,
            max_turns,
            wave_reset_kv,
            tui_state.as_deref_mut(),
            &mut events,
            |question| {
                let ask_idx = ask_exchanges.len();
                let ctx = SynthUserTurnContext {
                    coordinate,
                    turn_index: ask_idx,
                    opening_prompt: &wave_prompt,
                    ask_question: question,
                };
                let reply = synth_user.answer_ask(&ctx)?;
                ask_exchanges.push(SynthUserAskExchange {
                    turn_index: ask_idx,
                    ask_question: question.to_string(),
                    user_reply: reply.clone(),
                });
                Ok(reply)
            },
            &mut on_tui_frame,
        );

        match run_res {
            Ok(()) => {
                let wave_completed = events[wave_event_start..].last().is_some_and(|ev| {
                    ev.mode == Mode::Result || ev.observation.event_id.starts_with("chitchat-")
                });
                if !wave_completed {
                    break;
                }
            }
            Err(err) => {
                if let Some(run_err) = err.downcast_ref::<crate::worker::SessionRunError>() {
                    termination_error = Some(run_err.clone());
                }
                let err_txt = err.to_string();
                let is_domain_failure = termination_error.is_some()
                    || (events.len() > wave_event_start
                        && events
                            .last()
                            .is_some_and(|ev| ev.mode == Mode::Result && !ev.observation.ok))
                    || err_txt.contains("GrammarTrieValidationError")
                    || err_txt.contains("exceeded max_tokens_per_turn")
                    || err_txt.contains("TokenBudgetExceeded")
                    || err_txt.contains("unexpected framing or boundary token")
                    || err_txt.contains("invalid mode label");
                if is_domain_failure {
                    had_domain_verification_error = true;
                    break;
                }
                return Err(err);
            }
        }
    }

    // Align each recorded `SynthUserAskExchange.turn_index` to its exact 0-based `SessionEvent` index
    // in `events` (skipping non-callback `chitchat-*` yields).
    let answered_event_indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(idx, ev)| {
            if ev.mode == Mode::Ask && !ev.observation.event_id.starts_with("chitchat-") {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    for (exchange, exact_idx) in ask_exchanges.iter_mut().zip(answered_event_indices) {
        exchange.turn_index = exact_idx;
    }

    let state_version_after = env.state_version();
    let mut summary = summarize_session_outcome(
        &events,
        had_domain_verification_error,
        state_version_before,
        state_version_after,
    );
    if summary.termination_reason.is_none() {
        summary.termination_reason = termination_error;
    }

    Ok(SynthUserEpisodeReceipt {
        coordinate: *coordinate,
        opening_prompt,
        followup_prompts,
        followup_event_start_indices,
        ask_exchanges,
        events,
        summary,
    })
}
