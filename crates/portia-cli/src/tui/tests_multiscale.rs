//! Carbon-tier `TestBackend` verification for multi-scale `Qwen3.5` geometry (`0.8B`, `2B`, `4B`,
//! `9B`, `27B`, `--cartridge none`), on-disk `WireCartridgeManifestJson` slot-span probing, and
//! unicode-width-exact wrapping of `512`-token `Plan:` + multi-paragraph `<to_user>` / CJK prose.

use super::{
    build_left_interaction_lines, build_right_context_window_lines,
    coalesce_cartridge_epochs_for_scale, format_cartridge_bank_badge_for_scale,
    render_two_pane_tui_frame, summarize_loaded_cartridge_rack, ContextSpanKind, InteractionItem,
    SidebarMode, TuiModelScaleSpec, TwoPaneTuiState,
};
use crate::driver::ObservationOperation;
use crate::{DriverObservation, TableSchemaInfo};
use portia_core::bpe::TOKEN_IM_START;
use portia_core::domain::ColumnSchema;
use portia_core::relay::RelayPhase;
use portia_core::{CartridgeBank, Mode, PenOwner, StewardAuthState};
use serde_json::json;
use std::collections::BTreeMap;
use unicode_width::UnicodeWidthStr;

fn minimal_schema() -> Vec<TableSchemaInfo> {
    vec![TableSchemaInfo::new(
        "wallets",
        vec![
            ColumnSchema::with_sqlite_metadata(0, "wallet_id", "INTEGER", true, true),
            ColumnSchema::with_sqlite_metadata(1, "holder", "TEXT", true, false),
        ],
    )
    .with_row_count(8)]
}

fn attach_relay_spans(state: &mut TwoPaneTuiState) {
    state.record_user_prompt_envelope(
        0,
        "",
        "Inspect wallets",
        "<|im_start|>user\nInspect wallets\n",
        vec![TOKEN_IM_START, 872, 198],
    );
    state.push_context_span(
        ContextSpanKind::RelayPenSpan {
            turn_index: 0,
            span_index: 0,
            phase: RelayPhase::ModeLabel,
        },
        PenOwner::RouterPen,
        Some(CartridgeBank::OrientTrunk),
        CartridgeBank::OrientTrunk,
        "READ".to_string(),
        vec![18104],
    );
    state.push_context_span(
        ContextSpanKind::RelayPenSpan {
            turn_index: 0,
            span_index: 1,
            phase: RelayPhase::PayloadBody,
        },
        PenOwner::PayloadPen,
        Some(CartridgeBank::ModePayload(Mode::Read)),
        CartridgeBank::ModePayload(Mode::Read),
        "SELECT wallet_id, holder FROM wallets;".to_string(),
        vec![4102, 991],
    );
}

#[test]
fn multi_scale_header_banner_and_raw_token_trace_eliminate_08b_hardcodings() {
    let cases = [
        (
            "0.8b-base",
            "portia_v3_unified_cartridge_08b_base.json (default)",
            "Qwen/Qwen3.5-0.8B-Base",
            "0.8B-Base",
            24usize,
            1024usize,
            3584usize,
            18usize,
            6usize,
            22usize,
        ),
        (
            "2b-base",
            "portia_v3_unified_cartridge_2b_base.json (default)",
            "Qwen/Qwen3.5-2B-Base",
            "2B-Base",
            24,
            2048,
            6144,
            18,
            6,
            22,
        ),
        (
            "4b-base",
            "portia_v3_unified_cartridge_4b_base.json (default)",
            "Qwen/Qwen3.5-4B-Base",
            "4B-Base",
            32,
            2560,
            9216,
            24,
            8,
            30,
        ),
        (
            "9b-base",
            "portia_v3_unified_cartridge_9b_base.json (default)",
            "Qwen/Qwen3.5-9B-Base",
            "9B-Base",
            32,
            4096,
            12288,
            24,
            8,
            30,
        ),
        (
            "27b",
            "portia_v3_unified_cartridge_27b.json",
            "Qwen/Qwen3.5-27B",
            "27B",
            64,
            5120,
            17408,
            48,
            16,
            62,
        ),
    ];

    for (
        alias,
        cart_label,
        expected_family,
        expected_badge,
        expected_layers,
        expected_d,
        expected_m,
        expected_dn,
        expected_gqa,
        expected_router,
    ) in cases
    {
        let mut state = TwoPaneTuiState::new(
            "/srv/data/treasury.sqlite",
            alias,
            cart_label,
            minimal_schema(),
        );
        let spec = state.model_scale_spec();
        assert_eq!(spec.canonical_family, expected_family);
        assert_eq!(spec.scale_badge, expected_badge);
        assert_eq!(spec.num_layers, expected_layers);
        assert_eq!(spec.hidden_dim, expected_d);
        assert_eq!(spec.base_intermediate_size, expected_m);
        assert_eq!(spec.deltanet_layers, expected_dn);
        assert_eq!(spec.gqa_layers, expected_gqa);
        assert_eq!(spec.router_layer, expected_router);

        // 1. Welcome banner + HeaderStatusBar in default SchemaInspector view:
        let welcome_frame = render_two_pane_tui_frame(&state, 156, 30, false);
        assert!(
            welcome_frame.contains(expected_badge),
            "header missing badge {expected_badge} for {alias}"
        );
        assert!(
            welcome_frame.contains(&format!("(L{expected_router})")),
            "header missing router layer L{expected_router} for {alias}"
        );
        assert!(
            welcome_frame.contains(expected_family),
            "welcome banner missing {expected_family}"
        );

        // 2. RawTokenTrace rack + context window lines after relay turns:
        attach_relay_spans(&mut state);
        state.set_sidebar_mode(SidebarMode::RawTokenTrace);
        let rack = summarize_loaded_cartridge_rack(&state);
        assert_eq!(rack.len(), 3);
        let epochs = coalesce_cartridge_epochs_for_scale(&state.context_spans, &spec);
        assert_eq!(epochs.len(), 3);

        let trace_text: String = build_right_context_window_lines(&state, 84)
            .into_iter()
            .map(|l| l.into_iter().map(|c| c.text).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            rack[0].bank_name.contains(expected_badge)
                && rack[0].physical_slots.contains(&format!(
                    "{expected_m} native SwiGLU + {expected_dn} DeltaNet + {expected_gqa} GQA"
                )),
            "rack[0] missing native SwiGLU scale geometry for {alias}: {:?}",
            rack[0]
        );
        assert!(
            trace_text.contains(&format!(
                "{expected_m} SwiGLU + {expected_dn} DeltaNet + {expected_gqa} GQA"
            )),
            "trace rack missing scale geometry for {alias}:\n{trace_text}"
        );
        assert!(
            trace_text.contains(&format!("L{expected_router}[")),
            "trace rack missing L{expected_router} slot coordinates for {alias}"
        );
        assert!(
            !trace_text.contains("Two-Cartridge Ask Split"),
            "stale Argon Ask Split string leaked into {alias}"
        );

        if expected_m != 3584 {
            assert!(
                !trace_text.contains("3584"),
                "hardcoded 0.8B M=3584 leaked into {alias}:\n{trace_text}"
            );
        }
        if expected_router != 22 {
            assert!(
                !trace_text.contains("L22["),
                "hardcoded 0.8B L22 slot badge leaked into {alias}:\n{trace_text}"
            );
            assert!(
                !trace_text.contains("18 DeltaNet"),
                "hardcoded 18 DeltaNet leaked into {alias}:\n{trace_text}"
            );
        }

        let trace_frame = render_two_pane_tui_frame(&state, 168, 42, false);
        for (row_idx, row) in trace_frame.lines().enumerate() {
            assert_eq!(
                UnicodeWidthStr::width(row),
                168,
                "row {row_idx} width mismatch on {alias}"
            );
        }
    }

    // `--cartridge none` unpatched baseline reports `widened_slots == 0` and `[unpatched W_0]`:
    let unpatched = TuiModelScaleSpec::resolve("4b-base", "none (unpatched baseline)");
    assert_eq!(unpatched.widened_slots, 0);
    assert_eq!(unpatched.format_orient_slots(), "L30[unpatched W_0]");
    assert!(
        format_cartridge_bank_badge_for_scale(CartridgeBank::OrientTrunk, &unpatched)
            .contains("L30[unpatched W_0]")
    );
}

#[test]
fn on_disk_wire_cartridge_manifest_probe_overlays_exact_layer_and_slot_ranges() {
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let manifest_path = tmp_dir
        .path()
        .join("portia_v3_unified_cartridge_4b_base.json");

    let synthetic_manifest = json!({
        "format": "portia-wire-cartridge-v2",
        "header": {
            "schema_version": 2,
            "compiler_version": "0.2.0",
            "model_family": "Qwen/Qwen3.5-4B-Base",
            "hidden_dim": 2560,
            "num_layers": 32,
            "vocab_size": 248320,
            "weight_fingerprint_sha256": "0123456789abcdef"
        },
        "model_id": "Qwen/Qwen3.5-4B-Base",
        "base_intermediate_size": 9216,
        "widened_slots": 128,
        "banks": {
            "OrientTrunk": {
                "30": [
                    { "slot_lo": 9216, "slot_hi": 9230, "w_gate": [[0.0]], "w_up": [[0.0]], "w_down": [[0.0]] },
                    { "slot_lo": 9230, "slot_hi": 9244, "w_gate": [[0.0]], "w_up": [[0.0]], "w_down": [[0.0]] }
                ]
            },
            "ModePayload:Read": {
                "28": [
                    { "slot_lo": 9248, "slot_hi": 9256, "w_gate": [[0.0]], "w_up": [[0.0]], "w_down": [[0.0]] }
                ],
                "30": [
                    { "slot_lo": 9256, "slot_hi": 9288, "w_gate": [[0.0]], "w_up": [[0.0]], "w_down": [[0.0]] }
                ]
            }
        }
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&synthetic_manifest).expect("serialize manifest"),
    )
    .expect("write manifest");

    let matched = TuiModelScaleSpec::resolve("4b-base", &manifest_path.display().to_string());
    assert!(matched.inspected_from_cartridge);
    assert_eq!(matched.router_layer, 30);
    assert_eq!(matched.orient_slot_range, (9216, 9244));
    assert_eq!(
        matched.payload_layer_ranges,
        vec![(28, 9248, 9256), (30, 9256, 9288)]
    );
    assert_eq!(matched.format_orient_slots(), "L30[9216..9244)");
    assert_eq!(
        matched.format_payload_slots(),
        "L28[9248..9256) · L30[9256..9288)"
    );

    // Mismatched model_name (`2b-base` vs `4B-Base` manifest) must NOT overwrite `2B-Base` geometry:
    let mismatched = TuiModelScaleSpec::resolve("2b-base", &manifest_path.display().to_string());
    assert!(!mismatched.inspected_from_cartridge);
    assert_eq!(mismatched.router_layer, 22);
    assert_eq!(mismatched.base_intermediate_size, 6144);
}

#[test]
fn unbounded_512_tok_plan_and_multiparagraph_cjk_prose_wrap_without_truncation_or_clipping() {
    let mut state = TwoPaneTuiState::new(
        "/srv/data/treasury.sqlite",
        "2b-base",
        "portia_v3_unified_cartridge_2b_base.json (default)",
        vec![],
    );

    let multi_sentence_plan = concat!(
        "First inspect `ledger_entries` joined against `campaigns` to verify foreign-key lineage before touching settlement balances. ",
        "Second check whether any matching row carries `policy_decision = 'ASK'` or `lock_state = 'LOCKED'` under Steward governance. ",
        "Third compute exact cent totals grouped by settlement region so reconciliation matches the audit trail. ",
        "Fourth emit the verified summary table with SENTENCE_FOUR_VERIFIED_TAIL."
    );

    let multi_para_result = concat!(
        "<to_user>\n",
        "### Settlement Reconciliation Summary\n\n",
        "Completed multi-region balance verification across all active campaigns without mutating locked rows:\n\n",
        "- 東京-NRT 決済監査ゲートウェイ整合性検証パス完了_CJK_TAIL recorded 450 cents awaiting Steward sign-off.\n",
        "- Artifact digest path `artifacts/sha256_0123456789abcdef0123456789abcdef0123456789abcdef_OVERLONG_TAIL` verified byte-identical.\n\n",
        "All unlocked settlement ledgers balanced cleanly at db v9 with PARA_THREE_VERIFIED_TAIL.\n",
        "</to_user>"
    );

    let multi_para_abstain = concat!(
        "<to_user>\n",
        "Refusing destructive `DROP TABLE ledger_entries` request under active workspace governance.\n\n",
        "Use explicit row-scoped `UPDATE` queries after Steward authorization with ABSTAIN_SECOND_PARAGRAPH_TAIL.\n",
        "</to_user>"
    );

    state.interactions.push(InteractionItem::UserPrompt {
        turn_index: 0,
        schema_header: String::new(),
        user_text: "Run full multi-region settlement audit across 東京-NRT and Chamonix ledgers and report complete findings with USER_PROMPT_TAIL.".to_string(),
    });
    state.interactions.push(InteractionItem::AssistantTurn {
        turn_index: 0,
        mode: Mode::Result,
        route: Mode::Result.route_tag(),
        plan: multi_sentence_plan.to_string(),
        payload: multi_para_result.to_string(),
        single_pen_verified: true,
    });
    state.interactions.push(InteractionItem::AssistantTurn {
        turn_index: 1,
        mode: Mode::Abstain,
        route: Mode::Abstain.route_tag(),
        plan: "Decline schema-destroying DDL and state safe alternative.".to_string(),
        payload: multi_para_abstain.to_string(),
        single_pen_verified: true,
    });
    state.interactions.push(InteractionItem::Observation {
        turn_index: 1,
        observation: DriverObservation {
            event_id: "ev-err".to_string(),
            mode: Mode::Write,
            ok: false,
            db_version: 9,
            table_versions: BTreeMap::new(),
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: 0,
            message: None,
            error: Some(
                "SQLITE_CONSTRAINT_TRIGGER: locked ledger entry 7 rejected write until Steward override token OBS_ERROR_TAIL".to_string(),
            ),
            operation: ObservationOperation::from_wire_str("UPDATE"),
            ask_reply: None,
            steward_auth: StewardAuthState::NotRequired,
            is_chat_continuation: false,
        },
    });
    state.interactions.push(InteractionItem::HarnessNote {
        title: "Model Switched -> 2B-Base".to_string(),
        detail: "Active model: Qwen/Qwen3.5-2B-Base (24L d=2048 M=6144->6272) · Cartridge: portia_v3_unified_cartridge_2b_base.json with HARNESS_NOTE_TAIL".to_string(),
    });

    for width in [92usize, 124usize] {
        let inner_w = width - 4;
        let lines = build_left_interaction_lines(&state, inner_w);
        assert!(!lines.is_empty());

        let frame = render_two_pane_tui_frame(&state, width, 68, false);
        for (row_idx, row) in frame.lines().enumerate() {
            assert_eq!(
                UnicodeWidthStr::width(row),
                width,
                "rendered frame row {row_idx} width mismatch at width={width}: {row:?}"
            );
        }

        // Normalize whitespace/newlines across wrapped rows to verify zero clipping of tail tokens:
        let squashed: String = frame
            .lines()
            .map(|r| r.trim_matches(|c| c == '│' || c == ' '))
            .collect::<Vec<_>>()
            .join(" ");
        let squashed_nospace: String = frame
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '│')
            .collect();

        for tail_marker in [
            "SENTENCE_FOUR_VERIFIED_TAIL",
            "PARA_THREE_VERIFIED_TAIL",
            "ABSTAIN_SECOND_PARAGRAPH_TAIL",
            "USER_PROMPT_TAIL",
            "OBS_ERROR_TAIL",
            "HARNESS_NOTE_TAIL",
        ] {
            assert!(
                squashed.contains(tail_marker) || squashed_nospace.contains(tail_marker),
                "tail token {tail_marker} was truncated or right-clipped at width={width}:\n{frame}"
            );
        }
        assert!(
            squashed_nospace.contains("CJK_TAIL"),
            "CJK tail clipped at width={width}:\n{frame}"
        );
        assert!(
            squashed_nospace.contains("OVERLONG_TAIL"),
            "overlong path tail clipped at width={width}:\n{frame}"
        );
    }
}
