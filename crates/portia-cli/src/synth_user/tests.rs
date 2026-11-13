use super::*;
use crate::driver::LiveSqliteDailyDriver;
use crate::rust_driver::LiveRustAstDailyDriver;
use crate::worker::SessionOutcomeStatus;
use portia_nursery::nursery_scenario::{
    arb_scenario_coordinate, render_sealed_witness_and_approved_crate_block, HyloCartridgeLevel,
    HyloCartridgePhase, RustHyloPipelineStage, RustTodoHoleSpec,
};
use portia_nursery::roguelite_lattice::RogueliteStressVector;
use proptest::prelude::*;
use rusqlite::Connection;
use std::collections::VecDeque;

#[derive(Debug, Default)]
struct MockRecordingChatTransport {
    requests: Vec<ChatCompletionRequest>,
    scripted_replies: VecDeque<Result<String, String>>,
}

impl MockRecordingChatTransport {
    fn with_replies(replies: impl IntoIterator<Item = Result<&'static str, &'static str>>) -> Self {
        Self {
            requests: Vec::new(),
            scripted_replies: replies
                .into_iter()
                .map(|r| r.map(ToOwned::to_owned).map_err(ToOwned::to_owned))
                .collect(),
        }
    }
}

impl ChatCompletionTransport for MockRecordingChatTransport {
    fn complete_chat(&mut self, req: &ChatCompletionRequest) -> Result<String> {
        self.requests.push(req.clone());
        match self.scripted_replies.pop_front() {
            Some(Ok(content)) => Ok(content),
            Some(Err(err)) => bail!("{err}"),
            None => bail!(
                "MockRecordingChatTransport ran out of scripted replies on call #{}",
                self.requests.len()
            ),
        }
    }
}

fn build_single_hole_cleanroom_rust_spec(coord: ScenarioCoordinate) -> CleanroomRustHyloSpec {
    let module_name = coord.domain_skin.primary_rust_module().to_string();
    let trait_name = coord.domain_skin.primary_rust_trait().to_string();
    let state_struct_name = format!("{trait_name}State");
    let fn_name = format!("{}_step_01", coord.domain_skin.slug());
    let signature = format!(
            "fn {fn_name}(&self, input_metric: u64, guard_clause: u32) -> Result<VerifiedDomainOutput, &'static str>"
        );
    let todo_stub = format!("todo!(\"{module_name}::{fn_name}\")");

    let skeleton = format!(
        "// Cleanroom 1-hole verification module ({module_name})\n\
             {}\n\n\
             pub struct {state_struct_name} {{\n\
             \x20   pub seed: u64,\n\
             }}\n\n\
             pub trait {trait_name} {{\n\
             \x20   {signature};\n\
             }}\n\n\
             impl {trait_name} for {state_struct_name} {{\n\
             \x20   {signature} {{\n\
             \x20       {todo_stub}\n\
             \x20   }}\n\
             }}\n\n\
             #[cfg(test)]\n\
             mod tests {{\n\
             \x20   use super::*;\n\n\
             \x20   #[test]\n\
             \x20   fn verify_single_hole_saturating_invariant() {{\n\
             \x20       let state = {state_struct_name} {{ seed: {} }};\n\
             \x20       assert_eq!(state.{fn_name}(100, 10).map(|v| v.metric()), Ok(90));\n\
             \x20       assert_eq!(state.{fn_name}(5, 20).map(|v| v.metric()), Ok(0));\n\
             \x20   }}\n\
             }}\n",
        render_sealed_witness_and_approved_crate_block(),
        coord.seed,
    );

    let (crate_name, crate_purpose) = coord.domain_skin.approved_ecosystem_crate();
    CleanroomRustHyloSpec {
        coordinate: coord,
        module_name,
        trait_name,
        state_struct_name,
        ast_valid_token_zero_skeleton: skeleton,
        todo_holes: vec![RustTodoHoleSpec {
            hole_index: 0,
            fn_name: fn_name.clone(),
            signature,
            todo_stub_literal: todo_stub,
            invariant_contract: "Enforce saturating floor conservation clause #1".to_string(),
        }],
        pipeline_schedule: vec![
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Scaffold,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::ForkFanout { hole_count: 1 },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::TypeCart,
            },
            RustHyloPipelineStage::CartridgeCell {
                phase: HyloCartridgePhase::Impl,
                level: HyloCartridgeLevel::ValueCart,
            },
            RustHyloPipelineStage::OctoMerge {
                merged_hole_count: 1,
            },
        ],
        dialogic_affordances: vec![
            DialogicCodingAffordance::ClarifyUnderspecifiedInvariant {
                ambiguous_aspect: "bgp_noc underflow floor policy".to_string(),
                user_clarification: "Saturate underflow at floor 0 and add hole_index.".to_string(),
            },
            DialogicCodingAffordance::RequestEcosystemCrate {
                crate_name: crate_name.to_string(),
                purpose: crate_purpose.to_string(),
                approved: true,
            },
            DialogicCodingAffordance::ReviewScaffoldBeforeFanout {
                todo_hole_names: vec![fn_name],
            },
        ],
        has_contrived_decoy_functions: false,
    }
}

#[test]
fn openai_synth_user_drives_5_turn_rust_ast_episode_with_real_rustc_verification() {
    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::BgpNoc,
        RogueliteStressVector::sweet_spot_baseline(),
        0x5174_0001,
    );
    let spec = build_single_hole_cleanroom_rust_spec(coord);
    assert!(spec.verify_ast_valid_from_token_zero());

    let mut rust_env = LiveRustAstDailyDriver::from_spec(spec)
        .expect("init LiveRustAstDailyDriver")
        .with_rustc_verification(true);

    let transport = MockRecordingChatTransport::with_replies([
            Ok(
                "Hey, could you inspect `bgp_route_dampener` (`RouteDampeningPolicy`) and fill in its `todo!()` stub so tests pass? Let me know if the underflow behavior or crate policy is unclear.",
            ),
            Ok(
                "Saturate `input_metric - guard_clause` at floor 0 (+ `hole_index`), and yes — ecosystem crate `ipnet` is approved.",
            ),
        ]);
    let mut synth_user = OpenAiChatSynthUserEnv::new(transport, "gpt-4.1-mini");

    let mock_worker = tempfile::NamedTempFile::new().expect("temp mock worker");
    let captured_reqs = tempfile::NamedTempFile::new().expect("temp worker log");
    let mock_py_code = r####"import json, sys
log_path = sys.argv[1]
turns = [
    [
        (5140, " PRO", [5140]),
        (11261, "BE", [11261]),
        (60, "]", [60]),
        (1001, "Inspect Rust AST outline first.\n", [1001, 198]),
        (2001, "inspect rust ast outline", [2001]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (21541, " ASK", [21541]),
        (60, "]", [60]),
        (1002, "Clarify underflow invariant and request ipnet crate.\n", [1002, 198]),
        (
            2002,
            "Please clarify the underflow invariant and confirm if crate ipnet is approved [Options: Saturate at 0 and approve ipnet | Hand-roll custom parser]",
            [2002],
        ),
        (198, "\n</to_user>", [198, 523]),
    ],
    [
        (18104, " READ", [18104]),
        (60, "]", [60]),
        (1003, "Read bgp_noc_step_01 hole signature.\n", [1003, 198]),
        (2003, "fn bgp_noc_step_01", [2003]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (37272, " WRITE", [37272]),
        (60, "]", [60]),
        (1004, "Splice bgp_noc_step_01 using approved domain helper.\n", [1004, 198]),
        (
            2004,
            "bgp_noc_step_01 => approved_crate::eval_domain_step(input_metric, guard_clause, 0usize)",
            [2004],
        ),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (27604, " RESULT", [27604]),
        (60, "]", [60]),
        (1005, "Report rustc-verified completion.\n", [1005, 198]),
        (
            2005,
            "{\"status\":\"SPLICED_AND_RUSTC_VERIFIED\",\"module\":\"bgp_route_dampener\"}",
            [2005],
        ),
        (198, "\n</to_user>", [198, 523]),
    ],
]
turn_idx = 0
tok_idx = 0
with open(log_path, "w", encoding="utf-8") as log_f:
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        log_f.write(json.dumps(req) + "\n")
        log_f.flush()
        op = req.get("op")
        if op in ("reset_session", "prefill", "begin_task"):
            resp = {"status": "prefilled", "token_ids": [1, 2]}
        elif op == "sample_token":
            tid, piece, tids = turns[turn_idx][tok_idx]
            tok_idx += 1
            if tok_idx >= len(turns[turn_idx]):
                turn_idx += 1
                tok_idx = 0
            resp = {"status": "sampled_token", "token_id": tid, "token_piece": piece, "token_ids": tids}
        elif op == "consume_token":
            resp = {"status": "consumed_token", "token_ids": [int(req["token_id"])]}
        else:
            resp = {"status": "error", "message": f"unknown op {op}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
"####;
    std::fs::write(mock_worker.path(), mock_py_code).expect("write mock worker script");
    let cmd = format!(
        "python3 {:?} {:?}",
        mock_worker.path(),
        captured_reqs.path()
    );
    let mut worker = StdioQwenWorkerClient::spawn_shell(&cmd).expect("spawn mock worker");

    let receipt = run_synth_user_epistemic_episode(
        &mut rust_env,
        &mut synth_user,
        &mut worker,
        &coord,
        8,
        true,
    )
    .expect("5-turn synth-user coding-agent episode should complete and pass rustc --test");

    assert_eq!(receipt.summary.status, SessionOutcomeStatus::Completed);
    assert_eq!(receipt.summary.turns_executed, 5);
    assert_eq!(receipt.summary.writes_committed, 1);
    assert_eq!(
        receipt.events.iter().map(|e| e.mode).collect::<Vec<_>>(),
        vec![
            Mode::Probe,
            Mode::Ask,
            Mode::Read,
            Mode::Write,
            Mode::Result
        ]
    );
    assert_eq!(receipt.ask_exchanges.len(), 1);
    assert_eq!(receipt.ask_exchanges[0].turn_index, 1);
    assert!(receipt.ask_exchanges[0].ask_question.contains("ipnet"));
    assert!(receipt.ask_exchanges[0].user_reply.contains("ipnet"));

    // Verify zero ground-truth leakage in opening prompt and full multi-turn history in OpenAiChatSynthUserEnv:
    validate_opening_prompt_no_leakage(&coord, &receipt.opening_prompt).unwrap();
    assert_eq!(synth_user.transport().requests.len(), 2);
    assert_eq!(synth_user.history().len(), 5);
    assert_eq!(synth_user.history()[0].role, "system");
    assert_eq!(synth_user.history()[2].role, "assistant");
    assert_eq!(synth_user.history()[2].content, receipt.opening_prompt);
    assert_eq!(synth_user.history()[4].role, "assistant");
    assert_eq!(
        synth_user.history()[4].content,
        receipt.ask_exchanges[0].user_reply
    );

    // Verify cleanroom episode summary has 0 wipeout diagnoses and 0 remaining todo!() holes:
    let ep_summary = rust_env
        .last_turn_outcome()
        .and_then(|o| o.episode_summary.as_ref())
        .expect("Mode::Result turn must populate ScenarioEpisodeSummary");
    assert!(
        ep_summary.cleared,
        "episode must clear cleanly: {ep_summary:?}"
    );
    assert!(ep_summary.wipeout_diagnoses.is_empty());
    assert!(rust_env.remaining_todo_holes().unwrap().is_empty());
}

#[test]
fn curl_chat_completion_transport_posts_headers_stdin_body_and_surfaces_api_errors_over_loopback() {
    // 1. Missing API key fails closed immediately:
    let missing_err =
        CurlChatCompletionTransport::new("http://127.0.0.1:9/v1", "   ", "gpt-4.1-mini", 5)
            .expect_err("blank API key must fail closed");
    assert!(missing_err.to_string().contains("API key"));

    // 2. Spawn a local Python loopback HTTP server on `127.0.0.1:0` handling 2 requests:
    //    Request #1 -> HTTP 200 OpenAI chat.completion payload
    //    Request #2 -> HTTP 401 OpenAI JSON error payload
    let server_py = tempfile::NamedTempFile::new().expect("temp http server script");
    let capture_json = tempfile::NamedTempFile::new().expect("temp http capture");
    let py_code = r#"import http.server, json, sys

capture_path = sys.argv[1]
records = []

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        records.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "body": json.loads(raw_body),
        })
        with open(capture_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        if len(records) == 1:
            payload = {
                "id": "chatcmpl-test-001",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "  Please inspect `solaris_seal_verifier` and clarify underflow before editing.  "
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            out = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            err_payload = {
                "error": {
                    "message": "Simulated quota or authentication rejection",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
            out = json.dumps(err_payload).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def log_message(self, fmt, *args):
        return

httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
port = httpd.server_address[1]
sys.stdout.write(f"{port}\n")
sys.stdout.flush()
httpd.handle_request()
httpd.handle_request()
"#;
    std::fs::write(server_py.path(), py_code).expect("write loopback server");

    let mut server_child = Command::new("python3")
        .arg("-u")
        .arg(server_py.path())
        .arg(capture_json.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 loopback server");

    let port: u16 = {
        use std::io::{BufRead, BufReader};
        let stdout = server_child.stdout.as_mut().expect("server stdout");
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .expect("read bound loopback port");
        line.trim().parse().expect("parse port u16")
    };

    let base_url = format!("http://127.0.0.1:{port}/v1/");
    let mut transport = CurlChatCompletionTransport::new(
        &base_url,
        "sk-test-loopback-secret-99",
        "gpt-4.1-mini",
        10,
    )
    .expect("init CurlChatCompletionTransport");
    assert_eq!(
        transport.chat_completions_endpoint(),
        format!("http://127.0.0.1:{port}/v1/chat/completions")
    );

    let req1 = ChatCompletionRequest {
        model: String::new(),
        messages: vec![
            ChatMessage::new("system", "Act as pair-programming partner."),
            ChatMessage::new("user", "Generate opening prompt."),
        ],
        temperature: Some(0.25),
        max_completion_tokens: 128,
    };
    let reply = transport
        .complete_chat(&req1)
        .expect("HTTP 200 loopback chat completion should succeed");
    assert_eq!(
        reply,
        "Please inspect `solaris_seal_verifier` and clarify underflow before editing."
    );

    let err = transport
        .complete_chat(&req1)
        .expect_err("HTTP 401 JSON error must surface cleanly");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("Simulated quota or authentication rejection"),
        "unexpected error string: {err_msg}"
    );
    assert!(
        err_msg.contains("401"),
        "expected HTTP 401 in error: {err_msg}"
    );

    let _ = server_child.wait();

    let captured_raw =
        std::fs::read_to_string(capture_json.path()).expect("read captured HTTP requests");
    let captured: serde_json::Value =
        serde_json::from_str(&captured_raw).expect("parse captured JSON");
    let arr = captured.as_array().expect("2 captured POST requests");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["path"], "/v1/chat/completions");
    assert_eq!(arr[0]["auth"], "Bearer sk-test-loopback-secret-99");
    assert_eq!(arr[0]["content_type"], "application/json");
    assert_eq!(arr[0]["body"]["model"], "gpt-4.1-mini");
    assert_eq!(arr[0]["body"]["max_completion_tokens"], 128);
    assert_eq!(arr[0]["body"]["messages"][0]["role"], "system");
    assert_eq!(arr[0]["body"]["messages"][1]["role"], "user");
}

#[test]
fn sql_steward_synth_user_episode_handles_permission_grant_and_midflight_m3_pivot() {
    let db_file = tempfile::NamedTempFile::new().expect("temp sqlite db");
    {
        let conn = Connection::open(db_file.path()).expect("open temp sqlite");
        conn.execute_batch(
                "CREATE TABLE bgp_peering_sessions (
                    record_id INTEGER PRIMARY KEY,
                    entity_slug TEXT NOT NULL,
                    balance_credits INTEGER NOT NULL,
                    policy_decision TEXT NOT NULL,
                    lock_state TEXT NOT NULL,
                    steward_handle TEXT NOT NULL
                );
                INSERT INTO bgp_peering_sessions VALUES (101, 'peer-as64512-01a', 900, 'ASK', 'CLEAR', 'noc-steward');",
            )
            .expect("seed sqlite table");
    }

    let coord = ScenarioCoordinate::new(
        AppType::SqlSteward,
        ScenarioLanguage::Spanish,
        DomainSkin::BgpNoc,
        RogueliteStressVector::sweet_spot_baseline(),
        0x5174_0099,
    );

    // Verify mid-flight pivot (`M3`) normalizes permission refusal cleanly:
    let mut pivot_user = DeterministicStewardUserEnv::new()
        .with_trap_modifiers([UnifiedScenarioTrapModifier::MidFlightIntentPivotToReadOnlyM3]);
    let pivot_opening = pivot_user.opening_prompt(&coord).unwrap();
    let pivot_reply = pivot_user
        .answer_ask(&SynthUserTurnContext {
            coordinate: &coord,
            turn_index: 1,
            opening_prompt: &pivot_opening,
            ask_question: "[permission] May I update bgp_peering_sessions record 101?",
        })
        .unwrap();
    assert!(pivot_reply.starts_with("DENY"));
    assert!(pivot_reply.contains("hold off on writing"));

    // Run a full 5-turn SqlSteward episode (`Probe -> Read -> Ask(Permission) -> Write -> Result`):
    let mut sql_env = LiveSqliteDailyDriver::open(db_file.path(), 10)
        .expect("open LiveSqliteDailyDriver")
        .with_steward_policy_columns(["policy_decision"]);
    let transport = MockRecordingChatTransport::with_replies([
            Ok(
                "Por favor revise la sesión `peer-as64512-01a` en `bgp_peering_sessions` y actualice `balance_credits` a 1350 si procede.",
            ),
            Ok("Sí, adelante con la actualización autorizada."),
        ]);
    let mut synth_user = OpenAiChatSynthUserEnv::new(transport, "gpt-4.1-mini");

    let mock_worker = tempfile::NamedTempFile::new().expect("temp sql mock worker");
    let mock_py_code = r####"import json, sys
turns = [
    [
        (5140, " PRO", [5140]),
        (11261, "BE", [11261]),
        (60, "]", [60]),
        (1001, "Inspect bgp_peering_sessions schema.\n", [1001, 198]),
        (2001, "PRAGMA table_info(bgp_peering_sessions);", [2001]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (18104, " READ", [18104]),
        (60, "]", [60]),
        (1002, "Read row 101.\n", [1002, 198]),
        (2002, "SELECT record_id, entity_slug, balance_credits, policy_decision, lock_state FROM bgp_peering_sessions WHERE record_id = 101;", [2002]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (21541, " ASK", [21541]),
        (60, "]", [60]),
        (1003, "Request steward permission for ASK row.\n", [1003, 198]),
        (2003, "[permission] Authorize updating bgp_peering_sessions record_id=101 to 1350?", [2003]),
        (198, "\n</to_user>", [198, 523]),
    ],
    [
        (37272, " WRITE", [37272]),
        (60, "]", [60]),
        (1004, "Apply authorized balance update.\n", [1004, 198]),
        (2004, "UPDATE bgp_peering_sessions SET balance_credits = 1350 WHERE record_id = 101;", [2004]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (27604, " RESULT", [27604]),
        (60, "]", [60]),
        (1005, "Report updated balance.\n", [1005, 198]),
        (2005, "{\"record_id\":101,\"balance_credits\":1350}", [2005]),
        (198, "\n</to_user>", [198, 523]),
    ],
]
turn_idx = 0
tok_idx = 0
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    op = req.get("op")
    if op in ("reset_session", "prefill", "begin_task"):
        resp = {"status": "prefilled", "token_ids": [1, 2]}
    elif op == "sample_token":
        tid, piece, tids = turns[turn_idx][tok_idx]
        tok_idx += 1
        if tok_idx >= len(turns[turn_idx]):
            turn_idx += 1
            tok_idx = 0
        resp = {"status": "sampled_token", "token_id": tid, "token_piece": piece, "token_ids": tids}
    elif op == "consume_token":
        resp = {"status": "consumed_token", "token_ids": [int(req["token_id"])]}
    else:
        resp = {"status": "error", "message": "unknown op"}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"####;
    std::fs::write(mock_worker.path(), mock_py_code).unwrap();
    let cmd = format!("python3 {:?}", mock_worker.path());
    let mut worker = StdioQwenWorkerClient::spawn_shell(&cmd).unwrap();

    let receipt = run_synth_user_epistemic_episode(
        &mut sql_env,
        &mut synth_user,
        &mut worker,
        &coord,
        8,
        true,
    )
    .expect("5-turn SQL steward synth-user episode should complete");

    assert_eq!(receipt.summary.status, SessionOutcomeStatus::Completed);
    assert_eq!(receipt.summary.writes_committed, 1);
    assert_eq!(receipt.ask_exchanges.len(), 1);
    assert_eq!(receipt.ask_exchanges[0].turn_index, 2);
    assert!(receipt.ask_exchanges[0].user_reply.starts_with("ALLOW"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(36))]
    #[test]
    fn prop_deterministic_steward_user_env_covers_full_lattice_with_zero_leakage(
        coord in arb_scenario_coordinate()
    ) {
        let mut env = DeterministicStewardUserEnv::new();
        let opening = env.opening_prompt(&coord).expect("deterministic opening_prompt");
        prop_assert!(!opening.trim().is_empty());
        validate_opening_prompt_no_leakage(&coord, &opening)
            .expect("zero ground-truth leakage across full coordinate lattice");

        let ask_q = match coord.app_type {
            AppType::RustAstHylo => {
                "Please clarify the underflow invariant, confirm ecosystem crate approval, and review the todo scaffold."
            }
            AppType::SqlSteward => {
                "[permission] May we proceed with the requested entity update?"
            }
        };
        let reply = env
            .answer_ask(&SynthUserTurnContext {
                coordinate: &coord,
                turn_index: 1,
                opening_prompt: &opening,
                ask_question: ask_q,
            })
            .expect("deterministic answer_ask");
        prop_assert!(!reply.trim().is_empty());
        prop_assert_eq!(env.ask_rounds_answered(), 1);

        if coord.app_type == AppType::SqlSteward {
            let parsed = parse_ask_user_reply_for_kind(AskPromptKind::Permission, &reply)
                .expect("SQL permission reply must parse cleanly as AskUserReply::Decision");
            let is_decision = matches!(parsed, AskUserReply::Decision { .. });
            prop_assert!(is_decision);
        } else {
            let (crate_name, _) = coord.domain_skin.approved_ecosystem_crate();
            prop_assert!(reply.contains(crate_name));
        }

        let followup = env
            .followup_prompt(&coord, &[])
            .expect("deterministic followup_prompt");
        prop_assert!(!followup.trim().is_empty());
        validate_opening_prompt_no_leakage(&coord, &followup)
            .expect("zero ground-truth leakage on deterministic followup_prompt");
    }
}

#[test]
fn reasoning_and_gpt5_models_omit_temperature_and_default_to_4096_completion_tokens() {
    assert_eq!(DEFAULT_OPENAI_SYNTH_MODEL, "gpt-5-mini");
    assert_eq!(DEFAULT_OPENAI_MAX_COMPLETION_TOKENS, 4096);

    for reasoning_model in [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "openai/gpt-5-mini",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "codex-default",
    ] {
        assert!(
            !model_supports_custom_temperature(reasoning_model),
            "{reasoning_model} must omit custom temperature"
        );
    }

    for classic_model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "local/qwen-coder"] {
        assert!(
            model_supports_custom_temperature(classic_model),
            "{classic_model} should support custom temperature"
        );
    }

    let coord = ScenarioCoordinate::new(
        AppType::RustAstHylo,
        ScenarioLanguage::English,
        DomainSkin::BgpNoc,
        RogueliteStressVector::sweet_spot_baseline(),
        0x5174_7777,
    );

    // Default `gpt-5-mini` env omits `"temperature"` in JSON and sets `max_completion_tokens == 4096`:
    let transport_gpt5 = MockRecordingChatTransport::with_replies([Ok(
            "Please inspect `bgp_route_dampener` (`RouteDampeningPolicy`) and implement its `todo!()` methods.",
        )]);
    let mut env_gpt5 = OpenAiChatSynthUserEnv::new(transport_gpt5, "");
    assert_eq!(env_gpt5.model(), "gpt-5-mini");
    assert_eq!(env_gpt5.max_completion_tokens(), 4096);
    let _ = env_gpt5.opening_prompt(&coord).expect("gpt-5-mini opening");
    let sent_gpt5 = &env_gpt5.transport().requests[0];
    assert_eq!(sent_gpt5.temperature, None);
    assert_eq!(sent_gpt5.max_completion_tokens, 4096);
    let json_gpt5 = serde_json::to_value(sent_gpt5).unwrap();
    assert!(
        json_gpt5.get("temperature").is_none(),
        "serialized reasoning request must omit `temperature`: {json_gpt5}"
    );

    // `o3-mini` also omits `"temperature"`:
    let transport_o3 = MockRecordingChatTransport::with_replies([Ok(
            "Inspect `bgp_route_dampener` (`RouteDampeningPolicy`) and ask if edge cases need clarification.",
        )]);
    let mut env_o3 = OpenAiChatSynthUserEnv::new(transport_o3, "o3-mini");
    let _ = env_o3.opening_prompt(&coord).expect("o3-mini opening");
    let json_o3 = serde_json::to_value(&env_o3.transport().requests[0]).unwrap();
    assert!(json_o3.get("temperature").is_none());

    // `gpt-4.1-mini` includes `"temperature"`:
    let transport_41 = MockRecordingChatTransport::with_replies([Ok(
        "Inspect `bgp_route_dampener` (`RouteDampeningPolicy`) and fill in its `todo!()` stubs.",
    )]);
    let mut env_41 =
        OpenAiChatSynthUserEnv::new(transport_41, "gpt-4.1-mini").with_temperature(0.35);
    let _ = env_41.opening_prompt(&coord).expect("gpt-4.1-mini opening");
    let json_41 = serde_json::to_value(&env_41.transport().requests[0]).unwrap();
    let temp_val = json_41
        .get("temperature")
        .and_then(|v| v.as_f64())
        .expect("gpt-4.1-mini request must serialize `temperature`");
    assert!((temp_val - 0.35).abs() < 1e-6);
}

#[test]
fn codex_exec_chat_transport_invokes_isolated_read_only_cli_and_reads_output_file() {
    let script_dir = tempfile::tempdir().expect("tempdir for mock codex script");
    let script_path = script_dir.path().join("mock_codex.py");
    let capture_path = script_dir.path().join("codex_capture.jsonl");

    let mock_codex_py = format!(
        r#"#!/usr/bin/env python3
import json, os, sys

capture_file = {capture_path:?}
argv = sys.argv[1:]
stdin_text = sys.stdin.read()

c_dir = argv[argv.index("-C") + 1] if "-C" in argv else ""
out_path = argv[argv.index("-o") + 1] if "-o" in argv else ""
c_entries = os.listdir(c_dir) if c_dir and os.path.isdir(c_dir) else ["<missing>"]

record = {{
    "argv": argv,
    "c_dir": c_dir,
    "c_entries": c_entries,
    "stdin_text": stdin_text,
}}
with open(capture_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")

with open(out_path, "w", encoding="utf-8") as out_f:
    out_f.write("  Inspect `bgp_route_dampener` and ask before changing underflow behavior.  \n")
"#
    );
    std::fs::write(&script_path, mock_codex_py).expect("write mock codex script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).unwrap();
    }

    // 1. Default `codex-default` omits `-m` so `codex` uses its configured flagship subscription model:
    let mut default_codex = CodexExecChatTransport::new(
        script_path.to_string_lossy().into_owned(),
        None,
        Duration::from_secs(10),
    );
    let req_default = ChatCompletionRequest {
        model: "codex-default".to_string(),
        messages: vec![
            ChatMessage::new("system", "Pair-programming simulated user."),
            ChatMessage::new("user", "Compose opening task."),
        ],
        temperature: None,
        max_completion_tokens: 4096,
    };
    let reply1 = default_codex
        .complete_chat(&req_default)
        .expect("mock codex exec default run");
    assert_eq!(
        reply1,
        "Inspect `bgp_route_dampener` and ask before changing underflow behavior."
    );

    // 2. Explicit `model_override` passes `-m gpt-5`:
    let mut override_codex = CodexExecChatTransport::new(
        script_path.to_string_lossy().into_owned(),
        Some("gpt-5".to_string()),
        Duration::from_secs(10),
    );
    let _ = override_codex
        .complete_chat(&req_default)
        .expect("mock codex exec override run");

    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&capture_path)
        .expect("read codex capture")
        .lines()
        .map(|l| serde_json::from_str(l).expect("parse capture line"))
        .collect();
    assert_eq!(lines.len(), 2);

    let argv0: Vec<String> = serde_json::from_value(lines[0]["argv"].clone()).unwrap();
    assert_eq!(argv0[0], "exec");
    assert!(argv0.contains(&"--ephemeral".to_string()));
    assert!(argv0.contains(&"--skip-git-repo-check".to_string()));
    let s_pos = argv0.iter().position(|a| a == "-s").expect("-s flag");
    assert_eq!(argv0[s_pos + 1], "read-only");
    assert!(
        !argv0.contains(&"-m".to_string()),
        "default codex run must omit `-m`: {argv0:?}"
    );
    assert_eq!(argv0.last().map(String::as_str), Some("-"));
    assert_eq!(lines[0]["c_entries"], serde_json::json!([]));
    let stdin0 = lines[0]["stdin_text"].as_str().unwrap();
    assert!(stdin0
        .starts_with("Do NOT run shell commands; reply directly in plain text as instructed.\n\n"));
    assert!(stdin0.contains("[SYSTEM]\nPair-programming simulated user."));
    assert!(stdin0.contains("[USER]\nCompose opening task."));

    let argv1: Vec<String> = serde_json::from_value(lines[1]["argv"].clone()).unwrap();
    let m_pos = argv1
        .iter()
        .position(|a| a == "-m")
        .expect("explicit model override must pass `-m`");
    assert_eq!(argv1[m_pos + 1], "gpt-5");
}

#[test]
fn multi_wave_followup_preserves_worker_kv_and_coroner_parses_fenced_json_diagnosis() {
    let db_file = tempfile::NamedTempFile::new().expect("temp sqlite db");
    {
        let conn = Connection::open(db_file.path()).expect("open temp sqlite");
        conn.execute_batch(
                "CREATE TABLE bgp_peering_sessions (
                    record_id INTEGER PRIMARY KEY,
                    entity_slug TEXT NOT NULL,
                    balance_credits INTEGER NOT NULL,
                    policy_decision TEXT NOT NULL,
                    lock_state TEXT NOT NULL,
                    steward_handle TEXT NOT NULL
                );
                INSERT INTO bgp_peering_sessions VALUES (101, 'peer-as64512-01a', 900, 'ALLOW', 'CLEAR', 'noc-steward');",
            )
            .expect("seed sqlite table");
    }
    let coord = ScenarioCoordinate::new(
        AppType::SqlSteward,
        ScenarioLanguage::English,
        DomainSkin::BgpNoc,
        RogueliteStressVector::sweet_spot_baseline(),
        0x5174_0202,
    );

    let mut sql_env = LiveSqliteDailyDriver::open(db_file.path(), 10)
        .unwrap()
        .with_steward_policy_columns(["policy_decision"]);
    let transport = MockRecordingChatTransport::with_replies([
            Ok("Check `peer-as64512-01a` in `bgp_peering_sessions` and report its current balance."),
            Ok("Great — now re-read `peer-as64512-01a` in `bgp_peering_sessions` to confirm nothing drifted."),
        ]);
    let mut synth_user = OpenAiChatSynthUserEnv::new(transport, "gpt-5-mini");

    let mock_worker = tempfile::NamedTempFile::new().unwrap();
    let worker_log = tempfile::NamedTempFile::new().unwrap();
    let mock_py = r####"import json, sys
log_path = sys.argv[1]
turns = [
    [
        (18104, " READ", [18104]),
        (60, "]", [60]),
        (1001, "Read peer row.\n", [1001, 198]),
        (2001, "SELECT record_id, balance_credits FROM bgp_peering_sessions WHERE entity_slug = 'peer-as64512-01a';", [2001]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (27604, " RESULT", [27604]),
        (60, "]", [60]),
        (1002, "Return Wave 0 balance.\n", [1002, 198]),
        (2002, "{\"balance_credits\":900}", [2002]),
        (198, "\n</to_user>", [198, 523]),
    ],
    [
        (18104, " READ", [18104]),
        (60, "]", [60]),
        (1003, "Re-read peer row in Wave 1.\n", [1003, 198]),
        (2003, "SELECT record_id, balance_credits FROM bgp_peering_sessions WHERE record_id = 101;", [2003]),
        (198, "\n</sql>", [198, 522]),
    ],
    [
        (27604, " RESULT", [27604]),
        (60, "]", [60]),
        (1004, "Confirm Wave 1 balance.\n", [1004, 198]),
        (2004, "{\"verified_balance_credits\":900}", [2004]),
        (198, "\n</to_user>", [198, 523]),
    ],
]
turn_idx = 0
tok_idx = 0
with open(log_path, "w", encoding="utf-8") as lf:
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        lf.write(json.dumps(req) + "\n")
        lf.flush()
        op = req.get("op")
        if op in ("reset_session", "prefill", "begin_task"):
            resp = {"status": "prefilled", "token_ids": [1, 2]}
        elif op == "sample_token":
            tid, piece, tids = turns[turn_idx][tok_idx]
            tok_idx += 1
            if tok_idx >= len(turns[turn_idx]):
                turn_idx += 1
                tok_idx = 0
            resp = {"status": "sampled_token", "token_id": tid, "token_piece": piece, "token_ids": tids}
        elif op == "consume_token":
            resp = {"status": "consumed_token", "token_ids": [int(req["token_id"])]}
        else:
            resp = {"status": "error", "message": "unknown op"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
"####;
    std::fs::write(mock_worker.path(), mock_py).unwrap();
    let cmd = format!("python3 {:?} {:?}", mock_worker.path(), worker_log.path());
    let mut worker = StdioQwenWorkerClient::spawn_shell(&cmd).unwrap();

    let receipt = run_synth_user_epistemic_episode_with_followups(
        &mut sql_env,
        &mut synth_user,
        &mut worker,
        &coord,
        4,
        true,
        1,
    )
    .expect("2-wave stateful episode should succeed");

    assert_eq!(receipt.summary.status, SessionOutcomeStatus::Completed);
    assert_eq!(receipt.events.len(), 4);
    assert_eq!(receipt.followup_prompts.len(), 1);
    assert_eq!(receipt.followup_event_start_indices, vec![2]);

    // Verify `reset_session` was sent ONLY once (before Wave 0), preserving KV across Wave 1!
    let logged_ops: Vec<serde_json::Value> = std::fs::read_to_string(worker_log.path())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let reset_count = logged_ops
        .iter()
        .filter(|v| v.get("op").and_then(|o| o.as_str()) == Some("reset_session"))
        .count();
    assert_eq!(
        reset_count, 1,
        "follow-up wave must preserve worker KV without resetting session"
    );

    // Now verify `diagnose_failed_episode` parses fenced ```json Coroner output cleanly:
    let coroner_transport = MockRecordingChatTransport::with_replies([Ok(
            "```json\n{\n  \"divergence_turn_index\": 1,\n  \"observed_mode\": \"WRITE\",\n  \"expected_mode\": \"ASK\",\n  \"root_cause\": \"Committed UPDATE on ASK-gated row without requesting Steward permission\",\n  \"shrink_contrast_hint\": \"Contrast WRITE vs ASK after SELECT reveals policy_decision='ASK'\"\n}\n```",
        )]);
    let mut coroner_env = OpenAiChatSynthUserEnv::new(coroner_transport, "gpt-5-mini");
    let mut failed_receipt = receipt.clone();
    failed_receipt.summary.status = SessionOutcomeStatus::FailedAfterChanges;
    let diag = coroner_env
        .diagnose_failed_episode(&failed_receipt)
        .expect("parse fenced Coroner JSON");
    assert_eq!(diag.divergence_turn_index, 1);
    assert_eq!(diag.observed_mode, Mode::Write);
    assert_eq!(diag.expected_mode, Mode::Ask);
    assert!(diag.root_cause.contains("Steward permission"));
    let serialized_diag = serde_json::to_value(&diag).unwrap();
    assert_eq!(serialized_diag["observed_mode"], "WRITE");
    assert_eq!(serialized_diag["expected_mode"], "ASK");
}

#[test]
fn codex_exec_luna_transport_passes_gpt_6_luna_and_output_schema_file() {
    assert_eq!(DEFAULT_CODEX_LUNA_MODEL, "gpt-6-luna");
    assert!(!model_supports_custom_temperature("gpt-6-luna"));
    assert!(!model_supports_custom_temperature("openai/gpt-6-luna"));

    let args_log = tempfile::NamedTempFile::new().expect("args log");
    let schema_copy = tempfile::NamedTempFile::new().expect("schema copy");
    let mock_codex_py = tempfile::NamedTempFile::new().expect("mock codex py");
    let mock_codex_sh = tempfile::Builder::new()
        .prefix("mock_codex_luna_")
        .suffix(".sh")
        .tempfile()
        .expect("mock codex sh");

    let py_code = r#"import json, shutil, sys
args = sys.argv[1:]
log_path = sys.argv[1]
schema_dst = sys.argv[2]
cli_args = sys.argv[3:]
stdin_txt = sys.stdin.read()
out_path = cli_args[cli_args.index("-o") + 1]
if "--output-schema" in cli_args:
    schema_src = cli_args[cli_args.index("--output-schema") + 1]
    shutil.copyfile(schema_src, schema_dst)
with open(log_path, "w", encoding="utf-8") as f:
    json.dump({"args": cli_args, "stdin": stdin_txt}, f)
with open(out_path, "w", encoding="utf-8") as out:
    json.dump({
        "direct_cooperative": "Please inspect active node `bgp-peer-01` and update its balance if permitted.",
        "curveball_or_constraint_shift": "Hold off on writing for a moment and check `bgp-peer-01` read-only first.",
        "messy_underspecified_or_chitchat": "yo quick check on `bgp-peer-01` before we touch anything"
    }, out)
"#;
    std::fs::write(mock_codex_py.path(), py_code).unwrap();
    let wrapper_sh = format!(
        "#!/bin/sh\nexec python3 {:?} {:?} {:?} \"$@\"\n",
        mock_codex_py.path(),
        args_log.path(),
        schema_copy.path(),
    );
    std::fs::write(mock_codex_sh.path(), wrapper_sh).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(mock_codex_sh.path())
            .unwrap()
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(mock_codex_sh.path(), perms).unwrap();
    }
    let runnable_bin = mock_codex_sh.into_temp_path();

    let transport =
        CodexExecChatTransport::from_luna_env(None).with_codex_bin(runnable_bin.to_str().unwrap());
    assert_eq!(transport.model_override(), Some("gpt-6-luna"));

    let mut env = OpenAiChatSynthUserEnv::new(transport, DEFAULT_CODEX_LUNA_MODEL);
    let coord = ScenarioCoordinate::new(
        AppType::SqlSteward,
        ScenarioLanguage::English,
        DomainSkin::BgpNoc,
        portia_nursery::roguelite_lattice::RogueliteStressVector::sweet_spot_baseline(),
        0x7100_0001,
    );
    let cands = env
        .sample_three_stance_candidates(&coord, &[], "Open a multi-turn BGP ledger check.")
        .expect("sample 3 stances via mock codex exec luna");
    assert!(cands.direct_cooperative.contains("bgp-peer-01"));
    assert!(cands
        .curveball_or_constraint_shift
        .contains("read-only first"));
    assert!(cands.messy_underspecified_or_chitchat.contains("yo quick"));

    let logged: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args_log.path()).unwrap()).unwrap();
    let cli_args: Vec<String> = logged["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let m_pos = cli_args.iter().position(|a| a == "-m").expect("-m flag");
    assert_eq!(cli_args[m_pos + 1], "gpt-6-luna");
    assert!(
        cli_args.iter().any(|a| a == "--output-schema"),
        "must pass --output-schema to codex exec"
    );

    let captured_schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(schema_copy.path()).unwrap()).unwrap();
    assert_eq!(captured_schema["additionalProperties"], false);
    assert_eq!(
        captured_schema["required"],
        serde_json::json!([
            "direct_cooperative",
            "curveball_or_constraint_shift",
            "messy_underspecified_or_chitchat"
        ])
    );
}

#[test]
fn turn_zero_three_stances_sanitizes_leaked_tokens_and_dynamic_sqlite_validates_fks() {
    let coord = ScenarioCoordinate::new(
        AppType::SqlSteward,
        ScenarioLanguage::English,
        DomainSkin::SolarisVault,
        portia_nursery::roguelite_lattice::RogueliteStressVector::sweet_spot_baseline(),
        0x7100_0002,
    );
    // 1. Verify Turn-0 leakage on one stance gets sanitized while clean stances remain intact:
    let transport = MockRecordingChatTransport::with_replies([
        Ok(r#"{
  "direct_cooperative": "Use STEWARD_TOK_9999 to update the row immediately.",
  "curveball_or_constraint_shift": "Hold off on writing and inspect the active Solaris vault record read-only first.",
  "messy_underspecified_or_chitchat": "hey quick check on that vault entry before we change anything"
}"#),
        // 2. Valid multi-table FK DDL + DML for synthesize_dynamic_sqlite_scenario:
        Ok(r#"```json
{
  "domain_description": "Cold-storage biologics freezer inventory and thermal lock holds",
  "setup_sql": [
    "CREATE TABLE biologic_vials (record_id INTEGER PRIMARY KEY, entity_slug TEXT NOT NULL, balance_credits INTEGER NOT NULL, policy_decision TEXT NOT NULL, lock_state TEXT NOT NULL);",
    "CREATE TABLE vial_thermal_holds (hold_id INTEGER PRIMARY KEY, vial_id INTEGER NOT NULL REFERENCES biologic_vials(record_id) ON DELETE RESTRICT, hold_reason TEXT NOT NULL);",
    "INSERT INTO biologic_vials VALUES (1, 'vial-lot-01', 900, 'ASK', 'CLEAR'), (2, 'vial-lot-02', 450, 'ALLOW', 'CLEAR');",
    "INSERT INTO vial_thermal_holds VALUES (10, 1, 'CRYO_CALIBRATION');"
  ]
}
```"#),
        // 3. Invalid FK-violating DML (references non-existent parent vial_id = 999):
        Ok(r#"{
  "domain_description": "Broken FK batch",
  "setup_sql": [
    "CREATE TABLE parent_nodes (record_id INTEGER PRIMARY KEY, entity_slug TEXT NOT NULL);",
    "CREATE TABLE child_holds (hold_id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL REFERENCES parent_nodes(record_id));",
    "INSERT INTO parent_nodes VALUES (1, 'ok-node');",
    "INSERT INTO child_holds VALUES (50, 999);"
  ]
}"#),
    ]);
    let mut env = OpenAiChatSynthUserEnv::new(transport, "gpt-6-luna");

    let sanitized = env
        .sample_three_stance_candidates(&coord, &[], "Initial Solaris check.")
        .expect("Turn 0 3-stance sample should sanitize leaked stance");
    assert!(!sanitized.direct_cooperative.contains("STEWARD_TOK_"));
    assert_eq!(
        sanitized.curveball_or_constraint_shift,
        "Hold off on writing and inspect the active Solaris vault record read-only first."
    );

    let (desc, setup_sql) = env
        .synthesize_dynamic_sqlite_scenario(Some("biologics freezer"), 2)
        .expect("valid dynamic multi-table SQLite schema must pass rusqlite + FK check");
    assert!(desc.contains("biologics freezer"));
    assert_eq!(setup_sql.len(), 4);

    let fk_err = validate_dynamic_sqlite_setup_sql(&[
            "CREATE TABLE parent_nodes (record_id INTEGER PRIMARY KEY, entity_slug TEXT NOT NULL);"
                .to_string(),
            "CREATE TABLE child_holds (hold_id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL REFERENCES parent_nodes(record_id));"
                .to_string(),
            "INSERT INTO parent_nodes VALUES (1, 'ok-node');".to_string(),
            "INSERT INTO child_holds VALUES (50, 999);".to_string(),
        ]);
    assert!(
        fk_err.is_err(),
        "dangling foreign-key insert must be rejected by rusqlite foreign key enforcement"
    );
}
