use super::*;
use portia_core::relay::TypedTurnAction;
use portia_core::{Mode, StewardAuthState};
use rusqlite::Connection;
use std::collections::BTreeSet;

#[test]
fn ask_reply_parser_preserves_unicode_without_inventing_permission() {
    for reply in ["请选择第二行。", "🙂🙂", "ééééé", "الصف الثاني"] {
        assert_eq!(
            parse_ask_user_reply_for_kind(AskPromptKind::Clarification, reply).unwrap(),
            AskUserReply::Clarification {
                clarification: reply.to_string(),
            }
        );
        assert!(parse_ask_user_reply_for_kind(AskPromptKind::Permission, reply).is_err());
    }
    assert_eq!(
        parse_ask_user_reply_for_kind(AskPromptKind::Clarification, "CLARIFY: 请选择第二行。")
            .unwrap(),
        AskUserReply::Clarification {
            clarification: "请选择第二行。".to_string(),
        }
    );
    for (reply, decision) in [
        ("ALLOW: 批准修改", AskDecision::Allow),
        ("DENY — 拒绝修改", AskDecision::Deny),
    ] {
        assert_eq!(
            parse_ask_user_reply(reply).unwrap(),
            AskUserReply::Decision {
                decision,
                raw_reply: reply.to_string(),
            }
        );
    }
}

#[test]
fn ask_reply_parser_enforces_prompt_kind_and_normalizes_negative_replies() {
    // 1. Permission replies accept exact tokens ("no", "rejected", "ALLOW", "yes") and explicit structured headers:
    assert_eq!(
        parse_ask_user_reply("no").unwrap(),
        AskUserReply::Decision {
            decision: AskDecision::Deny,
            raw_reply: "no".to_string(),
        }
    );
    assert_eq!(
        parse_ask_user_reply("DENY: not approved").unwrap(),
        AskUserReply::Decision {
            decision: AskDecision::Deny,
            raw_reply: "DENY: not approved".to_string(),
        }
    );
    assert_eq!(
        parse_ask_user_reply("DENY  — Reject and cancel write").unwrap(),
        AskUserReply::Decision {
            decision: AskDecision::Deny,
            raw_reply: "DENY  — Reject and cancel write".to_string(),
        }
    );
    assert_eq!(
        parse_ask_user_reply("ALLOW").unwrap(),
        AskUserReply::Decision {
            decision: AskDecision::Allow,
            raw_reply: "ALLOW".to_string(),
        }
    );
    assert_eq!(
        parse_ask_user_reply("ALLOW — Authorize database write").unwrap(),
        AskUserReply::Decision {
            decision: AskDecision::Allow,
            raw_reply: "ALLOW — Authorize database write".to_string(),
        }
    );

    // 2. Casual sentences starting with yes/no/allow/deny or unstructured negations return Err:
    for invalid in [
        "maybe",
        "use account 42",
        "yes please tell me more",
        "no problem",
        "allow me to explain",
        "not approved",
    ] {
        assert!(
            parse_ask_user_reply_for_kind(AskPromptKind::Permission, invalid).is_err(),
            "expected Permission parser to reject casual reply {invalid:?}"
        );
    }

    // 3. Clarification prompts classify from the question and preserve free-text clarification without turning it into ALLOW:
    let clar_kind = classify_ask_prompt_kind("Which wallet record should be updated: 101 or 102?");
    assert_eq!(clar_kind, AskPromptKind::Clarification);
    assert_eq!(
        parse_ask_user_reply_for_kind(clar_kind, "wallet 102").unwrap(),
        AskUserReply::Clarification {
            clarification: "wallet 102".to_string(),
        }
    );

    let which_row_kind = classify_ask_prompt_kind("Which row should we update: 101 or 102?");
    assert_eq!(which_row_kind, AskPromptKind::Clarification);
    assert_eq!(
        parse_ask_user_reply_for_kind(which_row_kind, "101").unwrap(),
        AskUserReply::Clarification {
            clarification: "101".to_string(),
        }
    );

    let clarify_modify_kind = classify_ask_prompt_kind("Could you clarify which record to modify?");
    assert_eq!(clarify_modify_kind, AskPromptKind::Clarification);

    let perm_kind = classify_ask_prompt_kind(
        r#"{"kind": "permission", "question": "May we update locked wallet 101?"}"#,
    );
    assert_eq!(perm_kind, AskPromptKind::Permission);
}

#[test]
fn driver_tracks_multi_row_candidate_choices_and_formats_ask_payloads() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
            "CREATE TABLE campaigns (campaign_id INTEGER PRIMARY KEY, slug TEXT NOT NULL, status TEXT NOT NULL, policy TEXT NOT NULL);
             INSERT INTO campaigns VALUES (101, 'MATTERHORN-STARTER-7', 'ACTIVE', 'ALLOW');
             INSERT INTO campaigns VALUES (102, 'ZERMATT-RYE-4', 'ACTIVE', 'ALLOW');",
        )
        .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_candidate_disambiguation(true);

    // 1. Before any multi-row read, clarification question uses ConversationalClarify:
    assert_eq!(
        driver.ask_interaction_mode("Which campaign should we inspect?"),
        AskInteractionMode::ConversationalClarify
    );

    // 2. Multi-row READ populates `task_candidate_choices` and upgrades clarification prompts to `MultiSelect`:
    let read_obs =
        driver.execute_turn(
            Mode::Read,
            "SELECT * FROM campaigns",
            |_| Ok("ALLOW".into()),
        );
    assert!(read_obs.ok);
    assert_eq!(
        driver.task_candidate_choices(),
        &[
            "101 (MATTERHORN-STARTER-7 · ACTIVE)".to_string(),
            "102 (ZERMATT-RYE-4 · ACTIVE)".to_string(),
        ]
    );
    assert_eq!(
        driver.ask_interaction_mode("Which campaign(s) should we update?"),
        AskInteractionMode::MultiSelect {
            question: "Which campaign(s) should we update?".to_string(),
            choices: vec![
                "101 (MATTERHORN-STARTER-7 · ACTIVE)".to_string(),
                "102 (ZERMATT-RYE-4 · ACTIVE)".to_string(),
                NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string(),
            ],
        }
    );

    // 3. Explicit permission cues still route to StewardPermissionChoice:
    assert_eq!(
        driver.ask_interaction_mode("[permission] May we update campaign 101?"),
        AskInteractionMode::StewardPermissionChoice {
            question: "[permission] May we update campaign 101?".to_string(),
            choices: vec![
                "ALLOW — Authorize database write".to_string(),
                "DENY  — Reject and cancel write".to_string(),
                NONE_OF_ABOVE_ENTER_TEXT_CHOICE.to_string(),
            ],
        }
    );

    // 4. `format_ask_payload_for_display` renders element-as-key JSON cleanly:
    assert_eq!(
        format_ask_payload_for_display(
            r#"{"multi": "Which campaigns?", "options": ["101", "102"]}"#
        ),
        "Which campaigns? [101 | 102]"
    );
    assert_eq!(
        format_ask_payload_for_display(
            r#"{"select": "Authorize write?", "options": ["ALLOW", "DENY"]}"#
        ),
        "Authorize write? [ALLOW | DENY]"
    );
    assert_eq!(
        format_ask_payload_for_display(r#"{"input": "What campaign ID?"}"#),
        "What campaign ID?"
    );

    // 5. `begin_new_task` clears `task_candidate_choices`:
    driver.begin_new_task();
    assert!(driver.task_candidate_choices().is_empty());
}

#[test]
fn driver_tracks_insert_update_delete_operations_and_authorizer_provenance_on_union_and_subqueries()
{
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
            "CREATE TABLE accounts (id INTEGER PRIMARY KEY, holder TEXT NOT NULL, balance INT NOT NULL);
             CREATE TABLE audit_flags (account_id INTEGER PRIMARY KEY, flagged INT NOT NULL);
             INSERT INTO accounts VALUES (1, 'Inanna', 500);
             INSERT INTO audit_flags VALUES (1, 0);",
        )
        .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_candidate_disambiguation(true);

    // Metadata inspection runs 0 SELECT COUNT(*) queries (row_count == 0):
    let meta = driver.inspect_schema_metadata().unwrap();
    assert_eq!(meta.len(), 2);
    assert_eq!(meta[0].row_count, 0);

    // Read query with UNION and correlated EXISTS subquery records both `accounts` and `audit_flags` via SQLite authorizer:
    let union_subquery_sql = "
            SELECT id, holder FROM accounts WHERE EXISTS (SELECT 1 FROM audit_flags WHERE audit_flags.account_id = accounts.id)
            UNION
            SELECT account_id AS id, 'flag' AS holder FROM audit_flags
        ";
    let read_obs = driver.execute_turn(Mode::Read, union_subquery_sql, |_| Ok("ALLOW".into()));
    assert!(read_obs.ok, "read failed: {:?}", read_obs.error);

    // Because `audit_flags` was read inside the EXISTS/UNION query, an INSERT on `audit_flags` now succeeds and reports operation = "INSERT":
    let ins_obs = driver.execute_turn(Mode::Write, "INSERT INTO audit_flags VALUES (2, 1)", |_| {
        Ok("ALLOW".into())
    });
    assert!(ins_obs.ok, "insert failed: {:?}", ins_obs.error);
    assert_eq!(
        ins_obs.operation,
        Some(ObservationOperation::SqlWrite(SqlWriteOp::Insert))
    );

    // Re-read `audit_flags`, then DELETE and verify operation = "DELETE":
    let reread = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM audit_flags WHERE account_id = 2",
        |_| Ok("ALLOW".into()),
    );
    assert!(reread.ok);
    let del_obs = driver.execute_turn(
        Mode::Write,
        "DELETE FROM audit_flags WHERE account_id = 2",
        |_| Ok("ALLOW".into()),
    );
    assert!(del_obs.ok, "delete failed: {:?}", del_obs.error);
    assert_eq!(
        del_obs.operation,
        Some(ObservationOperation::SqlWrite(SqlWriteOp::Delete))
    );
}

#[test]
fn write_budget_counts_after_update_trigger_side_effects_and_rolls_back_savepoint() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
            "CREATE TABLE wallets (id INTEGER PRIMARY KEY, balance INT NOT NULL);
             CREATE TABLE shadow_ledger (id INTEGER PRIMARY KEY, wallet_id INT NOT NULL, delta INT NOT NULL);
             INSERT INTO wallets VALUES (1, 100);
             INSERT INTO shadow_ledger VALUES (10, 1, 0), (11, 1, 0), (12, 1, 0);
             CREATE TRIGGER trg_fanout_after_update AFTER UPDATE ON wallets
             BEGIN
                 UPDATE shadow_ledger SET delta = NEW.balance - OLD.balance WHERE wallet_id = NEW.id;
             END;",
        )
        .unwrap();
    // max_write_rows = 1: direct UPDATE touches 1 row in `wallets`, but AFTER UPDATE trigger modifies 3 rows in `shadow_ledger` (4 total changes > 1)
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 1).unwrap();

    let read_obs = driver.execute_turn(Mode::Read, "SELECT * FROM wallets WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert!(read_obs.ok);
    assert_eq!(
        read_obs.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );

    let write_obs = driver.execute_turn(
        Mode::Write,
        "UPDATE wallets SET balance = 250 WHERE id = 1",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        !write_obs.ok,
        "expected trigger fanout to exceed max_write_rows=1"
    );
    assert_eq!(write_obs.db_version, 0);
    assert!(write_obs
        .error
        .as_deref()
        .unwrap_or("")
        .contains("MaxWriteRowsExceeded"));

    // Verify SAVEPOINT rolled back both `wallets` and `shadow_ledger`:
    let verify_read = driver.execute_turn(
        Mode::Read,
        "SELECT balance FROM wallets WHERE id = 1",
        |_| Ok("ALLOW".into()),
    );
    assert_eq!(verify_read.rows[0][0], serde_json::json!(100));
    assert_eq!(
        verify_read.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );
}

#[test]
fn begin_new_task_resets_read_purpose_to_candidate_after_prior_task_write() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, qty INT NOT NULL);
             INSERT INTO items VALUES (1, 10), (2, 20);",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_candidate_disambiguation(true);

    // Task 1: Read candidate -> Write -> Read verification
    driver.begin_new_task();
    let r1 = driver.execute_turn(Mode::Read, "SELECT * FROM items WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert_eq!(
        r1.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );

    let w1 = driver.execute_turn(
        Mode::Write,
        "UPDATE items SET qty = 15 WHERE id = 1",
        |_| Ok("ALLOW".into()),
    );
    assert!(w1.ok);
    assert_eq!(driver.db_version(), 1);
    assert_eq!(driver.task_writes_committed(), 1);

    let r2 = driver.execute_turn(Mode::Read, "SELECT * FROM items WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert_eq!(
        r2.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
    );

    // Task 2 in same session: db_version is still 1, but begin_new_task() resets task_writes_committed to 0
    // so the initial READ of Task 2 is classified as "candidate", not "verification"!
    driver.begin_new_task();
    assert_eq!(driver.db_version(), 1);
    assert_eq!(driver.task_writes_committed(), 0);
    let task2_r1 = driver.execute_turn(Mode::Read, "SELECT * FROM items WHERE id = 2", |_| {
        Ok("ALLOW".into())
    });
    assert_eq!(
        task2_r1.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );
}

#[test]
fn writes_increment_versions_on_all_trigger_touched_tables_without_requiring_rereads() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, val INT NOT NULL);
             CREATE TABLE b (id INTEGER PRIMARY KEY, val INT NOT NULL);
             INSERT INTO a VALUES (1, 10);
             INSERT INTO b VALUES (1, 20);
             CREATE TRIGGER trg_a_updates_b AFTER UPDATE ON a
             BEGIN
                 UPDATE b SET val = NEW.val WHERE id = NEW.id;
             END;",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_candidate_disambiguation(true);

    // 1. Discover both tables at version 0:
    let r_a = driver.execute_turn(Mode::Read, "SELECT * FROM a WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert!(r_a.ok);
    let r_b = driver.execute_turn(Mode::Read, "SELECT * FROM b WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert!(r_b.ok);

    // 2. UPDATE `a`, whose AFTER UPDATE trigger modifies `b`:
    let w_a = driver.execute_turn(Mode::Write, "UPDATE a SET val = 99 WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert!(w_a.ok, "write on `a` failed: {:?}", w_a.error);
    assert_eq!(w_a.table_versions.get("a"), Some(&1));
    assert_eq!(w_a.table_versions.get("b"), Some(&1));

    // A subsequent write to the trigger-touched table needs no additional read.
    let w_b = driver.execute_turn(Mode::Write, "UPDATE b SET val = 100 WHERE id = 1", |_| {
        Ok("ALLOW".into())
    });
    assert!(w_b.ok, "write on `b` failed: {:?}", w_b.error);
    assert_eq!(w_b.table_versions.get("b"), Some(&2));
}

#[test]
fn causal_epistemic_schema_discovery_and_prompt_blind_ask_classification() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
            "CREATE TABLE workspace_accounts (account_id INTEGER PRIMARY KEY, slug TEXT NOT NULL, tier TEXT NOT NULL, policy TEXT NOT NULL);
             CREATE TABLE api_rate_policies (policy_id INTEGER PRIMARY KEY, name TEXT NOT NULL, rpm INT NOT NULL);
             INSERT INTO workspace_accounts VALUES (101, 'CHAMONIX-RYE', 'ENTERPRISE', 'ASK');
             INSERT INTO workspace_accounts VALUES (102, 'ZERMATT-STARTER', 'PRO', 'ALLOW');
             INSERT INTO api_rate_policies VALUES (1, 'DEFAULT-TIER', 60);",
        )
        .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_steward_policy_columns(["policy"])
        .with_candidate_disambiguation(true);

    // 1. Before any `Probe` or `Read`, `catalog_discovered` is false, `probed_table_names` is empty,
    //    and Turn-0 `Mode::Ask` prose yields to prompt (`ConversationalClarify`):
    assert!(!driver.catalog_discovered());
    assert!(driver.probed_table_names().is_empty());
    assert!(!driver.task_steward_grant_approved());
    let (all_tbls, probed_0) = driver.inspect_epistemic_schema_state().unwrap();
    assert_eq!(
        all_tbls,
        vec![
            "api_rate_policies".to_string(),
            "workspace_accounts".to_string()
        ]
    );
    assert!(probed_0.is_empty());
    assert_eq!(
        driver.classify_coroutine_step_effect(
            Mode::Ask,
            "Sure, I'll help you delete the chamonix rye account.",
        ),
        CoroutineStepEffect::YieldToPrompt
    );

    // 2. Probing `sqlite_master` sets `catalog_discovered = true` without marking user tables probed:
    let cat_obs = driver.execute_turn(
        Mode::Probe,
        "SELECT name FROM sqlite_master WHERE type = 'table';",
        |_| Ok("ALLOW".into()),
    );
    assert!(cat_obs.ok);
    assert!(driver.catalog_discovered());
    assert!(driver.probed_table_names().is_empty());

    // 3. `PRAGMA table_info(workspace_accounts)` records `workspace_accounts` into `probed_table_names`:
    let pragma_obs = driver.execute_turn(
        Mode::Probe,
        "PRAGMA table_info(workspace_accounts);",
        |_| Ok("ALLOW".into()),
    );
    assert!(pragma_obs.ok);
    assert!(driver.probed_table_names().contains("workspace_accounts"));
    let (_, probed_1) = driver.inspect_epistemic_schema_state().unwrap();
    assert_eq!(probed_1.len(), 1);
    assert_eq!(probed_1[0].table_name, "workspace_accounts");

    // 4. Multi-row `Mode::Read` populates `task_candidate_choices` -> `MultiSelect`:
    let r0 = driver.execute_turn(Mode::Read, "SELECT * FROM workspace_accounts;", |_| {
        Ok("ALLOW".into())
    });
    assert!(r0.ok);
    assert!(matches!(
        driver
            .classify_coroutine_step_effect(Mode::Ask, "Which workspace account should we update?"),
        CoroutineStepEffect::SuspendForGuiChoice {
            gui_mode: AskInteractionMode::MultiSelect { .. },
            ..
        }
    ));

    // 5. Even when `session_last_read_rows` has >= 2 rows from Task 1, starting Task 2 (`has_task_db_read == false`)
    //    never turns a Turn-0 Chitchat turn into `StewardPermissionChoice` or `MultiSelect`!
    driver.begin_new_task();
    assert_eq!(driver.session_last_read_rows().len(), 2);
    assert_eq!(
        driver.classify_coroutine_step_effect(Mode::Ask, "Hi there! How can I help you today?",),
        CoroutineStepEffect::YieldToPrompt
    );
    assert_eq!(
        driver.ask_interaction_mode("Which workspace account should I delete?"),
        AskInteractionMode::ConversationalClarify
    );

    // 6. Reading an `'ASK'` policy row sets `has_pending_steward_ask_row`, and recording `ALLOW` vs `DENY`
    //    toggles `task_steward_grant_approved` (resetting on `begin_new_task`):
    let r_ask = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM workspace_accounts WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(r_ask.ok);
    assert!(driver.has_pending_steward_ask_row());
    assert!(matches!(
        driver.classify_coroutine_step_effect(Mode::Ask, "Proceed with update?"),
        CoroutineStepEffect::SuspendForGuiChoice {
            gui_mode: AskInteractionMode::StewardPermissionChoice { .. },
            ..
        }
    ));
    let allow_obs = driver.execute_turn(Mode::Ask, "Proceed with update?", |_| Ok("ALLOW".into()));
    assert!(allow_obs.ok);
    assert!(driver.task_steward_grant_approved());
    let deny_obs = driver.execute_turn(Mode::Ask, "Proceed with update?", |_| Ok("DENY".into()));
    assert!(deny_obs.ok);
    assert!(!driver.task_steward_grant_approved());

    // 7. `clear_session_candidate_rows` clears `catalog_discovered` and `probed_table_names`,
    //    and `mark_all_tables_probed` marks all user tables probed:
    driver.clear_session_candidate_rows();
    assert!(!driver.catalog_discovered());
    assert!(driver.probed_table_names().is_empty());
    driver.mark_all_tables_probed().unwrap();
    assert!(driver.catalog_discovered());
    assert_eq!(driver.probed_table_names().len(), 2);
}

#[test]
fn driver_executes_natural_language_abstain_and_multi_table_join_group_by_or_queries() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE campaigns (
                campaign_id INTEGER PRIMARY KEY,
                slug TEXT NOT NULL,
                tier TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE ledger_entries (
                entry_id INTEGER PRIMARY KEY,
                campaign_id INTEGER NOT NULL,
                amount INTEGER NOT NULL
            );
            INSERT INTO campaigns VALUES
                (101, 'MATTERHORN-7', 'ENTERPRISE', 'ACTIVE'),
                (102, 'ZERMATT-4', 'ENTERPRISE', 'PENDING'),
                (103, 'CHAMONIX-2', 'STARTER', 'ARCHIVED');
            INSERT INTO ledger_entries VALUES
                (1, 101, 300),
                (2, 101, 200),
                (3, 102, 400),
                (4, 103, 100);",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5)
        .unwrap()
        .with_candidate_disambiguation(true);

    // 1. Natural-language and JSON `Mode::Abstain` refusals both execute cleanly and yield to prompt:
    let prose_refusal = "Refusing write: campaign 103 is ARCHIVED and locked against modification.";
    let abstain_prose_obs =
        driver.execute_turn(Mode::Abstain, prose_refusal, |_| Ok("ALLOW".into()));
    assert!(abstain_prose_obs.ok);
    assert_eq!(abstain_prose_obs.mode, Mode::Abstain);
    assert_eq!(abstain_prose_obs.message.as_deref(), Some(prose_refusal));
    assert_eq!(
        driver.classify_coroutine_step_effect(Mode::Abstain, prose_refusal),
        CoroutineStepEffect::YieldToPrompt
    );

    let typed_prose_action = TypedTurnAction::parse_for_mode(Mode::Abstain, prose_refusal).unwrap();
    let typed_abstain_obs =
        driver.execute_typed_action(&typed_prose_action, |_| Ok("ALLOW".into()));
    assert!(typed_abstain_obs.ok);
    assert_eq!(typed_abstain_obs.mode, Mode::Abstain);

    let json_refusal = r#"{"status": "ABSTAINED", "reason": "campaign locked"}"#;
    let abstain_json_obs = driver.execute_turn(Mode::Abstain, json_refusal, |_| Ok("ALLOW".into()));
    assert!(abstain_json_obs.ok);
    assert_eq!(abstain_json_obs.message.as_deref(), Some(json_refusal));

    // 2. Multi-table `SELECT ... JOIN ... WHERE ... OR ... GROUP BY ... ORDER BY ...` executes
    //    and records fresh read provenance for BOTH `campaigns` and `ledger_entries`:
    let join_group_sql = "
            SELECT c.tier, COUNT(*) AS entry_count, SUM(l.amount) AS total_amount
            FROM campaigns c
            JOIN ledger_entries l ON c.campaign_id = l.campaign_id
            WHERE c.status = 'ACTIVE' OR c.status = 'PENDING'
            GROUP BY c.tier
            ORDER BY total_amount DESC;
        ";
    let read_obs = driver.execute_turn(Mode::Read, join_group_sql, |_| Ok("ALLOW".into()));
    assert!(
        read_obs.ok,
        "multi-table JOIN + GROUP BY failed: {:?}",
        read_obs.error
    );
    assert_eq!(
        read_obs.columns,
        vec![
            "tier".to_string(),
            "entry_count".to_string(),
            "total_amount".to_string()
        ]
    );
    assert_eq!(
        read_obs.rows,
        vec![vec![
            serde_json::json!("ENTERPRISE"),
            serde_json::json!(3),
            serde_json::json!(900),
        ]]
    );

    // Both `campaigns` and `ledger_entries` were read in the JOIN query, so writes on both succeed:
    let write_campaign = driver.execute_turn(
        Mode::Write,
        "UPDATE campaigns SET status = 'ACTIVE' WHERE campaign_id = 102;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        write_campaign.ok,
        "expected write on `campaigns` after JOIN read to succeed: {:?}",
        write_campaign.error
    );
    let write_ledger = driver.execute_turn(
        Mode::Write,
        "UPDATE ledger_entries SET amount = 450 WHERE entry_id = 3;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        write_ledger.ok,
        "expected write on `ledger_entries` after JOIN read to succeed: {:?}",
        write_ledger.error
    );
}

#[test]
fn probe_steward_ask_external_mutation_and_session_reset_enforce_fail_closed_guards() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let setup = Connection::open(tmp.path()).unwrap();
        setup
            .execute_batch(
                "CREATE TABLE accounts (
                        account_id INTEGER PRIMARY KEY,
                        slug TEXT NOT NULL,
                        balance INTEGER NOT NULL,
                        policy_decision TEXT NOT NULL
                    );
                    CREATE TABLE audit_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        account_id INTEGER NOT NULL REFERENCES accounts(account_id),
                        note TEXT NOT NULL
                    );
                    CREATE TRIGGER trg_accounts_audit
                    AFTER UPDATE ON accounts
                    BEGIN
                        INSERT INTO audit_log (account_id, note)
                        VALUES (NEW.account_id, 'updated:' || NEW.balance);
                    END;
                    INSERT INTO accounts VALUES
                        (101, 'MATTERHORN-7', 500, 'ALLOW'),
                        (102, 'CHAMONIX-3', 900, 'ASK'),
                        (103, 'ZERMATT-9', 300, 'ALLOW');",
            )
            .unwrap();
    }

    let mut driver = LiveSqliteDailyDriver::open(tmp.path(), 2)
        .unwrap()
        .with_steward_policy_columns(["policy_decision"])
        .with_candidate_disambiguation(true);

    // 1. Schema discovery makes WRITE available without a row read.
    let probe_obs = driver.execute_turn(Mode::Probe, "PRAGMA table_info(accounts);", |_| {
        Ok("ALLOW".into())
    });
    assert!(probe_obs.ok);
    assert!(driver.probed_table_names().contains("accounts"));
    let premature_write = driver.execute_turn(
        Mode::Write,
        "UPDATE accounts SET balance = 550 WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        premature_write.ok && premature_write.rows_affected == 2,
        "schema discovery permits a guarded WRITE without a prior row read"
    );

    // 2. Bug D2 regression (both direct ASK read and sneaky ALLOW-read -> ASK-write):
    let read_allow = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM accounts WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(read_allow.ok);
    assert!(!driver.has_pending_steward_ask_row());
    let sneaky_ask_write = driver.execute_turn(
        Mode::Write,
        "UPDATE accounts SET balance = 999 WHERE account_id = 102;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        !sneaky_ask_write.ok && sneaky_ask_write.rows_affected == 0,
        "unapproved ASK row write must fail closed and roll back trigger side effects"
    );
    let check_conn = Connection::open(tmp.path()).unwrap();
    let bal_102: i64 = check_conn
        .query_row(
            "SELECT balance FROM accounts WHERE account_id = 102;",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let audit_cnt: i64 = check_conn
        .query_row("SELECT COUNT(*) FROM audit_log;", [], |r| r.get(0))
        .unwrap();
    assert_eq!(bal_102, 900);
    assert_eq!(audit_cnt, 1);

    // Reading row 102 directly sets `pending_steward_ask`; Steward ALLOW unlocks the write:
    let read_ask = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM accounts WHERE account_id = 102;",
        |_| Ok("ALLOW".into()),
    );
    assert!(read_ask.ok && driver.has_pending_steward_ask_row());
    let blocked_ask_write = driver.execute_turn(
        Mode::Write,
        "UPDATE accounts SET balance = 925 WHERE account_id = 102;",
        |_| Ok("ALLOW".into()),
    );
    assert!(!blocked_ask_write.ok && blocked_ask_write.rows_affected == 0);

    let grant = driver.execute_turn(
        Mode::Ask,
        "[permission] Authorize balance update on account 102?",
        |_| Ok("ALLOW".into()),
    );
    assert!(grant.ok && driver.task_steward_grant_approved());
    let approved_write = driver.execute_turn(
        Mode::Write,
        "UPDATE accounts SET balance = 925 WHERE account_id = 102;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        approved_write.ok,
        "expected Steward-approved ASK write to succeed: {:?}",
        approved_write.error
    );
    assert_eq!(approved_write.rows_affected, 2); // 1 direct + 1 trigger row in `audit_log`

    // 3. External mutations advance the causal version without imposing a reread.
    let re_read = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM accounts WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(re_read.ok);
    let v_before_ext = EpistemicDomainEnv::state_version(&driver);
    check_conn
        .execute(
            "UPDATE accounts SET balance = 333 WHERE account_id = 103;",
            [],
        )
        .unwrap();
    assert_eq!(
        EpistemicDomainEnv::state_version(&driver),
        v_before_ext + 1,
        "external SQLite commit must immediately bump state_version via PRAGMA data_version"
    );
    let stale_write_after_ext = driver.execute_turn(
        Mode::Write,
        "UPDATE accounts SET balance = 510 WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        stale_write_after_ext.ok && stale_write_after_ext.rows_affected == 2,
        "external mutation does not impose a mandatory row read"
    );

    // 4. Reset clears discovered schemas, so generated WRITE is unavailable until discovery.
    let fresh_again = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM accounts WHERE account_id = 101;",
        |_| Ok("ALLOW".into()),
    );
    assert!(fresh_again.ok);
    EpistemicDomainEnv::reset_epistemic_session(&mut driver);
    assert!(!EpistemicDomainEnv::legal_modes(&driver)
        .unwrap()
        .contains(&Mode::Write));
    assert!(!driver.task_steward_grant_approved());
}

#[test]
fn discovered_empty_tables_allow_insert_and_zero_row_updates_without_reads() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INTEGER);
                            CREATE TABLE hidden (id INTEGER PRIMARY KEY);",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 1).unwrap();
    assert!(!driver.legal_modes().unwrap().contains(&Mode::Write));
    assert!(
        driver
            .execute_turn(Mode::Probe, "PRAGMA table_info(entries);", |_| Ok(
                String::new()
            ))
            .ok
    );
    assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
    let portia_core::GrammarCartridgeSpec::LarkGrammar(grammar) =
        driver.payload_grammar_spec(Mode::Write).unwrap()
    else {
        panic!("expected schema-constrained SQL grammar");
    };
    assert!(grammar.contains("entries"));
    assert!(!grammar.contains("hidden"));
    let empty_read =
        driver.execute_turn(Mode::Read, "SELECT * FROM entries WHERE id = 99;", |_| {
            Ok(String::new())
        });
    assert!(empty_read.ok && empty_read.rows.is_empty());
    assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
    let no_match = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 5 WHERE id = 99;",
        |_| Ok(String::new()),
    );
    assert!(no_match.ok);
    assert_eq!(no_match.rows_affected, 0);
    assert_eq!(driver.db_version(), 0);
    assert_eq!(driver.task_writes_committed(), 0);
    let insert = driver.execute_turn(Mode::Write, "INSERT INTO entries VALUES (99, 5);", |_| {
        Ok(String::new())
    });
    assert!(insert.ok);
    assert_eq!(insert.rows_affected, 1);
    assert_eq!(driver.db_version(), 1);
}

#[test]
fn idempotent_updates_preserve_epochs_and_sqlite_affected_counts() {
    for in_memory in [true, false] {
        let file = tempfile::NamedTempFile::new().unwrap();
        let conn = if in_memory {
            Connection::open_in_memory().unwrap()
        } else {
            Connection::open(file.path()).unwrap()
        };
        conn.execute_batch(
                "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INT, note TEXT, payload BLOB, ratio REAL, optional TEXT);
                 INSERT INTO entries VALUES (1, 10, 'same', X'0001', 1.25, NULL);",
            ).unwrap();
        let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5).unwrap();
        for _ in 0..2 {
            let unchanged = driver.execute_turn(
                    Mode::Write,
                    "UPDATE entries SET value = 10, note = note, payload = payload, ratio = ratio, optional = optional WHERE id = 1;",
                    |_| Ok(String::new()),
                );
            assert!(unchanged.ok, "{:?}", unchanged.error);
            assert_eq!(unchanged.rows_affected, 1);
            assert_eq!(driver.db_version(), 0);
            assert_eq!(unchanged.table_versions["entries"], 0);
        }
        assert_eq!(driver.task_writes_committed(), 2);
        let read = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
        assert_eq!(
            read.operation,
            Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
        );
        let changed = driver.execute_turn(
            Mode::Write,
            "UPDATE entries SET value = 11 WHERE id = 1;",
            |_| Ok(String::new()),
        );
        assert!(changed.ok);
        assert_eq!(driver.db_version(), 1);
        let read = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
        assert_eq!(
            read.operation,
            Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
        );
        let unchanged = driver.execute_turn(
            Mode::Write,
            "UPDATE entries SET value = 11 WHERE id = 1;",
            |_| Ok(String::new()),
        );
        assert!(unchanged.ok);
        assert_eq!(driver.db_version(), 1);
        let read = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
        assert_eq!(
            read.operation,
            Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
        );
        let follow_up_read =
            driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
        assert_eq!(
            follow_up_read.operation,
            Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
        );
    }
}

#[test]
fn trigger_changes_advance_epoch_even_when_target_values_are_identical() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INT);
             CREATE TABLE counters (id INTEGER PRIMARY KEY, value INT);
             INSERT INTO entries VALUES (1, 10);
             INSERT INTO counters VALUES (1, 0);
             CREATE TRIGGER count_updates AFTER UPDATE ON entries BEGIN
                 UPDATE counters SET value = value + 1 WHERE id = 1;
             END;",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 2).unwrap();
    for epoch in 1..=2 {
        let write = driver.execute_turn(
            Mode::Write,
            "UPDATE entries SET value = 10 WHERE id = 1;",
            |_| Ok(String::new()),
        );
        assert!(write.ok, "{:?}", write.error);
        assert_eq!(write.rows_affected, 2);
        assert_eq!(driver.db_version(), epoch);
        assert_eq!(write.table_versions["entries"], 0);
        assert_eq!(write.table_versions["counters"], epoch);
    }
    driver.max_write_rows = 1;
    let rollback = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 10 WHERE id = 1;",
        |_| Ok(String::new()),
    );
    assert!(!rollback.ok);
    assert_eq!(driver.db_version(), 2);
    let read = driver.execute_turn(Mode::Read, "SELECT value FROM counters;", |_| {
        Ok(String::new())
    });
    assert_eq!(read.rows, vec![vec![serde_json::json!(2)]]);
}

#[test]
fn mutation_epochs_cover_without_rowid_blobs_and_trigger_only_rowid_changes() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE tokens (key TEXT PRIMARY KEY, payload BLOB) WITHOUT ROWID;
             INSERT INTO tokens VALUES ('a', X'0001');
             CREATE TABLE entries (id INTEGER PRIMARY KEY, value INT);
             INSERT INTO entries VALUES (1, 10);
             CREATE TABLE labels (label TEXT);
             INSERT INTO labels VALUES ('same');
             CREATE TRIGGER move_row AFTER UPDATE ON entries BEGIN
                 UPDATE labels SET rowid = rowid + 1;
             END;",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 2).unwrap();
    let unchanged = driver.execute_turn(
        Mode::Write,
        "UPDATE tokens SET payload = X'0001' WHERE key = 'a';",
        |_| Ok(String::new()),
    );
    assert!(unchanged.ok, "{:?}", unchanged.error);
    assert_eq!(driver.db_version(), 0);
    let changed = driver.execute_turn(
        Mode::Write,
        "UPDATE tokens SET payload = X'0002' WHERE key = 'a';",
        |_| Ok(String::new()),
    );
    assert!(changed.ok, "{:?}", changed.error);
    assert_eq!(driver.db_version(), 1);
    let rowid = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 10 WHERE id = 1;",
        |_| Ok(String::new()),
    );
    assert!(rowid.ok);
    assert_eq!(driver.db_version(), 2);
    assert_eq!(rowid.table_versions["labels"], 1);
    assert_eq!(rowid.table_versions["entries"], 0);
}

#[test]
fn ignored_autoincrement_insert_still_advances_metadata_epoch() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT UNIQUE);
             INSERT INTO entries (value) VALUES ('same');",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 1).unwrap();
    let ignored = driver.execute_turn(
        Mode::Write,
        "INSERT OR IGNORE INTO entries (value) VALUES ('same');",
        |_| Ok(String::new()),
    );
    assert!(ignored.ok, "{:?}", ignored.error);
    assert_eq!(ignored.rows_affected, 0);
    assert_eq!(
        driver.db_version(),
        1,
        "sqlite_sequence changed despite zero inserted rows"
    );
    assert_eq!(ignored.table_versions["entries"], 0);
    assert_eq!(driver.task_writes_committed(), 0);
}

#[test]
fn read_only_and_verification_purposes_follow_observed_table_versions() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INTEGER);
                            CREATE TABLE unrelated (id INTEGER PRIMARY KEY);
                            INSERT INTO entries VALUES (1, 10);",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 1).unwrap();
    driver.set_read_only(true).unwrap();
    let readonly = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
    assert_eq!(
        readonly.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::ReadOnly))
    );
    assert!(!driver.legal_modes().unwrap().contains(&Mode::Write));
    let denied = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 11 WHERE id = 1;",
        |_| Ok(String::new()),
    );
    assert!(!denied.ok);
    assert!(denied.error.unwrap().contains("ReadOnlySessionViolation"));
    driver.set_read_only(false).unwrap();
    assert!(
        driver
            .execute_turn(
                Mode::Write,
                "UPDATE entries SET value = 11 WHERE id = 1;",
                |_| Ok(String::new())
            )
            .ok
    );
    let other = driver.execute_turn(
        Mode::Read,
        "SELECT * FROM unrelated;",
        |_| Ok(String::new()),
    );
    assert_eq!(
        other.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );
    let verify = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
    assert_eq!(
        verify.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
    );
    assert_eq!(verify.rows[0][1], serde_json::json!(11));
    let reread = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
    assert_eq!(
        reread.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );
    driver.begin_new_task();
    let next_task =
        driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
    assert_eq!(
        next_task.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Candidate))
    );
}

#[test]
fn unobserved_steward_write_rolls_back_then_structured_allow_resumes() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE entries (id INTEGER PRIMARY KEY, value INTEGER, policy TEXT);
                            CREATE TABLE audit (id INTEGER);
                            INSERT INTO entries VALUES (1, 10, 'ALLOW'), (2, 20, 'ASK');
                            CREATE TRIGGER audit_write AFTER UPDATE ON entries BEGIN INSERT INTO audit VALUES (NEW.id); END;").unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 2)
        .unwrap()
        .with_steward_policy_columns(["policy"]);
    assert!(
        driver
            .execute_turn(Mode::Probe, "PRAGMA table_info(entries);", |_| Ok(
                String::new()
            ))
            .ok
    );
    assert!(
        driver
            .execute_turn(
                Mode::Write,
                "UPDATE entries SET value = 11 WHERE id = 1;",
                |_| Ok(String::new())
            )
            .ok
    );
    driver.begin_new_task();
    let blocked = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 21 WHERE id = 2;",
        |_| Ok(String::new()),
    );
    assert!(!blocked.ok);
    assert!(driver.has_pending_steward_ask_row());
    assert!(!driver.legal_modes().unwrap().contains(&Mode::Write));
    assert_eq!(
        driver
            .conn
            .query_row("SELECT count(*) FROM audit;", [], |row| row
                .get::<_, u64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        driver
            .conn
            .query_row("SELECT value FROM entries WHERE id = 2;", [], |row| row
                .get::<_, u64>(0))
            .unwrap(),
        20
    );
    let question = r#"{"select":"Authorize the requested change?","options":["ALLOW","DENY"]}"#;
    assert!(matches!(
        driver.classify_coroutine_step_effect(Mode::Ask, question),
        CoroutineStepEffect::SuspendForGuiChoice { .. }
    ));
    let grant = driver.execute_turn(Mode::Ask, question, |_| Ok("ALLOW".into()));
    assert!(grant.ok);
    assert!(driver.task_steward_grant_approved());
    assert!(driver.legal_modes().unwrap().contains(&Mode::Write));
    let write = driver.execute_turn(
        Mode::Write,
        "UPDATE entries SET value = 21 WHERE id = 2;",
        |_| Ok(String::new()),
    );
    assert!(write.ok);
    assert_eq!(write.rows_affected, 2);
    let verify = driver.execute_turn(Mode::Read, "SELECT * FROM entries WHERE id = 2;", |_| {
        Ok(String::new())
    });
    assert_eq!(
        verify.operation,
        Some(ObservationOperation::SqlRead(ReadPurpose::Verification))
    );
    assert_eq!(verify.rows[0][1], serde_json::json!(21));
    driver.begin_new_task();
    assert!(!driver.task_steward_grant_approved());
}

#[test]
fn monotonic_steward_auth_latches_on_mixed_read_and_opened_read_only_refuses_unlock() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INT, \"a.b\" TEXT);
                 INSERT INTO entries VALUES (1, 10, 'ALLOW'), (2, 20, 'ASK');",
        )
        .unwrap();
    }
    let mut ro = LiveSqliteDailyDriver::open_read_only(tmp.path()).unwrap();
    assert!(ro.set_read_only(false).is_err());

    let mut driver = LiveSqliteDailyDriver::open(tmp.path(), 2)
        .unwrap()
        .with_steward_policy_columns(["\"a.b\""]);
    let mixed = driver.execute_turn(Mode::Read, "SELECT * FROM entries;", |_| Ok(String::new()));
    assert!(mixed.ok);
    assert_eq!(mixed.steward_auth, StewardAuthState::PendingAsk);
    assert_eq!(driver.steward_auth(), StewardAuthState::PendingAsk);

    // Subsequent SELECT on ALLOW-only row must NOT clobber PendingAsk back to NotRequired:
    let allow_only = driver.execute_turn(Mode::Read, "SELECT * FROM entries WHERE id = 1;", |_| {
        Ok(String::new())
    });
    assert!(allow_only.ok);
    assert_eq!(allow_only.steward_auth, StewardAuthState::PendingAsk);
    assert_eq!(driver.steward_auth(), StewardAuthState::PendingAsk);
}

#[test]
fn invalid_sql_and_constraint_failures_are_observations() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY, value INTEGER NOT NULL);
                            INSERT INTO entries VALUES (1, 10);",
    )
    .unwrap();
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 1).unwrap();
    for sql in [
        "UPDATE entries SET missing = 1;",
        "INSERT INTO entries VALUES (1, 20);",
        "DELETE FROM entries; DELETE FROM entries;",
    ] {
        let action = TypedTurnAction::WriteMutation {
            sql: sql.to_string(),
        };
        let obs = driver.execute_turn_action(&action, &mut |_| Ok(String::new()));
        assert!(!obs.ok, "unexpected success for {sql}");
        assert!(obs.error.is_some());
        assert_eq!(obs.rows_affected, 0);
        assert_eq!(driver.db_version(), 0);
    }
    let read = driver.execute_turn(Mode::Read, "SELECT missing FROM entries;", |_| {
        Ok(String::new())
    });
    assert!(!read.ok && read.error.is_some());
    assert_eq!(
        driver
            .conn
            .query_row("SELECT value FROM entries WHERE id = 1;", [], |row| row
                .get::<_, u64>(0))
            .unwrap(),
        10
    );
}

mod stateful_proptests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    enum DriverTestOp {
        ProbeCatalog,
        ProbeTableAccounts,
        ProbeTableAudit,
        ReadSingleAllowRow,
        ReadSingleAskRow,
        ReadMultiRowAmbiguous,
        ReadJoinAccountsAndAudit,
        AskStewardAllow,
        AskStewardDeny,
        AskClarifyCandidate { pick_ask_row: bool },
        WriteUpdateAllowRow,
        WriteUpdateAskRow,
        WriteInsertAuditRow,
        WriteExceedRowBudget,
        ExternalConcurrentMutateAccounts,
        BeginNewTask,
        ResetEpistemicSession,
    }

    fn arb_driver_test_op() -> impl Strategy<Value = DriverTestOp> {
        prop_oneof![
            Just(DriverTestOp::ProbeCatalog),
            Just(DriverTestOp::ProbeTableAccounts),
            Just(DriverTestOp::ProbeTableAudit),
            Just(DriverTestOp::ReadSingleAllowRow),
            Just(DriverTestOp::ReadSingleAskRow),
            Just(DriverTestOp::ReadMultiRowAmbiguous),
            Just(DriverTestOp::ReadJoinAccountsAndAudit),
            Just(DriverTestOp::AskStewardAllow),
            Just(DriverTestOp::AskStewardDeny),
            any::<bool>()
                .prop_map(|pick_ask_row| DriverTestOp::AskClarifyCandidate { pick_ask_row }),
            Just(DriverTestOp::WriteUpdateAllowRow),
            Just(DriverTestOp::WriteUpdateAskRow),
            Just(DriverTestOp::WriteInsertAuditRow),
            Just(DriverTestOp::WriteExceedRowBudget),
            Just(DriverTestOp::ExternalConcurrentMutateAccounts),
            Just(DriverTestOp::BeginNewTask),
            Just(DriverTestOp::ResetEpistemicSession),
        ]
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ShadowReadKind {
        SingleAllow101,
        SingleAsk102,
        MultiAmbiguous101And102,
        JoinProjectionNoPolicy,
    }

    #[derive(Debug, Default, Clone)]
    struct ShadowSqliteEpistemicModel {
        catalog_discovered: bool,
        probed_tables: BTreeSet<String>,
        task_last_read_kind: Option<ShadowReadKind>,
        session_last_read_kind: Option<ShadowReadKind>,
        steward_auth: StewardAuthState,
        db_version: u64,
    }

    impl ShadowSqliteEpistemicModel {
        fn expected_legal_modes(&self) -> Vec<Mode> {
            if self.probed_tables.is_empty() {
                return vec![Mode::Probe, Mode::Ask, Mode::Abstain];
            }
            let mut modes = vec![Mode::Probe, Mode::Read, Mode::Ask];
            if !self.steward_auth.blocks_write() {
                modes.push(Mode::Write);
            }
            modes.push(Mode::Result);
            modes.push(Mode::Abstain);
            modes
        }

        fn resolve_clarification(&mut self, pick_ask_row: bool) {
            let active = self.task_last_read_kind.or(self.session_last_read_kind);
            match active {
                None => {}
                Some(ShadowReadKind::MultiAmbiguous101And102) => {
                    if pick_ask_row {
                        self.steward_auth = self.steward_auth.latch_ask_requirement();
                    }
                }
                Some(ShadowReadKind::SingleAllow101) => {}
                Some(ShadowReadKind::SingleAsk102) => {
                    if pick_ask_row {
                        self.steward_auth = self.steward_auth.latch_ask_requirement();
                    }
                }
                Some(ShadowReadKind::JoinProjectionNoPolicy) => {}
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_stateful_sqlite_daily_driver_and_shadow_epistemic_model_commute(
            ops in proptest::collection::vec(arb_driver_test_op(), 1..=24),
        ) {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            {
                let init_conn = Connection::open(tmp.path()).unwrap();
                init_conn
                    .execute_batch(
                        "CREATE TABLE accounts (
                                account_id INTEGER PRIMARY KEY,
                                slug TEXT NOT NULL,
                                balance INTEGER NOT NULL,
                                policy_decision TEXT NOT NULL
                            );
                            CREATE TABLE audit_log (
                                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                account_id INTEGER NOT NULL REFERENCES accounts(account_id),
                                note TEXT NOT NULL
                            );
                            CREATE TRIGGER trg_accounts_audit
                            AFTER UPDATE ON accounts
                            BEGIN
                                INSERT INTO audit_log (account_id, note)
                                VALUES (NEW.account_id, 'audit:' || NEW.balance);
                            END;
                            INSERT INTO accounts VALUES
                                (101, 'MATTERHORN-7', 500, 'ALLOW'),
                                (102, 'CHAMONIX-3', 900, 'ASK'),
                                (103, 'ZERMATT-9', 300, 'ALLOW');",
                    )
                    .unwrap();
            }

            let ext_conn = Connection::open(tmp.path()).unwrap();
            // `max_write_rows = 2` allows a single-row `accounts` update (1 direct + 1 `audit_log` trigger row = 2)
            // while rejecting multi-row `WriteExceedRowBudget` (3 direct + 3 trigger rows = 6 > 2).
            let mut driver = LiveSqliteDailyDriver::open(tmp.path(), 2)
                .unwrap()
                .with_steward_policy_columns(["policy_decision"])
                .with_candidate_disambiguation(true);
            let mut shadow = ShadowSqliteEpistemicModel::default();

            for (step_idx, op) in ops.iter().enumerate() {
                match op {
                    DriverTestOp::ProbeCatalog => {
                        let obs = driver.execute_turn(
                            Mode::Probe,
                            "SELECT name FROM sqlite_master WHERE type = 'table';",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ProbeCatalog failed: {:?}", obs.error);
                        shadow.catalog_discovered = true;
                    }
                    DriverTestOp::ProbeTableAccounts => {
                        let obs = driver.execute_turn(
                            Mode::Probe,
                            "PRAGMA table_info(accounts);",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ProbeTableAccounts failed: {:?}", obs.error);
                        shadow.probed_tables.insert("accounts".to_string());
                    }
                    DriverTestOp::ProbeTableAudit => {
                        let obs = driver.execute_turn(
                            Mode::Probe,
                            "PRAGMA table_info(audit_log);",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ProbeTableAudit failed: {:?}", obs.error);
                        shadow.probed_tables.insert("audit_log".to_string());
                    }
                    DriverTestOp::ReadSingleAllowRow => {
                        let obs = driver.execute_turn(
                            Mode::Read,
                            "SELECT account_id, slug, balance, policy_decision FROM accounts WHERE account_id = 101;",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ReadSingleAllowRow failed: {:?}", obs.error);
                        shadow.probed_tables.insert("accounts".to_string());
                        shadow.task_last_read_kind = Some(ShadowReadKind::SingleAllow101);
                        shadow.session_last_read_kind = Some(ShadowReadKind::SingleAllow101);
                    }
                    DriverTestOp::ReadSingleAskRow => {
                        let obs = driver.execute_turn(
                            Mode::Read,
                            "SELECT account_id, slug, balance, policy_decision FROM accounts WHERE account_id = 102;",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ReadSingleAskRow failed: {:?}", obs.error);
                        shadow.probed_tables.insert("accounts".to_string());
                        shadow.steward_auth = shadow.steward_auth.latch_ask_requirement();
                        shadow.task_last_read_kind = Some(ShadowReadKind::SingleAsk102);
                        shadow.session_last_read_kind = Some(ShadowReadKind::SingleAsk102);
                    }
                    DriverTestOp::ReadMultiRowAmbiguous => {
                        let obs = driver.execute_turn(
                            Mode::Read,
                            "SELECT account_id, slug, balance, policy_decision FROM accounts WHERE account_id IN (101, 102) ORDER BY account_id;",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ReadMultiRowAmbiguous failed: {:?}", obs.error);
                        shadow.probed_tables.insert("accounts".to_string());
                        shadow.task_last_read_kind = Some(ShadowReadKind::MultiAmbiguous101And102);
                        shadow.session_last_read_kind = Some(ShadowReadKind::MultiAmbiguous101And102);
                    }
                    DriverTestOp::ReadJoinAccountsAndAudit => {
                        let obs = driver.execute_turn(
                            Mode::Read,
                            "SELECT a.account_id, COUNT(l.log_id) AS audit_cnt FROM accounts a LEFT JOIN audit_log l ON a.account_id = l.account_id GROUP BY a.account_id ORDER BY a.account_id;",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} ReadJoinAccountsAndAudit failed: {:?}", obs.error);
                        shadow.probed_tables.insert("accounts".to_string());
                        shadow.probed_tables.insert("audit_log".to_string());
                        shadow.task_last_read_kind = Some(ShadowReadKind::JoinProjectionNoPolicy);
                        shadow.session_last_read_kind = Some(ShadowReadKind::JoinProjectionNoPolicy);
                    }
                    DriverTestOp::AskStewardAllow => {
                        let obs = driver.execute_turn(
                            Mode::Ask,
                            "[permission] Authorize account mutation?",
                            |_| Ok("ALLOW".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} AskStewardAllow failed: {:?}", obs.error);
                        shadow.steward_auth = shadow.steward_auth.record_decision(true);
                    }
                    DriverTestOp::AskStewardDeny => {
                        let obs = driver.execute_turn(
                            Mode::Ask,
                            "[permission] Authorize account mutation?",
                            |_| Ok("DENY".into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} AskStewardDeny failed: {:?}", obs.error);
                        shadow.steward_auth = shadow.steward_auth.record_decision(false);
                    }
                    DriverTestOp::AskClarifyCandidate { pick_ask_row } => {
                        let reply = if *pick_ask_row {
                            "account 102"
                        } else {
                            "account 101"
                        };
                        let obs = driver.execute_turn(
                            Mode::Ask,
                            "{\"multi\":\"Which account should be updated?\",\"options\":[\"101\",\"102\"]}",
                            |_| Ok(reply.into()),
                        );
                        prop_assert!(obs.ok, "step {step_idx} AskClarifyCandidate failed: {:?}", obs.error);
                        shadow.resolve_clarification(*pick_ask_row);
                    }
                    DriverTestOp::WriteUpdateAllowRow => {
                        let before_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        let can_write = !shadow.steward_auth.blocks_write();
                        let obs = driver.execute_turn(
                            Mode::Write,
                            "UPDATE accounts SET balance = balance + 10 WHERE account_id = 101;",
                            |_| Ok("ALLOW".into()),
                        );
                        let after_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        if can_write {
                            prop_assert!(
                                obs.ok && obs.rows_affected == 2,
                                "step {step_idx} expected ALLOW-row write to succeed: {:?}",
                                obs
                            );
                            prop_assert_ne!(before_snap, after_snap);
                            shadow.db_version += 1;
                            // Both `accounts` and `audit_log` (via AFTER UPDATE trigger) were mutated!
                        } else {
                            prop_assert!(
                                !obs.ok && obs.rows_affected == 0,
                                "step {step_idx} expected ALLOW-row write to fail closed (steward_auth={:?}): {:?}",
                                shadow.steward_auth,
                                obs
                            );
                            prop_assert_eq!(
                                before_snap,
                                after_snap,
                                "step {} failed write mutated SQLite!",
                                step_idx
                            );
                        }
                    }
                    DriverTestOp::WriteUpdateAskRow => {
                        let before_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        // The target row requires an explicit Steward grant even without a prior read.
                        let can_write_ask = shadow.steward_auth.is_granted();
                        let obs = driver.execute_turn(
                            Mode::Write,
                            "UPDATE accounts SET balance = balance + 25 WHERE account_id = 102;",
                            |_| Ok("ALLOW".into()),
                        );
                        let after_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        if can_write_ask {
                            prop_assert!(
                                obs.ok && obs.rows_affected == 2,
                                "step {step_idx} expected approved ASK-row write to succeed: {:?}",
                                obs
                            );
                            prop_assert_ne!(before_snap, after_snap);
                            shadow.db_version += 1;
                        } else {
                            prop_assert!(
                                !obs.ok && obs.rows_affected == 0,
                                "step {step_idx} unapproved/unread ASK-row write MUST fail closed: {:?}",
                                obs
                            );
                            shadow.steward_auth = shadow.steward_auth.latch_ask_requirement();
                            prop_assert_eq!(
                                before_snap,
                                after_snap,
                                "step {} rejected ASK write mutated SQLite!",
                                step_idx
                            );
                        }
                    }
                    DriverTestOp::WriteInsertAuditRow => {
                        let before_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        let can_write_audit = !shadow.steward_auth.blocks_write();
                        let obs = driver.execute_turn(
                            Mode::Write,
                            "INSERT INTO audit_log (account_id, note) VALUES (101, 'manual-check');",
                            |_| Ok("ALLOW".into()),
                        );
                        let after_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        if can_write_audit {
                            prop_assert!(
                                obs.ok && obs.rows_affected == 1,
                                "step {step_idx} expected audit_log insert to succeed: {:?}",
                                obs
                            );
                            prop_assert_ne!(before_snap, after_snap);
                            shadow.db_version += 1;
                        } else {
                            prop_assert!(
                                !obs.ok && obs.rows_affected == 0,
                                "step {step_idx} expected audit_log insert to fail closed: {:?}",
                                obs
                            );
                            prop_assert_eq!(before_snap, after_snap);
                        }
                    }
                    DriverTestOp::WriteExceedRowBudget => {
                        let before_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        let obs = driver.execute_turn(
                            Mode::Write,
                            "UPDATE accounts SET balance = balance + 1;",
                            |_| Ok("ALLOW".into()),
                        );
                        let after_snap = portia_sql::engine::dump_all_tables(&ext_conn).unwrap();
                        prop_assert!(
                            !obs.ok && obs.rows_affected == 0,
                            "step {step_idx} WriteExceedRowBudget must ALWAYS fail closed: {:?}",
                            obs
                        );
                        prop_assert_eq!(
                            before_snap,
                            after_snap,
                            "step {} WriteExceedRowBudget left partial mutations in SQLite!",
                            step_idx
                        );
                    }
                    DriverTestOp::ExternalConcurrentMutateAccounts => {
                        ext_conn
                            .execute(
                                "UPDATE accounts SET balance = balance + 5 WHERE account_id = 103;",
                                [],
                            )
                            .unwrap();
                        shadow.db_version += 1;
                    }
                    DriverTestOp::BeginNewTask => {
                        EpistemicDomainEnv::begin_new_task(&mut driver);
                        shadow.steward_auth = StewardAuthState::NotRequired;
                        shadow.task_last_read_kind = None;
                    }
                    DriverTestOp::ResetEpistemicSession => {
                        EpistemicDomainEnv::reset_epistemic_session(&mut driver);
                        shadow.catalog_discovered = false;
                        shadow.probed_tables.clear();
                        shadow.steward_auth = StewardAuthState::NotRequired;
                        shadow.task_last_read_kind = None;
                        shadow.session_last_read_kind = None;
                    }
                }

                // Universal Post-Step Invariant 1: `legal_modes()` matches `ShadowSqliteEpistemicModel`!
                let actual_modes = EpistemicDomainEnv::legal_modes(&driver).unwrap();
                let expected_modes = shadow.expected_legal_modes();
                prop_assert_eq!(
                    &actual_modes,
                    &expected_modes,
                    "step {} after {:?}: legal_modes mismatch (shadow={:?})",
                    step_idx,
                    op,
                    shadow
                );

                // Universal Post-Step Invariant 2: `payload_grammar_spec(mode)` succeeds (`Ok(LarkGrammar)`)
                // for EVERY mode in `legal_modes()`!
                for &legal_mode in &actual_modes {
                    let spec = EpistemicDomainEnv::payload_grammar_spec(&driver, legal_mode)
                        .unwrap_or_else(|e| {
                            panic!(
                                "step {step_idx} payload_grammar_spec({legal_mode:?}) failed: {e}"
                            )
                        });
                    prop_assert!(
                        matches!(spec, portia_core::GrammarCartridgeSpec::LarkGrammar(_)),
                        "step {step_idx} expected LarkGrammar for legal mode {legal_mode:?}, got {spec:?}"
                    );
                }

                // Universal Post-Step Invariant 3: Epistemic & version metadata match shadow model!
                prop_assert_eq!(
                    EpistemicDomainEnv::state_version(&driver),
                    shadow.db_version,
                    "step {} after {:?}: state_version mismatch",
                    step_idx,
                    op
                );
                prop_assert_eq!(driver.catalog_discovered(), shadow.catalog_discovered);
                prop_assert_eq!(driver.probed_table_names(), &shadow.probed_tables);
            }
        }
    }
}

#[test]
fn ordinary_sqlite_session_ignores_incidental_ask_cells_and_skips_unconfigured_multiselect() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE support_tickets (
                ticket_id INTEGER PRIMARY KEY,
                customer_slug TEXT NOT NULL,
                intent_tag TEXT NOT NULL,
                priority INTEGER NOT NULL
            );
            INSERT INTO support_tickets VALUES (1, 'ACME-01', 'ASK', 2);
            INSERT INTO support_tickets VALUES (2, 'ACME-02', 'ASK', 3);",
    )
    .unwrap();

    // Default `LiveSqliteDailyDriver::from_connection` has zero steward policy columns
    // and `candidate_disambiguation_enabled == false`:
    let mut driver = LiveSqliteDailyDriver::from_connection(conn, 5).unwrap();
    assert!(driver.steward_policy_columns().is_empty());
    assert!(!driver.candidate_disambiguation_enabled());

    let read_obs = driver.execute_turn(Mode::Read, "SELECT * FROM support_tickets;", |_| {
        Ok("ALLOW".into())
    });
    assert!(read_obs.ok);
    assert_eq!(read_obs.rows.len(), 2);

    // Neither `has_pending_steward_ask_row` nor `task_candidate_choices` is synthesized:
    assert!(!driver.has_pending_steward_ask_row());
    assert!(driver.task_candidate_choices().is_empty());
    assert_eq!(
        driver.ask_interaction_mode("Which ticket should we update?"),
        AskInteractionMode::ConversationalClarify
    );

    // `Mode::Write` on row 1 (whose `intent_tag` happens to equal `'ASK'`) succeeds cleanly
    // without `StewardAuthorizationViolation`:
    let write_obs = driver.execute_turn(
        Mode::Write,
        "UPDATE support_tickets SET priority = 1 WHERE ticket_id = 1;",
        |_| Ok("ALLOW".into()),
    );
    assert!(
        write_obs.ok,
        "ordinary table write should not be blocked by incidental 'ASK' cell: {:?}",
        write_obs.error
    );
    assert_eq!(write_obs.rows_affected, 1);
}
