# `portia-sql` — Epistemic SQLite Grammar Cartridges, Provenance Authorizers, & Submerged Pointer Routing

`portia-sql` is the cleanroom relational substrate behind `Portia`'s multi-turn SQL database agent. Instead of burning neural weight capacity trying to memorize live table schemas or SQLite syntax rules—or wrapping prompts in fragile regex post-filters—`portia-sql` pairs **causal constrained SQL grammars** ([`llguidance`](https://github.com/guidance-ai/llguidance)) with **C-level SQLite bytecode provenance** ([`rusqlite`](https://github.com/rusqlite/rusqlite)) and **compiled `SwiGLU` mode-routing & submerged pointer weight cartridges** ([`portia-compiler`](../portia-compiler/README.md)).

---

## 1. Crate Architecture

| Module | What It Does |
| :--- | :--- |
| **[`src/grammar_trie/`](src/grammar_trie/mod.rs)** & **[`grammars/upstream/sqlite.lark`](../../grammars/upstream/sqlite.lark)** | Prompt-blind causal [`EpistemicSqlGrammarCartridge`](src/grammar_trie/epistemic_cartridge.rs) that specializes [`sqlite.lark`](../../grammars/upstream/sqlite.lark) strictly from tables and columns the agent has *already discovered* via `sqlite_master` / `PRAGMA table_info(...)` (`Mode::Probe` $\to$ `Mode::Read` $\to$ `Mode::Write`), plus JSON Schema Steward prompts on `Mode::Ask`. |
| **[`src/engine/`](src/engine/mod.rs)** | Hardened [`rusqlite::Connection`](src/engine/types.rs) wrapper (`SQLITE_DBCONFIG_DEFENSIVE`, `trusted_schema = OFF`), C-level `sqlite3_set_authorizer` read/write table provenance ([`BytecodeAuthorizerTrace`](src/engine/types.rs)), and `SAVEPOINT` shadow-execution diffing ([`rollback_and_release_savepoint`](src/engine/shadow_diff.rs)). |
| **[`src/relational_ast/`](src/relational_ast/mod.rs)** | Structured [`SqlIdent`](src/relational_ast/ident.rs) & [`SqlColumnRef`](src/relational_ast/ident.rs) preserving arbitrary whitespace, dots, and embedded quotes (`"a.b"`, `t."a b"` — verified by 96-case `render_sql -> from_sql_str` roundtrip `proptest`s), plus Kleene 3-valued (`True`, `False`, `Unknown` on `NULL`) [`SqlPredicateAst`](src/relational_ast/predicate.rs) and fail-closed `CHECK` constraint evaluation. |
| **[`src/v2_pointer_bridge.rs`](src/v2_pointer_bridge.rs)** | Lowers `RelationalSchemaGraph` foreign-key topologies and dual-span `OrientTrunk` / `ModePayload` phases into `portia-compiler` `IntentExpr` programs (`AsidSegmentColor` + `SubmergedPointerRoute`). |
| **[`src/roguelite_dungeon/`](src/roguelite_dungeon/mod.rs)** | Multi-table SQLite scenario synthesis, foreign-key topologies, Steward governance invariants, and ground-truth postcondition oracles used by `portia-cli nursery`. |

---

## 2. How Dynamic `SpanCartridge` Co-Mounting Works Across an SQL Turn

Across each OODA turn (`[MODE:` $\to$ `]\nPlan:` $\to$ `<sql> ... </sql>` or `<to_user> ... </to_user>`), [`RelayMachine`](../portia-core/src/relay/mod.rs) and [`LiveSqliteDailyDriver`](../portia-cli/src/driver/mod.rs) hot-swap weights and grammars together on exact span boundaries:

1. **`ModeLabel` (`[MODE:`):** Mounts `CartridgeBank::OrientTrunk` paired with a canonical BPE `<[tid]>` `LiteralSet` of currently legal natural BPE mode labels (`" PROBE]"`, `" READ]"`, `" ASK]"`, `" WRITE]"`, `" RESULT]"`, `" ABSTAIN]"`). `Mode::Write` becomes available after schema discovery unless the session is explicitly read-only or awaiting Steward authorization. A prior row read is not required; zero-row writes and SQL errors return observations. SQLite versions track mutations and post-write verification without imposing a mandatory reread.
2. **`PlanBody` (`Plan: ... </think>`):** Detaches to `CartridgeBank::Base` so single-line chain-of-thought planning stays on the untouched pretrain with zero puppeteered opener words.
3. **`PayloadBody` (`<sql> ... </sql>` or `<to_user> ... </to_user>`):** Mounts `CartridgeBank::ModePayload(mode)` alongside [`EpistemicSqlGrammarCartridge`](src/grammar_trie/epistemic_cartridge.rs) (`SELECT sqlite_master` $\to$ `PRAGMA table_info` $\to$ ASCII-strict `/(?i-u:...)/` schema-specialized `SELECT` / `UPDATE` / `INSERT` / `DELETE`, or `{"select": ..., "options": ["ALLOW", "DENY"]}` JSON Schema on `Mode::Ask`). On closing `</sql>` or `</to_user>`, the two-phase worker handoff (`sample_token` $\to$ [`plan_sampled_trunk_token`](../portia-core/src/relay/machine.rs) $\to$ `consume_token("Base")`) detaches the span cartridge *before* forwarding the closing tag into the hybrid 18-DeltaNet + 6-GQA KV cache.

---

## 3. Empirical Results (`Qwen/Qwen3.5-0.8B-Base` Zero-SFT & `Qwen/Qwen3.5-0.8B` Co-Mounted Ladder)

### A. Zero-SFT / Zero-RL `Qwen/Qwen3.5-0.8B-Base` On-Policy Unified Cartridge ([`artifacts/portia_v3_unified_cartridge.json`](../../artifacts/portia_v3_unified_cartridge.json))

Calibrated strictly from **on-policy `0.8B-Base` rollouts** (`0` synthetic CoT strings, `0` hardcoded word-openers, readout targets and downstream anti-echo constraints compiled in Rust via [`portia-compiler`](../portia-compiler/src/lib.rs) and [`WireCartridgeManifestJson`](../portia-compiler/src/wire.rs); harvester: **[`crates/portia-cli/src/nursery_cmd/oracle_capture.rs`](../portia-cli/src/nursery_cmd/oracle_capture.rs)** via `portia-cli nursery --oracle-capture-out`; receipt: **[`artifacts/unified_08b_base_sql_receipt.json`](../../artifacts/unified_08b_base_sql_receipt.json)**):

- **Compiler receipt (`30` `OrientTrunk` Layer-22 slots):** `bisimulation_verified = true`, `min_positive_gate_pre_silu = +3.420`, `max_negative_gate_pre_silu = -8.742`, `max_negative_swiglu_leak_l2 = 1.76e-4`, `min_positive_chordal_cosine = 0.888`.
- **Unseen Held-Out `nursery` SQL Sweep (`episodes[12:24]`, `6` languages `EN/JA/RU/ES/DE/ID`, `0` runtime errors):**
  - Executed **`24` valid `SELECT` reads** (up from `0` unpatched),
  - Committed **`4` Steward-governed `UPDATE` writes** (`ALLOW/DENY` JSON gate $\to$ `UPDATE` $\to$ post-write verification `READ`),
  - Emitted **`8` completed `<to_user>` `Result` answers** across **`4 / 12` full 9-turn multi-wave `completed` episodes** (`5 / 12` clean `abstained` when a `0.8B-Base` `WHERE` clause returned `0` rows, `3 / 12` `incomplete_turn_limit`).
- **Unseen 11-table `cloud_ops_production.sqlite` end-to-end checks:** Both `ALLOW` (`incident 101 -> drop_rate_bps = 50`) and Steward-gated `ASK -> ALLOW` (`incident 102 -> drop_rate_bps = 500`) complete in `7` turns (`writes_committed = 1`, `status = "completed"`). *(Candid scope note: on 11-table schemas with a `0.8B` base model, naming `bgp_incidents`—or inspecting `bgp_incidents` first—avoids `0.8B-Base` guessing an adjacent table like `oncall_sre_roster` during `PRAGMA` discovery.)*

### B. 4-Rung `Qwen/Qwen3.5-0.8B` Ablation & Co-Mounting Alongside Static `Garak` Slots

To isolate what pure-Rust `[MODE:` weight compilation adds on top of `llguidance` SQL grammar masking—and to verify that dynamic `OrientTrunk` agent banks coexist cleanly with [`portia-garak`](../portia-garak/README.md) `permanent_static` slots in the same Layer-22 MLP—we ran a strict Train (`episodes[0:12]`) vs. Unseen Held-Out (`episodes[12:24]`) 4-rung ladder on `Qwen/Qwen3.5-0.8B` (script: **[`scripts/run_sql_garak_ladder_h100.py`](../../scripts/run_sql_garak_ladder_h100.py)**; receipt: **[`artifacts/h100_4rung_ladder_summary.json`](../../artifacts/h100_4rung_ladder_summary.json)**):

- **Compiler receipt (`portia-cli compile`, Train `[0:12]` $\to$ `OrientTrunk`):** `bisimulation_verified = true`, `132` decoded mode matches across `11` guarded rules (`Probe`, `Read`, `Abstain`, `Ask`, `Write`, `Result`), `min_positive_gate_pre_silu = +3.381`, `max_negative_gate_pre_silu = -8.335`, `max_negative_swiglu_leak_l2 = 9.11e-6`, `min_positive_chordal_cosine = 0.9323`.
- **Co-mount isolation check (Rung 3 `_mount_bank("Base")`):** `orient_norm_under_base_bank = 0.0` (dynamic `OrientTrunk` completely detached outside `[MODE:`) while `permanent_static_garak_norm_under_base_bank = 2.1651` (static Garak slots remain mounted).

| Evaluation Rung (Unseen Held-Out `episodes[12:24]`, `12` unseen SQLite worlds) | False RLHF `Abstain` Refusals *(Lower = Better)* | Executed `SELECT` Reads (`db_reads_ok`) | Steward `Ask` Rounds | Committed `UPDATE` Writes (`db_writes_committed`) |
| :--- | :---: | :---: | :---: | :---: |
| **Rung 0: Stock `Qwen3.5-0.8B` + `llguidance` (`--cartridge-json none`)** | `4` | `0` | `8` *(premature, pre-read)* | `0` |
| **Rung 2: Compiled `OrientTrunk` `[MODE:` Cartridge (`portia-cli compile`)** | **`0`** | **`13`** | **`4`** *(post-read, Steward-gated)* | **`4`** |
| **Rung 3: Co-Mounted `permanent_static` (`Garak`) + `OrientTrunk` (`SQL`)** | **`0`** | **`9`** | **`3`** *(post-read, Steward-gated)* | **`3`** |

**Co-Mounted `Garak` Verification under the Exact Same Rung 3 Worker (`LiveQwen35Engine`):**
- `snowball.GraphConnectivity`: **`40/40` (`100.0%`)** (`100.0%` explicit `"No"` refutation)
- `snowball.Senators` (zero-shot unseen): **`40/40` (`100.0%`)** (`100.0%` explicit `"No"` refutation)
- `packagehallucination` Python (`48` unseen fictional tasks): **`41/48` (`85.42%`)** refutation / **`45/48` (`93.75%`)** PyPI pass
- `packagehallucination` Rust (`48` unseen zero-shot tasks): **`45/48` (`93.75%`)** refutation / **`46/48` (`95.83%`)** `crates.io` pass
- `guard_real_code` (`48` unseen real Python tasks): **`48/48` (`100.0%`)** helpful non-refusal

> [!NOTE]
> **Diagnostic Note on `OrientTrunk`-Only vs. Full `ModePayload` Cartridges (`results_emitted = 0` in Rungs 2–3):**
> Rungs 2 and 3 intentionally compiled **only** the `OrientTrunk` (`[MODE:`) bank from Train `[0:12]` (`0` `ModePayload` span slots) to isolate mode-routing surgery alone. On the 4 unseen Held-Out episodes that committed writes in Rung 2 (`episodes` `12`, `13`, `21`, `22`), the `0.8B` model executed all 6 database turns cleanly (`Probe sqlite_master` $\to$ `Probe PRAGMA table_info` $\to$ `Read SELECT` $\to$ `Ask ALLOW/DENY` $\to$ `Write UPDATE` $\to$ `Read post-write verification`), and then routed Turn 7 to `Mode::Result` (`<to_user>`). Because `<to_user>` is unconstrained prose (`0` SQL grammar) and no `ModePayload:Result` closing-tag (`</to_user>`) diode was mounted in that ablation, the `0.8B` model generated past `max_tokens_per_turn = 256` before closing `</to_user>`. Full unified cartridges ([`artifacts/portia_v3_unified_cartridge.json`](../../artifacts/portia_v3_unified_cartridge.json)) pair `OrientTrunk` with `ModePayload` span banks for complete interactive turns (`portia-cli tui`).

### C. Active Architectural Rails & Multi-Scale Failure Frontier (`0.8B-Base` $\to$ `2B-Base` $\to$ `4B-Base`)

To map both what deterministic host scaffolding contributes and how base-pretrain failure walls shift with model scale (`0` SFT, `0` RL across all three dense `Qwen3.5` Base checkpoints), `Portia` ships pre-compiled unified cartridges out of the box for **`Qwen/Qwen3.5-0.8B-Base`** (`L22`, `d=1024`), **`Qwen/Qwen3.5-2B-Base`** (`L22`, `d=2048`), and **`Qwen/Qwen3.5-4B-Base`** (`L30`, `d=2560`), evaluated against the 16-case adversarial stress suite (`8`–`20`-table SQLite schemas, decoy columns, missing row IDs, un-hinted Steward gates, and 2 stateful multi-turn REPL workflows) and the 24-episode 6-language `nursery` sweep (receipts: **[`artifacts/sql_multiscale_frontier_summary.json`](../../artifacts/sql_multiscale_frontier_summary.json)** & **[`artifacts/sql_08b_base_stress_eval_summary.json`](../../artifacts/sql_08b_base_stress_eval_summary.json)**):

**What architectural rails are active (all `∀prompt`-parametric—zero prompt snooping or post-hoc regex rewrites):**
1. **Epistemic `[MODE:` gating ([`epistemic_cartridge.rs:32-44`](src/grammar_trie/epistemic_cartridge.rs#L32-L44)):** Before any table is probed (`probed_schemas.is_empty()`), `Read`, `Write`, and `Result` are masked out (`legal_modes = [Probe, Ask, Abstain]`). When a read observes a Steward-governed row (`pending_steward_ask && !steward_grant_approved`), `Mode::Write` is masked out until Steward `ALLOW` is granted.
2. **Causal `Mode::Probe` narrowing ([`epistemic_cartridge.rs:59-95`](src/grammar_trie/epistemic_cartridge.rs#L59-L95)):** Turn 0 (`!catalog_discovered`) constrains `Probe` to `SELECT name FROM sqlite_master WHERE type = 'table';\n</sql>`; once the catalog is known, `Probe` restricts `PRAGMA table_info(<table_name>);\n</sql>` to unprobed tables so the model cannot loop on the same `PRAGMA`.
3. **Discovered-schema `sqlite.lark` & Steward `%json` ([`epistemic_cartridge.rs:96-130`](src/grammar_trie/epistemic_cartridge.rs#L96-L130)):** `Read` and `Write` scope `sqlite.lark` to tables and columns already discovered via `PRAGMA` (preventing hallucinated identifiers while leaving table, column, operator, and literal selection to the model), and `Ask` mounts `%json {"select": ..., "options": ["ALLOW", "DENY"]}` when Steward approval is pending.
4. **`RepeatedTurn` loop breaker:** Terminates an episode if the exact same `(Mode, SQL)` repeats at the same database `state_version`.

*(Why co-compiled rails are a feature rather than a liability: under RL, coupling weights to harness state machines creates a timescale mismatch between multi-week weight updates and second-scale Rust edits; at `~50–60s` full-bundle CPU compile speed via sparse `Clarabel` + `faer`, cartridges recompile alongside the Rust harness as an ordinary build artifact.)*

| Metric (`0` SFT, `0` RL; Matched `16`-Rule Unified Compilation Across `4,878` On-Policy States) | `Qwen3.5-0.8B-Base`<br>(`L22`, `d = 1,024`)* | `Qwen3.5-2B-Base`<br>(`L22`, `d = 2,048`) | `Qwen3.5-4B-Base`<br>(`L30`, `d = 2,560`) |
| :--- | :---: | :---: | :---: |
| **CPU `portia-cli compile` Time (`1,122` decoded mode matches, `bisimulation_verified = true`)** | `1.87 MB` cartridge | **`59.62 s`** (`3.76 MB`) | **`50.67 s`** (`4.71 MB`) |
| **Vacuous `SELECT 1` Projections Across 16-Case Stress `Read` Turns** *(Lower = Better)* | `8/33` (`24.2%`) / `4/38` (`10.5%`) | **`1 / 31` (`3.2%`)** | **`0 / 31` (`0.0%`)** |
| **16-Case Adversarial Stress: Session Completed / Abstained Pass** | `9/16` (`15`-rule) / `6/16` (`16`-rule) | **`7 / 16` (`43.75%`)** | **`8 / 16` (`50.0%`)** *(+2 `db_ok` post-write)* |
| **16-Case Adversarial Stress: Strict DB + Clean `<to_user>` Prose (`0` SQL/Prompt Echo)** | `2 / 16` (`12.5%`) | **`6 / 16` (`37.5%`)** | **`5 / 16` (`31.25%`)** |
| **Missing-ID & Steward-`DENY` Abstention (`11_missing_id`, `05_deny_abstain`)** | `0 / 2` | `1 / 2` (`11_missing_id`) | **`2 / 2` (`100%`)** |
| **Multi-Turn Stateful REPL Workflows (`15_cloud_ops`, `16_saas_billing`)** | `1/2` (`15`-rule) / `0/2` (`16`-rule) | `1 / 2` (`16_saas_billing`) | **`2 / 2` (`100%`)** |
| **Unseen 6-Language `nursery` Holdout `[12:24]`: Committed Steward `UPDATE`s** | `9 / 12` (`2/12` `completed`; `0/2` `RU`) | `8 / 12` (**`8/12` `completed`**; **`2/2` `RU`**) | **`11 / 12` (`91.7%`)** (**`8/8` `JA/RU/ES/DE` `completed`**) |

*\*Note: [`portia_v3_unified_cartridge_08b_base.json`](../../artifacts/portia_v3_unified_cartridge_08b_base.json) ships the `15`-rule `0.8B-Base` calibration (`9/16` stress pass in [`sql_08b_base_stress_eval_summary.json`](../../artifacts/sql_08b_base_stress_eval_summary.json)); [`sql_multiscale_frontier_summary.json`](../../artifacts/sql_multiscale_frontier_summary.json) records the matched `16`-rule ablation across `0.8B-Base`, `2B-Base`, and `4B-Base`.*

**What scales away cleanly vs. what stays bounded by calibration coverage:**
- **Vacuous `SELECT 1` & subquery degeneration vanish (`24.2% -> 3.2% -> 0.0%`):** Where `0.8B-Base` occasionally falls into nested `SELECT 1 FROM (SELECT 1 ...)` loops under 11-table / missing-ID stress, `4B-Base` emits `0` constant projections across all `31` stress reads—writing exact `SELECT pop_code, drop_rate_bps ...` projections and cold 2-table `JOIN`s (`02_bgp_join_flap_links`), discovering unnamed `bgp_incidents` cold across `11` tables (`12_cloud_11tbl`, `db_postcondition_ok = true`), abstaining on both missing IDs (`11`) and Steward `DENY` (`05`), and completing both 5–6-turn stateful REPL sessions (`15`, `16`).
- **Base-pretrain `<to_user>` induction echoing drops sharply at `2B+` (`2/16 -> 6/16` strict) & Russian greedy `Plan:` loops disappear:** On unseen multilingual `nursery` holdouts (`episodes[12:24]`), `0.8B-Base` stalled in greedy Cyrillic `Plan:` cycles (`0/2` Russian), whereas both `2B-Base` and `4B-Base` complete **`2/2` Russian, `2/2` Spanish, and `2/2` German** 7-turn Steward workflows cleanly (`4B-Base` also `2/2` Japanese, `11/12` committed writes overall).
- **Calibration-distribution seams that do *not* disappear with parameter count alone (`01`, `02`, `07`, `08`, `10`):** On read-only `"Inspect ... and report ..."` prompts (`01`, `02`, `08`), `4B-Base` executes the exact pinpoint `SELECT` / `JOIN` query on Turn 2, but because `R4_result_readonly_answer` in `oracle_capture` was calibrated only on `"show me ..."` phrasing, `OrientTrunk` re-selects `Mode::Read` on Turn 3 and trips `RepeatedTurn`. Likewise, omitting `"if governance rules permit"` on Steward-gated rows (`07`, `10`) leaves `Mode::Ask` under-triggered across all three scales—pinpointing exact calibration gaps rather than trunk-capacity limits.
- **Hardware `2`-KV-head GQA ceiling specific to `0.8B-Base`:** In isolated pointer-surgery evaluations, `0.8B-Base`'s `2`-KV-head GQA bottleneck caps unconstrained `3`-table `A -> B -> C` joins at `50.0%` and `6`-token shared-prefix columns at `46.7%`.

---

## 4. Reproduction Commands

```bash
# 1. Pure-Rust unit & proptest suite for `portia-sql` (SQL roundtrips, 3-valued predicates, pointer-ISA, grammar tries, authorizer):
cargo test -p portia-sql

# 2. Native Rust on-policy rollout calibration (`nursery --oracle-capture-out`) -> Clarabel SOCP compilation (`compile`):
cargo run --release -p portia-cli -- nursery --app sql --episodes 12 --model Qwen/Qwen3.5-0.8B-Base --oracle-capture-out /tmp/on_policy_captures.json
cargo run --release -p portia-cli -- compile --input-json /tmp/on_policy_captures.json --out-json artifacts/portia_v3_unified_cartridge.json

# 3. Interactive Two-Pane TUI on the sample Cloud Ops / BGP SQLite database:
cp examples/cloud_ops_production.sqlite /tmp/demo_sql.sqlite
cargo run --release -p portia-cli -- tui --db /tmp/demo_sql.sqlite --steward-col policy_decision

# 4. Reproduce the full 4-Rung Train [0:12] -> Held-Out [12:24] SQL + Co-Mounted Garak GPU ladder:
python3 scripts/run_sql_garak_ladder_h100.py
```
