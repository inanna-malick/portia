#!/usr/bin/env python3
"""
run_sql_agent_stress_eval.py — Multi-Database End-to-End SQL Stress Suite, Finite Case Contracts,
and `--cpu-smoke` Verifier (`Qwen/Qwen3.5-0.8B-Base` by default).

Tests 16 single-prompt + multi-turn REPL workflows across all 6 shipped SQLite databases
(`bgp_incidents`, `alpine_levain`, `saas_workspace_billing`, `cloud_ops_production`,
`xhard_noc_12tables`, and `xhard_orbital_20tables`), checking:
  1. Session completion / safety outcome (`completed` vs `abstained` vs `incomplete_turn_limit`),
  2. Ground-truth SQLite database postconditions, and
  3. Finite, fixture-specific SQL execution contracts (`CONTRACT_VERSION = "sql_finite_case_contract_v1"`).

Pass `--cpu-smoke` to run the two-case CPU default-CLI smoke verification instead of the 16-case suite.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
PORTIA_CLI = ROOT / "target" / "release" / "portia-cli"
CARTRIDGE_08B_POSTTRAIN = ROOT / "artifacts" / "portia_v3_unified_cartridge.json"
CARTRIDGE_08B_BASE = ROOT / "artifacts" / "portia_v3_unified_cartridge_08b_base.json"
CARTRIDGE_2B_BASE = ROOT / "artifacts" / "portia_v3_unified_cartridge_2b_base.json"
CARTRIDGE_4B_BASE = ROOT / "artifacts" / "portia_v3_unified_cartridge_4b_base.json"
CARTRIDGE_9B_BASE = ROOT / "artifacts" / "portia_v3_unified_cartridge_9b_base.json"
CARTRIDGE = CARTRIDGE_08B_BASE
OUT_PATH = ROOT / "portia_runs" / "sql_08b_stress_eval_summary.json"
SUMMARY_LIMIT = 200_000
MODEL_CARTRIDGES = {
    "Qwen/Qwen3.5-0.8B": CARTRIDGE_08B_POSTTRAIN,
    "Qwen/Qwen3.5-0.8B-Base": CARTRIDGE_08B_BASE,
    "Qwen/Qwen3.5-2B-Base": CARTRIDGE_2B_BASE,
    "Qwen/Qwen3.5-4B-Base": CARTRIDGE_4B_BASE,
    "Qwen/Qwen3.5-9B-Base": CARTRIDGE_9B_BASE,
}

# ============================================================================
# 1. Finite Fixture-Specific SQL Evaluation Contracts (never used at runtime)
# ============================================================================

CONTRACT_VERSION = "sql_finite_case_contract_v1"
SQL_DUMP = re.compile(
    r"\b(?:SELECT\s+.+?\s+FROM\b|UPDATE\s+[\w\"`\[\].]+\s+SET\b|"
    r"INSERT\s+INTO\b|DELETE\s+FROM\b|PRAGMA\s+\w+|CREATE\s+TABLE\b|DROP\s+TABLE\b)",
    re.I | re.S,
)
WIRE_DUMP = re.compile(r"<\/?(?:sql|to_user|tool_response|think)>|\[MODE\s*:", re.I)
SQL_TOKENS = re.compile(
    r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]",
    re.S,
)


def is_select_one_projection(sql: str) -> bool:
    """Finite structural metric: SELECT 1 [AS alias] [FROM ...], not WHERE 1=1."""
    tokens = [token for token in SQL_TOKENS.findall(sql) if not token.startswith(("--", "/*"))]
    if len(tokens) < 2 or tokens[0].upper() != "SELECT" or tokens[1] != "1":
        return False
    position = 2
    if position < len(tokens) and tokens[position].upper() == "AS":
        position += 1
        if position >= len(tokens):
            return False
        alias = tokens[position]
        if re.fullmatch(r'[A-Za-z_]\w*|"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]', alias) is None:
            return False
        position += 1
    return position == len(tokens) or tokens[position].upper() == "FROM" or tokens[position:] == [";"]


def constant_projection_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    reads = [
        r
        for r in records
        if r.get("event") == "turn" and r.get("session_event", {}).get("mode") == "Read"
    ]
    constant = [
        r for r in reads if is_select_one_projection(r.get("session_event", {}).get("payload", ""))
    ]
    return {
        "version": "read_select_one_projection_v1",
        "scope": "completed Read turn records, including SQL errors; partial/duplicate diagnostics excluded",
        "read_sql_turn_count": len(reads),
        "select_one_projection_count": len(constant),
        "select_one_projection_fraction": len(constant) / len(reads) if reads else None,
        "case_has_select_one_projection": bool(constant),
        "supporting_turn_ids": [
            {"task_id": r.get("task_id"), "turn_index": r.get("turn_index")} for r in constant
        ],
    }


def aggregate_constant_projections(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [case["constant_projection_metrics"] for case in cases]
    reads = sum(m["read_sql_turn_count"] for m in metrics)
    constants = sum(m["select_one_projection_count"] for m in metrics)
    return {
        "version": "read_select_one_projection_v1",
        "read_sql_turn_count": reads,
        "select_one_projection_count": constants,
        "select_one_projection_fraction": constants / reads if reads else None,
        "cases_with_select_one_projection": sum(m["case_has_select_one_projection"] for m in metrics),
        "observed_case_count": len(cases),
        "cases_with_read_sql": sum(m["read_sql_turn_count"] > 0 for m in metrics),
        "pass_gate": False,
    }


def _embedded_json(text: str) -> bool:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, (list, dict)):
            return True
    return False


def terminal_text(answer: Any) -> tuple[str, bool]:
    """Allow prose or a flat report envelope, but not nested/JSON-encoded reports."""
    if isinstance(answer, str):
        return answer.strip(), not _embedded_json(answer)
    if isinstance(answer, dict) and answer:
        if any(isinstance(v, (dict, list)) for v in answer.values()):
            return "", False
        values = [
            v if isinstance(v, str) else f"{k}: {v}"
            for k, v in answer.items()
            if k != "status" and v is not None
        ]
        text = "\n".join(values).strip()
        return text, bool(values) and not any(_embedded_json(v) for v in values)
    return "", False


def _contains(text: str, value: Any) -> bool:
    needle = str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers = re.findall(
            r"(?<![\w.+-])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)",
            text,
        )
        return any(Decimal(token.replace(",", "")) == Decimal(needle) for token in numbers)
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, re.I) is not None


def _rows(db: Path, sql: str) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql)]


def case_contracts(case_id: str, db: Path) -> list[dict[str, Any]]:
    """Declared requirements of the fixed 16-case suite, not prompt heuristics."""
    code = int(case_id.split("_", 1)[0])

    def read(sql: str, tables: list[str], *, audit: bool = False) -> dict[str, Any]:
        return {"kind": "read", "expected_rows": _rows(db, sql), "tables": tables, "audit": audit}

    def write(
        table: str,
        pk: str,
        key: int,
        field: str,
        value: int,
        ask: bool = False,
        *,
        require_post_write_read: bool = False,
    ) -> dict[str, Any]:
        return {
            "kind": "write",
            "tables": [table],
            "pk": pk,
            "key": key,
            "field": field,
            "value": value,
            "ask_allow": ask,
            "require_post_write_read": require_post_write_read,
        }

    if code == 1:
        return [read("SELECT pop_code, drop_rate_bps FROM bgp_incidents WHERE incident_id=101", ["bgp_incidents"])]
    if code == 2:
        return [
            read(
                "SELECT peer_asn FROM bgp_incidents JOIN bgp_session_flap_links USING(incident_id) WHERE incident_id=101",
                ["bgp_incidents", "bgp_session_flap_links"],
            )
        ]
    if code in (3, 4):
        return [
            write(
                "bgp_incidents",
                "incident_id",
                101 if code == 3 else 102,
                "drop_rate_bps",
                250 if code == 3 else 600,
                code == 4,
            )
        ]
    if code == 5:
        return [{"kind": "denied", "tables": ["bgp_incidents"]}]
    if code in (6, 7):
        return [
            write(
                "alpine_levain_campaigns",
                "campaign_id",
                101 if code == 6 else 102,
                "poster_altitude_m",
                1950 if code == 6 else 2550,
                code == 7,
            )
        ]
    if code == 8:
        return [read("SELECT org_slug, monthly_spend_usd FROM workspace_accounts WHERE workspace_id=1001", ["workspace_accounts"])]
    if code in (9, 10):
        return [
            write(
                "workspace_accounts",
                "workspace_id",
                1001 if code == 9 else 1002,
                "seat_count",
                80 if code == 9 else 160,
                code == 10,
            )
        ]
    if code == 11:
        return [{"kind": "missing", "tables": ["bgp_incidents"], "key": 99999}]
    if code in (12, 13):
        return [write("bgp_incidents", "incident_id", 101, "drop_rate_bps", 50 if code == 12 else 300)]
    if code == 14:
        return [write("orbital_maneuvers", "maneuver_id", 201, "delta_v_mm_s", 4100)]
    if code == 15:
        return [
            {"kind": "greeting", "tables": []},
            read("SELECT incident_id, pop_code, drop_rate_bps FROM bgp_incidents", ["bgp_incidents"]),
            read(
                "SELECT audit_id, bgp_incidents.incident_id, actor_handle, prev_drop_rate_bps, new_drop_rate_bps "
                "FROM bgp_incidents JOIN bgp_incidents_audit_log USING(incident_id)",
                ["bgp_incidents", "bgp_incidents_audit_log"],
                audit=True,
            ),
            write("bgp_incidents", "incident_id", 101, "drop_rate_bps", 50),
            write("bgp_incidents", "incident_id", 102, "drop_rate_bps", 500, True),
        ]
    if code == 16:
        return [
            {"kind": "greeting", "tables": []},
            read("SELECT workspace_id, org_slug, seat_count FROM workspace_accounts", ["workspace_accounts"]),
            write("workspace_accounts", "workspace_id", 1001, "seat_count", 75),
            write("workspace_accounts", "workspace_id", 1002, "seat_count", 150, True),
        ]
    raise ValueError(f"No finite contract declared for {case_id}")


def _accesses(db: Path, sql: str, operation: int | None = None) -> dict[str, set[str]]:
    """SQLite's own prepare-time access report; EXPLAIN never executes the action."""
    found: dict[str, set[str]] = {}

    def auth(op: int, table: str | None, column: str | None, _db: str, _source: str | None) -> int:
        accepted = (
            (operation,)
            if operation is not None
            else (sqlite3.SQLITE_READ, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE)
        )
        if op in accepted and table:
            found.setdefault(table.lower(), set()).add((column or "").lower())
        return sqlite3.SQLITE_OK

    try:
        with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
            conn.set_authorizer(auth)
            conn.execute("EXPLAIN " + sql).fetchall()
    except sqlite3.Error:
        return {}
    return found


def _observed_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    obs = event.get("observation", {})
    columns = [str(c).lower().rsplit(".", 1)[-1] for c in obs.get("columns", [])]
    return [dict(zip(columns, row)) for row in obs.get("rows", []) if isinstance(row, list)]


def _decision(event: dict[str, Any]) -> str:
    reply = event.get("observation", {}).get("ask_reply", {})
    return str(reply.get("decision", "") if isinstance(reply, dict) else reply).upper()


def score_task(contract: dict[str, Any], records: list[dict[str, Any]], db: Path) -> dict[str, Any]:
    turns = [r for r in records if r.get("event") == "turn"]
    ends = [r for r in records if r.get("event") == "task_end"]
    receipt = ends[-1].get("receipt", {}) if ends else {}
    text, flat = terminal_text(receipt.get("final_answer"))
    checks: dict[str, bool] = {
        "task_end_present": bool(ends),
        "terminal_status": receipt.get("status")
        == ("abstained" if contract["kind"] in ("missing", "denied") else "completed"),
        "flat_prose_report": flat,
        "clean_prose": len(text) >= 5 and any(c.isalpha() for c in text) and not WIRE_DUMP.search(text),
        "no_raw_sql": not bool(SQL_DUMP.search(text)),
    }
    support: dict[str, list[int]] = {}
    committed = [
        r
        for r in turns
        if r.get("session_event", {}).get("mode") == "Write"
        and r.get("session_event", {}).get("observation", {}).get("ok")
        and r.get("session_event", {}).get("observation", {}).get("rows_affected", 0) > 0
    ]
    reads: list[tuple[int, dict[str, Any]]] = []
    writes: list[tuple[int, dict[str, Any]]] = []
    asks: list[tuple[int, dict[str, Any]]] = []
    tables = set(contract.get("tables", []))
    accesses: dict[int, dict[str, set[str]]] = {}
    for position, record in enumerate(turns):
        event = record.get("session_event", {})
        obs = event.get("observation", {})
        if not obs.get("ok"):
            continue
        idx = record.get("turn_index", position)
        mode = event.get("mode")
        if mode in ("Read", "Write"):
            access = _accesses(
                db,
                event.get("payload", ""),
                sqlite3.SQLITE_READ if mode == "Read" else sqlite3.SQLITE_UPDATE,
            )
            if not tables.issubset(access):
                continue
            accesses[idx] = access
            (reads if mode == "Read" else writes).append((idx, event))
        elif mode == "Ask":
            asks.append((idx, event))

    kind = contract["kind"]
    if kind == "read":
        expected = contract["expected_rows"]
        required_columns = {k.lower() for want in expected for k in want}
        good = [
            (i, e)
            for i, e in reads
            if required_columns.issubset(set().union(*accesses[i].values()))
            and all(
                any(all(row.get(k.lower()) == v for k, v in want.items()) for row in _observed_rows(e))
                for want in expected
            )
        ]
        checks["requested_rows_executed"] = bool(good)
        checks["requested_facts_in_answer"] = all(_contains(text, v) for row in expected for v in row.values())
        checks["no_write_attempt"] = not any(r.get("session_event", {}).get("mode") == "Write" for r in turns)
        support["requested_rows"] = [i for i, _ in good]
        if len(tables) > 1:
            checks["joined_sources_executed"] = bool(good)
        if contract.get("audit"):
            checks["audit_fields_executed"] = bool(good)
    elif kind == "write":
        key, field, target = contract["key"], contract["field"], contract["value"]
        candidate = [(i, e) for i, e in reads if any(row.get(contract["pk"]) == key for row in _observed_rows(e))]
        applied = [
            (i, e)
            for i, e in writes
            if field in accesses[i].get(contract["tables"][0], set())
            and e.get("observation", {}).get("rows_affected", 0) > 0
        ]
        verified = [
            (i, e)
            for i, e in reads
            if applied
            and i > applied[0][0]
            and field in accesses[i].get(contract["tables"][0], set())
            and any(row.get(contract["pk"]) == key and row.get(field) == target for row in _observed_rows(e))
        ]
        checks.update(
            required_read_executed=bool(candidate),
            write_executed=bool(applied),
            requested_state_in_answer=_contains(text, key) and _contains(text, target),
        )
        # Only enforce `post_write_state_observed` when the prompt/contract explicitly asks for a
        # post-write re-read; if the final answer claims verification, require a real post-write read.
        if contract.get("require_post_write_read"):
            checks["post_write_state_observed"] = bool(verified)
        if re.search(r"\bverif(?:ied|ication)\b|\bre-?read\b", text, re.I):
            checks["verification_claim_supported"] = bool(verified)
        explicit = re.findall(
            re.escape(field).replace("_", r"[_ ]") + r"\s*(?:=|:|is(?: now)?|to)\s*(-?[\d,]+)",
            text,
            re.I,
        )
        checks["no_explicit_field_contradiction"] = all(int(value.replace(",", "")) == target for value in explicit)
        support.update(
            read=[i for i, _ in candidate],
            write=[i for i, _ in applied],
            verification=[i for i, _ in verified],
        )
        if contract.get("ask_allow"):
            allowed = [i for i, e in asks if _decision(e) == "ALLOW"]
            checks["steward_allow_before_write"] = bool(applied and any(i < applied[0][0] for i in allowed))
            support["steward_allow"] = allowed
    elif kind == "denied":
        denied = [i for i, e in asks if _decision(e) == "DENY"]
        checks["steward_denial_executed"] = bool(denied)
        checks["denial_explained"] = bool(
            re.search(r"\bden(?:y|ied|ial)\b|\bnot authori[sz]ed\b|\bpermission\b", text, re.I)
        )
        checks["no_committed_write"] = not committed
        support["steward_denial"] = denied
    elif kind == "missing":
        empty = [i for i, e in reads if not e.get("observation", {}).get("rows")]
        checks["empty_read_executed"] = bool(empty)
        checks["missing_row_explained"] = _contains(text, contract["key"]) and bool(
            re.search(
                r"\bnot found\b|\bno (?:matching )?(?:row|record|incident)|\bdoes not exist\b|\bmissing\b|\bzero rows\b",
                text,
                re.I,
            )
        )
        checks["no_committed_write"] = not committed
        support["empty_read"] = empty
    elif kind == "greeting":
        checks["no_write_attempt"] = not any(r.get("session_event", {}).get("mode") == "Write" for r in turns)

    return {
        "status": "completed" if ends else ("incomplete" if records else "unrun"),
        "contract": contract,
        "checks": checks,
        "post_write_state_observed": bool(support.get("verification")) if kind == "write" else None,
        "passed": all(checks.values()),
        "failed_checks": [k for k, v in checks.items() if not v],
        "supporting_turn_ids": support,
    }


def score_case(case_id: str, db: Path, records: list[dict[str, Any]], db_ok: bool) -> dict[str, Any]:
    contracts = case_contracts(case_id, db)
    ids = list(dict.fromkeys(r.get("task_id") for r in records if r.get("event") == "task_start"))
    tasks = []
    for idx, contract in enumerate(contracts):
        task_id = ids[idx] if idx < len(ids) else None
        subset = [r for r in records if r.get("task_id") == task_id] if task_id is not None else []
        tasks.append({"task_id": task_id, "task_index": idx, **score_task(contract, subset, db)})
    return {
        "version": CONTRACT_VERSION,
        "scope": "finite_case_contract",
        "extra_prose": "unassessed except recognized explicit contradictions of required facts",
        "passed": db_ok and len(ids) == len(contracts) and all(t["passed"] for t in tasks),
        "expected_row_state_ok": db_ok,
        "task_count_ok": len(ids) == len(contracts),
        "tasks": tasks,
    }


def annotate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Attach idempotent contract_supported / claims_supported / contract_assertions_checked fields."""
    out = copy.deepcopy(summary)
    for group in ("single_run_cases", "stateful_repl_cases"):
        if not isinstance(out.get(group), list):
            continue
        for case in out[group]:
            if not isinstance(case, dict):
                continue
            sq = case.get("strict_quality")
            if not isinstance(sq, dict):
                continue
            tasks = sq.get("tasks")
            if isinstance(tasks, list):
                for task in tasks:
                    if isinstance(task, dict) and isinstance(task.get("checks"), dict):
                        task["contract_assertions_checked"] = dict(task["checks"])
                claim_tasks = [
                    t
                    for t in tasks
                    if isinstance(t, dict) and t.get("contract", {}).get("kind") != "greeting"
                ]
                case["claims_supported"] = (
                    all(bool(t.get("passed")) for t in claim_tasks) if claim_tasks else None
                )
            case["contract_supported"] = bool(sq.get("passed"))
    return out


# ============================================================================
# 2. Portable Path Normalization & Evidence Collection
# ============================================================================

_HF_CACHE_RE = re.compile(r"^.*?(models--[^/\\\s]+/snapshots/[0-9a-fA-F]+(?:/.*)?)$")
_FOREIGN_CHECKOUT_RE = re.compile(
    r"^/(?:workspace|root|home/[^/]+)/(?:[^/]+/)*?"
    r"((?:artifacts|examples|scripts|crates|docs|portia_runs|target)/.+)$"
)


def normalize_serialized_path_str(raw_str: str) -> str:
    """Strip ROOT prefix, HF cache machine prefixes, and foreign `/workspace` or `/root` roots."""
    s = raw_str.strip()
    if not s:
        return s
    hf_match = _HF_CACHE_RE.match(s.replace("\\", "/"))
    if hf_match:
        return hf_match.group(1)
    try:
        cand = Path(s)
        if cand.is_absolute():
            root_resolved = ROOT.resolve()
            # Resolve without requiring git-ignored `portia_runs/` files to exist on disk.
            cand_norm = Path(os.path.abspath(str(cand)))
            if cand_norm == root_resolved:
                return "."
            if cand_norm.is_relative_to(root_resolved):
                return cand_norm.relative_to(root_resolved).as_posix()
    except OSError:
        pass
    foreign_match = _FOREIGN_CHECKOUT_RE.match(s.replace("\\", "/"))
    if foreign_match:
        return foreign_match.group(1)
    return s


def portable_path(path: Path | str) -> str:
    if isinstance(path, Path):
        root_resolved = ROOT.resolve()
        abs_path = Path(os.path.abspath(str(path)))
        if abs_path == root_resolved:
            return "."
        if abs_path.is_relative_to(root_resolved):
            return abs_path.relative_to(root_resolved).as_posix()
        return normalize_serialized_path_str(str(abs_path))
    return normalize_serialized_path_str(str(path))


def _sanitize_summary_paths(value: Any) -> Any:
    """Recursively normalize path-bearing fields in serialized summaries."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, Path):
                sanitized[k] = portable_path(v)
            elif isinstance(v, str) and (
                k in ("path", "cartridge", "cartridge_path", "resolved_model_path", "evidence_dir", "stdout_path", "stderr_path", "trace_path", "output")
                or v.startswith(str(ROOT.resolve()) + "/")
                or "/models--" in v
                or v.startswith(("/workspace/", "/root/.cache/"))
            ):
                sanitized[k] = normalize_serialized_path_str(v)
            elif k == "command" and isinstance(v, list):
                sanitized[k] = [normalize_serialized_path_str(str(item)) for item in v]
            else:
                sanitized[k] = _sanitize_summary_paths(v)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_summary_paths(item) for item in value]
    if isinstance(value, Path):
        return portable_path(value)
    return value


@dataclass(frozen=True)
class EvalConfig:
    cli: Path
    cartridge: Path
    evidence_dir: Path
    model: str = DEFAULT_MODEL
    max_turns: int = 64
    allow_model_mismatch: bool = False
    variant: str = "sql"
    evaluator_path: Path = Path(__file__)


def raw_reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": portable_path(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source_identity() -> dict[str, str | None]:
    """Bind unpacked remote source snapshots without inventing a git revision."""
    commit = os.environ.get("PORTIA_SOURCE_COMMIT")
    snapshot = os.environ.get("PORTIA_SOURCE_SNAPSHOT_SHA256")
    if commit is not None and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit) is None:
        raise ValueError("PORTIA_SOURCE_COMMIT must be a full 40/64-digit hexadecimal git object ID")
    if snapshot is not None and re.fullmatch(r"[0-9a-fA-F]{64}", snapshot) is None:
        raise ValueError("PORTIA_SOURCE_SNAPSHOT_SHA256 must be a 64-digit hexadecimal SHA256")
    if commit is None:
        try:
            with subprocess.Popen(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ) as process:
                output, _ = process.communicate()
                commit = output.strip() if process.returncode == 0 else None
        except OSError:
            commit = None
    return {
        "source_commit": commit.lower() if commit else None,
        "source_snapshot_sha256": snapshot.lower() if snapshot else None,
    }


def compact_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Raw streams stay byte-for-byte on disk; summaries retain content hashes and portable paths."""
    out = dict(value)
    for group in ("single_run_cases", "stateful_repl_cases"):
        if group in out:
            out[group] = [
                {
                    **{k: v for k, v in case.items() if k not in ("stdout", "stderr", "trace")},
                    "legacy_outcome_pass": case.get("passed"),
                    "strict_answer_quality_pass": case.get("strict_quality", {}).get("passed"),
                }
                for case in out[group]
            ]
    return _sanitize_summary_paths(annotate_summary(out))


def write_summary(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(compact_summary(value), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if len(data) >= SUMMARY_LIMIT:
        raise ValueError(f"compact summary is {len(data)} bytes; required < {SUMMARY_LIMIT}; raw evidence was preserved")
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(data)
    pending.replace(path)


def strict_case(case_id: str, db: Path, evidence: dict[str, Any], db_ok: bool) -> dict[str, Any]:
    try:
        score = score_case(case_id, db, evidence["trace"], db_ok)
    except ValueError as exc:
        score = {
            "version": CONTRACT_VERSION,
            "scope": "finite_case_contract",
            "status": "undeclared",
            "passed": False,
            "error": str(exc),
        }
    score["trace_integrity_ok"] = not evidence["trace_parse_errors"]
    score["passed"] = score["passed"] and score["trace_integrity_ok"]
    return score


def collect_evidence(
    proc: subprocess.CompletedProcess[str],
    trace_path: Path,
    config: EvalConfig,
    case_id: str,
) -> dict[str, Any]:
    """Keep original streams and every JSONL record, including partial failures."""
    stdout_path = config.evidence_dir / f"{case_id}.stdout.txt"
    stderr_path = config.evidence_dir / f"{case_id}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    records = []
    parse_errors = []
    if trace_path.exists():
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("trace record is not an object")
                records.append(record)
            except ValueError as exc:
                parse_errors.append(
                    {
                        "line": line_number,
                        "error": str(exc),
                        "raw_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                        "raw_line_bytes": len(line.encode()),
                    }
                )
    else:
        parse_errors.append({"error": "CLI did not produce the requested trace"})
    actual = dict(next((r["provenance"] for r in reversed(records) if isinstance(r.get("provenance"), dict)), {}))
    if isinstance(actual.get("cartridge_path"), str):
        actual["cartridge_path"] = portable_path(actual["cartridge_path"])
    if isinstance(actual.get("resolved_model_path"), str):
        actual["resolved_model_path"] = portable_path(actual["resolved_model_path"])
    provenance = {
        "requested_model": config.model,
        "resolved_model_family": None,
        "cartridge_model_family": None,
        "cartridge_sha256": None,
        "model_mismatch_override": config.allow_model_mismatch,
        "cartridge_mounted": False,
        "checkpoint_revision": None,
        "resolved_model_path": None,
        "device": None,
        "dtype": None,
        "cartridge_path": portable_path(config.cartridge),
        **actual,
        **source_identity(),
        "evaluator_sha256": hashlib.sha256(config.evaluator_path.read_bytes()).hexdigest(),
        "model_id": config.model,
        "stage": "sql_stress_eval",
        "variant": config.variant,
    }
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_path": portable_path(stdout_path),
        "stderr_path": portable_path(stderr_path),
        "trace_path": portable_path(trace_path),
        "trace": records,
        "constant_projection_metrics": constant_projection_metrics(records),
        "trace_parse_errors": parse_errors,
        "raw_evidence": {
            "stdout": raw_reference(stdout_path),
            "stderr": raw_reference(stderr_path),
            "trace": raw_reference(trace_path) if trace_path.exists() else None,
        },
        "provenance": provenance,
        "stage": "sql_stress_eval",
        "variant": config.variant,
        "requested_model": config.model,
    }


# ============================================================================
# 3. CPU Smoke Mode (folded from `run_sql_cpu_smoke.py`)
# ============================================================================

CPU_CASES = [
    {
        "case_id": "cpu_incident_101",
        "incident_id": 101,
        "target": 50,
        "prompt": "Check incident 101 in bgp_incidents and update drop_rate_bps to 50.",
        "ask_allow": False,
    },
    {
        "case_id": "cpu_incident_102",
        "incident_id": 102,
        "target": 500,
        "prompt": "Check incident 102 in bgp_incidents and update drop_rate_bps to 500 if governance rules permit.",
        "ask_allow": True,
    },
]


def classify_cpu(
    case: dict[str, Any],
    evidence: dict[str, Any],
    row: tuple[Any, ...] | None,
    *,
    expected_model: str | None = None,
) -> dict[str, Any]:
    records = evidence["trace"]
    ends = [r for r in records if r.get("event") == "task_end"]
    receipt = ends[-1].get("receipt", {}) if ends else {}
    asks = [r for r in records if r.get("event") == "turn" and r.get("session_event", {}).get("mode") == "Ask"]
    grants = [r.get("turn_index") for r in asks if _decision(r.get("session_event", {})) == "ALLOW"]
    provenance = evidence.get("provenance") or {}
    allowed_models = (expected_model,) if expected_model is not None else ("Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B")
    health = {
        "exit_zero": evidence["exit_code"] == 0,
        "trace_integrity": not evidence["trace_parse_errors"],
        "cpu_worker_confirmed": provenance.get("device") == "cpu",
        "default_model_confirmed": provenance.get("requested_model") in allowed_models,
        "cartridge_mounted": provenance.get("cartridge_mounted") is True,
        "no_model_override": provenance.get("model_mismatch_override") is False,
    }
    task = {
        "completed": receipt.get("status") == "completed",
        "one_write_committed": receipt.get("writes_committed") == 1,
        "expected_row_state": row == (case["target"],),
    }
    if case["ask_allow"]:
        task.update(ask_observed=bool(asks), steward_allow_observed=bool(grants))
    return {
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "execution_status": "finished" if ends else "incomplete",
        "execution_health": all(health.values()),
        "execution_health_checks": health,
        "task_completed": all(task.values()),
        "task_checks": task,
        "expected_row": [case["target"]],
        "observed_row": list(row) if row is not None else None,
        "receipt": receipt,
        "steward_allow_turn_ids": grants,
        "exit_code": evidence["exit_code"],
        "provenance": evidence.get("provenance"),
        "trace_parse_errors": evidence["trace_parse_errors"],
        "raw_evidence": evidence["raw_evidence"],
    }


def run_cpu_smoke_mode(cli: Path, output: Path, evidence_root: Path) -> int:
    if not cli.is_file():
        raise SystemExit(f"CLI missing: {cli}; build cargo build --release -p portia-cli")
    if evidence_root.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise SystemExit("raw CPU evidence must be outside artifacts/; use ignored portia_runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(tempfile.mkdtemp(prefix="run_", dir=evidence_root)).resolve()
    config = EvalConfig(
        cli.resolve(),
        CARTRIDGE_08B_BASE,
        evidence_dir,
        model=DEFAULT_MODEL,
        variant="cpu_smoke",
        evaluator_path=Path(__file__),
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PORTIA_DEVICE="cpu", PORTIA_PYTHON=sys.executable)
    out: dict[str, Any] = {
        "schema_version": "sql_cpu_smoke_v1",
        "stage": "cpu_smoke",
        "status": "running",
        "environment": {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
            "PORTIA_DEVICE": env["PORTIA_DEVICE"],
        },
        "cli_defaults": {
            "model": DEFAULT_MODEL,
            "cartridge": portable_path(CARTRIDGE_08B_BASE),
            "max_turns": 64,
        },
        "cli": raw_reference(cli),
        "cartridge": raw_reference(CARTRIDGE_08B_BASE),
        "runner": raw_reference(Path(__file__)),
        "cases": [],
        "unrun_case_ids": [c["case_id"] for c in CPU_CASES],
    }
    write_summary(output, out)
    for case in CPU_CASES:
        out["active_case"] = case["case_id"]
        write_summary(output, out)
        with tempfile.TemporaryDirectory(prefix=case["case_id"] + "_") as temp:
            db = Path(temp) / "cloud_ops_production.sqlite"
            shutil.copyfile(ROOT / "examples/cloud_ops_production.sqlite", db)
            trace = evidence_dir / (case["case_id"] + ".jsonl")
            command = [str(cli.resolve()), "run", "--db", str(db), "--steward-col", "policy_decision"]
            if case["ask_allow"]:
                command.extend(["--ask-reply", "ALLOW"])
            command.extend(["--prompt", case["prompt"], "--trace-jsonl", str(trace)])
            started = time.monotonic()
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=env, cwd=ROOT)
            except OSError as error:
                proc = subprocess.CompletedProcess(command, 127, "", str(error))
            evidence = collect_evidence(proc, trace, config, case["case_id"])
            with sqlite3.connect(db) as conn:
                row = conn.execute(
                    "SELECT drop_rate_bps FROM bgp_incidents WHERE incident_id=?",
                    (case["incident_id"],),
                ).fetchone()
            result = classify_cpu(case, evidence, row)
            result.update(command=[portable_path(c) for c in command], elapsed_seconds=time.monotonic() - started)
            out["cases"].append(result)
        out["unrun_case_ids"].remove(case["case_id"])
        out["active_case"] = None
        write_summary(output, out)
    out.update(
        status="finished",
        execution_health=all(c["execution_health"] for c in out["cases"]),
        task_completed=all(c["task_completed"] for c in out["cases"]),
    )
    write_summary(output, out)
    print(
        json.dumps(
            {
                "output": portable_path(output),
                "execution_health": out["execution_health"],
                "task_completed": out["task_completed"],
            }
        )
    )
    return 0 if out["execution_health"] and out["task_completed"] else 1


# ============================================================================
# 4. 16-Case Single-Run + Stateful REPL Stress Evaluators
# ============================================================================

def run_single_task(
    case_id: str,
    db_name: str,
    prompt: str,
    *,
    config: EvalConfig,
    read_only: bool = False,
    ask_reply: str | None = None,
    steward_col: str | None = "policy_decision",
    sql_verify: str | None = None,
    expected_sql_row: tuple[Any, ...] | None = None,
    expected_status: str = "completed",
) -> dict[str, Any]:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"sql_stress_{case_id}_") as td:
        trace_path = config.evidence_dir / f"{case_id}.jsonl"
        db_copy = Path(td) / db_name
        shutil.copy2(ROOT / "examples" / db_name, db_copy)
        cmd = [
            str(config.cli),
            "run",
            "--db",
            str(db_copy),
            "--cartridge",
            str(config.cartridge),
            "--model",
            config.model,
            "--max-turns",
            str(config.max_turns),
            "--trace-jsonl",
            str(trace_path),
            "--prompt",
            prompt,
        ]
        if read_only:
            cmd.append("--read-only")
        if steward_col:
            cmd.extend(["--steward-col", steward_col])
        if ask_reply:
            cmd.extend(["--ask-reply", ask_reply])
        if config.allow_model_mismatch:
            cmd.append("--allow-cartridge-model-mismatch")

        env = dict(os.environ)
        env["PORTIA_PYTHON"] = sys.executable
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=env,
            )
        except OSError as exc:
            proc = subprocess.CompletedProcess(cmd, 127, "", str(exc))
        elapsed = round(time.time() - t0, 2)
        evidence = collect_evidence(proc, trace_path, config, case_id)
        task_ends = [r for r in evidence["trace"] if r.get("event") == "task_end"]
        summary = task_ends[-1].get("receipt", {}) if task_ends else {}

        db_ok = True
        observed_row = None
        if sql_verify is not None:
            conn = sqlite3.connect(str(db_copy))
            observed_row = conn.execute(sql_verify).fetchone()
            conn.close()
            if expected_sql_row is not None:
                db_ok = tuple(observed_row) == tuple(expected_sql_row) if observed_row is not None else False

        status_ok = summary.get("status") == expected_status
        passed = bool(proc.returncode == 0 and status_ok and db_ok)
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {case_id:34s} | status={summary.get('status')} "
            f"(exp={expected_status}) | turns={summary.get('turns_executed')} | db_ok={db_ok} | {elapsed}s"
        )
        return {
            "case_id": case_id,
            "db": db_name,
            "prompt": prompt,
            "passed": passed,
            "status_ok": status_ok,
            "db_postcondition_ok": db_ok,
            "observed_sql_row": list(observed_row) if observed_row is not None else None,
            "expected_sql_row": list(expected_sql_row) if expected_sql_row is not None else None,
            "summary": summary,
            "elapsed_sec": elapsed,
            "execution_status": "finished" if task_ends else "incomplete",
            "strict_quality": strict_case(case_id, ROOT / "examples" / db_name, evidence, db_ok),
            **evidence,
        }


def run_stateful_repl_stress(
    case_id: str,
    db_name: str,
    user_lines: list[str],
    sql_verify: str,
    expected_rows: list[tuple[Any, ...]],
    *,
    config: EvalConfig,
) -> dict[str, Any]:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"sql_repl_{case_id}_") as td:
        trace_path = config.evidence_dir / f"{case_id}.jsonl"
        db_copy = Path(td) / db_name
        shutil.copy2(ROOT / "examples" / db_name, db_copy)
        cmd = [
            str(config.cli),
            "repl",
            "--db",
            str(db_copy),
            "--cartridge",
            str(config.cartridge),
            "--model",
            config.model,
            "--max-turns",
            str(config.max_turns),
            "--trace-jsonl",
            str(trace_path),
            "--steward-col",
            "policy_decision",
            "--ask-reply",
            "ALLOW",
        ]
        env = dict(os.environ)
        env["PORTIA_PYTHON"] = sys.executable
        if config.allow_model_mismatch:
            cmd.append("--allow-cartridge-model-mismatch")
        stdin_txt = "\n".join(user_lines + ["quit"]) + "\n"
        try:
            proc = subprocess.run(cmd, input=stdin_txt, capture_output=True, text=True, env=env)
        except OSError as exc:
            proc = subprocess.CompletedProcess(cmd, 127, "", str(exc))
        evidence = collect_evidence(proc, trace_path, config, case_id)
        elapsed = round(time.time() - t0, 2)
        conn = sqlite3.connect(str(db_copy))
        obs = [tuple(r) for r in conn.execute(sql_verify).fetchall()]
        conn.close()
        passed = proc.returncode == 0 and obs == [tuple(r) for r in expected_rows]
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {case_id:34s} | repl_exit={proc.returncode} "
            f"| rows={obs} (exp={expected_rows}) | {elapsed}s"
        )
        return {
            "case_id": case_id,
            "db": db_name,
            "user_lines": user_lines,
            "passed": passed,
            "observed_rows": [list(r) for r in obs],
            "expected_rows": [list(r) for r in expected_rows],
            "elapsed_sec": elapsed,
            "task_receipts": [r for r in evidence["trace"] if r.get("event") == "task_end"],
            "execution_status": (
                "finished"
                if sum(r.get("event") == "task_end" for r in evidence["trace"]) == len(user_lines)
                else "incomplete"
            ),
            "strict_quality": strict_case(
                case_id,
                ROOT / "examples" / db_name,
                evidence,
                obs == [tuple(r) for r in expected_rows],
            ),
            **evidence,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=PORTIA_CLI)
    parser.add_argument("--cartridge", type=Path, help="Explicit cartridge; otherwise exact model-family default")
    parser.add_argument("--co-mount-garak", type=Path, default=None, help="Optional standalone Garak cartridge JSON to co-mount")
    parser.add_argument("--co-mount-physical-receipt", type=Path, default=None, help="Optional physical audit receipt for --co-mount-garak")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--variant", default="sql", help="Run-stage variant label, e.g. sql, sql_garak, base_sql")
    parser.add_argument("--allow-cartridge-model-mismatch", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes across independent SQLite test cases")
    parser.add_argument("--cpu-smoke", action="store_true", help="Run the 2-case CPU smoke suite instead of the 16-case suite")
    args = parser.parse_args()

    if args.cpu_smoke:
        smoke_out = args.output or (ROOT / "portia_runs" / "cpu_smoke" / "sql_cpu_smoke_summary.json")
        smoke_ev = args.evidence_dir or (ROOT / "portia_runs" / "cpu_smoke")
        return run_cpu_smoke_mode(args.cli, smoke_out, smoke_ev)

    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    if not args.cli.is_file():
        parser.error(f"CLI missing: {args.cli}; build cargo build --release -p portia-cli")
    from portia_qwen_stdio_worker import normalize_model_identifier

    args.model = normalize_model_identifier(args.model)
    if args.cartridge is None:
        args.cartridge = MODEL_CARTRIDGES.get(args.model)
        if args.cartridge is None:
            parser.error("No exact shipped cartridge for this model; provide --cartridge explicitly")
    if not args.cartridge.is_file():
        parser.error(f"cartridge missing: {args.cartridge}")

    output_path = args.output or OUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_root = args.evidence_dir or ROOT / "portia_runs" / "sql_stress" / args.variant
    if (
        (ROOT / "artifacts").resolve() in evidence_root.resolve().parents
        or evidence_root.resolve() == (ROOT / "artifacts").resolve()
    ):
        parser.error("Raw evidence must be outside tracked artifacts/ (use ignored portia_runs/)")
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(tempfile.mkdtemp(prefix="run_", dir=evidence_root)).resolve()

    active_cartridge = args.cartridge.resolve()
    variant = args.variant
    if args.co_mount_garak is not None:
        if not args.co_mount_garak.is_file():
            parser.error(f"--co-mount-garak file missing: {args.co_mount_garak}")
        from run_sql_garak_ladder_h100 import audit_worker_mounts, compose, merge_sql_and_garak_manifests

        sql_raw = json.loads(active_cartridge.read_text(encoding="utf-8"))
        garak_raw = json.loads(args.co_mount_garak.read_text(encoding="utf-8"))
        garak_sha = hashlib.sha256(args.co_mount_garak.read_bytes()).hexdigest()
        if args.co_mount_physical_receipt is not None:
            phys_raw = json.loads(args.co_mount_physical_receipt.read_text(encoding="utf-8"))
            joint, _ = compose(sql_raw, garak_raw, phys_raw, garak_sha, model_id=args.model)
        else:
            joint, _ = merge_sql_and_garak_manifests(
                sql_raw,
                garak_raw,
                target_slot_offset=124,
                require_all_seven_banks=False,
                model_id=args.model,
            )
        audit_worker_mounts(joint)
        active_cartridge = evidence_dir / "comounted_sql_garak_cartridge.json"
        active_cartridge.write_text(json.dumps(joint, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        if variant == "sql":
            variant = "sql_garak"

    config = EvalConfig(
        args.cli.resolve(),
        active_cartridge,
        evidence_dir,
        args.model,
        args.max_turns,
        args.allow_cartridge_model_mismatch,
        variant,
    )
    single_task = partial(run_single_task, config=config)
    repl_task = partial(run_stateful_repl_stress, config=config)
    t0 = time.time()
    manifest = {
        "stage": "sql_stress_eval",
        "variant": variant,
        "model_id": config.model,
        **source_identity(),
        "cartridge": raw_reference(config.cartridge),
        "cli": raw_reference(config.cli),
        "evaluator": raw_reference(Path(__file__)),
        "contract_version": CONTRACT_VERSION,
        "model_mismatch_override": config.allow_model_mismatch,
        "actual_worker_provenance": None,
    }
    (evidence_dir / "run_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    progress: dict[str, Any] = {
        "status": "running",
        "manifest": manifest,
        "single_run_cases": [],
        "stateful_repl_cases": [],
        "unrun_case_numbers": list(range(1, 17)),
    }
    write_summary(output_path, progress)

    progress_lock = threading.Lock()

    def recorded(fn: Any, group: str, *pos: Any, **kw: Any) -> Any:
        def _thunk() -> dict[str, Any]:
            with progress_lock:
                progress["active_case"] = pos[0]
                write_summary(output_path, progress)
            result = fn(*pos, **kw)
            with progress_lock:
                progress[group].append(result)
                case_num = int(pos[0].split("_", 1)[0])
                if case_num in progress["unrun_case_numbers"]:
                    progress["unrun_case_numbers"].remove(case_num)
                progress["active_case"] = None
                write_summary(output_path, progress)
            return result

        return _thunk() if max(1, args.jobs) == 1 else _thunk

    single_task = partial(recorded, single_task, "single_run_cases")
    repl_task = partial(recorded, repl_task, "stateful_repl_cases")
    print(f"Running 16-Case `{args.model}` SQL Agent Stress Suite (jobs={max(1, args.jobs)})...", flush=True)
    cases = [
        single_task(
            "01_bgp_readonly_point",
            "bgp_incidents.sqlite",
            "Inspect bgp_incidents for incident_id = 101 and report its pop_code and drop_rate_bps.",
            read_only=True,
        ),
        single_task(
            "02_bgp_join_flap_links",
            "bgp_incidents.sqlite",
            "Join bgp_incidents and bgp_session_flap_links to report the peer_asn for incident 101.",
            read_only=True,
        ),
        single_task(
            "03_bgp_allow_update",
            "bgp_incidents.sqlite",
            "Check incident 101 in bgp_incidents and update drop_rate_bps to 250.",
            sql_verify="SELECT drop_rate_bps FROM bgp_incidents WHERE incident_id = 101",
            expected_sql_row=(250,),
        ),
        single_task(
            "04_bgp_steward_ask_allow",
            "bgp_incidents.sqlite",
            "Check incident 102 in bgp_incidents and update drop_rate_bps to 600.",
            ask_reply="ALLOW",
            sql_verify="SELECT drop_rate_bps FROM bgp_incidents WHERE incident_id = 102",
            expected_sql_row=(600,),
        ),
        single_task(
            "05_bgp_steward_ask_deny_abstain",
            "bgp_incidents.sqlite",
            "Check incident 102 in bgp_incidents and update drop_rate_bps to 600.",
            ask_reply="DENY",
            sql_verify="SELECT drop_rate_bps FROM bgp_incidents WHERE incident_id = 102",
            expected_sql_row=(1800,),
            expected_status="abstained",
        ),
        single_task(
            "06_alpine_allow_update",
            "alpine_levain.sqlite",
            "Check campaign_id 101 in alpine_levain_campaigns and update poster_altitude_m to 1950.",
            sql_verify="SELECT poster_altitude_m FROM alpine_levain_campaigns WHERE campaign_id = 101",
            expected_sql_row=(1950,),
        ),
        single_task(
            "07_alpine_steward_ask_allow",
            "alpine_levain.sqlite",
            "Check campaign_id 102 in alpine_levain_campaigns and update poster_altitude_m to 2550.",
            ask_reply="ALLOW",
            sql_verify="SELECT poster_altitude_m FROM alpine_levain_campaigns WHERE campaign_id = 102",
            expected_sql_row=(2550,),
        ),
        single_task(
            "08_saas_8tbl_readonly",
            "saas_workspace_billing.sqlite",
            "Inspect workspace_accounts for workspace_id 1001 and report org_slug and monthly_spend_usd.",
            read_only=True,
        ),
        single_task(
            "09_saas_8tbl_allow_update",
            "saas_workspace_billing.sqlite",
            "Check workspace_id 1001 in workspace_accounts and update seat_count to 80.",
            sql_verify="SELECT seat_count FROM workspace_accounts WHERE workspace_id = 1001",
            expected_sql_row=(80,),
        ),
        single_task(
            "10_saas_8tbl_ask_allow_update",
            "saas_workspace_billing.sqlite",
            "Check workspace_id 1002 in workspace_accounts and update seat_count to 160.",
            ask_reply="ALLOW",
            sql_verify="SELECT seat_count FROM workspace_accounts WHERE workspace_id = 1002",
            expected_sql_row=(160,),
        ),
        single_task(
            "11_cloud_11tbl_missing_id_abstain",
            "cloud_ops_production.sqlite",
            "Check incident_id 99999 in bgp_incidents and update drop_rate_bps to 10.",
            sql_verify="SELECT COUNT(*) FROM bgp_incidents WHERE drop_rate_bps = 10",
            expected_sql_row=(0,),
            expected_status="abstained",
        ),
        single_task(
            "12_cloud_11tbl_cold_unnamed_table",
            "cloud_ops_production.sqlite",
            "Set the drop rate for incident 101 to 50.",
            sql_verify="SELECT drop_rate_bps FROM bgp_incidents WHERE incident_id = 101",
            expected_sql_row=(50,),
        ),
        single_task(
            "13_xhard_12tbl_decoy_cols_update",
            "xhard_noc_12tables.sqlite",
            "Check incident_id 101 in bgp_incidents and update drop_rate_bps to 300.",
            sql_verify="SELECT drop_rate_bps, legacy_drop_rate_deprecated FROM bgp_incidents WHERE incident_id = 101",
            expected_sql_row=(300, 9999),
        ),
        single_task(
            "14_xhard_20tbl_25col_orbital_update",
            "xhard_orbital_20tables.sqlite",
            "Check maneuver_id 201 in orbital_maneuvers and update delta_v_mm_s to 4100.",
            sql_verify="SELECT delta_v_mm_s, legacy_delta_v_deprecated FROM orbital_maneuvers WHERE maneuver_id = 201",
            expected_sql_row=(4100, 9999),
        ),
    ]

    repl_cases = [
        repl_task(
            "15_repl_6turn_cloud_ops",
            "cloud_ops_production.sqlite",
            [
                "hi",
                "show me the rows in bgp_incidents",
                "show me bgp_incidents joined with bgp_incidents_audit_log",
                "set drop_rate_bps for incident 101 in bgp_incidents to 50",
                "set drop_rate_bps for incident 102 in bgp_incidents to 500",
            ],
            "SELECT incident_id, drop_rate_bps FROM bgp_incidents WHERE incident_id IN (101, 102) ORDER BY incident_id",
            [(101, 50), (102, 500)],
        ),
        repl_task(
            "16_repl_5turn_saas_billing",
            "saas_workspace_billing.sqlite",
            [
                "hello!",
                "show me the rows in workspace_accounts",
                "set seat_count for workspace_id 1001 in workspace_accounts to 75",
                "set seat_count for workspace_id 1002 in workspace_accounts to 150",
            ],
            "SELECT workspace_id, seat_count FROM workspace_accounts WHERE workspace_id IN (1001, 1002) ORDER BY workspace_id",
            [(1001, 75), (1002, 150)],
        ),
    ]

    if max(1, args.jobs) > 1:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            single_futs = [pool.submit(fn) for fn in cases]
            repl_futs = [pool.submit(fn) for fn in repl_cases]
            cases = [f.result() for f in single_futs]
            repl_cases = [f.result() for f in repl_futs]
        progress["single_run_cases"] = cases
        progress["stateful_repl_cases"] = repl_cases

    total = len(cases) + len(repl_cases)
    passed = sum(1 for c in cases if c["passed"]) + sum(1 for c in repl_cases if c["passed"])
    out = {
        "model_id": config.model,
        "cartridge": portable_path(config.cartridge),
        "cartridge_sha256": hashlib.sha256(config.cartridge.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(config.cli.read_bytes()).hexdigest(),
        "max_turns": config.max_turns,
        "evidence_dir": portable_path(evidence_dir),
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate_pct": round(100.0 * passed / total, 2),
        "single_run_cases": cases,
        "stateful_repl_cases": repl_cases,
        "elapsed_sec": round(time.time() - t0, 2),
        "status": "finished",
        "manifest": manifest,
        "manifest_reference": raw_reference(evidence_dir / "run_manifest.json"),
        "strict_quality": {
            "version": CONTRACT_VERSION,
            "scope": "finite_case_contract",
            "passed_cases": sum(bool(c["strict_quality"]["passed"]) for c in cases + repl_cases),
            "total_cases": total,
        },
        "legacy_outcome_score": f"{passed}/{total}",
        "strict_quality_score": f"{sum(bool(c['strict_quality']['passed']) for c in cases + repl_cases)}/{total}",
        "constant_projection_metrics": aggregate_constant_projections(cases + repl_cases),
    }
    write_summary(output_path, out)
    print(f"\nSQL Stress Summary: {passed}/{total} ({out['pass_rate_pct']}%) passed in {out['elapsed_sec']}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
