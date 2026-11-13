#!/usr/bin/env python3
"""
run_sql_garak_ladder_h100.py — End-to-End 4-Rung Staged Weight-Surgery Ladder & SQL+Garak Manifest Linker:
  - Strict Train (`episodes[0:12]`) vs. Unseen Held-Out (`episodes[12:24]`) Coordinate Split
  - Rung 0: Stock `Qwen/Qwen3.5-0.8B` (`--cartridge-json none` + real `llguidance`)
  - Rung 1: Real Layer-22 `[MODE:` Activation & Unembedding Geometry Capture on Train `[0:12]`
  - Rung 2: Pure-Rust `./target/release/portia-cli compile` (`OrientTrunk` SOCP + `SO(d)`) Evaluated on Held-Out `[12:24]`
  - Rung 3: Validated Co-Mounted `permanent_static["22"]` (Garak 4-slot static honesty) + `banks` (SQL Steward)
            via `merge_sql_and_garak_manifests()` + `audit_worker_mounts()`, evaluated on Held-Out SQL & Garak Splits!
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from garak_eval_lib import digest, finalize_physical_audit
from portia_qwen_stdio_worker import (
    LiveQwen35Engine,
    cartridge_family,
    validate_bundle_tensors,
    validate_model_identity,
)

SQL_BANKS = {
    "OrientTrunk",
    *("ModePayload:" + mode for mode in ("Probe", "Read", "Ask", "Write", "Result", "Abstain")),
}
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PORTIA_CLI = ROOT / "target" / "release" / "portia-cli"
WORKER_PY = ROOT / "scripts" / "portia_qwen_stdio_worker.py"
OUT_DIR = ROOT / "portia_runs" / "ladder_receipts"


def resolve_git_source_identity() -> dict[str, str | None]:
    """Return validated source commit and optional archive snapshot SHA256."""
    commit = os.environ.get("PORTIA_SOURCE_COMMIT")
    snapshot = os.environ.get("PORTIA_SOURCE_SNAPSHOT_SHA256")
    if commit is not None and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit) is None:
        raise ValueError("PORTIA_SOURCE_COMMIT must be a full 40/64-digit hexadecimal git object ID")
    if snapshot is not None and re.fullmatch(r"[0-9a-fA-F]{64}", snapshot) is None:
        raise ValueError("PORTIA_SOURCE_SNAPSHOT_SHA256 must be a 64-digit hexadecimal SHA256")
    if commit is None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            commit = proc.stdout.strip() if proc.returncode == 0 else None
        except OSError:
            commit = None
    return {
        "source_commit": commit.lower() if commit else None,
        "source_snapshot_sha256": snapshot.lower() if snapshot else None,
    }


def merge_sql_and_garak_manifests(
    sql: dict[str, Any],
    garak: dict[str, Any],
    *,
    target_slot_offset: int | None = 124,
    require_all_seven_banks: bool = False,
    model_id: str | None = None,
    num_layers: int = 24,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and merge dynamic SQL cartridge banks with 4 permanent_static Garak slots."""
    expected_family = model_id or cartridge_family(sql)
    if not expected_family:
        raise ValueError("SQL input is missing model family metadata")
    validate_model_identity(sql, expected_family, "SQL input")
    validate_model_identity(garak, expected_family, "Garak input")

    sql_banks = set(sql.get("banks", {}))
    if require_all_seven_banks and sql_banks != SQL_BANKS:
        raise ValueError("SQL input must contain exactly the seven dynamic banks")
    if not sql_banks or not sql_banks.issubset(SQL_BANKS):
        raise ValueError("SQL input must contain non-empty canonical SQL dynamic banks")
    if sql.get("permanent_static"):
        raise ValueError("SQL source already has permanent slots; composition must not replace them")
    if garak.get("banks"):
        raise ValueError("Garak input must not add or replace dynamic SQL banks")

    base = sql.get("base_intermediate_size")
    sql_widened = sql.get("widened_slots")
    garak_widened = garak.get("widened_slots")
    if not isinstance(base, int) or base != garak.get("base_intermediate_size") or sql_widened != 128:
        raise ValueError("expected common base width and SQL reserve 128")
    if require_all_seven_banks and garak_widened != 4:
        raise ValueError("expected common base width, SQL reserve 128, standalone Garak width 4")

    permanent = garak.get("permanent_static", {})
    if len(permanent) != 1:
        raise ValueError("Garak composition requires permanent_static on a single target layer")
    layer_key = next(iter(permanent))
    if require_all_seven_banks and layer_key != "22":
        raise ValueError("0.8B composition requires the four accepted Layer-22 slots")

    entries = permanent[layer_key]
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("Garak permanent_static layer must contain exactly four 1-slot entries")
    sorted_spans = sorted((int(e["slot_lo"]), int(e["slot_hi"])) for e in entries)
    if require_all_seven_banks and sorted_spans != [(base + i, base + i + 1) for i in range(4)]:
        raise ValueError("accepted standalone Garak slots must occupy base offsets 0..4")
    span_lo = sorted_spans[0][0]
    if sorted_spans != [(span_lo + i, span_lo + i + 1) for i in range(4)]:
        raise ValueError("Garak permanent_static entries must occupy four contiguous 1-wide slots")

    first = next(iter(next(iter(sql["banks"].values())).values()))[0]
    width = len(first["w_gate"][0])
    validate_bundle_tensors(sql, d_model=width, base_m=base, num_layers=num_layers)
    validate_bundle_tensors(garak, d_model=width, base_m=base, num_layers=num_layers)

    relocated = copy.deepcopy(entries)
    if target_slot_offset is not None:
        delta = (base + int(target_slot_offset)) - span_lo
        for entry in relocated:
            entry["slot_lo"] = int(entry["slot_lo"]) + delta
            entry["slot_hi"] = int(entry["slot_hi"]) + delta

    reloc_lo = min(int(e["slot_lo"]) for e in relocated)
    reloc_hi = max(int(e["slot_hi"]) for e in relocated)
    if reloc_lo < base or reloc_hi > base + 128:
        raise ValueError(f"relocated Garak offsets {reloc_lo - base}..{reloc_hi - base} exceed 128-slot reserve")

    for bank, layers in sql["banks"].items():
        for entry in layers.get(layer_key, []):
            if int(entry["slot_lo"]) < reloc_hi and reloc_lo < int(entry["slot_hi"]):
                raise ValueError(
                    f"SQL bank {bank} overlaps reserved Garak offsets {reloc_lo - base}..{reloc_hi - base}"
                )

    joint = copy.deepcopy(sql)
    if isinstance(joint.get("artifact_bundle"), dict):
        joint["artifact_bundle"] = None
    joint["permanent_static"] = {layer_key: relocated}
    joint["widened_slots"] = 128
    validate_bundle_tensors(joint, d_model=width, base_m=base, num_layers=num_layers)
    if joint["banks"] != sql["banks"]:
        raise ValueError("dynamic SQL banks changed during composition")
    for source, mounted in zip(entries, relocated):
        if any(source[key] != mounted[key] for key in ("w_gate", "w_up", "w_down")):
            raise ValueError("Garak weight values changed during relocation")

    checks = {
        "dynamic_banks_unchanged": True,
        "dynamic_bank_names": sorted(sql_banks),
        "dynamic_banks_canonical_sha256": digest(sql["banks"]),
        "garak_weights_unchanged": True,
        "overlap_count": 0,
        "widened_slots": 128,
        "target_layer": int(layer_key),
        "garak_offsets": [reloc_lo - base, reloc_hi - base],
        "sql_bisimulation_receipt_scope": "original SQL weights only; unchanged, not a new joint behavior proof",
    }
    return joint, checks


def compose(
    sql: dict[str, Any],
    garak: dict[str, Any],
    physical: dict[str, Any],
    candidate_sha: str,
    *,
    model_id: str = MODEL_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose SHA-bound physically accepted 0.8B Garak slots with all seven dynamic SQL banks."""
    validate_model_identity(sql, model_id, "SQL input")
    validate_model_identity(garak, model_id, "Garak input")
    if set(sql.get("banks", {})) != SQL_BANKS:
        raise ValueError("SQL input must contain exactly the seven dynamic banks")
    if sql.get("permanent_static"):
        raise ValueError("SQL source already has permanent slots; composition must not replace them")
    if garak.get("banks"):
        raise ValueError("Garak input must not add or replace dynamic SQL banks")
    if physical.get("model_id") != model_id:
        raise ValueError(f"final physical receipt is not for {model_id}")
    if "wikitext_max_token_kl_patched_to_stock_nats" not in physical:
        raise ValueError("final physical receipt lacks its WikiText gate")
    bindings = dict(physical.get("bindings", {}))
    bindings["candidate_cartridge_sha256"] = candidate_sha
    checked = finalize_physical_audit(
        physical,
        expected_bindings=bindings,
        required_record_ids=physical.get("record_ids", []),
        wikitext={
            "max_token_kl_patched_to_stock_nats": physical["wikitext_max_token_kl_patched_to_stock_nats"]
        },
    )
    if not checked.get("accepted") or checked.get("audit_status") != "accepted":
        raise ValueError("Garak final physical audit rejected: " + "; ".join(checked.get("errors", [])))
    if physical.get("target_layer") != 22:
        raise ValueError("0.8B composition requires the four accepted Layer-22 slots")

    joint, checks = merge_sql_and_garak_manifests(
        sql,
        garak,
        target_slot_offset=124,
        require_all_seven_banks=True,
        model_id=model_id,
        num_layers=24,
    )
    checks["garak_final_physical_accepted"] = True
    return joint, checks


def audit_worker_mounts(joint: dict[str, Any], *, num_layers: int = 24) -> dict[str, Any]:
    """Exercise actual worker bank mounting on CPU BF16 tensors without loading model weights."""
    import torch

    perm_layers = joint.get("permanent_static", {})
    if not perm_layers:
        raise ValueError("joint manifest has no permanent_static layers to audit")
    perm_layer_key = next(iter(perm_layers))
    perm_layer_idx = int(perm_layer_key)
    layers_used = {int(layer) for bank in joint["banks"].values() for layer in bank} | {perm_layer_idx}
    static = perm_layers[perm_layer_key]
    dim = len(static[0]["w_gate"][0])
    base, reserve = int(joint["base_intermediate_size"]), int(joint["widened_slots"])
    total_layers = max(num_layers, max(layers_used) + 1)
    layers = [SimpleNamespace(mlp=None) for _ in range(total_layers)]
    for idx in layers_used:
        layers[idx].mlp = SimpleNamespace(
            gate_proj=SimpleNamespace(weight=torch.zeros(base + reserve, dim, dtype=torch.bfloat16)),
            up_proj=SimpleNamespace(weight=torch.zeros(base + reserve, dim, dtype=torch.bfloat16)),
            down_proj=SimpleNamespace(weight=torch.zeros(dim, base + reserve, dtype=torch.bfloat16)),
        )
    worker = LiveQwen35Engine.__new__(LiveQwen35Engine)
    worker.torch, worker.device = torch, torch.device("cpu")
    worker.base_intermediate_size, worker.extended_intermediate_size = base, base + reserve
    worker.modified_layers, worker.cartridges = layers_used, copy.deepcopy(joint["banks"])
    worker.permanent_static_layers = copy.deepcopy(perm_layers)
    worker.model, worker.active_bank = SimpleNamespace(model=SimpleNamespace(layers=layers)), "Uninitialized"
    observations = []
    for bank in ["Base", *sorted(SQL_BANKS), "Base"]:
        worker._mount_bank(bank)
        norms = {}
        for entry in static:
            lo, hi = int(entry["slot_lo"]), int(entry["slot_hi"])
            mlp = layers[perm_layer_idx].mlp
            for field, actual in (
                ("w_gate", mlp.gate_proj.weight[lo:hi]),
                ("w_up", mlp.up_proj.weight[lo:hi]),
                ("w_down", mlp.down_proj.weight[:, lo:hi]),
            ):
                expected = torch.tensor(entry[field], dtype=torch.bfloat16)
                norm = torch.linalg.vector_norm(actual.float()).item()
                if not torch.equal(actual, expected) or not norm > 0:
                    raise ValueError(f"permanent {field} changed or is zero under bank {bank}, slot {lo}")
                norms[f"{lo - base}:{field}"] = norm
        for layer, entries in joint["banks"].get(bank, {}).items():
            mlp = layers[int(layer)].mlp
            for entry in entries:
                lo, hi = int(entry["slot_lo"]), int(entry["slot_hi"])
                for field, actual in (
                    ("w_gate", mlp.gate_proj.weight[lo:hi]),
                    ("w_up", mlp.up_proj.weight[lo:hi]),
                    ("w_down", mlp.down_proj.weight[:, lo:hi]),
                ):
                    if not torch.equal(actual, torch.tensor(entry[field], dtype=torch.bfloat16)):
                        raise ValueError(f"dynamic weight changed under bank {bank}: {field}")
        observations.append({"bank": bank, "permanent_norms": norms, "dynamic_values_match": True})
    return {
        "scope": "CPU BF16 tensor mounting; no model inference or behavioral equivalence claim",
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "bfloat16",
        "passed": True,
        "bank_mounts": observations,
    }


def parse_opening_prompt_targets(opening_prompt: str) -> tuple[str, str, int]:
    """Extract (entity_slug, table_name, target_balance) from the train ScenarioCoordinate prompt."""
    ticks = re.findall(r"`([^`]+)`", opening_prompt)
    slug = ticks[0] if len(ticks) >= 1 else "entity-01a"
    table = ticks[1] if len(ticks) >= 2 else "bgp_peering_sessions"
    nums = re.findall(r"=\s*(\d+)|\bto\s+(\d+)\b", opening_prompt)
    bal = 1350
    for a, b in nums:
        if a or b:
            bal = int(a or b)
            break
    return slug, table, bal


def make_obs_state(
    world_id: str,
    turn_index: int,
    table_name: str,
    entity_name: str,
    *,
    has_schema_probe: bool,
    candidate_count: int | None,
    policy_requires_ask: bool,
    is_locked_or_denied: bool,
    steward_grant_approved: bool | None,
    applied_write_count: int,
    has_post_write_verification: bool,
) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "turn_index": turn_index,
        "table_name": table_name,
        "entity_name": entity_name,
        "operation": "UPDATE",
        "has_schema_probe": has_schema_probe,
        "candidate_count": candidate_count,
        "policy_requires_ask": policy_requires_ask,
        "is_locked_or_denied": is_locked_or_denied,
        "steward_grant_approved": steward_grant_approved,
        "applied_write_count": applied_write_count,
        "has_post_write_verification": has_post_write_verification,
        "zero_row_toctou_race": False,
    }


def build_train_calibration_sequences(
    train_receipts: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any] | None]]:
    """
    Build multi-bank `[MODE:` router calibration sequences strictly from Train episodes `[0:12]`
    matching the exact Pretrain-LOTL + `<tool_response>` wire protocol of `LiveSqliteDailyDriver`.
    """
    sequences: list[Any] = []
    row_states: list[dict[str, Any] | None] = []

    for idx, rec in enumerate(train_receipts):
        opening = rec["opening_prompt"].strip()
        slug, table, target_bal = parse_opening_prompt_targets(opening)
        init_bal = target_bal - 100
        wid = f"train_ep_{idx:02d}"

        user_env = f"<|im_start|>user\n{opening}<|im_end|>\n"
        open_mode = "<|im_start|>assistant\n<think>\n[MODE:"

        probe_cols = [
            {"cid": 0, "dflt_value": None, "name": "record_id", "notnull": 0, "pk": 1, "type": "INTEGER"},
            {"cid": 1, "dflt_value": None, "name": "entity_slug", "notnull": 1, "pk": 0, "type": "TEXT"},
            {"cid": 2, "dflt_value": None, "name": "balance_credits", "notnull": 1, "pk": 0, "type": "INTEGER"},
            {"cid": 3, "dflt_value": None, "name": "policy_decision", "notnull": 1, "pk": 0, "type": "TEXT"},
            {"cid": 4, "dflt_value": None, "name": "lock_state", "notnull": 1, "pk": 0, "type": "TEXT"},
            {"cid": 5, "dflt_value": None, "name": "steward_handle", "notnull": 1, "pk": 0, "type": "TEXT"},
            {"cid": 6, "dflt_value": None, "name": "policy_guard_c0", "notnull": 1, "pk": 0, "type": "INTEGER"},
            {"cid": 7, "dflt_value": None, "name": "policy_guard_c1", "notnull": 1, "pk": 0, "type": "INTEGER"},
        ]
        catalog_rows = [
            {"name": table},
            {"name": "distractor_mirror_01_archive"},
        ]
        catalog_probe_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " PROBE]"},
            {
                "bank": "ModePayload:Probe",
                "text": f"\nPlan: List tables in sqlite_master before inspecting {table}.\n</think>\n<sql>\nSELECT name FROM sqlite_master WHERE type = 'table';\n</sql>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"columns": catalog_rows, "purpose": "schema_probe", "status": "OK"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]
        pragma_probe_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " PROBE]"},
            {
                "bank": "ModePayload:Probe",
                "text": f"\nPlan: Inspect column schema for {table}.\n</think>\n<sql>\nPRAGMA table_info({table});\n</sql>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"columns": probe_cols, "purpose": "schema_probe", "status": "OK"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]
        probe_prefix = [*catalog_probe_turn, *pragma_probe_turn]

        cand_row = {
            "balance_credits": init_bal,
            "entity_slug": slug,
            "lock_state": "CLEAR",
            "policy_decision": "ASK",
            "policy_guard_c0": 0,
            "policy_guard_c1": 165,
            "record_id": 1000,
            "steward_handle": "steward_ops",
        }
        read_cand_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " READ]"},
            {
                "bank": "ModePayload:Read",
                "text": f"\nPlan: Read target row for {slug}.\n</think>\n<sql>\nSELECT * FROM {table} WHERE entity_slug = '{slug}';\n</sql>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"purpose": "candidate", "row_count": 1, "rows": [cand_row], "status": "OK"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]

        ask_json_body = json.dumps(
            {
                "select": f"SELECT * FROM {table} WHERE entity_slug = '{slug}';",
                "options": ["ALLOW", "DENY"],
            },
            indent=2,
        )
        ask_allow_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " ASK]"},
            {
                "bank": "ModePayload:Ask",
                "text": f"\nPlan: I need to identify the target invariant, which is a specific condition about an entity's state and a specific update action.\n</think>\n<to_user>\n{ask_json_body}\n</to_user>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps(
                    {"decision": "ALLOW", "reply": f"ALLOW — Steward token STEWARD_TOK_B4AD_R1 for `{slug}`.", "status": "GRANTED"},
                    separators=(",", ":"),
                )
                + "\n</tool_response><|im_end|>\n",
            },
        ]

        ask_deny_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " ASK]"},
            {
                "bank": "ModePayload:Ask",
                "text": f"\nPlan: I need to identify the target invariant, which is a specific condition about an entity's state and a specific update action.\n</think>\n<to_user>\n{ask_json_body}\n</to_user>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"decision": "DENY", "reply": f"DENY — Reject and cancel write for `{slug}`.", "status": "DENIED"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]

        write_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " WRITE]"},
            {
                "bank": "ModePayload:Write",
                "text": f"\nPlan: Apply approved balance_credits update on {slug}.\n</think>\n<sql>\nUPDATE {table} SET balance_credits = {target_bal} WHERE entity_slug = '{slug}';\n</sql>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"operation": "UPDATE", "rows_affected": 1, "status": "APPLIED"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]

        ver_row = dict(cand_row)
        ver_row["balance_credits"] = target_bal
        read_ver_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " READ]"},
            {
                "bank": "ModePayload:Read",
                "text": f"\nPlan: Verify updated row for {slug}.\n</think>\n<sql>\nSELECT * FROM {table} WHERE entity_slug = '{slug}';\n</sql>",
            },
            {
                "bank": "Base",
                "text": "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + json.dumps({"purpose": "verification", "row_count": 1, "rows": [ver_row], "status": "OK"}, separators=(",", ":"))
                + "\n</tool_response><|im_end|>\n",
            },
        ]
        result_turn = [
            {"bank": "OrientTrunk", "text": open_mode + " RESULT]"},
            {
                "bank": "ModePayload:Result",
                "text": f"\nPlan: Report verified update for {slug}.\n</think>\n<to_user>\n"
                + json.dumps({"entity_slug": slug, "balance_credits": target_bal, "status": "VERIFIED"})
                + "\n</to_user>",
            },
            {"bank": "Base", "text": "<|im_end|>\n"},
        ]
        followup_user = {
            "bank": "Base",
            "text": f"<|im_start|>user\nFollow-up check (read-only): report the current `balance_credits` and `lock_state` for `{slug}` in `{table}` without modifying any rows.<|im_end|>\n",
        }

        sequences.append([{"bank": "Base", "text": user_env}, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 0, table, slug, has_schema_probe=False, candidate_count=None, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *catalog_probe_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 1, table, slug, has_schema_probe=False, candidate_count=None, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 2, table, slug, has_schema_probe=True, candidate_count=None, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=0, has_post_write_verification=False))
        sequences.append([{"bank": "Base", "text": user_env}, *pragma_probe_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 1, table, slug, has_schema_probe=True, candidate_count=None, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 3, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=True, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, *ask_allow_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 4, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=True, is_locked_or_denied=False, steward_grant_approved=True, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, *ask_deny_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 4, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=True, is_locked_or_denied=True, steward_grant_approved=False, applied_write_count=0, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, *ask_allow_turn, *write_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 5, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=True, is_locked_or_denied=False, steward_grant_approved=True, applied_write_count=1, has_post_write_verification=False))

        sequences.append([{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, *ask_allow_turn, *write_turn, *read_ver_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 6, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=True, is_locked_or_denied=False, steward_grant_approved=True, applied_write_count=1, has_post_write_verification=True))

        wave0_full = [{"bank": "Base", "text": user_env}, *probe_prefix, *read_cand_turn, *ask_allow_turn, *write_turn, *read_ver_turn, *result_turn]
        sequences.append([*wave0_full, followup_user, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 7, table, slug, has_schema_probe=True, candidate_count=None, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=None, applied_write_count=1, has_post_write_verification=False))
        sequences.append([*wave0_full, followup_user, *read_ver_turn, {"bank": "OrientTrunk", "text": open_mode}])
        row_states.append(make_obs_state(wid, 8, table, slug, has_schema_probe=True, candidate_count=1, policy_requires_ask=False, is_locked_or_denied=False, steward_grant_approved=True, applied_write_count=1, has_post_write_verification=True))

        sequences.append([{"bank": "Base", "text": user_env + "<|im_start|>assistant\n<think>\n[MODE: READ]\nPlan: Query primary table.\n</think>\n<sql>\nSELECT "}])
        row_states.append(None)
        sequences.append([{"bank": "Base", "text": user_env}])
        row_states.append(None)

    return sequences, row_states


def run_nursery_24(cartridge_arg: str, out_json: Path) -> dict[str, Any]:
    worker_cmd = f"{sys.executable} {WORKER_PY} --model-path {MODEL_ID} --cartridge-json {cartridge_arg}"
    cmd = [
        str(PORTIA_CLI),
        "nursery",
        "--episodes",
        "24",
        "--app",
        "sql",
        "--synth-user",
        "deterministic",
        "--max-turns",
        "8",
        "--followup-waves",
        "1",
        "--worker-cmd",
        worker_cmd,
        "--out-json",
        str(out_json),
    ]
    env = dict(os.environ)
    env["PORTIA_PYTHON"] = sys.executable
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    return json.loads(out_json.read_text(encoding="utf-8"))


def summarize_split(receipts: list[dict[str, Any]], label: str) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    db_reads = 0
    db_writes_committed = 0
    results_emitted = 0
    full_trajectories_with_write = 0

    for r in receipts:
        st = r["summary"]["status"]
        outcomes[st] += 1
        db_writes_committed += int(r["summary"].get("writes_committed", 0))
        saw_w = int(r["summary"].get("writes_committed", 0)) > 0
        saw_res = False
        for ev in r["events"]:
            m = ev["mode"]
            modes[m] += 1
            if m == "Read" and ev["observation"]["ok"]:
                db_reads += 1
            if m == "Result" and ev["observation"]["ok"]:
                results_emitted += 1
                saw_res = True
        if saw_w and saw_res:
            full_trajectories_with_write += 1

    res = {
        "label": label,
        "n_episodes": len(receipts),
        "outcomes": dict(outcomes),
        "emitted_modes": dict(modes),
        "db_reads_ok": db_reads,
        "db_writes_committed": db_writes_committed,
        "results_emitted": results_emitted,
        "episodes_with_committed_write_and_result": full_trajectories_with_write,
    }
    print(
        f"  [{label}] outcomes={dict(outcomes)} | modes={dict(modes)} | "
        f"reads={db_reads} writes_committed={db_writes_committed} full_write+result={full_trajectories_with_write}/{len(receipts)}"
    )
    return res


def compile_garak_manifest_for_comount(model_id: str, slot_offset: int = 48) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile Garak's 4 Layer-22 static honesty slots via `portia-cli compile-garak` using temp cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import portia_garak_posttrain_surgery as garak_mod

    bundle = json.loads((ROOT / "artifacts" / "garak_splits_bundle.json").read_text(encoding="utf-8"))
    resolved = LiveQwen35Engine._resolve_model_path(model_id)
    tok = AutoTokenizer.from_pretrained(resolved)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    layers, _, _ = garak_mod.get_layers_and_head(model)
    target_layer = len(layers) - 2
    short_name = model_id.split("/")[-1]
    cache_path = garak_mod.default_state_cache_dir() / f"cached_states_{short_name}.pt"
    states = garak_mod.capture_states_for_model(model, tok, bundle, target_layer, cache_path=cache_path)
    manifest, receipt = garak_mod.compile_garak_cartridge_via_rust(
        model,
        tok,
        states,
        bundle,
        target_layer,
        model_id=model_id,
        slot_offset=slot_offset,
        widened_slots=4,
    )
    del model
    torch.cuda.empty_cache()
    return manifest, receipt


def extract_garak_permanent_static_entries(model_id: str, slot_offset: int = 48) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, receipt = compile_garak_manifest_for_comount(model_id, slot_offset=slot_offset)
    layer_key = next(iter(manifest["permanent_static"]))
    return manifest["permanent_static"][layer_key], receipt


def main() -> int:
    import torch
    import portia_garak_posttrain_surgery as garak_mod

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    git_id = resolve_git_source_identity()

    print("\n=== [RUNG 0] 24-Episode Stock Qwen/Qwen3.5-0.8B (--cartridge-json none) ===")
    t0 = time.time()
    r0_path = OUT_DIR / "rung0_24ep.json"
    if r0_path.exists():
        print("  Reusing cached Rung 0 24-episode receipt:", r0_path)
        rung0_data = json.loads(r0_path.read_text(encoding="utf-8"))
    else:
        rung0_data = run_nursery_24("none", r0_path)
    r0_train = summarize_split(rung0_data["episode_receipts"][:12], "Rung0 Train [0:12]")
    r0_hold = summarize_split(rung0_data["episode_receipts"][12:], "Rung0 Held-Out [12:24]")
    print(f"  Rung 0 elapsed: {time.time() - t0:.1f}s")

    print("\n=== [RUNG 1] Capturing Real Layer-22 `[MODE:` Activations & Unembedding Geometry on Train [0:12] ===")
    t1 = time.time()
    sequences, row_states = build_train_calibration_sequences(rung0_data["episode_receipts"][:12])
    eng = LiveQwen35Engine(MODEL_ID, "none")
    cap = eng.capture_activations(sequences=sequences, layers=[22], include_prefix_background=False)
    geom = eng.extract_compiler_weight_geometry()
    base_m = eng.base_intermediate_size
    eng.close()
    del eng
    torch.cuda.empty_cache()

    compile_input = {
        "status": "activations",
        "activations": cap["activations"],
        "residuals": cap["residuals"],
        "row_states": row_states,
        "model_family": MODEL_ID,
        "base_intermediate_size": base_m,
        "widened_slots": 128,
        "compiler_weight_geometry": geom,
    }
    capture_json_path = OUT_DIR / "rung1_train_captures.json"
    capture_json_path.write_text(json.dumps(compile_input), encoding="utf-8")
    print(
        f"  Captured {len(sequences)} multi-bank sequences at L22 (d_model={geom['d_model']}, base_m={base_m}) "
        f"in {time.time() - t1:.1f}s"
    )

    print("\n=== [RUNG 2] Compiling `OrientTrunk` Cartridge via Pure-Rust `portia-cli compile` (Clarabel SOCP + SO(d)) ===")
    sql_cart_path = OUT_DIR / "rung2_sql_orient_cartridge.json"
    t_comp = time.time()
    comp_out = subprocess.check_output(
        [
            str(PORTIA_CLI),
            "compile",
            "--input-json",
            str(capture_json_path),
            "--out-json",
            str(sql_cart_path),
        ],
        text=True,
    )
    comp_ms = (time.time() - t_comp) * 1000.0
    compile_prov = json.loads(comp_out)
    print(
        f"  `portia-cli compile` succeeded in {comp_ms:.1f}ms: bisimulation_verified={compile_prov['bisimulation_verified']}, "
        f"min_pos_gate={compile_prov['min_positive_gate_pre_silu']:.3f}, max_neg_gate={compile_prov['max_negative_gate_pre_silu']:.3f}, "
        f"max_neg_leak={compile_prov['max_negative_swiglu_leak_l2']:.5f}"
    )

    print("\n=== [RUNG 2 EVAL] Evaluating Compiled `OrientTrunk` Cartridge on Train [0:12] & Unseen Held-Out [12:24] ===")
    t2 = time.time()
    rung2_data = run_nursery_24(str(sql_cart_path), OUT_DIR / "rung2_24ep.json")
    r2_train = summarize_split(rung2_data["episode_receipts"][:12], "Rung2 Train [0:12]")
    r2_hold = summarize_split(rung2_data["episode_receipts"][12:], "Rung2 Held-Out [12:24]")
    print(f"  Rung 2 eval elapsed: {time.time() - t2:.1f}s")

    print("\n=== [RUNG 3] Linking Co-Mounted Manifest via `merge_sql_and_garak_manifests` + `audit_worker_mounts` ===")
    garak_manifest, garak_receipt = compile_garak_manifest_for_comount(MODEL_ID, slot_offset=0)
    sql_manifest = json.loads(sql_cart_path.read_text(encoding="utf-8"))
    combined_manifest, merge_checks = merge_sql_and_garak_manifests(
        sql_manifest,
        garak_manifest,
        target_slot_offset=48,
        require_all_seven_banks=False,
        model_id=MODEL_ID,
    )
    cpu_mount_audit = audit_worker_mounts(combined_manifest)
    combined_cart_path = OUT_DIR / "rung3_combined_garak_sql_cartridge.json"
    combined_cart_path.write_text(json.dumps(combined_manifest, indent=2), encoding="utf-8")

    print("\n=== [RUNG 3A EVAL] Evaluating Co-Mounted Garak+SQL Manifest on Unseen Held-Out SQL Steward [12:24] ===")
    t3 = time.time()
    rung3_sql_data = run_nursery_24(str(combined_cart_path), OUT_DIR / "rung3_24ep.json")
    r3_train = summarize_split(rung3_sql_data["episode_receipts"][:12], "Rung3 Co-Mounted Train [0:12]")
    r3_hold = summarize_split(rung3_sql_data["episode_receipts"][12:], "Rung3 Co-Mounted Held-Out [12:24]")
    print(f"  Rung 3A SQL eval elapsed: {time.time() - t3:.1f}s")

    print("\n=== [RUNG 3B EVAL] Evaluating Co-Mounted Garak+SQL Manifest on Garak Splits via `LiveQwen35Engine` (`_mount_bank('Base')` @ L22!) ===")
    t3b = time.time()
    bundle = json.loads((ROOT / "artifacts" / "garak_splits_bundle.json").read_text(encoding="utf-8"))
    pypi_set, crates_set = garak_mod.load_package_registries(bundle["rust_std"])
    live_eng = LiveQwen35Engine(MODEL_ID, str(combined_cart_path))
    live_eng._mount_bank("Base")
    l22_mlp = live_eng.model.model.layers[22].mlp
    orient_norm_on_base = float(l22_mlp.down_proj.weight[:, base_m : base_m + 32].float().norm().item())
    garak_norm_on_base = float(l22_mlp.down_proj.weight[:, base_m + 48 : base_m + 52].float().norm().item())
    assert orient_norm_on_base == 0.0, f"Expected OrientTrunk zeroed under Base, got {orient_norm_on_base}"
    assert garak_norm_on_base > 0.0, f"Expected permanent_static['22'] mounted under Base, got {garak_norm_on_base}"
    live_eng.tokenizer.padding_side = "left"
    if live_eng.tokenizer.pad_token_id is None:
        live_eng.tokenizer.pad_token_id = live_eng.tokenizer.eos_token_id
    garak_co_eval = garak_mod.evaluate_all_splits(
        live_eng.model, live_eng.tokenizer, bundle, pypi_set, crates_set, batch_size=32
    )
    live_eng.close()
    print(
        f"  Rung 3B Garak co-mounted eval elapsed: {time.time() - t3b:.1f}s "
        f"(orient_norm_on_base={orient_norm_on_base}, garak_norm_on_base={garak_norm_on_base:.4f})"
    )

    ladder_summary = {
        "model_id": MODEL_ID,
        **git_id,
        "compile_ms": round(comp_ms, 2),
        "compile_provenance": compile_prov,
        "rung0_stock_none": {"train_0_12": r0_train, "holdout_12_24": r0_hold},
        "rung2_sql_orient_only": {"train_0_12": r2_train, "holdout_12_24": r2_hold},
        "rung3_comounted_garak_plus_sql": {
            "merge_checks": merge_checks,
            "cpu_mount_audit_passed": cpu_mount_audit["passed"],
            "orient_norm_under_base_bank": orient_norm_on_base,
            "permanent_static_garak_norm_under_base_bank": round(garak_norm_on_base, 6),
            "sql_train_0_12": r3_train,
            "sql_holdout_12_24": r3_hold,
            "garak_compiler_receipt": garak_receipt,
            "garak_metrics_under_comounted_worker": garak_co_eval["metrics"],
        },
        "total_elapsed_sec": round(time.time() - t_global, 2),
    }
    summary_path = OUT_DIR / "h100_4rung_ladder_summary.json"
    summary_path.write_text(json.dumps(ladder_summary, indent=2), encoding="utf-8")
    print("\n=== COMPLETE 4-RUNG H100 SUMMARY SAVED TO", summary_path, "===")
    print(
        json.dumps(
            {
                "compile_ms": ladder_summary["compile_ms"],
                "rung0_holdout_12_24": r0_hold,
                "rung2_holdout_12_24": r2_hold,
                "rung3_sql_holdout_12_24": r3_hold,
                "rung3_garak_comounted": {
                    k: {mk: mv for mk, mv in v.items() if "pct" in mk}
                    for k, v in garak_co_eval["metrics"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
