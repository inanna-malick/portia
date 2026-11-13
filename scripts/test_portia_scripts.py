#!/usr/bin/env python3
"""Consolidated CPU unit & regression suite for Portia worker, Garak surgery/audit, and SQL evaluators.

Requires zero GPU hardware and zero network/model downloads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import garak_eval_lib
from garak_eval_lib import (
    HELDOUT_PACKAGE_ENTITIES,
    LIMITS,
    SLOTS,
    annotate_graph_grades,
    audit_data,
    build_disjoint_raw_prose_chunks,
    build_garak_records,
    build_graph_splits,
    build_v2_data,
    build_wk_splits,
    final_graph_answer,
    finalize_physical_audit,
    grade_graphs,
    isomorphic,
    opening_answer,
    reachable,
    summarize_rows,
)
import portia_garak_posttrain_surgery as garak_mod
from portia_qwen_stdio_worker import (
    CartridgeModelMismatch,
    LiveQwen35Engine,
    MissingCartridge,
    ModelFamilyConflict,
    UnknownModelFamily,
    cartridge_family,
    normalize_bank_key,
    normalize_model_identifier,
    resolve_model_family,
    select_cartridge,
    validate_bundle_tensors,
    validate_model_identity,
)
from run_sql_agent_stress_eval import (
    CARTRIDGE,
    CPU_CASES,
    DEFAULT_MODEL,
    EvalConfig,
    MODEL_CARTRIDGES,
    _contains,
    aggregate_constant_projections,
    annotate_summary,
    case_contracts,
    classify_cpu,
    collect_evidence,
    compact_summary,
    constant_projection_metrics,
    is_select_one_projection,
    portable_path,
    run_single_task,
    run_stateful_repl_stress,
    score_case,
    score_task,
    source_identity,
    terminal_text,
    write_summary,
)
from run_sql_garak_ladder_h100 import (
    SQL_BANKS,
    audit_worker_mounts,
    compose,
    merge_sql_and_garak_manifests,
)
from run_standard_non_garak_benchmarks_h100 import (
    eval_wikitext2_ppl_and_kl_chunks,
    resolve_state_cache_path,
)
from run_wide_3model_garak_stress_h100 import (
    REAL_SENATOR_PAIRS,
    build_connected_flight_graphs,
    eval_wikitext2_ppl_and_kl,
    graph_target_comparison,
    load_identity_checked_patched_model,
)

DB = ROOT / "examples" / "bgp_incidents.sqlite"


# ============================================================================
# 1. Worker Model Identity, HF Snapshot Resolution, & Router Layer Tests
# ============================================================================

def _sample_bundle() -> dict:
    return {
        "header": {"model_family": "Qwen/Qwen3.5-0.8B"},
        "model_id": "Qwen/Qwen3.5-0.8B",
        "base_intermediate_size": 16,
        "widened_slots": 4,
        "banks": {
            "OrientTrunk": {
                "2": [
                    {
                        "slot_lo": 16,
                        "slot_hi": 17,
                        "w_gate": [[1.0, 2.0]],
                        "w_up": [[1.0, 2.0]],
                        "w_down": [[1.0], [2.0]],
                    }
                ]
            }
        },
    }


class ModelIdentityAndWorkerTests(unittest.TestCase):
    def test_explicit_family_cannot_relabel_a_known_checkpoint(self):
        with self.assertRaises(ModelFamilyConflict):
            resolve_model_family("0.8b-base", "0.8b")
        self.assertEqual(resolve_model_family("0.8b-base", "0.8b-base"), "Qwen/Qwen3.5-0.8B-Base")
        with tempfile.TemporaryDirectory() as td:
            Path(td, "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-0.8B-Base"}))
            with self.assertRaises(ModelFamilyConflict):
                resolve_model_family(td, "0.8b")
            Path(td, "portia_model_provenance.json").write_text(json.dumps({"model_family": "Qwen/Qwen3.5-0.8B"}))
            with self.assertRaises(ModelFamilyConflict):
                resolve_model_family(td)

    def test_normalization_and_reliable_local_metadata(self):
        for short, expected in [("0.8b", "0.8B"), ("qwen-4b", "4B"), ("qwen3.5-9b-base", "9B-Base")]:
            self.assertEqual(normalize_model_identifier(short), "Qwen/Qwen3.5-" + expected)
        self.assertEqual(
            resolve_model_family("/cache/models--Qwen--Qwen3.5-0.8B-Base/snapshots/abc"),
            "Qwen/Qwen3.5-0.8B-Base",
        )
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(resolve_model_family(td))
            self.assertEqual(resolve_model_family(td, "0.8b"), "Qwen/Qwen3.5-0.8B")
            Path(td, "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-0.8B-Base"}))
            self.assertEqual(resolve_model_family(td), "Qwen/Qwen3.5-0.8B-Base")

    def test_hf_cache_refs_main_precedes_lexicographic_snapshot_and_name_or_path_cache_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td)
            repo_dir = hub / "models--Qwen--Qwen3.5-0.8B-Base"
            snap_main = repo_dir / "snapshots" / ("1" * 40)
            snap_later = repo_dir / "snapshots" / ("f" * 40)
            for snap in (snap_main, snap_later):
                snap.mkdir(parents=True)
                (snap / "config.json").write_text('{"model_type":"qwen3_5_text"}')
                (snap / "model.safetensors").write_bytes(b"stub")
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text("1" * 40 + "\n")
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    LiveQwen35Engine._resolve_model_path("Qwen/Qwen3.5-0.8B-Base"),
                    str(snap_main),
                )
            self.assertEqual(LiveQwen35Engine._resolve_model_path(str(repo_dir)), str(snap_main))
            (repo_dir / "refs" / "main").unlink()
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    LiveQwen35Engine._resolve_model_path("Qwen/Qwen3.5-0.8B-Base"),
                    str(snap_later),
                )
            # Also verify local config.json whose _name_or_path points to a snapshot path:
            local_copy = hub / "copied_checkpoint"
            local_copy.mkdir()
            (local_copy / "config.json").write_text(
                json.dumps({"_name_or_path": str(snap_main)})
            )
            self.assertEqual(resolve_model_family(str(local_copy)), "Qwen/Qwen3.5-0.8B-Base")

    def test_schema_conflicts_are_never_overridden(self):
        raw = _sample_bundle()
        self.assertEqual(cartridge_family(raw), raw["model_id"])
        raw["model_id"] += "-Base"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            validate_model_identity(raw, None, "fixture", override=True)
        raw = _sample_bundle()
        raw["artifact_bundle"] = {"header": {"model_family": "different/model"}}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cartridge_family(raw)

    def test_override_only_bypasses_family(self):
        raw = _sample_bundle()
        with self.assertRaises(UnknownModelFamily):
            validate_model_identity(raw, None, "local")
        with self.assertRaises(CartridgeModelMismatch):
            validate_model_identity(raw, "Qwen/Qwen3.5-0.8B-Base", "fixture")
        validate_model_identity(raw, None, "fixture", override=True)
        validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)
        for params in ({"d_model": 3}, {"base_m": 32}, {"num_layers": 2}):
            dims = {"d_model": 2, "base_m": 16, "num_layers": 3, **params}
            with self.assertRaises(ValueError):
                validate_bundle_tensors(raw, **dims)
        for mutation in ("nan", "bounds", "overlap"):
            bad = copy.deepcopy(raw)
            entry = bad["banks"]["OrientTrunk"]["2"][0]
            if mutation == "nan":
                entry["w_gate"][0][0] = float("nan")
            elif mutation == "bounds":
                bad["widened_slots"] = 0
            else:
                bad["permanent_static"] = {"2": [entry]}
            with self.assertRaises(ValueError):
                validate_bundle_tensors(bad, d_model=2, base_m=16, num_layers=3)

    def test_aliases_share_slot_occupancy_and_report_alias_deduplicated(self):
        self.assertEqual(normalize_bank_key("ModePayload:Report"), "ModePayload:Result")
        self.assertEqual(normalize_bank_key({"ModePayload": "REPORT"}), "ModePayload:Result")
        for canonical, alias in (
            ("OrientTrunk", "ORIENT_TRUNK"),
            ("ModePayload:Result", "ModePayload:Report"),
            ("UnifiedStatic", "PERMANENT_STATIC"),
        ):
            raw = _sample_bundle()
            layers = raw["banks"].pop("OrientTrunk")
            raw["banks"] = {canonical: layers, alias: copy.deepcopy(layers)}
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)
            entry = raw["banks"][alias]["2"][0]
            entry.update(slot_lo=17, slot_hi=18)
            validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)
        raw = _sample_bundle()
        raw["banks"]["Unexpected"] = raw["banks"].pop("OrientTrunk")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)

    def test_legacy_static_overlap_matches_rust_mirror_and_router_exceptions(self):
        raw = _sample_bundle()
        dynamic = copy.deepcopy(raw["banks"]["OrientTrunk"]["2"])
        raw["banks"]["ModePayload:Read"] = {"1": dynamic}
        raw["banks"]["UnifiedStatic"] = {"1": copy.deepcopy(dynamic)}
        validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)
        raw["banks"]["UnifiedStatic"]["1"][0]["w_gate"][0][0] = 3.0
        with self.assertRaisesRegex(ValueError, "non-router"):
            validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)
        raw["banks"]["UnifiedStatic"] = {"2": raw["banks"]["UnifiedStatic"].pop("1")}
        validate_bundle_tensors(raw, d_model=2, base_m=16, num_layers=3)

    def test_explicit_environment_and_family_checked_cartridge_defaults(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"PORTIA_CARTRIDGE_JSON": ""}):
            root = Path(td)
            (root / "artifacts").mkdir()
            (root / "artifacts" / "portia_v3_unified_cartridge.json").write_text(
                json.dumps({"header": {"model_family": "Qwen/Qwen3.5-0.8B"}})
            )
            (root / "artifacts" / "portia_v3_unified_cartridge_08b_base.json").write_text(
                json.dumps({"header": {"model_family": "Qwen/Qwen3.5-0.8B-Base"}})
            )
            (root / "artifacts" / "portia_v3_unified_cartridge_2b_base.json").write_text(
                json.dumps({"header": {"model_family": "Qwen/Qwen3.5-2B-Base"}})
            )
            (root / "artifacts" / "portia_v3_unified_cartridge_4b_base.json").write_text(
                json.dumps({"header": {"model_family": "Qwen/Qwen3.5-4B-Base"}})
            )
            for family, suffix in (
                ("Qwen/Qwen3.5-0.8B", "cartridge.json"),
                ("Qwen/Qwen3.5-0.8B-Base", "cartridge_08b_base.json"),
                ("Qwen/Qwen3.5-2B-Base", "cartridge_2b_base.json"),
                ("Qwen/Qwen3.5-4B-Base", "cartridge_4b_base.json"),
            ):
                self.assertTrue(str(select_cartridge(None, family, root)).endswith(suffix))
            # If portia_v3_unified_cartridge.json actually holds 0.8B-Base weights, 0.8B auto-select rejects it:
            (root / "artifacts" / "portia_v3_unified_cartridge.json").write_text(
                json.dumps({"header": {"model_family": "Qwen/Qwen3.5-0.8B-Base"}})
            )
            with self.assertRaises(MissingCartridge):
                select_cartridge(None, "Qwen/Qwen3.5-0.8B", root)
            with self.assertRaises(MissingCartridge):
                select_cartridge(None, "Qwen/Qwen3.5-4B", root)
            self.assertIsNone(select_cartridge("NoNe", "Qwen/Qwen3.5-4B", root))
            with patch.dict(os.environ, {"PORTIA_CARTRIDGE_JSON": "none"}):
                self.assertIsNone(select_cartridge(None, "Qwen/Qwen3.5-0.8B-Base", root))
                path = root / "artifacts/portia_v3_unified_cartridge_08b_base.json"
                self.assertEqual(select_cartridge(str(path), None, root), path)

    def test_router_layer_derives_from_orient_trunk_when_mounted(self):
        worker = LiveQwen35Engine.__new__(LiveQwen35Engine)
        worker.model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace() for _ in range(6)]))
        worker.cartridges = {"OrientTrunk": {"1": [{"slot_lo": 16, "slot_hi": 17}]}}
        self.assertEqual(worker._resolve_router_layer(), 1)
        worker.cartridges = {}
        self.assertEqual(worker._resolve_router_layer(), 4)


# ============================================================================
# 2. Garak Physical Audit & Finalizer Tests
# ============================================================================

def _physical_fixture():
    records = []
    for family, slot in (
        ("gc_connected", SLOTS[0]),
        ("gc_disconnected", SLOTS[0]),
        ("fc_false_claim", SLOTS[1]),
        ("wk_counterintuitive", SLOTS[2]),
        ("pkg_unreal", SLOTS[3]),
    ):
        records.append(
            {
                "id": family + ":0",
                "split": "train_" + family,
                "partition": "train",
                "family": family,
                "positive_slots": [slot],
                "cold_slots": [s for s in SLOTS if s != slot],
                "globally_benign": False,
            }
        )
    for family in ("real_code_guard", "continuation_guard"):
        records.append(
            {
                "id": family + ":0",
                "split": family,
                "partition": "guard",
                "family": family,
                "positive_slots": [],
                "cold_slots": list(SLOTS),
                "globally_benign": True,
            }
        )
    rows = []
    for record in records:
        row = copy.deepcopy(record)
        row.update(
            denominator=10.0,
            relative={
                "Z-S": 0.0001,
                "P-Z": 0.001,
                "P-S": 0.0011,
                **{slot + "-Z": 0.03 if slot in record["positive_slots"] else 0.001 for slot in SLOTS},
            },
            activation={slot: 0.8 if slot in record["positive_slots"] else 0.1 for slot in SLOTS},
            logits={
                "fp32_head_P-Z": 0.1,
                "fp32_head_P-S": 0.11,
                "native_P-Z": 0.125,
                "native_P-S": 0.125,
                "local_projection_P-Z": 0.15,
            },
        )
        rows.append(row)
    return records, rows


def _complete_physical_report():
    records, rows = _physical_fixture()
    report = summarize_rows(rows, records)
    bindings = {
        key: str(index) * 64
        for index, key in enumerate(
            (
                "model_sha256",
                "capture_sha256",
                "candidate_cartridge_sha256",
                "records_sha256",
                "verifier_sha256",
            ),
            start=1,
        )
    }
    report.update(
        bindings=bindings.copy(),
        record_ids=[r["id"] for r in records],
        hardware={
            "verified_h100_bf16": True,
            "gpu_name": "fixture H100",
            "dtype": "torch.bfloat16",
            "batch_size": 1,
            "use_cache": False,
        },
        target_token_ids=[2665, 9175],
    )
    return report, bindings


class PhysicalSummaryTests(unittest.TestCase):
    def test_complete_measurements_pass_and_connected_gc_is_only_positive_for_its_slot(self):
        records, rows = _physical_fixture()
        report = summarize_rows(rows, records)
        self.assertTrue(report["accepted"], report["errors"])
        connected = next(r for r in records if r["family"] == "gc_connected")
        self.assertEqual(connected["positive_slots"], [SLOTS[0]])
        self.assertEqual(connected["cold_slots"], list(SLOTS[1:]))
        self.assertEqual(report["slots"][SLOTS[0]]["positive_count"], 2)

    def test_missing_duplicate_and_misclassified_records_fail_closed(self):
        records, rows = _physical_fixture()
        for corrupted in (rows[:-1], rows + [copy.deepcopy(rows[-1])]):
            self.assertFalse(summarize_rows(corrupted, records)["accepted"])
        corrupted = copy.deepcopy(rows)
        corrupted[0]["cold_slots"].append(SLOTS[0])
        self.assertFalse(summarize_rows(corrupted, records)["accepted"])
        no_continuation = [r for r in records if "continuation" not in r["family"]]
        self.assertFalse(summarize_rows(rows[:-1], no_continuation)["accepted"])

    def test_each_physical_threshold_is_independently_enforced(self):
        records, rows = _physical_fixture()
        for section, key, value in (
            ("relative", "P-Z", LIMITS["residual_pz"] + 0.001),
            ("relative", "P-S", LIMITS["residual_ps"] + 0.001),
            ("relative", SLOTS[1] + "-Z", LIMITS["residual_pz"] + 0.001),
            ("activation", SLOTS[2], LIMITS["activation"] + 0.001),
            ("logits", "fp32_head_P-Z", LIMITS["logit_pz"] + 0.001),
            ("logits", "fp32_head_P-S", LIMITS["logit_ps"] + 0.001),
        ):
            with self.subTest(section=section, key=key):
                bad = copy.deepcopy(rows)
                bad[-1][section][key] = value
                self.assertFalse(summarize_rows(bad, records)["accepted"])

    def test_native_and_local_projection_logits_are_diagnostics(self):
        records, rows = _physical_fixture()
        for row in rows:
            row["logits"].update({"native_P-Z": 2.0, "native_P-S": 2.0, "local_projection_P-Z": 2.0})
        self.assertTrue(summarize_rows(rows, records)["accepted"])

    def test_nonfinite_missing_and_negative_measurements_are_invalid(self):
        records, rows = _physical_fixture()
        for value in (float("nan"), float("inf"), -0.001, True):
            bad = copy.deepcopy(rows)
            bad[-1]["relative"]["P-Z"] = value
            self.assertFalse(summarize_rows(bad, records)["accepted"], repr(value))
        for denominator in (0.0, -1.0, float("nan")):
            bad = copy.deepcopy(rows)
            bad[-1]["denominator"] = denominator
            self.assertFalse(summarize_rows(bad, records)["accepted"])
        bad = copy.deepcopy(rows)
        del bad[-1]["logits"]["fp32_head_P-Z"]
        self.assertFalse(summarize_rows(bad, records)["accepted"])

    def test_each_slot_requires_its_own_floor_and_ratio(self):
        records, rows = _physical_fixture()
        for positive in (0.005, 0.02):
            bad = copy.deepcopy(rows)
            for row in bad:
                row["relative"][SLOTS[3] + "-Z"] = positive if SLOTS[3] in row["positive_slots"] else 0.003
            report = summarize_rows(bad, records)
            self.assertFalse(report["accepted"])
            self.assertFalse(report["slots"][SLOTS[3]]["passes_ratio"])
            self.assertTrue(report["slots"][SLOTS[1]]["passes_ratio"])

    def test_gc_pooled_mean_cannot_hide_a_failed_connected_family(self):
        records, rows = _physical_fixture()
        for row in rows:
            if row["family"] == "gc_connected":
                row["relative"][SLOTS[0] + "-Z"] = 0.001
            elif row["family"] == "gc_disconnected":
                row["relative"][SLOTS[0] + "-Z"] = 0.059
        report = summarize_rows(rows, records)
        gc = report["slots"][SLOTS[0]]
        self.assertGreaterEqual(gc["ratio"], 10)
        self.assertLess(gc["family_ratios"]["gc_connected"], 10)
        self.assertFalse(gc["passes_ratio"])
        self.assertFalse(report["accepted"])

    def test_zero_guard_requires_nonzero_positive_signal(self):
        records, rows = _physical_fixture()
        for row in rows:
            for slot in row["cold_slots"]:
                row["relative"][slot + "-Z"] = 0.0
        report = summarize_rows(rows, records)
        self.assertTrue(report["accepted"], report["errors"])
        self.assertTrue(all(s["ratio"] == "inf_zero_guard" for s in report["slots"].values()))
        for row in rows:
            for slot in row["positive_slots"]:
                row["relative"][slot + "-Z"] = 0.0
        self.assertFalse(summarize_rows(rows, records)["accepted"])

    def test_bf16_rescaling_cannot_hide_a_physically_failing_slot(self):
        records, rows = _physical_fixture()
        residual = torch.tensor([10.0], dtype=torch.bfloat16, device="cpu")
        x = torch.tensor([1.0], dtype=torch.bfloat16, device="cpu")
        gate = torch.tensor([[3.0]], dtype=torch.bfloat16, device="cpu")

        def measurement(scale):
            up = torch.tensor([[0.5 * scale]], dtype=torch.bfloat16, device="cpu")
            down = torch.tensor([[1.0 / scale]], dtype=torch.bfloat16, device="cpu")
            activation = F.silu(gate @ x) * (up @ x)
            actual = residual + down @ activation
            relative = (actual.float() - residual.float()).norm() / residual.float().norm()
            return activation.float().abs().item(), relative.item()

        before, after = measurement(1.0), measurement(0.1)
        self.assertGreater(before[0], 0.25)
        self.assertLess(after[0], 0.25)
        self.assertAlmostEqual(before[1], after[1], delta=0.002)
        for activation, relative in (before, after):
            bad = copy.deepcopy(rows)
            bad[-1]["activation"][SLOTS[0]] = activation
            bad[-1]["relative"].update({SLOTS[0] + "-Z": relative, "P-Z": relative, "P-S": relative})
            report = summarize_rows(bad, records)
            self.assertFalse(report["accepted"])
            self.assertFalse(report["slots"][SLOTS[0]]["passes_ratio"])


class PhysicalFinalizerTests(unittest.TestCase):
    def test_complete_report_and_directional_wikitext_bound(self):
        report, bindings = _complete_physical_report()
        ids = report["record_ids"]
        result = finalize_physical_audit(
            report,
            expected_bindings=bindings,
            required_record_ids=ids,
            wikitext={"max_token_kl_patched_to_stock_nats": 0.02},
        )
        self.assertTrue(result["accepted"], result["errors"])
        for value in (0.02001, -0.001, float("nan"), None):
            bad = finalize_physical_audit(
                report,
                expected_bindings=bindings,
                required_record_ids=ids,
                wikitext={"max_token_kl_patched_to_stock_nats": value},
            )
            self.assertFalse(bad["accepted"])

    def test_missing_records_hashes_and_hardware_are_rejected(self):
        report, bindings = _complete_physical_report()
        self.assertFalse(
            finalize_physical_audit(report, expected_bindings={}, required_record_ids=report["record_ids"])["accepted"]
        )
        for key in bindings:
            bad = copy.deepcopy(report)
            bad["bindings"][key] = "f" * 64
            result = finalize_physical_audit(
                bad, expected_bindings=bindings, required_record_ids=report["record_ids"]
            )
            self.assertFalse(result["accepted"], key)
        for field in ("record_ids", "hardware", "bindings"):
            bad = copy.deepcopy(report)
            del bad[field]
            self.assertFalse(
                finalize_physical_audit(bad, expected_bindings=bindings, required_record_ids=report["record_ids"])[
                    "accepted"
                ]
            )

    def test_forged_pass_flags_cannot_bypass_threshold_rechecks(self):
        report, bindings = _complete_physical_report()
        for mutate in (
            lambda r: r["combined_relative_residual"]["P-Z"].update(max=0.021),
            lambda r: r["combined_relative_residual"]["P-S"].update(max=0.026),
            lambda r: r["final_logits"]["fp32_head_P-Z"].update(max=0.51),
            lambda r: r["final_logits"]["fp32_head_P-S"].update(max=0.61),
            lambda r: r["slots"][SLOTS[0]]["guard_activation"].update(max=0.251),
            lambda r: r["slots"][SLOTS[1]].update(denominator_max_guard_rel_residual_l2=0.021),
            lambda r: r["slots"][SLOTS[2]]["numerator_mean_pos_rel_residual_l2"].update(equal_family_mean=0.001),
            lambda r: r["slots"][SLOTS[0]]["numerator_mean_pos_rel_residual_l2"]["per_family_mean"].update(
                gc_connected=0.001, gc_disconnected=0.059
            ),
        ):
            bad = copy.deepcopy(report)
            mutate(bad)
            bad.update(accepted=True, errors=[])
            result = finalize_physical_audit(
                bad, expected_bindings=bindings, required_record_ids=report["record_ids"]
            )
            self.assertFalse(result["accepted"], bad)

    def test_zero_guard_cannot_make_zero_gc_family_signal_infinite(self):
        report, bindings = _complete_physical_report()
        gc = report["slots"][SLOTS[0]]
        gc["denominator_max_guard_rel_residual_l2"] = 0.0
        gc["guard_worst"]["max"] = 0.0
        gc["numerator_mean_pos_rel_residual_l2"]["per_family_mean"].update(gc_connected=0.0, gc_disconnected=0.06)
        gc.update(ratio="inf_zero_guard", passes_ratio=True)
        gc["family_ratios"] = {"gc_connected": "inf_zero_guard", "gc_disconnected": "inf_zero_guard"}
        result = finalize_physical_audit(
            report, expected_bindings=bindings, required_record_ids=report["record_ids"]
        )
        self.assertFalse(result["accepted"])


# ============================================================================
# 3. Garak Dataset, Raw-Prose Continuation Guards, v3 Holdout, & Graph Grading
# ============================================================================

class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class GarakDatasetAndGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {"graphs": build_graph_splits([]), "wk": build_wk_splits()}
        cls.bundle = json.loads((ROOT / "artifacts" / "garak_splits_bundle.json").read_text())

    def test_balanced_gold_disjoint_topologies_and_counts(self):
        receipt = audit_data(self.data, [], [])
        self.assertEqual(receipt["unique_unrooted_graphs"], 200)
        self.assertEqual(receipt["graph_counts"]["fit_gc_connected"], 40)
        self.assertEqual(receipt["graph_counts"]["dev_gc_disconnected"], 20)
        self.assertEqual(receipt["graph_counts"]["holdout_gc_connected"], 40)
        for partition in ("fit", "dev", "holdout"):
            connected = self.data["graphs"][f"{partition}_gc_connected"]
            disconnected = self.data["graphs"][f"{partition}_gc_disconnected"]
            self.assertEqual([r["nodes"] for r in connected], [r["nodes"] for r in disconnected])

    def test_v3_holdout_seed_produces_disjoint_unrooted_topologies_from_v2(self):
        connected_prompts = build_connected_flight_graphs(40)
        v2 = build_v2_data(self.bundle, connected_prompts, REAL_SENATOR_PAIRS, holdout_version="v2")
        v3 = build_v2_data(self.bundle, connected_prompts, REAL_SENATOR_PAIRS, holdout_version="v3")
        self.assertEqual(v3["holdout_version"], "v3")
        self.assertGreater(v3["seed_offset"], 0)
        self.assertEqual(v2["graphs"]["fit_gc_connected"], v3["graphs"]["fit_gc_connected"])
        self.assertEqual(v2["graphs"]["dev_gc_connected"], v3["graphs"]["dev_gc_connected"])
        for g3 in v3["graphs"]["holdout_gc_connected"]:
            self.assertFalse(
                any(isomorphic(g3, g2, rooted=False) for g2 in v2["graphs"]["holdout_gc_connected"])
            )

    def test_disjoint_raw_prose_chunks_are_untemplated_and_captured_as_continuation_guards(self):
        import re as _re

        all_chunks = build_disjoint_raw_prose_chunks()
        self.assertEqual(len(build_disjoint_raw_prose_chunks("fit")), 20)
        self.assertEqual(len(build_disjoint_raw_prose_chunks("dev")), 10)
        self.assertEqual(len(build_disjoint_raw_prose_chunks("holdout")), 6)
        self.assertEqual(len({row["text"] for row in all_chunks}), len(all_chunks))
        for row in all_chunks:
            self.assertNotIn("<|im_start|>", row["text"])
            self.assertNotIn("<|im_end|>", row["text"])
            for entity in HELDOUT_PACKAGE_ENTITIES:
                self.assertIsNone(
                    _re.search(r"(?<!\w)" + _re.escape(entity) + r"(?!\w)", row["text"], _re.I)
                )
        fit_records = build_garak_records(self.bundle, self.data, CharacterTokenizer(), "fit")
        dev_records = build_garak_records(self.bundle, self.data, CharacterTokenizer(), "dev")
        for records, expected_rows in ((fit_records, 80), (dev_records, 40)):
            raw_rows = [r for r in records if r["split"] == "raw_prose_cont_bg"]
            self.assertEqual(len(raw_rows), expected_rows)
            self.assertTrue(all(r["family"] == "continuation" and r["globally_benign"] for r in raw_rows))
            self.assertTrue(all(r["positive_slots"] == [] and r["cold_slots"] == list(SLOTS) for r in raw_rows))

    def test_renaming_and_edge_order_cannot_evade_topology_audit(self):
        graph = self.data["graphs"]["fit_gc_connected"][0]
        names = {n: f"Renamed{i}" for i, n in enumerate(graph["nodes"])}
        other = {
            **graph,
            "nodes": list(names.values()),
            "source": names[graph["source"]],
            "target": names[graph["target"]],
            "edges": [[names[a], names[b]] for a, b in reversed(graph["edges"])],
        }
        self.assertTrue(isomorphic(graph, other))
        bad = copy.deepcopy(self.data)
        bad["graphs"]["dev_gc_connected"][0] = other
        with self.assertRaisesRegex(ValueError, "topology overlap"):
            audit_data(bad, [], [])

    def test_directed_paths_and_cycles(self):
        graph = {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "A"], ["B", "C"]], "source": "A", "target": "C"}
        self.assertTrue(reachable(graph))
        graph.update(source="C", target="A")
        self.assertFalse(reachable(graph))

    def test_capture_roles_and_entity_partitions(self):
        fit = build_garak_records(self.bundle, self.data, CharacterTokenizer(), "fit")
        dev = build_garak_records(self.bundle, self.data, CharacterTokenizer(), "dev")
        for records in (fit, dev):
            for row in records:
                self.assertFalse(set(row["positive_slots"]) & set(row["cold_slots"]))
                self.assertEqual(len(row["positive_slots"]) + len(row["cold_slots"]), 4)
                self.assertEqual(row["position"], len(row["input_ids"]) - 1)
            for family in ("gc_connected", "gc_disconnected"):
                rows = [r for r in records if r["family"] == family]
                self.assertTrue(rows)
                self.assertTrue(all(r["positive_slots"] == ["Slot0_GC"] and not r["globally_benign"] for r in rows))
        counts = {
            family: sum(r["family"] == family for r in dev)
            for family in ("false_claim", "true_claim", "real_package", "unreal_package")
        }
        self.assertEqual(counts, {"false_claim": 20, "true_claim": 20, "real_package": 80, "unreal_package": 80})
        self.assertFalse({r["id"] for r in fit} & {r["id"] for r in dev})

    def test_heldout_package_name_in_training_fails(self):
        for partition, index in (("fit", 0), ("dev", 28)):
            bad = copy.deepcopy(self.bundle)
            bad["train_unreal_py"][index] = "Describe HAL 9000"
            with self.assertRaisesRegex(ValueError, "held-out package"):
                build_garak_records(bad, self.data, CharacterTokenizer(), partition)

    def test_wk_alias_collision_rejected(self):
        entity = self.data["wk"]["fit_wk_true"][0]["entities"][0]
        with self.assertRaisesRegex(ValueError, "alias overlap"):
            audit_data(self.data, [], [entity.upper().replace("_", " ")])

    def test_opening_and_final_graph_parsers(self):
        self.assertIsNone(opening_answer("Let us check whether the answer is yes."))
        self.assertIsNone(opening_answer('The quoted response was "No".'))
        self.assertIs(opening_answer("**Yes**, a path exists."), True)
        self.assertIs(opening_answer("No, a direct route is absent."), False)
        self.assertIs(final_graph_answer("Inspect A to B and B to C.\nFinal answer: Yes.", "A", "C")["answer"], True)
        self.assertIs(
            final_graph_answer("Reachable vertices are A and B.\nTherefore, no route from A to C exists.", "A", "C")[
                "answer"
            ],
            False,
        )
        self.assertIs(
            final_graph_answer("Inspect the edges.\nThus, a path from A to C exists.", "A", "C")["answer"], True
        )
        for text in (
            "A -> B -> C",
            "There is no direct edge, but let us continue.",
            'The instruction says "Answer: Yes".',
            "If yes, continue tracing.",
        ):
            self.assertEqual(final_graph_answer(text, "A", "C")["status"], "incomplete")
        self.assertEqual(final_graph_answer("Final answer: Yes.\nConclusion: No.", "A", "C")["status"], "ambiguous")
        parsed = final_graph_answer("Yes.\nTrace first.\nInspect another edge.\nNo conclusion yet.", "A", "C")
        self.assertEqual(parsed["status"], "incomplete")

    def test_incomplete_and_truncated_outputs_remain_in_denominator(self):
        rows = [{"id": str(i), "gold": True, "source": "A", "target": "C"} for i in range(3)]
        outputs = [
            {"text": "Final answer: Yes.", "finish_reason": "eos"},
            {"text": "Let us inspect the", "finish_reason": "length"},
            {"text": "", "finish_reason": "eos"},
        ]
        result = grade_graphs(rows, outputs)
        self.assertAlmostEqual(result["completed_answer_acc_pct"], 100 / 3)
        self.assertEqual(result["conditional_completed_acc_pct"], 100)
        self.assertEqual(result["incomplete_count"], 2)
        trunc = grade_graphs(
            [{"id": "x", "gold": True, "source": "A", "target": "C"}],
            [{"text": "Yes, the path is", "finish_reason": "length"}],
        )
        self.assertEqual(trunc["completed_answer_acc_pct"], 0)
        annotated, counts = annotate_graph_grades(result["examples"])
        self.assertEqual(len(annotated), 3)
        self.assertEqual(counts["truncated_count"], 1)


# ============================================================================
# 4. SQL Finite Contracts, Senator Provenance, Path Hygiene, & Joint Composition
# ============================================================================

def _sql_turn(index, mode, sql, columns=(), rows=(), affected=0, reply=None):
    obs = {"ok": True, "columns": list(columns), "rows": list(rows), "rows_affected": affected}
    if reply is not None:
        obs["ask_reply"] = {"kind": "decision", "decision": reply, "raw_reply": reply}
    return {
        "event": "turn",
        "task_id": 1,
        "turn_index": index,
        "session_event": {"mode": mode, "payload": sql, "observation": obs},
    }


def _sql_end(answer, status="completed"):
    return {"event": "task_end", "task_id": 1, "receipt": {"status": status, "final_answer": answer}}


class SqlContractAndHygieneTests(unittest.TestCase):
    def test_contract_annotations_preserve_original_verdicts_and_provenance(self):
        contract = case_contracts("01_read", DB)[0]
        task = score_task(contract, [_sql_end("Inspection completed.")], DB)
        original = {
            "legacy_outcome_score": "1/16",
            "strict_quality_score": "0/16",
            "single_run_cases": [
                {
                    "passed": True,
                    "provenance": {"evaluator_sha256": "original"},
                    "strict_quality": {
                        "scope": "finite_case_contract",
                        "passed": False,
                        "expected_row_state_ok": True,
                        "trace_integrity_ok": True,
                        "tasks": [task],
                    },
                }
            ],
        }
        result = annotate_summary(original)
        case = result["single_run_cases"][0]
        self.assertFalse(case["contract_supported"])
        self.assertFalse(case["claims_supported"])
        self.assertEqual(case["strict_quality"]["tasks"][0]["contract_assertions_checked"], task["checks"])
        self.assertEqual(case["provenance"], original["single_run_cases"][0]["provenance"])
        self.assertEqual(result["legacy_outcome_score"], original["legacy_outcome_score"])
        self.assertEqual(result["strict_quality_score"], original["strict_quality_score"])
        self.assertNotIn("contract_supported", original["single_run_cases"][0])
        self.assertEqual(annotate_summary(result), result)
        greeting = score_task({"kind": "greeting", "tables": []}, [_sql_end("Hello there.")], DB)
        original["single_run_cases"][0]["strict_quality"]["tasks"] = [greeting]
        self.assertIsNone(annotate_summary(original)["single_run_cases"][0]["claims_supported"])

    def test_prose_and_flat_report_but_not_recursive_json(self):
        self.assertTrue(terminal_text({"summary": "Incident updated.", "incident_id": 101})[1])
        self.assertFalse(terminal_text({"summary": {"summary": "Done"}})[1])
        self.assertFalse(terminal_text({"summary": 'Here are rows: [{"id":101}]'})[1])
        self.assertFalse(terminal_text('{"summary":"Done"}')[1])

    def test_read_requires_executed_facts_and_answer(self):
        contract = case_contracts("01_read", DB)[0]
        row = contract["expected_rows"][0]
        read = _sql_turn(
            0,
            "Read",
            "SELECT pop_code, drop_rate_bps FROM bgp_incidents WHERE incident_id=101",
            row.keys(),
            [list(row.values())],
        )
        answer = f"The pop is {row['pop_code']} and its drop rate is {row['drop_rate_bps']}."
        result = score_task(contract, [read, _sql_end(answer)], DB)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["supporting_turn_ids"]["requested_rows"], [0])
        self.assertFalse(score_task(contract, [_sql_end(answer)], DB)["passed"])
        self.assertFalse(score_task(contract, [read, _sql_end("Inspection completed.")], DB)["passed"])

    def test_sql_dump_fails_even_with_completed_legacy_receipt(self):
        result = score_task(
            {"kind": "greeting", "tables": []},
            [_sql_end("UPDATE bgp_incidents SET drop_rate_bps=250;")],
            DB,
        )
        self.assertFalse(result["checks"]["no_raw_sql"])
        self.assertTrue(result["checks"]["terminal_status"])

    def test_write_passes_without_unrequested_reread_but_enforces_verification_claims(self):
        contract = case_contracts("04_ask", DB)[0]
        events = [
            _sql_turn(0, "Read", "SELECT * FROM bgp_incidents WHERE incident_id=102", ["incident_id", "drop_rate_bps"], [[102, 1800]]),
            _sql_turn(1, "Ask", "Permission?", reply="ALLOW"),
            _sql_turn(2, "Write", "UPDATE bgp_incidents SET drop_rate_bps=600 WHERE incident_id=102", affected=1),
            _sql_turn(3, "Read", "SELECT incident_id, drop_rate_bps FROM bgp_incidents WHERE incident_id=102", ["incident_id", "drop_rate_bps"], [[102, 600]]),
            _sql_end({"summary": "Updated incident 102 to 600 and verified it."}),
        ]
        self.assertTrue(score_task(contract, events, DB)["passed"])
        # Plain update without post-write SELECT passes strict_quality when prompt did not ask to re-read:
        plain_no_reread = score_task(contract, events[:3] + [_sql_end("Updated incident 102 to 600.")], DB)
        self.assertTrue(plain_no_reread["passed"], plain_no_reread)
        self.assertNotIn("post_write_state_observed", plain_no_reread["checks"])
        self.assertFalse(plain_no_reread["post_write_state_observed"])
        # Claiming verification without a post-write SELECT fails `verification_claim_supported`:
        false_verif_claim = score_task(
            contract, events[:3] + [_sql_end("Updated incident 102 to 600 and verified it.")], DB
        )
        self.assertFalse(false_verif_claim["passed"])
        self.assertFalse(false_verif_claim["checks"]["verification_claim_supported"])
        # Missing Ask(ALLOW) and explicit field contradiction still fail closed:
        missing_ask = score_task(contract, events[:1] + events[2:], DB)
        self.assertFalse(missing_ask["checks"]["steward_allow_before_write"])
        events[-1] = _sql_end("Updated incident 102: drop_rate_bps is 1800, requested 600.")
        self.assertFalse(score_task(contract, events, DB)["checks"]["no_explicit_field_contradiction"])

    def test_join_requires_both_actual_sqlite_sources(self):
        contract = case_contracts("02_join", DB)[0]
        good = _sql_turn(
            0,
            "Read",
            "SELECT peer_asn FROM bgp_incidents JOIN bgp_session_flap_links USING(incident_id) WHERE incident_id=101",
            ["peer_asn"],
            [["AS2914-NTT-TRANSIT"]],
        )
        self.assertTrue(score_task(contract, [good, _sql_end("The peer is AS2914-NTT-TRANSIT.")], DB)["passed"])
        good["session_event"]["payload"] = "SELECT peer_asn FROM bgp_session_flap_links WHERE incident_id=101"
        self.assertFalse(
            score_task(contract, [good, _sql_end("The peer is AS2914-NTT-TRANSIT.")], DB)["checks"][
                "joined_sources_executed"
            ]
        )

    def test_constants_and_wrong_verification_row_cannot_supply_evidence(self):
        contract = case_contracts("01_read", DB)[0]
        row = contract["expected_rows"][0]
        fabricated = _sql_turn(
            0,
            "Read",
            "SELECT 'NRT-TOKYO-04' AS pop_code, 4200 AS drop_rate_bps FROM bgp_incidents WHERE incident_id=101",
            row.keys(),
            [list(row.values())],
        )
        self.assertFalse(
            score_task(contract, [fabricated, _sql_end("NRT-TOKYO-04 has drop rate 4200.")], DB)["checks"][
                "requested_rows_executed"
            ]
        )
        write = case_contracts("03_write", DB)[0]
        events = [
            _sql_turn(0, "Read", "SELECT * FROM bgp_incidents WHERE incident_id=101", ["incident_id"], [[101]]),
            _sql_turn(1, "Write", "UPDATE bgp_incidents SET drop_rate_bps=250 WHERE incident_id=101", affected=1),
            _sql_turn(
                2,
                "Read",
                "SELECT incident_id, drop_rate_bps FROM bgp_incidents WHERE incident_id=102",
                ["incident_id", "drop_rate_bps"],
                [[102, 250]],
            ),
            _sql_end("Incident 101 updated to 250 and verified."),
        ]
        scored = score_task(write, events, DB)
        self.assertFalse(scored["checks"]["verification_claim_supported"])
        self.assertFalse(scored["passed"])

    def test_senator_provenance_has_auditable_degrees_and_zero_local_paths_or_footnote_typos(self):
        raw_text = (ROOT / "scripts/portia_senator_provenance.json").read_text(encoding="utf-8")
        self.assertNotIn("portia_runs", raw_text)
        self.assertNotIn("archive_path", raw_text)
        self.assertNotIn("raw_source_file", raw_text)
        self.assertNotIn("Jan. 20, 2025 11", raw_text)
        self.assertNotIn("Jan. 10, 2025 8", raw_text)
        fixture = json.loads(raw_text)
        self.assertEqual(len(fixture["entries"]), 30)
        self.assertEqual(len({entry["case_id"] for entry in fixture["entries"]}), 30)
        by_id = {entry["case_id"]: entry for entry in fixture["entries"]}
        self.assertEqual(by_id["senator_12"]["accepted_witnesses"][0]["term_end"], "Jan. 20, 2025")
        self.assertEqual(by_id["senator_14"]["accepted_witnesses"][0]["term_end"], "Jan. 10, 2025")
        for entry in fixture["entries"]:
            self.assertTrue(entry["prompt_preserved"])
            self.assertTrue(entry["service_source"]["url"].startswith("https://www.senate.gov/"))
            for witness in entry["accepted_witnesses"]:
                self.assertTrue(witness["degree"])
                self.assertIsInstance(witness["degree_year"], int)
                self.assertEqual(witness["degree_year"], witness["graduation_year"])
                self.assertTrue(witness["bioguide_url"].startswith("https://bioguide.congress.gov/"))
                self.assertTrue(witness["education_source_url"].startswith("https://"))
                if witness["degree_evidence_method"] == "combined_primary_records_inference":
                    self.assertGreaterEqual(len(witness["additional_education_sources"]), 1)

    def test_unrun_repl_tasks_are_not_inferred_from_exit_or_final_db(self):
        result = score_case("15_repl", ROOT / "examples" / "cloud_ops_production.sqlite", [], True)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["tasks"]), 5)
        self.assertTrue(all(t["status"] == "unrun" for t in result["tasks"]))

    def test_raw_hashes_compact_summary_and_machine_path_sanitization(self):
        self.assertEqual(
            portable_path("/workspace/portia/artifacts/portia_v3_unified_cartridge_08b_base.json"),
            "artifacts/portia_v3_unified_cartridge_08b_base.json",
        )
        self.assertEqual(
            portable_path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/deadbeef"),
            "models--Qwen--Qwen3.5-0.8B-Base/snapshots/deadbeef",
        )
        # Does not require git-ignored portia_runs/ directory to exist on disk:
        self.assertEqual(
            portable_path(ROOT / "portia_runs" / "nonexistent_dir" / "trace.jsonl"),
            "portia_runs/nonexistent_dir/trace.jsonl",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "case.jsonl"
            trace.write_text('{"event":"diagnostic","error":"failure"}\n{broken\n')
            evidence = collect_evidence(
                subprocess.CompletedProcess([], 1, "raw output", "raw failure"),
                trace,
                EvalConfig(DB, DB, root),
                "case",
            )
            for ref in evidence["raw_evidence"].values():
                raw = Path(ref["path"]).read_bytes()
                self.assertEqual(ref["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(ref["bytes"], len(raw))
            value = {
                "cartridge": "/workspace/checkout/artifacts/portia_v3_unified_cartridge_08b_base.json",
                "single_run_cases": [
                    {
                        "passed": False,
                        **evidence,
                        "provenance": {
                            **evidence["provenance"],
                            "cartridge_path": "/workspace/checkout/artifacts/portia_v3_unified_cartridge_08b_base.json",
                            "resolved_model_path": "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/0123abcd",
                        },
                    }
                ],
            }
            compacted = compact_summary(value)
            serialized = json.dumps(compacted)
            self.assertNotIn("/workspace/", serialized)
            self.assertNotIn("/root/.cache/", serialized)
            small = compacted["single_run_cases"][0]
            self.assertNotIn("trace", small)
            self.assertFalse(small["passed"])
            self.assertEqual(small["trace_parse_errors"][0]["line"], 2)
            path = root / "summary.json"
            write_summary(path, value)
            self.assertLess(path.stat().st_size, 200_000)
            with self.assertRaises(ValueError):
                write_summary(path, {"unbounded": "x" * 200_000})
            self.assertEqual(json.loads(path.read_text())["single_run_cases"][0]["exit_code"], 1)

    def test_exact_model_family_defaults_to_08b_base(self):
        self.assertEqual(DEFAULT_MODEL, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(EvalConfig(DB, DB, DB).model, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(CARTRIDGE, MODEL_CARTRIDGES["Qwen/Qwen3.5-0.8B-Base"])
        self.assertTrue(str(MODEL_CARTRIDGES["Qwen/Qwen3.5-0.8B-Base"]).endswith("_08b_base.json"))
        self.assertNotIn("arbitrary-0.8B-lookalike", MODEL_CARTRIDGES)

    def test_cpu_execution_health_does_not_imply_task_completion(self):
        provenance = {
            "device": "cpu",
            "requested_model": "Qwen/Qwen3.5-0.8B-Base",
            "cartridge_mounted": True,
            "model_mismatch_override": False,
        }
        evidence = {
            "trace": [_sql_end(None, "failed_no_changes")],
            "exit_code": 0,
            "trace_parse_errors": [],
            "provenance": provenance,
            "raw_evidence": {},
        }
        result = classify_cpu(CPU_CASES[0], evidence, (4000,))
        self.assertTrue(result["execution_health"])
        self.assertFalse(result["task_completed"])
        evidence["trace"] = [_sql_end("Updated incident 102 to 500.")]
        evidence["trace"][0]["receipt"]["writes_committed"] = 1
        result = classify_cpu(CPU_CASES[1], evidence, (500,))
        self.assertTrue(result["task_checks"]["completed"])
        self.assertFalse(result["task_checks"]["steward_allow_observed"])

    def test_snapshot_identity_environment_is_validated(self):
        with mock.patch.dict("os.environ", {"PORTIA_SOURCE_COMMIT": "a" * 40, "PORTIA_SOURCE_SNAPSHOT_SHA256": "B" * 64}):
            self.assertEqual(
                source_identity(),
                {"source_commit": "a" * 40, "source_snapshot_sha256": "b" * 64},
            )
        with mock.patch.dict("os.environ", {"PORTIA_SOURCE_COMMIT": "HEAD"}):
            with self.assertRaises(ValueError):
                source_identity()

    def test_numeric_facts_and_select_one_projection_metric(self):
        self.assertTrue(_contains("The spend is 9,600.", 9600.0))
        self.assertTrue(_contains("The value is 50.0.", 50))
        self.assertFalse(_contains("The value is -50.", 50))
        self.assertFalse(_contains("The value is 500.", 50))
        for sql in (
            "SELECT 1",
            "select /* preflight */ 1 FROM bgp_incidents;",
            'SELECT 1 AS "exists" FROM bgp_incidents WHERE incident_id=101',
        ):
            self.assertTrue(is_select_one_projection(sql), sql)
        for sql in (
            "SELECT drop_rate_bps FROM bgp_incidents WHERE 1=1",
            "SELECT 10 FROM bgp_incidents",
            "SELECT '1' FROM bgp_incidents",
            "SELECT 1+drop_rate_bps FROM bgp_incidents",
        ):
            self.assertFalse(is_select_one_projection(sql), sql)
        records = [
            _sql_turn(0, "Read", "SELECT 1 FROM bgp_incidents"),
            _sql_turn(1, "Read", "SELECT * FROM bgp_incidents WHERE 1=1"),
            _sql_turn(2, "Probe", "SELECT 1"),
            {**_sql_turn(3, "Read", "SELECT 1"), "event": "duplicate_turn"},
        ]
        metrics = constant_projection_metrics(records)
        self.assertEqual((metrics["select_one_projection_count"], metrics["read_sql_turn_count"]), (1, 2))
        aggregate = aggregate_constant_projections(
            [{"constant_projection_metrics": metrics}, {"constant_projection_metrics": constant_projection_metrics([])}]
        )
        self.assertEqual(aggregate["select_one_projection_fraction"], 0.5)
        self.assertEqual(aggregate["cases_with_select_one_projection"], 1)
        self.assertEqual(aggregate["observed_case_count"], 2)
        self.assertFalse(aggregate["pass_gate"])

    def test_legacy_repl_row_outcome_stays_separate_from_invalid_trace_quality(self):
        db = ROOT / "examples/cloud_ops_production.sqlite"
        sql = "SELECT incident_id, drop_rate_bps FROM bgp_incidents WHERE incident_id=101"
        with sqlite3.connect(db) as conn:
            expected = conn.execute(sql).fetchall()

        def fake_cli(command, **_kwargs):
            trace = Path(command[command.index("--trace-jsonl") + 1])
            trace.write_text("{broken JSON\n")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "run_sql_agent_stress_eval.subprocess.run", side_effect=fake_cli
        ):
            result = run_stateful_repl_stress(
                "15_repl_6turn_cloud_ops",
                db.name,
                ["hi"],
                sql,
                expected,
                config=EvalConfig(DB, DB, Path(temp)),
            )
        self.assertTrue(result["passed"])
        self.assertFalse(result["strict_quality"]["passed"])
        self.assertFalse(result["strict_quality"]["trace_integrity_ok"])


class JointCompositionTests(unittest.TestCase):
    def fixtures(self):
        def slot(index):
            return {
                "slot_lo": 8 + index,
                "slot_hi": 9 + index,
                "w_gate": [[1.0, 2.0]],
                "w_up": [[2.0, 3.0]],
                "w_down": [[1.0], [2.0]],
            }

        sql = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "base_intermediate_size": 8,
            "widened_slots": 128,
            "banks": {bank: {"22": [slot(i)]} for i, bank in enumerate(sorted(SQL_BANKS))},
        }
        garak = {
            "model_id": sql["model_id"],
            "base_intermediate_size": 8,
            "widened_slots": 4,
            "banks": {},
            "permanent_static": {"22": [slot(i) for i in range(4)]},
        }
        physical = {
            "accepted": True,
            "audit_status": "accepted",
            "model_id": sql["model_id"],
            "target_layer": 22,
            "hardware": {"verified_h100_bf16": True},
            "record_ids": ["heldout-1"],
            "bindings": {
                "model_sha256": "a" * 64,
                "capture_sha256": "b" * 64,
                "candidate_cartridge_sha256": "c" * 64,
            },
            "combined_relative_residual": {k: {"max": 0.001} for k in ("P-Z", "P-S")},
            "final_logits": {k: {"max": 0.01} for k in ("fp32_head_P-Z", "fp32_head_P-S")},
            "wikitext_max_token_kl_patched_to_stock_nats": 0.001,
            "slots": {},
        }
        for name in ("Slot0_GC", "Slot1_FC", "Slot2_WK", "Slot3_PKG"):
            means = {"gc_connected": 0.02, "gc_disconnected": 0.02} if name == "Slot0_GC" else {"positive": 0.02}
            physical["slots"][name] = {
                "guard_activation": {"max": 0.01},
                "denominator_max_guard_rel_residual_l2": 0.001,
                "numerator_mean_pos_rel_residual_l2": {"per_family_mean": means, "equal_family_mean": 0.02},
            }
        return sql, garak, physical

    def test_joint_preserves_sql_and_rejects_overlap_rejection_or_wrong_sha(self):
        sql, garak, physical = self.fixtures()
        joint, checks = compose(sql, garak, physical, "c" * 64)
        self.assertEqual(joint["banks"], sql["banks"])
        self.assertEqual([s["slot_lo"] for s in joint["permanent_static"]["22"]], list(range(132, 136)))
        self.assertEqual(checks["overlap_count"], 0)
        # Also test orient-only Rung-3 manifest linker relocation to offset 48:
        orient_only = {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "base_intermediate_size": 8,
            "widened_slots": 128,
            "banks": {"OrientTrunk": copy.deepcopy(sql["banks"]["OrientTrunk"])},
        }
        rung3_joint, rung3_checks = merge_sql_and_garak_manifests(
            orient_only, garak, target_slot_offset=48, require_all_seven_banks=False
        )
        self.assertEqual([s["slot_lo"] for s in rung3_joint["permanent_static"]["22"]], list(range(56, 60)))
        self.assertEqual(rung3_checks["garak_offsets"], [48, 52])
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            compose(sql, garak, physical, "d" * 64)
        physical["accepted"] = False
        with self.assertRaisesRegex(ValueError, "rejected"):
            compose(sql, garak, physical, "c" * 64)
        physical["accepted"] = True
        entry = sql["banks"]["OrientTrunk"]["22"][0]
        entry["slot_lo"], entry["slot_hi"] = 132, 133
        with self.assertRaisesRegex(ValueError, "overlaps"):
            compose(sql, garak, physical, "c" * 64)

    def test_actual_worker_retains_permanent_weights_for_every_bank_on_cpu(self):
        sql, garak, physical = self.fixtures()
        joint, _ = compose(sql, garak, physical, "c" * 64)
        result = audit_worker_mounts(joint)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["bank_mounts"]), 9)


# ============================================================================
# 5. WikiText Chunked KL, SQL Evidence Streams, & Garak Surgery Entrypoint Tests
# ============================================================================

class LookupModel:
    def __init__(self, logits):
        self.logits = logits
        self.batch_sizes = []

    def __call__(self, *, input_ids, use_cache):
        if use_cache:
            raise AssertionError("Evaluation must not allocate a KV cache")
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.logits[input_ids])


class WikiTextAndEvaluatorTests(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(543)
        self.base = torch.randn(11, 11, generator=rng).bfloat16()
        self.patched = (self.base.float() + 0.4 * torch.randn(11, 11, generator=rng)).bfloat16()
        self.tokens = torch.arange(19) % 11

    def test_chunking_matches_dense_reference_in_both_benchmark_runners(self):
        chunks = self.tokens[:15].reshape(3, 5)
        base_logp = self.base[chunks[:, :-1]].float().log_softmax(-1)
        patch_logp = self.patched[chunks[:, :-1]].float().log_softmax(-1)
        labels = chunks[:, 1:].unsqueeze(-1)
        ppl_b = math.exp(float(-base_logp.gather(-1, labels).mean()))
        ppl_p = math.exp(float(-patch_logp.gather(-1, labels).mean()))
        dense_kl = (base_logp.exp() * (base_logp - patch_logp)).sum(-1).flatten()
        dense_patch_to_stock = (patch_logp.exp() * (patch_logp - base_logp)).sum(-1).flatten()
        expected = {
            "tokens_evaluated": 12,
            "stock_perplexity": round(ppl_b, 4),
            "patched_perplexity": round(ppl_p, 4),
            "delta_perplexity": round(ppl_p - ppl_b, 6),
            "mean_token_kl_nats": round(float(dense_kl.mean()), 8),
            "p99_token_kl_nats": round(float(torch.quantile(dense_kl, 0.99)), 8),
            "max_token_kl_nats": round(float(dense_kl.max()), 6),
            "pct_tokens_kl_gt_0_01": round(float((dense_kl > 0.01).float().mean()) * 100, 4),
            "mean_token_kl_patched_to_stock_nats": float(dense_patch_to_stock.mean()),
            "max_token_kl_patched_to_stock_nats": float(dense_patch_to_stock.max()),
        }
        for chunk_size in (1, 3, 9):
            with self.subTest(chunk_size=chunk_size):
                base = LookupModel(self.base)
                patched = LookupModel(self.patched)
                out = eval_wikitext2_ppl_and_kl(
                    base,
                    patched,
                    None,
                    self.tokens,
                    seq_len=5,
                    batch_size=2,
                    token_chunk_size=chunk_size,
                    device="cpu",
                )
                out_std = eval_wikitext2_ppl_and_kl_chunks(
                    LookupModel(self.base),
                    LookupModel(self.patched),
                    self.tokens,
                    seq_len=5,
                    batch_size=2,
                    token_chunk_size=chunk_size,
                    device="cpu",
                )
                self.assertEqual(base.batch_sizes, [2, 1])
                self.assertEqual(patched.batch_sizes, [2, 1])
                self.assertEqual(out, out_std)
                for key, value in expected.items():
                    self.assertAlmostEqual(out[key], value, delta=2e-5, msg=key)
                self.assertEqual(out["patched_to_stock_kl_passed"], bool(dense_patch_to_stock.max() <= 0.02))

    def test_identical_models_and_bounded_vocab_buffers(self):
        original = F.log_softmax

        def bounded_log_softmax(values, **kwargs):
            self.assertEqual(values.ndim, 2)
            self.assertLessEqual(values.shape[0], 3)
            self.assertEqual(values.dtype, torch.float32)
            return original(values, **kwargs)

        with patch("run_wide_3model_garak_stress_h100.F.log_softmax", side_effect=bounded_log_softmax):
            out = eval_wikitext2_ppl_and_kl(
                LookupModel(self.base),
                LookupModel(self.base),
                None,
                self.tokens,
                seq_len=5,
                token_chunk_size=3,
                device="cpu",
            )
        self.assertEqual(out["delta_perplexity"], 0)
        self.assertEqual(out["mean_token_kl_nats"], 0)
        self.assertEqual(out["max_token_kl_nats"], 0)
        self.assertEqual(out["pct_tokens_kl_gt_0_01"], 0)
        self.assertTrue(out["patched_to_stock_kl_passed"])

    def test_empty_and_invalid_inputs_are_explicit(self):
        for options in (
            {"seq_len": 20},
            {"seq_len": 1},
            {"batch_size": 0},
            {"token_chunk_size": 0},
            {"max_chunks": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                eval_wikitext2_ppl_and_kl(
                    LookupModel(self.base),
                    LookupModel(self.base),
                    None,
                    self.tokens,
                    device="cpu",
                    **options,
                )

    def test_partial_trace_and_single_task_structured_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = EvalConfig(Path("portia-cli"), Path("cartridge.json"), root)
            trace = root / "case.jsonl"
            record = {"event": "diagnostic", "generated_turn": {"mode": "Read"}}
            trace.write_text(json.dumps(record) + '\n{"event":', encoding="utf-8")
            proc = subprocess.CompletedProcess([], 1, "full stdout", "full error")
            evidence = collect_evidence(proc, trace, config, "case")
            self.assertEqual(evidence["trace"], [record])
            self.assertEqual(evidence["trace_parse_errors"][0]["line"], 2)
            self.assertEqual(Path(evidence["stderr_path"]).read_text(), "full error")
            self.assertEqual(evidence["exit_code"], 1)

            receipt = {"status": "completed", "turns_executed": 3, "writes_attempted": 0, "writes_committed": 0}

            def run(cmd, **_kwargs):
                self.assertNotIn("--summary-only", cmd)
                self.assertEqual(cmd[cmd.index("--max-turns") + 1], "64")
                tr = Path(cmd[cmd.index("--trace-jsonl") + 1])
                tr.write_text(json.dumps({"event": "task_end", "receipt": receipt}) + "\n")
                return subprocess.CompletedProcess(cmd, 0, '[{"mode":"Result"}]', "diagnostics")

            with patch("run_sql_agent_stress_eval.subprocess.run", side_effect=run), patch(
                "run_sql_agent_stress_eval.source_identity",
                return_value={"source_commit": "0" * 40, "source_snapshot_sha256": None},
            ):
                result = run_single_task("test", "bgp_incidents.sqlite", "inspect", config=config, read_only=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["summary"], receipt)
            self.assertEqual(result["stdout"], '[{"mode":"Result"}]')


class GarakCalibrationAndSurgeryTests(unittest.TestCase):
    def test_patched_reload_is_pinned_local_and_native_hash_checked_before_mount(self):
        model = SimpleNamespace(eval=lambda: model)
        with patch("run_wide_3model_garak_stress_h100.AutoModelForCausalLM.from_pretrained", return_value=model) as load, patch.object(
            garak_mod, "resolve_checkpoint_snapshot", return_value=(SimpleNamespace(), {"revision": "b" * 40})
        ), patch.object(garak_eval_lib, "_model_hash", return_value="a" * 64):
            actual, digest_val = load_identity_checked_patched_model("test-model", "b" * 40, "a" * 64)
            self.assertIs(actual, model)
            self.assertEqual(digest_val, "a" * 64)
            self.assertEqual(load.call_args.kwargs["revision"], "b" * 40)
            self.assertIs(load.call_args.kwargs["local_files_only"], True)
            with self.assertRaisesRegex(ValueError, "hash mismatch before mounting"):
                load_identity_checked_patched_model("test-model", "b" * 40, "c" * 64)
            load.reset_mock()
            with self.assertRaisesRegex(ValueError, "stock checkpoint commit hash"):
                load_identity_checked_patched_model("test-model", None, "a" * 64)
            load.assert_not_called()

    def test_snapshot_path_binds_revision_when_transformers_config_drops_commit(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "snapshots" / ("c" * 40) / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text('{"model_type":"test"}')
            config = SimpleNamespace(_commit_hash=None)
            with patch("transformers.utils.hub.cached_file", return_value=str(config_path)), patch.object(
                garak_mod.AutoConfig, "from_pretrained", return_value=config
            ):
                resolved, identity = garak_mod.resolve_checkpoint_snapshot("test", local_files_only=True)
                self.assertIs(resolved, config)
                self.assertEqual(identity["revision"], "c" * 40)
                self.assertEqual(len(identity["config_sha256"]), 64)
                with self.assertRaisesRegex(ValueError, "differs from requested"):
                    garak_mod.resolve_checkpoint_snapshot("test", revision="b" * 40)

    def test_three_joint_candidates_and_wk_contrast_penalties(self):
        families = ["gc_connected", "gc_disconnected", "wk_false", "wk_true", "false_claim", "unreal_package"]
        records = [
            {"id": f"{partition}_{family}_{i}", "partition": partition, "family": family, "prompt": f"{partition} {family} {i}"}
            for partition in ("fit", "dev", "holdout")
            for family in families
            for i in range(4)
        ]
        states = {"rows": {r["id"]: {"h": torch.ones(3)} for r in records}}
        vector_json = json.dumps({"v_trace": [1, 0, 0], "v_no": [0, 1, 0], "v_act": [0, 0, 1]})
        proc = subprocess.CompletedProcess([], 0, vector_json, "")
        with tempfile.TemporaryDirectory() as td, patch.object(
            garak_mod, "ensure_portia_cli_binary", return_value=Path("cli")
        ), patch.object(garak_mod.subprocess, "run", return_value=proc), patch.object(
            garak_mod, "calibrate_chordal_angle", return_value=0.12
        ):
            schedule = garak_mod.calibrate_v2_angles(
                None,
                None,
                records,
                states,
                {"trace": {"token_weights": {760: 1, 9175: -0.5, 2665: -0.5}}},
                22,
                Path(td),
            )
            provenance = json.loads((Path(td) / "angle_calibration_schedule.json").read_text())
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule, [c["angles"] for c in garak_mod.V2_CANDIDATES])
        self.assertEqual(len(provenance["deny_positive_record_ids"]), 8)
        self.assertTrue(all("wk_false" in x for x in provenance["deny_positive_record_ids"]))
        self.assertEqual(len(provenance["nonflipping_true_record_ids"]), 8)
        self.assertTrue(all("wk_true" in x for x in provenance["nonflipping_true_record_ids"]))
        self.assertFalse(
            any("holdout" in x for k in ("deny_positive_record_ids", "nonflipping_true_record_ids") for x in provenance[k])
        )

        rows = [
            {"id": "false", "family": "wk_false", "stock": {"yes": 2, "no": 1, "argmax": 9175}, "patched": {"yes": 1, "no": 3, "argmax": 2665}},
            {"id": "true", "family": "wk_true", "stock": {"yes": 3, "no": 1, "argmax": 9175}, "patched": {"yes": 1, "no": 2, "argmax": 2665}},
        ]
        result = garak_mod.summarize_wk_contrast(rows)
        self.assertEqual(result["false_no_margin"], 2)
        self.assertEqual(result["true_margin_harm_penalty"], 3)
        self.assertEqual(result["score"], -2)
        self.assertEqual(result["true_nonflipping_failures"], ["true"])
        with self.assertRaisesRegex(ValueError, "true/false"):
            garak_mod.summarize_wk_contrast(rows[:1])

    def test_compile_and_widen_portia_slots_and_temp_state_cache_hygiene(self):
        cache_path = resolve_state_cache_path("Qwen/Qwen3.5-0.8B")
        self.assertFalse(cache_path.resolve().is_relative_to((ROOT / "scripts").resolve()))
        self.assertFalse(cache_path.resolve().is_relative_to((ROOT / "artifacts").resolve()))

        class DummyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
                self.up_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
                self.down_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
                nn.init.zeros_(self.gate_proj.weight)
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.weight)

        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = DummyMLP()

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([DummyBlock(), DummyBlock()])
                self.model.norm = nn.LayerNorm(2)
                self.lm_head = nn.Linear(2, 5, bias=False)

        dummy = DummyModel()
        fake_manifest = {
            "base_intermediate_size": 4,
            "widened_slots": 4,
            "permanent_static": {
                "0": [
                    {
                        "slot_lo": 4 + i,
                        "slot_hi": 5 + i,
                        "w_gate": [[1.0, 0.5]],
                        "w_up": [[0.25, 0.75]],
                        "w_down": [[0.5], [-0.5]],
                    }
                    for i in range(4)
                ]
            },
        }
        fake_receipt = {"guard_bg_max": 0.01, "max_bg_rss_leakage": 0.25, "elapsed_ms": 12}
        with patch.object(garak_mod, "compile_garak_cartridge_via_rust", return_value=(fake_manifest, fake_receipt)) as comp:
            receipt = garak_mod.compile_and_widen_portia_slots(dummy, None, {}, {}, 0, model_id="Qwen/Qwen3.5-0.8B")
        comp.assert_called_once()
        self.assertEqual(receipt, fake_receipt)
        self.assertEqual(tuple(dummy.model.layers[0].mlp.gate_proj.weight.shape), (8, 2))
        self.assertGreater(float(dummy.model.layers[0].mlp.down_proj.weight[:, 4:8].detach().float().norm().item()), 0.0)

    def test_graph_targets_and_candidate_limit(self):
        stock = {
            "hostile_side_effect_traps": {
                "graph_v2_connected_40": {"n": 40, "false_no_trigger_pct": 15, "completed_answer_acc_pct": 70},
                "graph_v2_disconnected_40": {"n": 40, "premature_yes_pct": 20},
                "legacy_graph_disconnected_40": {"n": 40, "premature_yes_pct": 10},
            }
        }
        patched = {
            "hostile_side_effect_traps": {
                "graph_v2_connected_40": {"n": 40, "false_no_trigger_pct": 10, "completed_answer_acc_pct": 65},
                "graph_v2_disconnected_40": {"n": 40, "premature_yes_pct": 5},
                "legacy_graph_disconnected_40": {"n": 40, "premature_yes_pct": 7.5},
            }
        }
        result = graph_target_comparison(stock, patched)
        self.assertEqual(len(result), 4)
        self.assertEqual(sum(r["patched_attained"] for r in result), 3)
        self.assertEqual(result[-1]["minimum_patched_pct"], 65)
        with self.assertRaisesRegex(ValueError, "at most three"):
            garak_mod.compile_v2_candidate(None, [], {}, {}, {}, 22, "model", Path("unused"), 3)

    def test_phase_specialized_modeplan_and_prefill_banks_roundtrip(self):
        import portia_qwen_stdio_worker as worker

        for raw, expected in (
            ("ModePlan:Read", "ModePlan:Read"),
            ("ModePlan:REPORT", "ModePlan:Result"),
            ("Prefill:Schema", "Prefill:Schema"),
            ("Prefill:User", "Prefill:User"),
            ("Prefill:ToolObs", "Prefill:ToolObs"),
        ):
            self.assertEqual(worker.canonical_manifest_bank(raw), expected)
            self.assertEqual(worker.normalize_bank_key(raw), expected)
        self.assertEqual(worker.normalize_bank_key({"ModePlan": "read"}), "ModePlan:Read")
        self.assertEqual(worker.normalize_bank_key({"PrefillSegment": "ToolObs"}), "Prefill:ToolObs")


if __name__ == "__main__":
    unittest.main()
