#!/usr/bin/env python3
"""Stateless-protocol PyTorch + `llguidance` tensor & weight-bank worker for `portia-cli`."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# The Rust integration tests also load this entrypoint by absolute file path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _k, _v in (
    ("PYTORCH_ENABLE_MPS_FALLBACK", "1"),
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
    ("TQDM_DISABLE", "1"),
    ("TRANSFORMERS_VERBOSITY", "error"),
    ("TOKENIZERS_PARALLELISM", "false"),
):
    os.environ.setdefault(_k, _v)
warnings.filterwarnings("ignore")

try:
    import llguidance
    import llguidance.hf
    import llguidance.torch

    if hasattr(llguidance.torch.apply_token_bitmask_inplace_kernel, "_torchdynamo_orig_callable"):
        llguidance.torch.apply_token_bitmask_inplace_kernel = (
            llguidance.torch.apply_token_bitmask_inplace_kernel._torchdynamo_orig_callable
        )
except ImportError:
    llguidance = None  # type: ignore[assignment]

PROTOCOL_MODE_TIDS: dict[str, int] = {
    "Probe": 5140,
    "Read": 18104,
    "Ask": 5574,
    "Write": 37272,
    "Result": 27604,
    "Abstain": 13732,
}


# ============================================================================
# Deployment Identity & Cartridge Tensor Validation (folded from portia_model_identity)
# ============================================================================


class CartridgeModelMismatch(ValueError):
    pass


class UnknownModelFamily(ValueError):
    pass


class MissingCartridge(FileNotFoundError):
    pass


class ModelFamilyConflict(ValueError):
    pass


def normalize_model_identifier(raw: str) -> str:
    value = raw.strip()
    short = value.lower()
    for prefix in ("qwen3.5-", "qwen-"):
        if short.startswith(prefix):
            short = short[len(prefix) :]
            break
    base = short.endswith("-base")
    size = short[:-5] if base else short
    names = {"0.8b": "0.8B", "2b": "2B", "4b": "4B", "9b": "9B", "27b": "27B"}
    return f"Qwen/Qwen3.5-{names[size]}{'-Base' if base else ''}" if size in names else value


def _canonical_repo(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(
        p and p not in (".", "..") and all(c.isascii() and (c.isalnum() or c in "-_.") for c in p)
        for p in parts
    )


def _repo_from_hf_cache_parts(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    parts = Path(raw.strip()).parts
    for index, part in enumerate(parts):
        if part.startswith("models--") and (
            index + 1 >= len(parts) or parts[index + 1] in ("snapshots", "refs", "blobs")
        ):
            split = part[len("models--") :].split("--", 1)
            if len(split) == 2:
                candidate = normalize_model_identifier("/".join(split))
                if _canonical_repo(candidate):
                    return candidate
    return None


def resolve_model_family(model: str, explicit: str | None = None) -> str | None:
    declared = None
    if explicit is not None:
        declared = normalize_model_identifier(explicit)
        if not _canonical_repo(declared):
            raise ValueError(f"InvalidModelFamily: expected canonical org/repo, got {explicit!r}")
    model = normalize_model_identifier(model)
    path = Path(model)
    identities: list[str] = []
    if not path.exists() and _canonical_repo(model):
        identities.append(model)
    cache_repo = _repo_from_hf_cache_parts(model)
    if cache_repo is not None:
        identities.append(cache_repo)
    for filename, key in (("portia_model_provenance.json", "model_family"), ("config.json", "_name_or_path")):
        source = path / filename
        if source.is_file():
            raw_family = json.loads(source.read_text(encoding="utf-8")).get(key)
            if isinstance(raw_family, str) and raw_family.strip():
                norm_family = normalize_model_identifier(raw_family)
                if _canonical_repo(norm_family):
                    identities.append(norm_family)
                else:
                    snap_family = _repo_from_hf_cache_parts(raw_family)
                    if snap_family is not None:
                        identities.append(snap_family)
    if len(set(identities)) > 1:
        raise ModelFamilyConflict(f"ModelFamilyConflict: contradictory checkpoint provenance: {identities!r}")
    inferred = identities[0] if identities else None
    if declared is not None and inferred is not None and declared != inferred:
        raise ModelFamilyConflict(
            f"ModelFamilyConflict: declared {declared!r} contradicts known model {inferred!r}; "
            "--model-family may only identify an otherwise unknown model"
        )
    return inferred or declared


def cartridge_family(raw: dict[str, Any]) -> str | None:
    header = raw.get("header") or {}
    embedded = (raw.get("artifact_bundle") or {}).get("header") or {}
    identities = [
        v for v in (header.get("model_family"), raw.get("model_id"), embedded.get("model_family")) if v is not None
    ]
    if any(not isinstance(v, str) or not v.strip() for v in identities) or len(set(identities)) > 1:
        raise ValueError("Conflicting or empty cartridge identities: header.model_family, model_id, artifact_bundle.header.model_family")
    return identities[0] if identities else None


def validate_model_identity(raw: dict[str, Any], family: str | None, path: str, override: bool = False) -> str | None:
    artifact_family = cartridge_family(raw)  # Schema conflicts are never overridable.
    if not override:
        if family is None:
            raise UnknownModelFamily(
                "UnknownModelFamily: provide --model-family <org/repo> or --allow-cartridge-model-mismatch for this local directory"
            )
        if family != artifact_family:
            raise CartridgeModelMismatch(
                f"CartridgeModelMismatch: model={family!r}, cartridge={artifact_family!r}, path={path!r}; "
                "use --allow-cartridge-model-mismatch only for an intentional experiment"
            )
    return artifact_family


_DEFAULT_FAMILY_CARTRIDGES: dict[str, str] = {
    "Qwen/Qwen3.5-0.8B": "portia_v3_unified_cartridge.json",
    "Qwen/Qwen3.5-0.8B-Base": "portia_v3_unified_cartridge_08b_base.json",
    "Qwen/Qwen3.5-2B-Base": "portia_v3_unified_cartridge_2b_base.json",
    "Qwen/Qwen3.5-4B-Base": "portia_v3_unified_cartridge_4b_base.json",
    "Qwen/Qwen3.5-9B-Base": "portia_v3_unified_cartridge_9b_base.json",
}


def _cartridge_matches_family(path: Path, expected_family: str) -> bool:
    if not path.is_file():
        return False
    try:
        fam = cartridge_family(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return False
    return fam in (expected_family, None)


def select_cartridge(explicit: str | None, family: str | None, root: Path) -> Path | None:
    selected = explicit
    if selected is None:
        selected = os.environ.get("PORTIA_CARTRIDGE_JSON", "").strip() or None
    if selected is not None and selected.strip().lower() == "none":
        return None
    if selected is None and family in _DEFAULT_FAMILY_CARTRIDGES:
        cand = root / "artifacts" / _DEFAULT_FAMILY_CARTRIDGES[family]
        if _cartridge_matches_family(cand, family):
            selected = str(cand)
        elif family == "Qwen/Qwen3.5-0.8B-Base":
            alt = root / "artifacts" / "portia_v3_unified_cartridge.json"
            if _cartridge_matches_family(alt, family):
                selected = str(alt)
    if selected is None or not Path(selected).is_file():
        raise MissingCartridge(
            f"MissingCartridge: model={family!r}, path={selected!r}; "
            "pass --cartridge <path> or --cartridge none (direct worker: --cartridge-json)"
        )
    return Path(selected)


_SIMPLE_BANKS = {
    "BASE": "Base",
    "ORIENTTRUNK": "OrientTrunk",
    "ORIENT_TRUNK": "OrientTrunk",
    "UNIFIEDSTATIC": "UnifiedStatic",
    "UNIFIED_STATIC": "UnifiedStatic",
    "PERMANENTSTATIC": "UnifiedStatic",
    "PERMANENT_STATIC": "UnifiedStatic",
}
_CANONICAL_MODES = ("PROBE", "READ", "ASK", "WRITE", "RESULT", "ABSTAIN")
_PREFILL_BANKS = {
    "SCHEMA": "Prefill:Schema",
    "AUTHORITATIVESCHEMA": "Prefill:Schema",
    "USER": "Prefill:User",
    "PROMPT": "Prefill:User",
    "TOOLOBS": "Prefill:ToolObs",
    "TOOL_OBS": "Prefill:ToolObs",
    "OBSERVATION": "Prefill:ToolObs",
    "TOOLRESPONSE": "Prefill:ToolObs",
}


def _canonicalize_mode_token(raw_mode: str, prefix: str = "ModePayload") -> str | None:
    mode = raw_mode.strip().upper()
    if mode == "REPORT":
        mode = "RESULT"
    return f"{prefix}:{mode.capitalize()}" if mode in _CANONICAL_MODES else None


def canonical_manifest_bank(name: str) -> str:
    """Match Rust MountPolicy::from_wire_key for serialized weight bank names."""
    text = name.strip()
    if text.upper() in _SIMPLE_BANKS:
        return _SIMPLE_BANKS[text.upper()]
    for prefix in ("ModePayload", "ModePlan"):
        if text.startswith(f"{prefix}:"):
            resolved = _canonicalize_mode_token(text.split(":", 1)[1], prefix=prefix)
            if resolved is not None:
                return resolved
    if text.startswith("Prefill:"):
        seg = text.split(":", 1)[1].strip().upper()
        if seg in _PREFILL_BANKS:
            return _PREFILL_BANKS[seg]
    raise ValueError(f"Unsupported cartridge bank {name!r}")


def normalize_bank_key(bank: Any) -> str:
    if isinstance(bank, dict):
        for prefix in ("ModePayload", "ModePlan"):
            if prefix in bank:
                raw_mode = str(bank[prefix])
                return _canonicalize_mode_token(raw_mode, prefix=prefix) or f"{prefix}:{raw_mode.strip().capitalize()}"
        if "PrefillSegment" in bank:
            seg = str(bank["PrefillSegment"]).strip().upper()
            return _PREFILL_BANKS.get(seg, f"Prefill:{seg.capitalize()}")
    s = str(bank).strip()
    if s.upper() in _SIMPLE_BANKS:
        return _SIMPLE_BANKS[s.upper()]
    if s.startswith("ModePlan:"):
        resolved = _canonicalize_mode_token(s.split(":", 1)[1], prefix="ModePlan")
        return resolved if resolved is not None else s
    if s.startswith("Prefill:"):
        seg = s.split(":", 1)[1].strip().upper()
        return _PREFILL_BANKS.get(seg, s)
    candidate = s.split(":", 1)[1] if s.startswith("ModePayload:") else s
    resolved = _canonicalize_mode_token(candidate, prefix="ModePayload")
    return resolved if resolved is not None else s


def validate_bundle_tensors(raw: dict[str, Any], *, d_model: int, base_m: int, num_layers: int) -> None:
    """Validate every mounted entry before touching model storage."""
    cartridge_family(raw)
    capacity = raw.get("widened_slots")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0 or capacity > 128:
        raise ValueError("Invalid widened_slots: expected 1..128")
    if raw.get("base_intermediate_size", base_m) != base_m:
        raise ValueError("Cartridge base_intermediate_size does not match model")
    for header in (raw.get("header") or {}, (raw.get("artifact_bundle") or {}).get("header") or {}):
        for key in ("hidden_dim", "d_model"):
            if key in header and header[key] != d_model:
                raise ValueError(f"Cartridge {key} does not match model")
        if "base_intermediate_size" in header and header["base_intermediate_size"] != base_m:
            raise ValueError("Cartridge header base_intermediate_size mismatch")
    excluded = {
        "header",
        "banks",
        "permanent_static",
        "mode_grammars",
        "latent_interrupt_traps",
        "artifact_bundle",
        "bisimulation_receipt",
    }
    bank_map = raw.get("banks", {})
    permanent = raw.get("permanent_static", {})
    if not isinstance(bank_map, dict) or not isinstance(permanent, dict):
        raise ValueError("banks and permanent_static must be layer maps")
    groups = [
        (canonical_manifest_bank(name), layers)
        for name, layers in raw.items()
        if name not in excluded and isinstance(layers, dict) and any(str(k).isdigit() for k in layers)
    ]
    groups.extend((canonical_manifest_bank(name), layers) for name, layers in bank_map.items())
    groups.append(("__permanent_static__", permanent))
    occupied: dict[tuple[str, int], set[int]] = {}
    normalized_entries: dict[tuple[str, int], list[dict]] = {}
    router_layer: int | None = None
    entry_count = 0
    for bank, layers in groups:
        if not isinstance(layers, dict):
            raise ValueError(f"Cartridge bank {bank!r} must be a layer map")
        for layer, entries in layers.items():
            if not str(layer).isdigit() or not 0 <= int(layer) < num_layers:
                raise ValueError(f"Invalid layer {layer!r} for {num_layers} model layers")
            slots = occupied.setdefault((bank, int(layer)), set())
            for entry in entries if isinstance(entries, list) else [entries]:
                entry_count += 1
                if not isinstance(entry, dict):
                    raise ValueError("Cartridge slot entries must be objects")
                lo, hi = entry.get("slot_lo"), entry.get("slot_hi")
                if (
                    not isinstance(lo, int)
                    or isinstance(lo, bool)
                    or not isinstance(hi, int)
                    or isinstance(hi, bool)
                    or not base_m <= lo < hi <= base_m + capacity
                ):
                    raise ValueError("Cartridge slot outside declared widened bounds")
                if slots.intersection(range(lo, hi)):
                    raise ValueError("Overlapping slots in cartridge bank")
                slots.update(range(lo, hi))
                normalized_entries.setdefault((bank, int(layer)), []).append(entry)
                if bank == "OrientTrunk":
                    router_layer = max(router_layer if router_layer is not None else int(layer), int(layer))
                for key, rows, cols in (("w_gate", hi - lo, d_model), ("w_up", hi - lo, d_model), ("w_down", d_model, hi - lo)):
                    values = entry.get(key)
                    if (
                        not isinstance(values, list)
                        or len(values) != rows
                        or any(not isinstance(row, list) or len(row) != cols for row in values)
                    ):
                        raise ValueError(f"Invalid dimensions for {key}")
                    if any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                        for row in values
                        for v in row
                    ):
                        raise ValueError(f"Nonfinite or invalid weight in {key}")
    if not entry_count:
        raise ValueError("Empty executable cartridge")
    for (bank, layer), slots in occupied.items():
        if bank != "__permanent_static__" and slots.intersection(occupied.get(("__permanent_static__", layer), set())):
            raise ValueError("Permanent static and dynamic slots overlap")
    effective_router = router_layer if router_layer is not None else max(0, num_layers - 2)
    for (bank, layer), entries in normalized_entries.items():
        if bank in ("UnifiedStatic", "__permanent_static__") or layer == effective_router:
            continue
        static_entries = normalized_entries.get(("UnifiedStatic", layer), [])
        static_slots = occupied.get(("UnifiedStatic", layer), set())
        for entry in entries:
            if entry not in static_entries and static_slots.intersection(range(entry["slot_lo"], entry["slot_hi"])):
                raise ValueError("Legacy UnifiedStatic and dynamic slots overlap on a non-router layer")


def _is_unconstrained_grammar_spec(grammar_spec: Any) -> bool:
    if grammar_spec is None:
        return True
    if isinstance(grammar_spec, str):
        return grammar_spec.strip().lower() in ("", "unconstrained", "none")
    if isinstance(grammar_spec, dict):
        if "Unconstrained" in grammar_spec and len(grammar_spec) == 1:
            return True
        if "kind" in grammar_spec:
            return str(grammar_spec.get("kind", "")).strip().lower() in ("", "unconstrained", "none")
    return False


def _has_valid_snapshot_files(snap_dir: Path) -> bool:
    return snap_dir.is_dir() and (snap_dir / "config.json").is_file() and (
        any(snap_dir.glob("*.safetensors"))
        or any(snap_dir.glob("*.bin"))
        or (snap_dir / "model.safetensors.index.json").is_file()
    )


def _resolve_hf_repo_cache(repo_dir: Path) -> str | None:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    ref_main = snapshots_dir.parent / "refs" / "main"
    if ref_main.is_file():
        rev = ref_main.read_text(encoding="utf-8").strip()
        if rev and _has_valid_snapshot_files(snapshots_dir / rev):
            return str(snapshots_dir / rev)
    snaps = sorted(s for s in snapshots_dir.iterdir() if _has_valid_snapshot_files(s))
    return str(snaps[-1]) if snaps else None


class LiveQwen35Engine:
    """Pure PyTorch + `llguidance` Qwen3.5 tensor & intra-layer SwiGLU weight-bank engine."""

    def __init__(
        self,
        model_path: str,
        cartridge_json: str | None = None,
        *,
        model_family: str | None = None,
        allow_cartridge_model_mismatch: bool = False,
    ) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        import torch
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.utils.hub import cached_file

        self.torch = torch
        requested_model = model_path
        model_path = normalize_model_identifier(model_path)
        resolved = self._resolve_model_path(model_path)
        family = resolve_model_family(model_path, model_family)
        family = resolve_model_family(resolved, family)
        cpath = select_cartridge(cartridge_json, family, Path(__file__).resolve().parent.parent)
        cartridge_bytes = cpath.read_bytes() if cpath is not None else None
        raw_bundle = json.loads(cartridge_bytes) if cartridge_bytes is not None else None
        artifact_family = (
            validate_model_identity(raw_bundle, family, str(cpath), allow_cartridge_model_mismatch)
            if raw_bundle is not None
            else None
        )
        env_dev = os.environ.get("PORTIA_DEVICE", "").strip().lower()
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        use_cuda = torch.cuda.is_available() and env_dev not in ("cpu", "mps")
        use_mps = (env_dev == "mps" or (not use_cuda and env_dev != "cpu")) and has_mps
        self.device = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
        env_dtype = os.environ.get("PORTIA_DTYPE", "").strip().lower()
        if env_dtype in ("bf16", "bfloat16"):
            load_dtype = torch.bfloat16
        elif env_dtype in ("fp16", "float16"):
            load_dtype = torch.float16
        elif env_dtype in ("fp32", "float32"):
            load_dtype = torch.float32
        elif use_cuda:
            load_dtype = torch.bfloat16
        elif use_mps:
            try:
                _p = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
                _ = (_p @ _p).cpu()
                load_dtype = torch.bfloat16
            except Exception:
                load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        for local_only in (True, False):
            target_id = resolved if local_only else model_path
            try:
                config_file = cached_file(target_id, "config.json", local_files_only=local_only)
                config_directory = Path(config_file).parent
                loaded_config = AutoConfig.from_pretrained(config_directory, local_files_only=True)
                loaded_revision = (
                    config_directory.name
                    if config_directory.parent.name == "snapshots"
                    else getattr(loaded_config, "_commit_hash", None)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(target_id, revision=loaded_revision, local_files_only=local_only)
                self.model = AutoModelForCausalLM.from_pretrained(
                    target_id,
                    config=loaded_config,
                    revision=loaded_revision,
                    local_files_only=local_only,
                    dtype=load_dtype,
                    low_cpu_mem_usage=True,
                )
                resolved = str(config_directory)
                break
            except Exception:
                if not local_only:
                    raise
        self.model.to(self.device).eval()

        first_mlp = self.model.model.layers[0].mlp
        self.base_intermediate_size, self.d_model = int(first_mlp.gate_proj.weight.shape[0]), int(first_mlp.gate_proj.weight.shape[1])
        self.reserved_slots, self.extended_intermediate_size = 128, self.base_intermediate_size
        self.modified_layers, self.cartridges, self.active_bank = set(), {}, "Uninitialized"
        self.permanent_static_layers: dict[str, list[Any]] = {}
        self.manifest_mode_grammars: dict[str, Any] = {}
        self.latent_interrupt_traps: list[dict[str, Any]] = []
        self.active_interrupt_trap: dict[str, Any] | None = None
        self._active_grammar_span_key: tuple[str, str] | None = None
        self.past_key_values, self.last_logits, self.session_token_ids, self._last_step_token_ids = None, None, [], []
        self._pending_utf8_tids: list[int] = []
        self.ll_tokenizer: Any = llguidance.hf.from_tokenizer(self.tokenizer) if llguidance is not None else None
        self.active_ll_matcher: Any = None

        if raw_bundle is not None:
            validate_bundle_tensors(
                raw_bundle,
                d_model=self.d_model,
                base_m=self.base_intermediate_size,
                num_layers=len(self.model.model.layers),
            )
            self._load_and_widen_cartridge_bundle(raw_bundle)
        self.runtime_provenance = {
            "requested_model": requested_model,
            "resolved_model_family": family,
            "cartridge_model_family": artifact_family,
            "cartridge_sha256": hashlib.sha256(cartridge_bytes).hexdigest() if cartridge_bytes is not None else None,
            "cartridge_path": str(cpath.resolve()) if cpath else None,
            "model_mismatch_override": bool(allow_cartridge_model_mismatch),
            "cartridge_mounted": bool(self.cartridges or self.permanent_static_layers),
            "actual_model_loaded": True,
            "checkpoint_revision": loaded_revision,
            "resolved_model_path": str(resolved),
            "device": str(self.device),
            "dtype": str(load_dtype),
        }

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        path = Path(model_path)
        if path.exists():
            if (path / "snapshots").is_dir():
                resolved_repo = _resolve_hf_repo_cache(path)
                if resolved_repo is not None:
                    return resolved_repo
            elif path.name == "snapshots" and path.is_dir():
                resolved_repo = _resolve_hf_repo_cache(path.parent)
                if resolved_repo is not None:
                    return resolved_repo
            return str(path)
        hub_root = Path(os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache" / "huggingface" / "hub"))
        repo_dir = hub_root / ("models--" + model_path.strip().replace("/", "--"))
        resolved = _resolve_hf_repo_cache(repo_dir)
        return resolved if resolved is not None else model_path

    def _ensure_ll_tokenizer(self) -> Any:
        if getattr(self, "ll_tokenizer", None) is None:
            if llguidance is None:
                raise RuntimeError("llguidance is required to mount grammar_spec")
            self.ll_tokenizer = (
                self.tokenizer.to_ll_tokenizer()
                if hasattr(self.tokenizer, "to_ll_tokenizer")
                else llguidance.hf.from_tokenizer(self.tokenizer)
            )
        return self.ll_tokenizer

    def mount_grammar_cartridge(self, grammar_spec: dict[str, Any] | str | None) -> None:
        if grammar_spec is None or (isinstance(grammar_spec, str) and grammar_spec.strip().lower() in ("", "unconstrained", "none")):
            self.active_ll_matcher = None
            return
        if isinstance(grammar_spec, str):
            lark_or_schema = grammar_spec
        elif isinstance(grammar_spec, dict):
            if "LarkGrammar" in grammar_spec:
                kind, spec = "lark_grammar", grammar_spec["LarkGrammar"]
            elif "JsonSchema" in grammar_spec:
                kind, spec = "json_schema", grammar_spec["JsonSchema"]
            elif "LiteralSet" in grammar_spec:
                kind, spec = "literal_set", grammar_spec["LiteralSet"]
            elif "Unconstrained" in grammar_spec:
                self.active_ll_matcher = None
                return
            else:
                kind, spec = str(grammar_spec.get("kind", "")).strip().lower(), grammar_spec.get("spec")
            if kind in ("", "unconstrained", "none"):
                self.active_ll_matcher = None
                return
            if kind == "lark_grammar":
                lark_or_schema = str(spec)
            elif kind == "json_schema":
                if llguidance is None:
                    raise RuntimeError("llguidance is required to mount json_schema")
                lark_or_schema = llguidance.LLMatcher.grammar_from_json_schema(
                    spec if isinstance(spec, str) else json.dumps(spec)
                )
            elif kind == "literal_set":
                literals = [str(s) for s in (spec or []) if str(s)]
                if not literals:
                    self.active_ll_matcher = None
                    return
                branches: list[str] = []
                tok_obj = getattr(self, "tokenizer", None)
                for lit in literals:
                    tids = (
                        [int(t) for t in tok_obj.encode(lit, add_special_tokens=False)]
                        if tok_obj is not None and hasattr(tok_obj, "encode")
                        else []
                    )
                    if tids:
                        branches.append(" ".join(f"<[{t}]>" for t in tids))
                    else:
                        esc = lit.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
                        branches.append(f'"{esc}"')
                lark_or_schema = "start: " + " | ".join(branches)
            else:
                raise ValueError(f"Unsupported grammar_spec kind: {kind!r}")
        else:
            raise TypeError(f"Invalid grammar_spec type: {type(grammar_spec)!r}")
        matcher = llguidance.LLMatcher(self._ensure_ll_tokenizer(), lark_or_schema, log_level=0)
        if matcher.is_error():
            self.active_ll_matcher = None
            raise ValueError(f"Invalid llguidance grammar: {matcher.get_error()}")
        self.active_ll_matcher = matcher

    @staticmethod
    def _iter_bundle_layers(raw_bundle: dict[str, Any]) -> list[tuple[str, str, Any]]:
        out: list[tuple[str, str, Any]] = []
        excluded = {
            "header",
            "banks",
            "permanent_static",
            "mode_grammars",
            "latent_interrupt_traps",
            "artifact_bundle",
            "bisimulation_receipt",
        }
        for src in [raw_bundle] + ([raw_bundle["banks"]] if isinstance(raw_bundle.get("banks"), dict) else []):
            for bkey, bspec in src.items():
                if bkey not in excluded and isinstance(bspec, dict):
                    for lkey, entries in bspec.items():
                        if str(lkey).isdigit():
                            out.append((bkey, str(lkey), entries))
        return out

    def _validate_slot_entry_list(self, entries: Any) -> bool:
        entry_list = entries if isinstance(entries, list) else [entries]
        if not entry_list:
            return False
        for entry in entry_list:
            if not isinstance(entry, dict) or "slot_lo" not in entry or "slot_hi" not in entry:
                return False
            slot_lo, slot_hi = int(entry["slot_lo"]), int(entry["slot_hi"])
            span = slot_hi - slot_lo
            if slot_lo < self.base_intermediate_size or span <= 0 or slot_hi > self.base_intermediate_size + 128:
                return False
            for pk in ("w_gate", "w_up"):
                mat = entry.get(pk)
                if not isinstance(mat, list) or len(mat) != span or any(not isinstance(r, list) or len(r) != self.d_model for r in mat):
                    return False
            w_down = entry.get("w_down")
            if not isinstance(w_down, list) or len(w_down) != self.d_model or any(not isinstance(r, list) or len(r) != span for r in w_down):
                return False
        return True

    def _bundle_matches_model_dim(self, raw_bundle: dict[str, Any]) -> bool:
        if not isinstance(raw_bundle, dict):
            return False
        hdr = raw_bundle.get("header") if isinstance(raw_bundle.get("header"), dict) else {}
        if ("hidden_dim" in hdr and int(hdr["hidden_dim"]) != self.d_model) or (
            "d_model" in hdr and int(hdr["d_model"]) != self.d_model
        ):
            return False
        if any("base_intermediate_size" in s and int(s["base_intermediate_size"]) != self.base_intermediate_size for s in (hdr, raw_bundle)):
            return False
        for _bkey, _lkey, entries in self._iter_bundle_layers(raw_bundle):
            if not self._validate_slot_entry_list(entries):
                return False
        perm_static = raw_bundle.get("permanent_static")
        if isinstance(perm_static, dict):
            for lkey, entries in perm_static.items():
                if not str(lkey).isdigit() or not self._validate_slot_entry_list(entries):
                    return False
        return True

    def _widen_mlp_layers(self, base_m: int, added_slots: int) -> None:
        torch, nn = self.torch, self.torch.nn
        cur_m = int(self.model.model.layers[0].mlp.gate_proj.weight.shape[0])
        if cur_m >= base_m + added_slots:
            self.base_intermediate_size, self.reserved_slots, self.extended_intermediate_size = base_m, added_slots, cur_m
            return
        m_ext = base_m + added_slots
        with torch.no_grad():
            for block in self.model.model.layers:
                blk_mlp = block.mlp
                gw, uw, dw = blk_mlp.gate_proj.weight, blk_mlp.up_proj.weight, blk_mlp.down_proj.weight
                d_model, dtype, device = int(gw.shape[1]), gw.dtype, gw.device
                new_gate = nn.Linear(d_model, m_ext, bias=False, device=device, dtype=dtype)
                new_up = nn.Linear(d_model, m_ext, bias=False, device=device, dtype=dtype)
                new_down = nn.Linear(m_ext, d_model, bias=False, device=device, dtype=dtype)
                new_gate.weight.zero_()
                new_up.weight.zero_()
                new_down.weight.zero_()
                new_gate.weight[:base_m].copy_(gw.detach()[:base_m])
                new_up.weight[:base_m].copy_(uw.detach()[:base_m])
                new_down.weight[:, :base_m].copy_(dw.detach()[:, :base_m])
                if hasattr(blk_mlp, "intermediate_size"):
                    blk_mlp.intermediate_size = m_ext
                blk_mlp.gate_proj, blk_mlp.up_proj, blk_mlp.down_proj = new_gate, new_up, new_down
        self.base_intermediate_size, self.reserved_slots, self.extended_intermediate_size = base_m, m_ext - base_m, m_ext

    def _load_and_widen_cartridge_bundle(self, raw_bundle: dict[str, Any]) -> None:
        if not self._bundle_matches_model_dim(raw_bundle):
            raise ValueError("Cartridge bundle failed slot bounds or dimension validation")
        if not hasattr(self, "permanent_static_layers"):
            self.permanent_static_layers = {}
        if not hasattr(self, "manifest_mode_grammars"):
            self.manifest_mode_grammars = {}
        if not hasattr(self, "latent_interrupt_traps"):
            self.latent_interrupt_traps = []
        max_slot_hi = self.base_intermediate_size
        for bkey, lkey, slot_data in self._iter_bundle_layers(raw_bundle):
            self.modified_layers.add(int(lkey))
            entries = list(slot_data if isinstance(slot_data, list) else [slot_data])
            self.cartridges.setdefault(normalize_bank_key(bkey), {}).setdefault(str(int(lkey)), []).extend(entries)
            max_slot_hi = max([max_slot_hi] + [int(e["slot_hi"]) for e in entries if isinstance(e, dict) and "slot_hi" in e])
        perm_static = raw_bundle.get("permanent_static")
        if isinstance(perm_static, dict):
            for lkey, slot_data in perm_static.items():
                if str(lkey).isdigit():
                    self.modified_layers.add(int(lkey))
                    entries = list(slot_data if isinstance(slot_data, list) else [slot_data])
                    self.permanent_static_layers.setdefault(str(int(lkey)), []).extend(entries)
                    max_slot_hi = max([max_slot_hi] + [int(e["slot_hi"]) for e in entries if isinstance(e, dict) and "slot_hi" in e])
        raw_grammars = raw_bundle.get("mode_grammars")
        if isinstance(raw_grammars, dict):
            for mkey, gspec in raw_grammars.items():
                norm_bkey = normalize_bank_key(mkey)
                self.manifest_mode_grammars[norm_bkey] = gspec
                if norm_bkey.startswith("ModePayload:"):
                    mode_short = norm_bkey.split(":", 1)[1]
                    self.manifest_mode_grammars[mode_short] = gspec
                else:
                    self.manifest_mode_grammars[str(mkey).strip()] = gspec
        raw_traps = raw_bundle.get("latent_interrupt_traps")
        if isinstance(raw_traps, list):
            self.latent_interrupt_traps = [dict(t) for t in raw_traps if isinstance(t, dict)]
        if max_slot_hi > self.base_intermediate_size:
            req_widened = int(raw_bundle.get("widened_slots", 128))
            self._widen_mlp_layers(self.base_intermediate_size, max(req_widened, max_slot_hi - self.base_intermediate_size))
        self._prematerialize_bundle_tensors()

    def _prematerialize_bundle_tensors(self) -> None:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            return
        for l_str, entries in getattr(self, "permanent_static_layers", {}).items():
            if str(l_str).isdigit() and int(l_str) < len(layers):
                self._materialize_entry_list(int(l_str), entries)
        for bank_layers in self.cartridges.values():
            if isinstance(bank_layers, dict):
                for l_str, entries in bank_layers.items():
                    if str(l_str).isdigit() and int(l_str) < len(layers):
                        self._materialize_entry_list(int(l_str), entries)

    def _materialize_entry_list(self, layer_idx: int, slot_entries: Any) -> None:
        mlp = self.model.model.layers[int(layer_idx)].mlp
        dtype = mlp.gate_proj.weight.dtype
        for s in (slot_entries if isinstance(slot_entries, list) else [slot_entries]):
            if isinstance(s, dict) and all(k in s for k in ("w_gate", "w_up", "w_down")):
                for k in ("w_gate", "w_up", "w_down"):
                    v = s[k]
                    if not isinstance(v, self.torch.Tensor) or v.device != self.device or v.dtype != dtype:
                        s[k] = self.torch.as_tensor(v, dtype=dtype, device=self.device).contiguous()

    def resolve_manifest_grammar(self, bank: Any) -> Any | None:
        grammars = getattr(self, "manifest_mode_grammars", None) or {}
        if not grammars:
            return None
        bank_key = normalize_bank_key(bank)
        candidates = [bank_key]
        if bank_key.startswith("ModePayload:"):
            mode_short = bank_key.split(":", 1)[1]
            candidates.extend([mode_short, mode_short.upper(), mode_short.lower()])
        for k in candidates:
            if k in grammars and not _is_unconstrained_grammar_spec(grammars[k]):
                return grammars[k]
        return None

    def evaluate_latent_interrupt_traps(self, pre_mlp_vec: Any, layer_idx: int | None = None) -> dict[str, Any] | None:
        traps = getattr(self, "latent_interrupt_traps", None) or []
        if not traps and getattr(self, "active_interrupt_trap", None) is None:
            return None
        vec = [
            float(v)
            for v in (
                pre_mlp_vec.detach().float().cpu().tolist()
                if hasattr(pre_mlp_vec, "detach")
                else pre_mlp_vec
            )
        ]
        active = getattr(self, "active_interrupt_trap", None)
        if active is not None:
            if layer_idx is None or int(active.get("layer_idx", -1)) == int(layer_idx):
                w_gate = active.get("w_gate_f32") or []
                if len(w_gate) == len(vec):
                    dot_val = sum(float(w) * float(x) for w, x in zip(w_gate, vec))
                    if dot_val <= float(active["release_ceiling"]):
                        trap_name = str(active.get("trap_name", ""))
                        self.active_interrupt_trap = None
                        self.mount_grammar_cartridge(None)
                        self._mount_bank("Base")
                        return {"event": "released", "trap_name": trap_name, "dot": dot_val, "bank": self.active_bank}
                    return {"event": "held", "trap_name": str(active.get("trap_name", "")), "dot": dot_val, "bank": self.active_bank}
            return None
        for trap in traps:
            if layer_idx is not None and int(trap.get("layer_idx", -1)) != int(layer_idx):
                continue
            w_gate = trap.get("w_gate_f32") or []
            if len(w_gate) != len(vec):
                continue
            dot_val = sum(float(w) * float(x) for w, x in zip(w_gate, vec))
            if dot_val >= float(trap["fire_threshold"]):
                target = trap.get("target_cartridge", "Base")
                if isinstance(target, dict):
                    target_bank = target.get("weight_bank", target.get("bank", "Base"))
                    target_grammar = target.get("grammar")
                else:
                    target_bank, target_grammar = target, None
                self.active_interrupt_trap = trap
                self._mount_bank(target_bank)
                if target_grammar is not None:
                    self.mount_grammar_cartridge(target_grammar)
                return {"event": "fired", "trap_name": str(trap.get("trap_name", "")), "dot": dot_val, "bank": self.active_bank}
        return None

    def _pending_utf8_buffer(self) -> list[int]:
        buf = getattr(self, "_pending_utf8_tids", None)
        if buf is None:
            buf = []
            self._pending_utf8_tids = buf
        return buf

    def _decode_token_ids(self, tids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode(tids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        except TypeError:
            return str(self.tokenizer.decode(tids, skip_special_tokens=False))

    def _decode_incremental_utf8_step(self, next_tid: int) -> str:
        pending = self._pending_utf8_buffer()
        pending.append(int(next_tid))
        candidate = self._decode_token_ids(pending)
        if candidate.endswith("\ufffd") and len(pending) < 4:
            return ""
        pending.clear()
        return candidate

    def reset_task_execution_state(self) -> None:
        self._last_step_token_ids.clear()
        self._pending_utf8_buffer().clear()
        self.active_ll_matcher = None
        self.active_interrupt_trap = None
        self._active_grammar_span_key = None

    def reset_session(self) -> None:
        self.past_key_values, self.last_logits = None, None
        self.session_token_ids.clear()
        self._pending_utf8_buffer().clear()
        self.reset_task_execution_state()
        self._mount_bank("Base")
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        self.past_key_values, self.last_logits, self.active_ll_matcher = None, None, None
        try:
            self.model.to("cpu")
        except Exception:
            pass
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _zero_reserved_slots(self) -> None:
        lo, hi = self.base_intermediate_size, self.extended_intermediate_size
        if hi > lo:
            with self.torch.no_grad():
                layers = self.model.model.layers
                for layer_idx in (self.modified_layers or range(len(layers))):
                    mlp = layers[int(layer_idx)].mlp
                    if int(mlp.gate_proj.weight.shape[0]) >= hi:
                        mlp.gate_proj.weight[lo:hi].zero_()
                        mlp.up_proj.weight[lo:hi].zero_()
                        mlp.down_proj.weight[:, lo:hi].zero_()

    def _copy_slot_entries(self, layer_idx: int, slot_entries: Any) -> None:
        mlp = self.model.model.layers[int(layer_idx)].mlp
        dtype = mlp.gate_proj.weight.dtype
        for s in (slot_entries if isinstance(slot_entries, list) else [slot_entries]):
            lo, hi = int(s["slot_lo"]), int(s["slot_hi"])
            wg, wu, wd = s["w_gate"], s["w_up"], s["w_down"]
            if not isinstance(wg, self.torch.Tensor) or wg.device != self.device or wg.dtype != dtype:
                wg = self.torch.as_tensor(wg, dtype=dtype, device=self.device).contiguous()
                wu = self.torch.as_tensor(wu, dtype=dtype, device=self.device).contiguous()
                wd = self.torch.as_tensor(wd, dtype=dtype, device=self.device).contiguous()
                if isinstance(s, dict):
                    s["w_gate"], s["w_up"], s["w_down"] = wg, wu, wd
            mlp.gate_proj.weight[lo:hi].copy_(wg)
            mlp.up_proj.weight[lo:hi].copy_(wu)
            mlp.down_proj.weight[:, lo:hi].copy_(wd)

    active_bank_name = property(lambda self: self.active_bank, lambda self, v: setattr(self, "active_bank", v))

    def _resolve_router_layer(self) -> int:
        orient_layers = [
            int(k)
            for k, v in self.cartridges.get("OrientTrunk", {}).items()
            if str(k).isdigit() and v
        ]
        if orient_layers:
            return max(orient_layers)
        return max(0, len(self.model.model.layers) - 2)

    def _mount_bank(self, bank: Any) -> None:
        bank_key = normalize_bank_key(bank)
        if bank_key == self.active_bank:
            return
        self.active_bank = bank_key
        self._zero_reserved_slots()
        with self.torch.no_grad():
            router_layer = self._resolve_router_layer()
            for l_str, entries in getattr(self, "permanent_static_layers", {}).items():
                self._copy_slot_entries(int(l_str), entries)
            for l_str, entries in self.cartridges.get("UnifiedStatic", {}).items():
                entry_list = entries if isinstance(entries, list) else [entries]
                if int(l_str) != router_layer:
                    self._copy_slot_entries(int(l_str), entry_list)
                else:
                    dyn_intervals: list[tuple[int, int]] = []
                    for dyn_key, dyn_layers in self.cartridges.items():
                        if dyn_key not in ("UnifiedStatic", "Base") and l_str in dyn_layers:
                            dyn_entries = dyn_layers[l_str] if isinstance(dyn_layers[l_str], list) else [dyn_layers[l_str]]
                            for de in dyn_entries:
                                dyn_intervals.append((int(de["slot_lo"]), int(de["slot_hi"])))
                    non_overlapping = [
                        e
                        for e in entry_list
                        if not any(int(e["slot_lo"]) < d_hi and d_lo < int(e["slot_hi"]) for d_lo, d_hi in dyn_intervals)
                    ]
                    if non_overlapping:
                        self._copy_slot_entries(int(l_str), non_overlapping)
            if bank_key != "UnifiedStatic":
                for l_str, entries in self.cartridges.get(bank_key, {}).items():
                    self._copy_slot_entries(int(l_str), entries)

    def _capture_layer_pre_mlp(self, layer_idx: int | None, fn_call: Any) -> tuple[Any, list[float] | None]:
        layers = getattr(getattr(self.model, "model", None), "layers", None) if layer_idx is not None else None
        if layers is None or not (0 <= int(layer_idx) < len(layers)):  # type: ignore[arg-type]
            return fn_call(), None
        norm_mod = getattr(layers[int(layer_idx)], "post_attention_layernorm", None)  # type: ignore[arg-type]
        if norm_mod is None or not hasattr(norm_mod, "register_forward_hook"):
            return fn_call(), None
        captured: list[list[float]] = []
        handle = norm_mod.register_forward_hook(
            lambda _m, _i, out: captured.append(
                [
                    float(v)
                    for v in (out[0, -1, :] if getattr(out, "ndim", 0) == 3 else out[-1, :])
                    .detach()
                    .to("cpu", dtype=self.torch.float32)
                    .tolist()
                ]
            )
        )
        try:
            return fn_call(), (captured[-1] if captured else None)
        finally:
            handle.remove()

    def prefill(self, bank: Any, _pen: str = "HarnessPen", text: str = "", *, reset_kv: bool = False) -> list[int]:
        del _pen
        self.active_ll_matcher = None
        self._active_grammar_span_key = None
        self._pending_utf8_buffer().clear()
        self._mount_bank(bank)
        if bool(reset_kv):
            self.past_key_values = None
            self.session_token_ids.clear()
        ids = [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]
        if not ids:
            return []
        self.session_token_ids.extend(ids)
        with self.torch.no_grad():
            try:
                out = self.model(
                    input_ids=self.torch.tensor([ids], dtype=self.torch.long, device=self.device),
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            except (NotImplementedError, RuntimeError):
                if self.device.type != "mps":
                    raise
                self.device, self.past_key_values = self.torch.device("cpu"), None
                self._cached_ll_bitmasks = None
                self.model.to(self.device, dtype=self.torch.float32)
                self._prematerialize_bundle_tensors()
                self.active_bank = "Uninitialized"
                self._mount_bank(bank)
                out = self.model(
                    input_ids=self.torch.tensor([self.session_token_ids], dtype=self.torch.long, device=self.device),
                    use_cache=True,
                    logits_to_keep=1,
                )
        self.past_key_values, self.last_logits = out.past_key_values, out.logits[0, -1, :]
        return ids

    def append_prompt(self, text: str, bank: Any = "Base", pen: str = "HarnessPen", *, reset_kv: bool = False) -> list[int]:
        return self.prefill(bank, pen, text, reset_kv=bool(reset_kv))

    def _get_ll_bitmask_buffers(self, vocab_sz: int, dev: Any) -> tuple[Any, Any | None]:
        cached = getattr(self, "_cached_ll_bitmasks", None)
        dev_key = str(dev)
        if cached is not None and cached[0] == vocab_sz and cached[1] == dev_key:
            return cached[2], cached[3]
        cpu_mask = llguidance.torch.allocate_token_bitmask(1, vocab_sz)
        if getattr(dev, "type", "") == "cuda" and self.torch.cuda.is_available():
            cpu_mask = cpu_mask.pin_memory()
            gpu_mask = self.torch.empty_like(cpu_mask, device=dev)
        elif getattr(dev, "type", "cpu") != "cpu":
            gpu_mask = self.torch.empty_like(cpu_mask, device=dev)
        else:
            gpu_mask = None
        self._cached_ll_bitmasks = (vocab_sz, dev_key, cpu_mask, gpu_mask)
        return cpu_mask, gpu_mask

    def sample_token(
        self,
        bank: Any,
        _pen: str = "PayloadPen",
        phase: str = "PayloadBody",
        allowed_token_ids: list[int] | None = None,
        grammar_spec: dict[str, Any] | str | None = None,
    ) -> tuple[int, str]:
        del _pen
        if self.last_logits is None:
            raise RuntimeError("sample_token called before any prefill")
        bank_key = normalize_bank_key(bank)
        cur_span_key = (bank_key, str(phase))
        is_new_span = cur_span_key != getattr(self, "_active_grammar_span_key", None)
        self._active_grammar_span_key = cur_span_key
        if grammar_spec is not None and not _is_unconstrained_grammar_spec(grammar_spec):
            self.mount_grammar_cartridge(grammar_spec)
        elif str(phase) == "PayloadBody" and is_new_span and (manifest_grammar := self.resolve_manifest_grammar(bank_key)) is not None:
            self.mount_grammar_cartridge(manifest_grammar)
        elif grammar_spec is not None:
            self.mount_grammar_cartridge(grammar_spec)
        self._mount_bank(bank)
        logits = (
            self.last_logits.clone()
            if self.last_logits.device.type == "cuda"
            else self.last_logits.detach().float().cpu()
        )
        matcher = getattr(self, "active_ll_matcher", None)
        if matcher is not None:
            if llguidance is None:
                raise RuntimeError("llguidance is required when active_ll_matcher is mounted")
            ll_tok = self._ensure_ll_tokenizer()
            vocab_sz = int(getattr(ll_tok, "vocab_size", logits.shape[-1]))
            cpu_mask, gpu_mask = self._get_ll_bitmask_buffers(vocab_sz, logits.device)
            llguidance.torch.fill_next_token_bitmask(matcher, cpu_mask, 0)
            if gpu_mask is not None:
                gpu_mask.copy_(cpu_mask, non_blocking=True)
                llguidance.torch.apply_token_bitmask_inplace(logits, gpu_mask)
            else:
                llguidance.torch.apply_token_bitmask_inplace(logits, cpu_mask)
        elif allowed_token_ids is not None:
            valid_tids = [int(t) for t in allowed_token_ids if 0 <= int(t) < int(logits.shape[-1])]
            if not valid_tids:
                raise ValueError("Empty allowed token set after applying generation constraints")
            masked = logits.clone().fill_(float("-inf"))
            for t in valid_tids:
                masked[t] = logits[t]
            logits = masked

        next_tid = int(self.torch.argmax(logits).item())
        if matcher is not None:
            if not matcher.consume_token(next_tid):
                err, self.active_ll_matcher = matcher.get_error(), None
                raw_piece = self._decode_token_ids([next_tid])
                self._pending_utf8_buffer().clear()
                raise RuntimeError(f"llguidance matcher rejected token {next_tid} ({raw_piece!r}): {err}")
            if matcher.is_stopped():
                self.active_ll_matcher = None
        piece = self._decode_incremental_utf8_step(next_tid)
        self._last_step_token_ids = [next_tid]
        return next_tid, piece

    def consume_token(self, bank: Any, token_id: int) -> None:
        tid = int(token_id)
        self._mount_bank(bank)
        self.session_token_ids.append(tid)
        with self.torch.no_grad():
            try:
                out = self.model(
                    input_ids=self.torch.tensor([[tid]], dtype=self.torch.long, device=self.device),
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            except (NotImplementedError, RuntimeError):
                if self.device.type != "mps":
                    raise
                self.device, self.past_key_values = self.torch.device("cpu"), None
                self._cached_ll_bitmasks = None
                self.model.to(self.device, dtype=self.torch.float32)
                self._prematerialize_bundle_tensors()
                self.active_bank = "Uninitialized"
                self._mount_bank(bank)
                out = self.model(
                    input_ids=self.torch.tensor([self.session_token_ids], dtype=self.torch.long, device=self.device),
                    use_cache=True,
                    logits_to_keep=1,
                )
        self.past_key_values, self.last_logits = out.past_key_values, out.logits[0, -1, :]

    def step_token(
        self,
        bank: Any,
        _pen: str = "PayloadPen",
        phase: str = "PayloadBody",
        allowed_token_ids: list[int] | None = None,
        grammar_spec: dict[str, Any] | str | None = None,
    ) -> tuple[int, str]:
        next_tid, piece = self.sample_token(bank, _pen, phase, allowed_token_ids=allowed_token_ids, grammar_spec=grammar_spec)
        self.consume_token(bank, next_tid)
        return next_tid, piece

    def _encode_sequence_chunks(self, item: Any) -> list[tuple[str, list[int]]]:
        enc = lambda s: [int(t) for t in self.tokenizer.encode(str(s), add_special_tokens=False)]
        if isinstance(item, str):
            return [("Base", enc(item))]
        if isinstance(item, dict):
            if any(k in item for k in ("chunks", "turns", "messages")):
                return self._encode_sequence_chunks(item.get("chunks") or item.get("turns") or item.get("messages"))
            return [(str(item.get("bank", "Base")), [int(t) for t in item["token_ids"]] if "token_ids" in item else enc(item.get("text", "")))]
        if isinstance(item, list) and item:
            if all(isinstance(x, int) for x in item):
                return [("Base", [int(x) for x in item])]
            if all(isinstance(x, str) for x in item):
                return [("Base", enc("".join(item)))]
            if all(isinstance(x, dict) and "role" in x and "content" in x for x in item):
                has_tpl = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
                txt = (
                    self.tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=False)
                    if has_tpl
                    else "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in item)
                )
                return [("Base", enc(txt))]
            if all(isinstance(x, dict) for x in item):
                return [
                    (normalize_bank_key(c.get("bank", "Base")), [int(t) for t in c["token_ids"]] if "token_ids" in c else enc(c.get("text", "")))
                    for c in item
                    if c.get("text") or c.get("token_ids")
                ]
        raise ValueError(f"Unsupported capture sequence item: {item!r}")

    def capture_activations(
        self,
        prompts: list[Any] | None = None,
        layers: list[int] | None = None,
        include_prefix_background: bool = False,
        background_stride: int = 8,
        *,
        sequences: list[Any] | None = None,
    ) -> dict[str, Any]:
        target_layers = [int(l) for l in (layers or [])]
        activations: dict[str, list[list[float]]] = {str(l): [] for l in target_layers}
        residuals: dict[str, list[list[float]]] = {str(l): [] for l in target_layers}
        background_activations: dict[str, list[list[float]]] = {str(l): [] for l in target_layers}
        cur_x, cur_h, cur_bg, stride = {}, {}, {}, max(1, int(background_stride))

        def _make_hook(lid: int) -> Any:
            def _hook(_mod: Any, inp: Any, out: Any) -> None:
                x_in, x_out = (inp[0] if isinstance(inp, tuple) else inp), (out[0] if isinstance(out, tuple) else out)
                cur_h[lid], cur_x[lid] = x_in[0, -1, :].detach().float().cpu().tolist(), x_out[0, -1, :].detach().float().cpu().tolist()
                cur_bg[lid] = (
                    x_out[0, :-1:stride, :].detach().float().cpu().tolist()
                    if include_prefix_background and int(x_out.shape[1]) > 1
                    else []
                )

            return _hook

        hooks = [self.model.model.layers[l].post_attention_layernorm.register_forward_hook(_make_hook(l)) for l in target_layers]
        try:
            for item in (sequences if sequences is not None else (prompts or [])):
                chunks = [c for c in self._encode_sequence_chunks(item) if c[1]]
                if len(chunks) == 1 and chunks[0][0] == "Base":
                    self._mount_bank("Base")
                    with self.torch.no_grad():
                        self.model(
                            input_ids=self.torch.tensor([chunks[0][1]], dtype=self.torch.long, device=self.device),
                            use_cache=False,
                            logits_to_keep=1,
                        )
                else:
                    pkv = None
                    for bank_name, tids in chunks:
                        self._mount_bank(bank_name)
                        with self.torch.no_grad():
                            pkv = self.model(
                                input_ids=self.torch.tensor([tids], dtype=self.torch.long, device=self.device),
                                past_key_values=pkv,
                                use_cache=True,
                                logits_to_keep=1,
                            ).past_key_values
                for l in target_layers:
                    activations[str(l)].append(cur_x[l])
                    residuals[str(l)].append(cur_h[l])
                    if include_prefix_background:
                        background_activations[str(l)].extend(cur_bg[l])
        finally:
            for h in hooks:
                h.remove()
            self._mount_bank("Base")
        return {
            "activations": activations,
            "residuals": residuals,
            **({"background_activations": background_activations} if include_prefix_background else {}),
        }

    def extract_compiler_weight_geometry(
        self,
        qk_attn_layers: list[int] | None = None,
        deltanet_layers: list[int] | None = None,
        *,
        target_layer: int | None = None,
        token_ids: list[int] | None = None,
        protected_token_ids: list[int] | None = None,
        qk_head_partitions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        torch, F = self.torch, self.torch.nn.functional
        num_layers = len(self.model.model.layers)
        resolved_target_layer = int(target_layer) if target_layer is not None else self._resolve_router_layer()
        if num_layers > 0 and (resolved_target_layer < 0 or resolved_target_layer >= num_layers):
            raise ValueError(f"target_layer {resolved_target_layer} out of range for model with {num_layers} layers")
        qk_attn_layers = qk_attn_layers or ([15, 19] if num_layers <= 24 else [23, 27])
        deltanet_layers = deltanet_layers or ([21] if num_layers <= 24 else [29])
        with torch.no_grad():
            emb_w, lm_w = self.model.model.embed_tokens.weight.detach(), self.model.lm_head.weight.detach()
            vocab_rows = int(emb_w.shape[0])
            tokenizer_len = min(
                vocab_rows,
                len(self.tokenizer) if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "__len__") else 248077,
            )
            norm_w = self.model.model.norm.weight.detach().float().cpu().tolist()
            w_nl = emb_w[:tokenizer_len].float()
            _, eigvecs = torch.linalg.eigh((w_nl.T @ w_nl).double())
            requested_tids: list[int] = list(PROTOCOL_MODE_TIDS.values())
            for raw_tid in (token_ids or []):
                tid_int = int(raw_tid)
                if tid_int not in requested_tids:
                    requested_tids.append(tid_int)
            named_token_unembeddings = {
                str(tid): lm_w[int(tid)].float().cpu().tolist()
                for tid in requested_tids
                if 0 <= int(tid) < vocab_rows
            }
            named_token_unembeddings.update(
                {
                    alias: named_token_unembeddings[str(tid)]
                    for alias, tid in PROTOCOL_MODE_TIDS.items()
                    if str(tid) in named_token_unembeddings
                }
            )
            protected_tids = [int(t) for t in (protected_token_ids or []) if 0 <= int(t) < vocab_rows]
            protected_sql_embeddings = emb_w[protected_tids].float().cpu().tolist() if protected_tids else []
            qk_layer_vectors: dict[str, dict[str, list[float]]] = {}
            if isinstance(qk_head_partitions, dict) and qk_head_partitions:
                for l_attn in qk_attn_layers:
                    spec = (
                        qk_head_partitions.get(str(int(l_attn)))
                        or qk_head_partitions.get(str(int(l_attn) - 1))
                        or qk_head_partitions.get("default")
                    )
                    if isinstance(spec, dict) and 0 < int(l_attn) < num_layers:
                        pos_idx = [int(h) for h in spec.get("pos_heads", [])]
                        neg_idx = [int(h) for h in spec.get("neg_heads", [])]
                        set_idx = [int(h) for h in spec.get("set_heads", [])]
                        where_idx = [int(h) for h in spec.get("where_heads", [])]
                        if not (pos_idx and neg_idx and set_idx and where_idx):
                            continue
                        attn_blk = self.model.model.layers[int(l_attn)]
                        gamma_in = 1.0 + attn_blk.input_layernorm.weight.detach().float().cpu()
                        q_proj_w = attn_blk.self_attn.q_proj.weight.detach().float().cpu()
                        head_dim = int(getattr(attn_blk.self_attn, "head_dim", 256))
                        num_heads = int(q_proj_w.shape[0]) // (2 * head_dim)
                        q_query = q_proj_w.view(num_heads, 2 * head_dim, -1)[:, :head_dim, :]
                        q_gate = q_proj_w.view(num_heads, 2 * head_dim, -1)[:, head_dim:, :]
                        cfg = getattr(self.model, "config", None)
                        rotary_dim = max(
                            0,
                            min(
                                head_dim // 2,
                                int(getattr(cfg, "rotary_dim", int(head_dim * float(getattr(cfg, "partial_rotary_factor", 0.25))))),
                            ),
                        )
                        mid_dim = rotary_dim + (head_dim - rotary_dim) // 2
                        v_pos, v_neg = q_gate[pos_idx].mean(dim=(0, 1)) * gamma_in, q_gate[neg_idx].mean(dim=(0, 1)) * gamma_in
                        v_qs = F.normalize(q_query[set_idx, rotary_dim:mid_dim, :].mean(dim=(0, 1)) * gamma_in, dim=0)
                        v_qw = F.normalize(q_query[where_idx, mid_dim:head_dim, :].mean(dim=(0, 1)) * gamma_in, dim=0)
                        entry_vecs: dict[str, list[float]] = {
                            "v_pos_gate": v_pos.tolist(),
                            "v_neg_gate": v_neg.tolist(),
                            "v_qk_set": v_qs.tolist(),
                            "v_qk_where": v_qw.tolist(),
                        }
                        if "gate_weight" in spec and "query_weight" in spec:
                            gw, qw = float(spec["gate_weight"]), float(spec["query_weight"])
                            v_gc = F.normalize(v_pos - v_neg, dim=0)
                            entry_vecs["v_set"] = F.normalize(gw * v_gc + qw * v_qs, dim=0).tolist()
                            entry_vecs["v_where"] = F.normalize(gw * v_gc + qw * v_qw, dim=0).tolist()
                        qk_layer_vectors[str(int(l_attn) - 1)] = entry_vecs
            deltanet_layer_weights: dict[str, dict[str, Any]] = {}
            for l_dn in deltanet_layers:
                if 0 < int(l_dn) < num_layers and hasattr(self.model.model.layers[int(l_dn)], "linear_attn"):
                    blk = self.model.model.layers[int(l_dn)]
                    la = blk.linear_attn
                    deltanet_layer_weights[str(int(l_dn) - 1)] = {
                        "in_proj_a_weight": la.in_proj_a.weight.detach().float().cpu().tolist(),
                        "in_proj_b_weight": la.in_proj_b.weight.detach().float().cpu().tolist(),
                        "a_log": la.A_log.detach().float().cpu().tolist(),
                        "dt_bias": la.dt_bias.detach().float().cpu().tolist(),
                        "input_layernorm_weight": blk.input_layernorm.weight.detach().float().cpu().tolist(),
                    }
            downstream_mlp_anti_echo: dict[str, Any] | None = None
            anti_layer_idx = resolved_target_layer + 1
            if num_layers >= 2 and 0 <= anti_layer_idx < num_layers:
                last_blk = self.model.model.layers[anti_layer_idx]
                if hasattr(getattr(last_blk, "post_attention_layernorm", None), "weight") and hasattr(last_blk, "mlp"):
                    post_norm_w = last_blk.post_attention_layernorm.weight.detach().float().cpu()
                    wg_last = last_blk.mlp.gate_proj.weight.detach().float().cpu()
                    wu_last = last_blk.mlp.up_proj.weight.detach().float().cpu()
                    wd_last = last_blk.mlp.down_proj.weight.detach().float().cpu()
                    valid_req_tids = [int(t) for t in requested_tids if 0 <= int(t) < vocab_rows]
                    if len(valid_req_tids) >= 2 and int(wg_last.shape[0]) > 0:
                        gamma_final = 1.0 + self.model.model.norm.weight.detach().float().cpu()
                        u_mat = lm_w[valid_req_tids].float().cpu() * gamma_final.unsqueeze(0)
                        selected_neurons: set[int] = set()
                        for i_tok in range(u_mat.shape[0]):
                            u_promo = u_mat[i_tok]
                            u_others = torch.cat([u_mat[:i_tok], u_mat[i_tok + 1 :]], dim=0).mean(dim=0)
                            oppose_scores = wd_last.T @ (u_others - u_promo)
                            pos_idx_t = torch.nonzero(oppose_scores > 0.0).squeeze(-1)
                            if pos_idx_t.numel() > 0:
                                k_take = min(24, int(pos_idx_t.numel()))
                                top_n = pos_idx_t[torch.topk(oppose_scores[pos_idx_t], k_take).indices].tolist()
                                selected_neurons.update(int(j) for j in top_n)
                        if selected_neurons:
                            sel_sorted = sorted(selected_neurons)
                            downstream_mlp_anti_echo = {
                                "w_gate": wg_last[sel_sorted].tolist(),
                                "w_up": wu_last[sel_sorted].tolist(),
                                "w_down": wd_last[:, sel_sorted].tolist(),
                                "post_attn_norm_weight": post_norm_w.tolist(),
                                "top_k_anti_echo_neurons": 16,
                            }
        return {
            "target_layer": resolved_target_layer,
            "d_model": self.d_model,
            "num_hidden_layers": num_layers,
            "base_intermediate_size": self.base_intermediate_size,
            "vocab_rows": vocab_rows,
            "tokenizer_len": tokenizer_len,
            "norm_weight": norm_w,
            "mean_natural_token_norm": float(w_nl.norm(dim=-1).mean().item()),
            "vocab_top_k_eigvecs": eigvecs[:, -min(256, eigvecs.shape[1]) :].T.float().cpu().tolist(),
            "protected_sql_embeddings": protected_sql_embeddings,
            "pretrained_unused_row": emb_w[min(tokenizer_len, vocab_rows - 1)].float().cpu().tolist(),
            "named_token_unembeddings": named_token_unembeddings,
            "qk_layer_vectors": qk_layer_vectors,
            "deltanet_layer_weights": deltanet_layer_weights,
            "downstream_mlp_anti_echo": downstream_mlp_anti_echo,
        }

    def capture_payload_activations(
        self,
        wire_prefix: str,
        payload: str,
        closing_tag: str,
        target_layer: int,
    ) -> dict[str, Any]:
        """Teacher-force calibration payloads and capture real next-token boundaries."""
        if not 0 <= target_layer < len(self.model.model.layers):
            raise ValueError("payload capture target layer out of range")
        encode = lambda text: [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]
        prefix_ids, payload_ids, closing_ids = encode(wire_prefix), encode(payload), encode(closing_tag)
        if not prefix_ids or not payload_ids or not closing_ids:
            raise ValueError("payload capture requires nonempty prefix, payload and closing tag")
        positions = {0: "opener", len(payload_ids): "close_0"}
        if payload.lstrip().startswith("{") and len(payload_ids) > 1:
            positions[1] = "field"
            for index in range(1, len(payload_ids)):
                decoded_prefix = self.tokenizer.decode(payload_ids[:index], skip_special_tokens=False)
                if decoded_prefix.endswith(('":"', ': "')):
                    positions[index] = "value"
        if len(payload_ids) > 3:
            positions.setdefault(len(payload_ids) // 2, "background")
        for i in range(1, len(closing_ids)):
            positions[len(payload_ids) + i] = f"close_{i}"
        continuation = payload_ids + closing_ids
        captured: dict[str, list[float]] = {}

        def hook(_module: Any, inputs: Any, outputs: Any) -> None:
            residual = inputs[0] if isinstance(inputs, tuple) else inputs
            pre_mlp = outputs[0] if isinstance(outputs, tuple) else outputs
            captured["residual"] = residual[0, -1].detach().float().cpu().tolist()
            captured["pre_mlp"] = pre_mlp[0, -1].detach().float().cpu().tolist()

        self.reset_session()
        self._mount_bank("Base")
        handle = self.model.model.layers[target_layer].post_attention_layernorm.register_forward_hook(hook)
        rows = []
        try:
            previous = 0
            for position, boundary in sorted(positions.items()):
                tids = prefix_ids if position == 0 else continuation[previous:position]
                with self.torch.no_grad():
                    output = self.model(
                        input_ids=self.torch.tensor([tids], dtype=self.torch.long, device=self.device),
                        past_key_values=self.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                self.past_key_values = output.past_key_values
                logits = output.logits[0, -1].detach()
                target = continuation[position]
                natural_limit = (
                    min(len(self.tokenizer), int(logits.shape[-1]))
                    if hasattr(self.tokenizer, "__len__")
                    else int(logits.shape[-1])
                )
                candidates = logits[:natural_limit].topk(2).indices.tolist()
                competitor = next(int(t) for t in candidates if int(t) != target)
                rows.append({"boundary": boundary, **captured, "target_token_id": target, "competitor_token_id": competitor})
                previous = position
        finally:
            handle.remove()
            self.reset_session()
        return {"rows": rows}

    def capture_and_generate_on_policy_plan(
        self,
        wire_prefix: str,
        oracle_mode: str,
        *,
        target_layer: int | None = None,
        max_plan_tokens: int = 512,
    ) -> dict[str, Any]:
        """
        Capture `(pre_mlp, residual)` at `[MODE:` on `target_layer` (prefilling `wire_prefix` under
        `Base` without polluting `router_background`), consume non-closing `ModeLabel` tokens under
        `OrientTrunk`, detach closing `]` under `Base`, and generate on-policy `Plan:` under `Base`.
        """
        num_layers = len(self.model.model.layers)
        resolved_layer = int(target_layer) if target_layer is not None else self._resolve_router_layer()
        if resolved_layer < 0 or resolved_layer >= num_layers:
            raise ValueError(f"target_layer {resolved_layer} out of range for model with {num_layers} layers")
        mode_upper = str(oracle_mode).strip().upper()
        if mode_upper not in _CANONICAL_MODES:
            raise ValueError(f"Unsupported oracle_mode: {oracle_mode!r}")

        captured: dict[str, list[float]] = {}
        router_background: list[dict[str, list[float]]] = []
        capturing_opener = True

        def _hook(_mod: Any, inp: Any, out: Any) -> None:
            x_in = inp[0] if isinstance(inp, tuple) else inp
            x_out = out[0] if isinstance(out, tuple) else out
            residuals = x_in[0].detach().float().cpu().tolist()
            pre_mlp = x_out[0].detach().float().cpu().tolist()
            states = [{"pre_mlp": pre, "residual": res} for pre, res in zip(pre_mlp, residuals)]
            if capturing_opener:
                captured.update(states[-1])
            else:
                router_background.extend(states)

        self.reset_session()
        mode_tids = [int(t) for t in self.tokenizer.encode(f" {mode_upper}]", add_special_tokens=False)]
        hook = self.model.model.layers[resolved_layer].post_attention_layernorm.register_forward_hook(_hook)
        try:
            self.prefill("Base", "HarnessPen", wire_prefix, reset_kv=True)
            capturing_opener = False
            # Capture only genuine OrientTrunk-mounted ModeLabel continuation positions before `]`.
            for tid in mode_tids[:-1]:
                self.consume_token("OrientTrunk", tid)
        finally:
            hook.remove()
        if mode_tids:
            self.consume_token("Base", mode_tids[-1])

        self.prefill("Base", "HarnessPen", "\nPlan:", reset_kv=False)
        plan_lark = "start: / [^\\r\\n<>]{3,160}\\n/"
        plan_pieces: list[str] = []
        for step_i in range(max(1, int(max_plan_tokens))):
            gspec = {"kind": "lark_grammar", "spec": plan_lark} if step_i == 0 else None
            tid, piece = self.sample_token("Base", "PayloadPen", "PlanBody", grammar_spec=gspec)
            self.consume_token("Base", tid)
            plan_pieces.append(piece)
            if "\n" in "".join(plan_pieces):
                break
        self.active_ll_matcher = None
        self._active_grammar_span_key = None
        generated_plan_line = "".join(plan_pieces).split("\n", 1)[0].strip()
        if not generated_plan_line:
            raise ValueError("on-policy Plan generation produced no text")

        return {
            "target_layer": resolved_layer,
            "pre_mlp": captured["pre_mlp"],
            "residual": captured["residual"],
            "generated_plan_line": generated_plan_line,
            "router_background": router_background,
        }


QwenStdioWorker = LiveQwen35Engine


def run_self_test() -> dict[str, Any]:
    """Deterministic `llguidance` + incremental KV + bank-mounting self-test."""
    import torch
    from transformers import AutoTokenizer

    assert llguidance is not None, "llguidance must be installed in the worker environment"
    tokenizer_source, logits_vocab = "Qwen/Qwen3.5-0.8B-Base", 248320
    try:
        tok: Any = AutoTokenizer.from_pretrained(LiveQwen35Engine._resolve_model_path("Qwen/Qwen3.5-0.8B-Base"), local_files_only=True)
        ll_tok = llguidance.hf.from_tokenizer(tok)
    except Exception:
        tokenizer_source, logits_vocab = "byte_fallback", 256

        class ByteFallbackTokenizer:
            eos_token_id, bos_token_id, special_token_ids, tokens = 0, None, [0], [b"<eos>"] + [bytes([i]) for i in range(1, 256)]

            def __call__(self, s: str) -> list[int]:
                return [int(b) for b in s.encode("utf-8")]

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [int(b) for b in text.encode("utf-8")]

            def decode(self, tids: list[int], skip_special_tokens: bool = False) -> str:
                return bytes(int(t) for t in tids if 0 < int(t) < 256).decode("utf-8", errors="replace")

            def to_ll_tokenizer(self) -> Any:
                return llguidance.LLTokenizer(llguidance.TokenizerWrapper(self))

        tok = ByteFallbackTokenizer()
        ll_tok = tok.to_ll_tokenizer()

    class RecordingModel:
        def __init__(self, worker_ref: LiveQwen35Engine) -> None:
            self.worker_ref, self.calls = worker_ref, []
            mk_mlp = lambda: type(
                "Mlp",
                (),
                {
                    "gate_proj": type("L", (), {"weight": torch.zeros(16, 8)})(),
                    "up_proj": type("L", (), {"weight": torch.zeros(16, 8)})(),
                    "down_proj": type("L", (), {"weight": torch.zeros(8, 16)})(),
                },
            )()
            self.model = type(
                "Inner",
                (),
                {
                    "layers": [
                        type("Layer", (), {"mlp": mk_mlp(), "post_attention_layernorm": torch.nn.Identity()})()
                        for _ in range(4)
                    ],
                    "embed_tokens": type("Emb", (), {"weight": torch.ones(logits_vocab, 8)})(),
                    "norm": type("Norm", (), {"weight": torch.zeros(8)})(),
                },
            )()
            self.lm_head = type("Head", (), {"weight": torch.ones(logits_vocab, 8)})()

        def __call__(self, input_ids: Any, past_key_values: Any = None, use_cache: bool = True, logits_to_keep: int = 1) -> Any:
            flat_ids = [int(x) for x in input_ids[0].tolist()]
            new_kv = (
                {"accumulated_ids": (list(past_key_values["accumulated_ids"]) if isinstance(past_key_values, dict) else []) + flat_ids}
                if use_cache
                else None
            )
            for idx, lyr in enumerate(self.model.layers):
                lyr.post_attention_layernorm(torch.full((1, max(1, len(flat_ids)), 8), float(idx + 1), dtype=torch.float32))
            self.calls.append(
                {
                    "input_ids": flat_ids,
                    "received_past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "bank": self.worker_ref.active_bank_name,
                }
            )
            return type("Out", (), {"past_key_values": new_kv, "logits": torch.full((1, 1, logits_vocab), -10.0)})()

    worker = QwenStdioWorker.__new__(QwenStdioWorker)
    worker.torch, worker.device, worker.tokenizer, worker.ll_tokenizer, worker.active_ll_matcher = (
        torch,
        torch.device("cpu"),
        tok,
        ll_tok,
        None,
    )
    worker.model = RecordingModel(worker)
    worker.base_intermediate_size, worker.d_model, worker.reserved_slots, worker.extended_intermediate_size = 8, 8, 8, 16
    worker.modified_layers, worker.cartridges = (
        {0, 1, 2, 3},
        {"Base": {}, "OrientTrunk": {}, "ModePayload:Probe": {}, "ModePayload:Read": {}, "ModePayload:Result": {}},
    )
    worker.permanent_static_layers, worker.manifest_mode_grammars, worker.latent_interrupt_traps = {}, {}, []
    worker.active_interrupt_trap, worker._active_grammar_span_key = None, None
    worker.active_bank, worker.past_key_values, worker.last_logits, worker.session_token_ids, worker._last_step_token_ids = (
        "Uninitialized",
        None,
        None,
        [],
        [],
    )
    worker._pending_utf8_tids = []

    def _drive_span(
        bank: str,
        phase: str,
        prefill_txt: str,
        grammar_spec: dict[str, Any],
        target_text: str,
        decoy_str: str,
        *,
        reset_kv: bool = False,
    ) -> list[str]:
        worker.prefill(bank, "HarnessPen", prefill_txt, reset_kv=reset_kv)
        decoy_tid = int(tok.encode(decoy_str, add_special_tokens=False)[0])
        target_tids = [int(t) for t in tok.encode(target_text, add_special_tokens=False)]
        pieces: list[str] = []
        for idx, tid in enumerate(target_tids):
            worker.last_logits = torch.full((logits_vocab,), -10.0, dtype=torch.float32)
            if idx == 0:
                worker.last_logits[decoy_tid] = 95.0
                if logits_vocab > 248077:
                    worker.last_logits[248100] = 999.0
            worker.last_logits[tid] = 25.0
            step_tid, piece = worker.step_token(bank, "PayloadPen", phase, grammar_spec=grammar_spec if idx == 0 else None)
            assert step_tid == tid and (idx + 1 == len(target_tids) or worker.active_ll_matcher is not None)
            pieces.append(piece)
        assert worker.active_ll_matcher is None and "".join(pieces) == target_text and "\ufffd" not in "".join(pieces)
        return pieces

    mode_pieces = _drive_span(
        "OrientTrunk",
        "ModeLabel",
        "<|im_start|>user\nInspect incidents.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:",
        {"kind": "literal_set", "spec": [" PROBE]", " ASK]"]},
        " PROBE]",
        " ",
        reset_kv=True,
    )
    assert tokenizer_source != "Qwen/Qwen3.5-0.8B-Base" or mode_pieces == [" PRO", "BE", "]"]
    plan_pieces = _drive_span(
        "Base",
        "PlanBody",
        "\nPlan:",
        {"kind": "lark_grammar", "spec": "start: / [^\\r\\n]{3,}\\n/"},
        " Inspect schema.\n",
        "\n",
    )
    json_pieces = _drive_span(
        "ModePayload:Result",
        "PayloadBody",
        "</think>\n<to_user>\n",
        {
            "kind": "json_schema",
            "spec": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
        '{"ok":true}',
        "SELECT",
    )
    sql_pieces = _drive_span(
        "ModePayload:Read",
        "PayloadBody",
        "</think>\n<sql>\n",
        {"kind": "lark_grammar", "spec": 'start: "SELECT * FROM incidents" ";\\n</sql>"'},
        "SELECT * FROM incidents;\n</sql>",
        "UPDATE",
    )

    two_phase_sql = "SELECT 1;\n</sql>"
    worker.prefill("ModePayload:Read", "HarnessPen", "</think>\n<sql>\n", reset_kv=True)
    two_phase_tids = [int(t) for t in tok.encode(two_phase_sql, add_special_tokens=False)]
    accumulated_sql = ""
    body_consume_banks: list[str] = []
    for idx, tid in enumerate(two_phase_tids):
        worker.last_logits = torch.full((logits_vocab,), -10.0, dtype=torch.float32)
        worker.last_logits[tid] = 30.0
        calls_before_sample = len(worker.model.calls)
        sampled_tid, sampled_piece = worker.sample_token(
            "ModePayload:Read",
            "PayloadPen",
            "PayloadBody",
            grammar_spec={"kind": "lark_grammar", "spec": 'start: "SELECT 1;\\n</sql>"'} if idx == 0 else None,
        )
        assert sampled_tid == tid and len(worker.model.calls) == calls_before_sample
        accumulated_sql += sampled_piece
        consume_bank = "Base" if accumulated_sql.endswith("</sql>") else "ModePayload:Read"
        worker.consume_token(consume_bank, sampled_tid)
        assert len(worker.model.calls) == calls_before_sample + 1
        body_consume_banks.append(worker.model.calls[-1]["bank"])
    assert accumulated_sql == two_phase_sql
    assert body_consume_banks[-1] == "Base" and all(b == "ModePayload:Read" for b in body_consume_banks[:-1])

    squid_sql = "SELECT '🦑';\n</sql>"
    squid_pieces = _drive_span(
        "ModePayload:Read",
        "PayloadBody",
        "</think>\n<sql>\n",
        {"kind": "lark_grammar", "spec": 'start: "SELECT \'🦑\';\\n</sql>"'},
        squid_sql,
        "UPDATE",
    )
    assert "".join(squid_pieces) == squid_sql and "\ufffd" not in "".join(squid_pieces)
    if tokenizer_source == "Qwen/Qwen3.5-0.8B-Base":
        assert tok.encode(squid_sql, add_special_tokens=False)[:5] == [4703, 359, 9008, 99, 239]
        assert squid_pieces[:5] == ["SELECT", " '", "", "", "🦑"]

    jp_sql = "SELECT * FROM 注文 WHERE 名前 = '東京🦑カフェ';\n</sql>"
    jp_pieces = _drive_span(
        "ModePayload:Read",
        "PayloadBody",
        "</think>\n<sql>\n",
        {"kind": "lark_grammar", "spec": 'start: "SELECT * FROM 注文 WHERE 名前 = \'東京🦑カフェ\';\\n</sql>"'},
        jp_sql,
        "UPDATE",
    )
    assert "".join(jp_pieces) == jp_sql and "\ufffd" not in "".join(jp_pieces)

    ru_ask = "Пожалуйста, проверьте баланс 🚀!"
    ru_pieces = _drive_span(
        "ModePayload:Result",
        "PayloadBody",
        "</think>\n<to_user>\n",
        {"kind": "literal_set", "spec": [ru_ask]},
        ru_ask,
        "\n",
    )
    assert "".join(ru_pieces) == ru_ask and "\ufffd" not in "".join(ru_pieces)
    if tokenizer_source == "Qwen/Qwen3.5-0.8B-Base":
        assert " 🚀" in ru_pieces

    class MockPrefixPlusIncompleteByteTokenizer:
        _map = {1: b"SELECT", 2: b" '\xf0", 3: b"\x9f\xa6", 4: b"\x91';"}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1] if text else []

        def decode(self, tids: list[int], skip_special_tokens: bool = False) -> str:
            return b"".join(self._map[int(t)] for t in tids).decode("utf-8", errors="replace")

    saved_tok, saved_ll_tok = worker.tokenizer, worker.ll_tokenizer
    try:
        worker.tokenizer, worker.ll_tokenizer, worker.active_ll_matcher = MockPrefixPlusIncompleteByteTokenizer(), None, None
        worker.reset_session()
        worker.prefill("Base", "HarnessPen", "seed", reset_kv=True)
        prefix_pieces: list[str] = []
        for mock_tid in (1, 2, 3, 4):
            worker.last_logits = torch.full((16,), -10.0, dtype=torch.float32)
            worker.last_logits[mock_tid] = 10.0
            _, mock_piece = worker.step_token("Base", "PayloadPen", "PayloadBody")
            prefix_pieces.append(mock_piece)
        assert prefix_pieces == ["SELECT", "", "", " '🦑';"] and "".join(prefix_pieces) == "SELECT '🦑';"
        worker.last_logits = torch.full((16,), -10.0, dtype=torch.float32)
        worker.last_logits[2] = 10.0
        assert worker.step_token("Base", "PayloadPen", "PayloadBody") == (2, "")
        assert worker._pending_utf8_tids == [2]
        worker.prefill("Base", "HarnessPen", "seed")
        assert worker._pending_utf8_tids == []
        worker.last_logits = torch.full((16,), -10.0, dtype=torch.float32)
        worker.last_logits[2] = 10.0
        worker.step_token("Base", "PayloadPen", "PayloadBody")
        assert worker._pending_utf8_tids == [2]
        worker.reset_session()
        assert worker._pending_utf8_tids == []
    finally:
        worker.tokenizer, worker.ll_tokenizer = saved_tok, saved_ll_tok

    try:
        worker.mount_grammar_cartridge({"kind": "lark_grammar", "spec": "start: [[[invalid"})
        raise AssertionError("Expected ValueError on malformed Lark grammar")
    except ValueError:
        pass

    slot_entry = {"slot_lo": 8, "slot_hi": 9, "w_gate": [[1.0] * 8], "w_up": [[1.0] * 8], "w_down": [[2.5] for _ in range(8)]}
    perm_router_entry = {"slot_lo": 10, "slot_hi": 11, "w_gate": [[1.0] * 8], "w_up": [[1.0] * 8], "w_down": [[4.25] for _ in range(8)]}
    static_router_nonoverlap = {"slot_lo": 11, "slot_hi": 12, "w_gate": [[1.0] * 8], "w_up": [[1.0] * 8], "w_down": [[6.5] for _ in range(8)]}
    static_router_overlap = {"slot_lo": 8, "slot_hi": 9, "w_gate": [[1.0] * 8], "w_up": [[1.0] * 8], "w_down": [[9.0] for _ in range(8)]}
    manifest_bundle = {
        "base_intermediate_size": 8,
        "banks": {
            "UnifiedStatic": {"1": [static_router_nonoverlap, static_router_overlap]},
            "OrientTrunk": {"1": [slot_entry]},
            "ModePayload:Read": {"0": [slot_entry]},
        },
        "permanent_static": {"1": [perm_router_entry]},
        "mode_grammars": {"Read": {"LarkGrammar": 'start: "SELECT 1;\\n</sql>"'}},
        "latent_interrupt_traps": [
            {
                "trap_name": "sql_read_fault",
                "layer_idx": 1,
                "target_cartridge": {
                    "weight_bank": {"ModePayload": "Read"},
                    "grammar": {"kind": "lark_grammar", "spec": 'start: "SELECT 1;\\n</sql>"'},
                },
                "w_gate_f32": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "fire_threshold": 2.0,
                "release_ceiling": 0.5,
            }
        ],
    }
    assert worker._bundle_matches_model_dim(manifest_bundle) is True
    worker.cartridges = {}
    worker.permanent_static_layers = {}
    worker._load_and_widen_cartridge_bundle(manifest_bundle)
    # OrientTrunk is mounted on layer 1 (not len(layers)-2 == 2), verifying router_layer derives from OrientTrunk:
    assert worker._resolve_router_layer() == 1
    worker.active_bank = "Uninitialized"
    worker._mount_bank("Base")
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 10].item()) == 4.25
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 11].item()) == 6.5
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 8].item()) == 0.0
    worker._mount_bank("OrientTrunk")
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 10].item()) == 4.25
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 11].item()) == 6.5
    assert float(worker.model.model.layers[1].mlp.down_proj.weight[0, 8].item()) == 2.5

    worker.prefill("ModePayload:Read", "HarnessPen", "</think>\n<sql>\n", reset_kv=True)
    sel_tid = int(tok.encode("SELECT 1;\n</sql>", add_special_tokens=False)[0])
    upd_tid = int(tok.encode("UPDATE", add_special_tokens=False)[0])
    worker.last_logits = torch.full((logits_vocab,), -10.0, dtype=torch.float32)
    worker.last_logits[upd_tid] = 90.0
    worker.last_logits[sel_tid] = 20.0
    auto_tid, _ = worker.sample_token("ModePayload:Read", "PayloadPen", "PayloadBody", grammar_spec=None)
    assert auto_tid == sel_tid
    worker.reset_session()
    fired = worker.evaluate_latent_interrupt_traps([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], layer_idx=1)
    assert fired is not None and fired["event"] == "fired" and worker.active_bank == "ModePayload:Read" and worker.active_ll_matcher is not None
    released = worker.evaluate_latent_interrupt_traps([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], layer_idx=1)
    assert released is not None and released["event"] == "released" and worker.active_bank == "Base" and worker.active_ll_matcher is None

    geom = worker.extract_compiler_weight_geometry(target_layer=2)
    assert geom["target_layer"] == 2 and geom["protected_sql_embeddings"] == [] and geom["qk_layer_vectors"] == {}
    if logits_vocab > max(PROTOCOL_MODE_TIDS.values()):
        assert set(PROTOCOL_MODE_TIDS.keys()).issubset(set(geom["named_token_unembeddings"].keys()))

    calls_before_on_pol = len(worker.model.calls)
    on_pol = worker.capture_and_generate_on_policy_plan(
        "<|im_start|>user\nshow tables<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:",
        "PROBE",
        target_layer=2,
        max_plan_tokens=8,
    )
    probe_tids = tok.encode(" PROBE]", add_special_tokens=False)
    expected_background = max(0, len(probe_tids) - 1)
    assert len(on_pol["router_background"]) == expected_background
    on_pol_calls = worker.model.calls[calls_before_on_pol:]
    assert on_pol_calls[0]["bank"] == "Base"
    for c_i in range(expected_background):
        assert on_pol_calls[1 + c_i]["bank"] == "OrientTrunk"
    assert on_pol_calls[1 + expected_background]["bank"] == "Base"
    assert all(state["pre_mlp"] == [3.0] * 8 and state["residual"] == [3.0] * 8 for state in on_pol["router_background"])
    assert on_pol["target_layer"] == 2
    assert on_pol["pre_mlp"] == [3.0] * 8 and on_pol["residual"] == [3.0] * 8
    assert isinstance(on_pol["generated_plan_line"], str) and len(on_pol["generated_plan_line"]) > 0
    assert worker.active_ll_matcher is None

    payload_capture = worker.capture_payload_activations(
        "<|im_start|>assistant\n</think>\n<to_user>\n",
        '{"summary":"done"}',
        "\n</to_user>",
        2,
    )
    assert {"opener", "field", "value", "background", "close_0"}.issubset({row["boundary"] for row in payload_capture["rows"]})
    assert all(row["pre_mlp"] == [3.0] * 8 and row["residual"] == [3.0] * 8 for row in payload_capture["rows"])
    assert all(row["target_token_id"] != row["competitor_token_id"] for row in payload_capture["rows"])
    assert worker.past_key_values is None

    kv_before = worker.past_key_values
    _, cap_vec = worker._capture_layer_pre_mlp(
        2,
        lambda: worker.prefill("Base", "HarnessPen", "<|im_start|>user\nfollow up<|im_end|>\n", reset_kv=False),
    )
    assert worker.model.calls[-1]["received_past_key_values"] is kv_before and cap_vec == [3.0] * 8
    worker.prefill("Base", "HarnessPen", "<|im_start|>user\nfresh task<|im_end|>\n", reset_kv=True)
    assert worker.model.calls[-1]["received_past_key_values"] is None
    assert worker.capture_activations(layers=[1], sequences=[[{"role": "user", "content": "hi"}]])["activations"]["1"] == [[2.0] * 8]

    summary = {
        "status": "OK",
        "tokenizer_source": tokenizer_source,
        "mode_pieces": mode_pieces,
        "plan_text": "".join(plan_pieces),
        "sql_payload": "".join(sql_pieces),
        "json_payload": "".join(json_pieces),
        "two_phase_close_tag_base_verified": True,
        "router_layer_permanent_static_verified": True,
        "manifest_grammar_and_traps_verified": True,
        "capture_layer_verified": True,
        "reset_kv_verified": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Portia Qwen3.5 stateless-protocol PyTorch + llguidance worker")
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--cartridge-json", default=None)
    parser.add_argument("--model-family")
    parser.add_argument("--allow-cartridge-model-mismatch", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run real llguidance + KV + bank-mounting self-test.")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    engine = LiveQwen35Engine(
        args.model_path,
        args.cartridge_json,
        model_family=args.model_family,
        allow_cartridge_model_mismatch=args.allow_cartridge_model_mismatch,
    )
    for raw_line in sys.stdin:
        if not (line := raw_line.strip()):
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "runtime_metadata":
                resp = {"status": "runtime_metadata", "provenance": engine.runtime_provenance}
            elif op in ("reset_session", "begin_task", "reset_task"):
                (engine.reset_session if op == "reset_session" else engine.reset_task_execution_state)()
                resp: dict[str, Any] = {"status": "prefilled", "token_ids": []}
            elif op == "prefill":
                tids, cap = engine._capture_layer_pre_mlp(
                    req.get("capture_layer"),
                    lambda: engine.prefill(
                        req.get("bank", "Base"),
                        req.get("pen", "HarnessPen"),
                        req.get("text", ""),
                        reset_kv=bool(req.get("reset_kv", False)),
                    ),
                )
                if req.get("grammar_spec") is not None:
                    engine.mount_grammar_cartridge(req.get("grammar_spec"))
                resp = {"status": "prefilled", "token_ids": tids, **({"captured_pre_mlp": cap} if cap is not None else {})}
            elif op == "sample_token":
                tid, piece = engine.sample_token(
                    req.get("bank", "Base"),
                    req.get("pen", "PayloadPen"),
                    req.get("phase", "PayloadBody"),
                    req.get("allowed_token_ids"),
                    req.get("grammar_spec"),
                )
                resp = {
                    "status": "sampled_token",
                    "token_id": int(tid),
                    "token_piece": str(piece),
                    "token_ids": list(engine._last_step_token_ids),
                }
            elif op == "consume_token":
                cid = int(req["token_id"])
                _, cap = engine._capture_layer_pre_mlp(
                    req.get("capture_layer"),
                    lambda: engine.consume_token(req.get("bank", "Base"), cid),
                )
                resp = {"status": "consumed_token", "token_ids": [cid], **({"captured_pre_mlp": cap} if cap is not None else {})}
            elif op == "step_token":
                (tid, piece), cap = engine._capture_layer_pre_mlp(
                    req.get("capture_layer"),
                    lambda: engine.step_token(
                        req.get("bank", "Base"),
                        req.get("pen", "PayloadPen"),
                        req.get("phase", "PayloadBody"),
                        req.get("allowed_token_ids"),
                        req.get("grammar_spec"),
                    ),
                )
                resp = {
                    "status": "token",
                    "token_id": int(tid),
                    "token_piece": str(piece),
                    "token_ids": list(engine._last_step_token_ids),
                    **({"captured_pre_mlp": cap} if cap is not None else {}),
                }
            elif op == "capture_activations":
                resp = {
                    "status": "activations",
                    **engine.capture_activations(
                        req.get("prompts"),
                        req["layers"],
                        bool(req.get("include_prefix_background", False)),
                        int(req.get("background_stride", 8)),
                        sequences=req.get("sequences"),
                    ),
                }
            elif op == "extract_compiler_weight_geometry":
                resp = {
                    "status": "weight_geometry",
                    **engine.extract_compiler_weight_geometry(
                        req.get("qk_attn_layers"),
                        req.get("deltanet_layers"),
                        target_layer=req.get("target_layer"),
                        token_ids=req.get("token_ids"),
                        protected_token_ids=req.get("protected_token_ids"),
                        qk_head_partitions=req.get("qk_head_partitions"),
                    ),
                }
            elif op == "capture_payload_activations":
                resp = {
                    "status": "payload_activations",
                    **engine.capture_payload_activations(
                        req["wire_prefix"],
                        req["payload"],
                        req["closing_tag"],
                        int(req["target_layer"]),
                    ),
                }
            elif op == "capture_and_generate_on_policy_plan":
                resp = {
                    "status": "on_policy_plan_captured",
                    **engine.capture_and_generate_on_policy_plan(
                        req["wire_prefix"],
                        req["oracle_mode"],
                        target_layer=req.get("target_layer"),
                        max_plan_tokens=int(req.get("max_plan_tokens", 512)),
                    ),
                }
            else:
                resp = {"status": "error", "message": f"Unknown op: {op!r}"}
        except Exception as exc:  # noqa: BLE001
            resp = {"status": "error", "message": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
