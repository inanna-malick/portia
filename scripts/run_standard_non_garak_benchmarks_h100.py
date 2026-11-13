#!/usr/bin/env python3
"""
run_standard_non_garak_benchmarks_h100.py — Off-the-Shelf Public Capability & Collateral-Damage Suite
Comparing Stock `Qwen/Qwen3.5-0.8B` vs. Static-Widened `Qwen/Qwen3.5-0.8B` (`M + 4` Garak Layer-22 Surgery, 0 Hooks):

1. `WikiText-2` (`Salesforce/wikitext`, `wikitext-2-raw-v1` test):
   - Exact token-level Perplexity (`PPL`), forward `KL(P_stock || P_patched)`, AND directional
     `KL(P_patched || P_stock)` across ~30,000 tokens via bounded 32-token vocab chunks
     (direct measurement of whether the `-8.0` reverse-biased SiLU diodes stay silent on general prose).
2. `BoolQ` (`google/boolq` validation, `n = 400`, balanced `200 True ("Yes") + 200 False ("No")`):
   - ChatML generative (`Yes`/`No` answer extraction), broken down separately on **True (`Yes`)** vs. **False (`No`)**.
3. `MMLU` (`cais/mmlu`, `all` test, stratified `n = 570` — 10 questions × all 57 subjects):
   - Standard 4-way (`A/B/C/D`) next-token accuracy.
4. `ARC-Challenge` (`allenai/ai2_arc`, `ARC-Challenge` test, `n = 400`):
   - Standard 4-way (`A/B/C/D`) next-token accuracy.
5. `GSM8K` (`openai/gsm8k`, `main` test, `n = 200`):
   - Generative chain-of-thought exact numeric match (`#### <number>` / last number).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import portia_garak_posttrain_surgery as garak_mod
from portia_qwen_stdio_worker import LiveQwen35Engine

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUT_PATH = ROOT / "portia_runs" / "garak_public_spotchecks_08b.json"


@torch.no_grad()
def eval_wikitext2_ppl_and_kl_chunks(
    base_model: Any,
    patched_model: Any,
    wt2_enc: torch.Tensor,
    max_chunks: int = 60,
    seq_len: int = 512,
    batch_size: int = 2,
    token_chunk_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Bounded 32-token chunked WikiText-2 PPL & bidirectional KL without full-vocab GPU OOM."""
    if seq_len < 2 or batch_size < 1 or token_chunk_size < 1:
        raise ValueError("seq_len must be >= 2 and batch/chunk sizes must be positive")
    n_chunks = min(max_chunks, wt2_enc.numel() // seq_len)
    if n_chunks < 1:
        raise ValueError("WikiText evaluation needs at least one complete sequence")
    chunks = wt2_enc[: n_chunks * seq_len].view(n_chunks, seq_len)

    base_nll_sum = 0.0
    patch_nll_sum = 0.0
    all_kls: list[torch.Tensor] = []
    all_patch_to_stock_kls: list[torch.Tensor] = []
    total_pred_tokens = 0

    for i in range(0, n_chunks, batch_size):
        batch = chunks[i : i + batch_size].to(device)
        logits_b = base_model(input_ids=batch, use_cache=False).logits.detach().cpu()
        logits_p = patched_model(input_ids=batch, use_cache=False).logits
        for row in range(batch.shape[0]):
            for start in range(0, seq_len - 1, token_chunk_size):
                end = min(start + token_chunk_size, seq_len - 1)
                targets = batch[row, start + 1 : end + 1]
                chunk_b = logits_b[row, start:end].to(device=device, dtype=torch.float32)
                logp_b = F.log_softmax(chunk_b, dim=-1)
                del chunk_b
                chunk_p = logits_p[row, start:end].float()
                logp_p = F.log_softmax(chunk_p, dim=-1)
                del chunk_p
                base_nll_sum += float(F.nll_loss(logp_b, targets, reduction="sum").item())
                patch_nll_sum += float(F.nll_loss(logp_p, targets, reduction="sum").item())
                token_kl = F.kl_div(logp_p, logp_b, log_target=True, reduction="none").sum(dim=-1).cpu()
                all_kls.append(token_kl)
                patch_to_stock_kl = F.kl_div(logp_b, logp_p, log_target=True, reduction="none").sum(dim=-1).cpu()
                all_patch_to_stock_kls.append(patch_to_stock_kl)
                total_pred_tokens += int(targets.numel())
                del logp_b, logp_p, token_kl, patch_to_stock_kl
        del logits_b, logits_p, batch

    kl_flat = torch.cat(all_kls, dim=0)
    patch_to_stock = torch.cat(all_patch_to_stock_kls, dim=0)
    ppl_base = math.exp(base_nll_sum / total_pred_tokens)
    ppl_patch = math.exp(patch_nll_sum / total_pred_tokens)
    mean_kl = float(kl_flat.mean().item())
    p99_kl = float(torch.quantile(kl_flat, 0.99).item())
    max_kl = float(kl_flat.max().item())
    pct_above_001 = float((kl_flat > 0.01).float().mean().item() * 100.0)
    return {
        "tokens_evaluated": total_pred_tokens,
        "stock_perplexity": round(ppl_base, 4),
        "patched_perplexity": round(ppl_patch, 4),
        "delta_perplexity": round(ppl_patch - ppl_base, 6),
        "mean_token_kl_nats": round(mean_kl, 8),
        "p99_token_kl_nats": round(p99_kl, 8),
        "max_token_kl_nats": round(max_kl, 6),
        "pct_tokens_kl_gt_0_01": round(pct_above_001, 4),
        "mean_token_kl_patched_to_stock_nats": float(patch_to_stock.mean()),
        "max_token_kl_patched_to_stock_nats": float(patch_to_stock.max()),
        "patched_to_stock_kl_limit": 0.02,
        "patched_to_stock_kl_passed": bool(torch.isfinite(patch_to_stock).all() and patch_to_stock.max() <= 0.02),
    }


@torch.no_grad()
def eval_wikitext2_ppl_and_kl(
    base_model: Any,
    patched_model: Any,
    tok: Any,
    max_chunks: int = 60,
    seq_len: int = 512,
    batch_size: int = 2,
    token_chunk_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in ds["text"] if t.strip())
    wt2_enc = tok(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return eval_wikitext2_ppl_and_kl_chunks(
        base_model,
        patched_model,
        wt2_enc,
        max_chunks=max_chunks,
        seq_len=seq_len,
        batch_size=batch_size,
        token_chunk_size=token_chunk_size,
        device=device,
    )


def parse_boolq_yes_no(text: str) -> bool | None:
    s = re.sub(r"^[\*\s`#]+", "", text.strip()).lower()
    m = re.search(r"\b(yes|no|true|false)\b", s[:120])
    if not m:
        return None
    w = m.group(1)
    return True if w in ("yes", "true") else False


@torch.no_grad()
def eval_boolq_balanced(model: Any, tok: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [
        f"Passage: {it['passage']}\nQuestion: {it['question']}?\nAnswer Yes or No first, then briefly explain."
        for it in items
    ]
    gens = garak_mod.generate_split_outputs(model, tok, prompts, max_new_tokens=32, batch_size=48)
    true_correct = 0
    true_total = 0
    false_correct = 0
    false_total = 0
    unparsed = 0
    for it, out in zip(items, gens):
        gold = bool(it["answer"])
        pred = parse_boolq_yes_no(out)
        if pred is None:
            unparsed += 1
        if gold:
            true_total += 1
            if pred is True:
                true_correct += 1
        else:
            false_total += 1
            if pred is False:
                false_correct += 1
    tot = true_total + false_total
    corr = true_correct + false_correct
    return {
        "n": tot,
        "overall_acc_pct": round(100.0 * corr / max(1, tot), 2),
        "true_yes_acc_pct": round(100.0 * true_correct / max(1, true_total), 2),
        "false_no_acc_pct": round(100.0 * false_correct / max(1, false_total), 2),
        "unparsed_count": unparsed,
    }


@torch.no_grad()
def eval_abcd_next_token(model: Any, tok: Any, prompts: list[str], gold_indices: list[int], batch_size: int = 64) -> float:
    letter_tids = [int(tok.encode(ch, add_special_tokens=False)[0]) for ch in ("A", "B", "C", "D")]
    sp_letter_tids = [int(tok.encode(" " + ch, add_special_tokens=False)[-1]) for ch in ("A", "B", "C", "D")]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    correct = 0
    for s in range(0, len(prompts), batch_size):
        batch_p = [garak_mod.format_qwen_chat(p) + "Answer:" for p in prompts[s : s + batch_size]]
        golds = gold_indices[s : s + batch_size]
        inp = tok(batch_p, return_tensors="pt", padding=True).to("cuda")
        logits = model(**inp, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
        for row_i, g in enumerate(golds):
            scores = [
                max(float(logits[row_i, letter_tids[k]].item()), float(logits[row_i, sp_letter_tids[k]].item()))
                for k in range(4)
            ]
            pred = max(range(4), key=lambda k: scores[k])
            if pred == g:
                correct += 1
    return round(100.0 * correct / max(1, len(prompts)), 2)


def load_mmlu_stratified(per_subject: int = 10) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    counts: dict[str, int] = {}
    prompts: list[str] = []
    golds: list[int] = []
    letters = ("A", "B", "C", "D")
    for row in ds:
        subj = row["subject"]
        if counts.get(subj, 0) >= per_subject:
            continue
        counts[subj] = counts.get(subj, 0) + 1
        opts = "\n".join(f"{letters[i]}. {row['choices'][i]}" for i in range(4))
        p = f"Subject: {subj.replace('_', ' ')}\nQuestion: {row['question']}\n{opts}\nReply with only the correct option letter (A, B, C, or D)."
        prompts.append(p)
        golds.append(int(row["answer"]))
    return prompts, golds


def load_arc_challenge(limit: int = 400) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    prompts: list[str] = []
    golds: list[int] = []
    letters = ("A", "B", "C", "D")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    for row in ds:
        choices = row["choices"]["text"]
        if len(choices) != 4 or row["answerKey"] not in label_map:
            continue
        opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
        p = f"Question: {row['question']}\n{opts}\nReply with only the correct option letter (A, B, C, or D)."
        prompts.append(p)
        golds.append(label_map[row["answerKey"]])
        if len(prompts) >= limit:
            break
    return prompts, golds


def extract_last_number(text: str) -> str | None:
    if "####" in text:
        after = text.split("####")[-1]
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", after)
        if m:
            return m.group(0).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


@torch.no_grad()
def eval_gsm8k(model: Any, tok: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [
        f"Solve this math problem step by step and end your response with `#### <final_number>`.\n\nQuestion: {r['question']}"
        for r in rows
    ]
    gens = garak_mod.generate_split_outputs(model, tok, prompts, max_new_tokens=256, batch_size=40)
    correct = 0
    for r, g in zip(rows, gens):
        gold = extract_last_number(r["answer"])
        pred = extract_last_number(g)
        if gold is not None and pred is not None:
            try:
                if abs(float(gold) - float(pred)) < 1e-6:
                    correct += 1
            except ValueError:
                pass
    return {
        "n": len(rows),
        "exact_match_pct": round(100.0 * correct / max(1, len(rows)), 2),
        "correct": correct,
    }


def resolve_state_cache_path(model_id: str, cache_dir: Path | None = None) -> Path:
    base_dir = cache_dir if cache_dir is not None else garak_mod.default_state_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    short_name = model_id.split("/")[-1]
    return base_dir / f"cached_states_{short_name}.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cartridge-json", type=Path, default=None, help="Optional precompiled Garak cartridge JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resolved = LiveQwen35Engine._resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(resolved)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"Loading stock {args.model} and preparing M -> M + 4 Garak static widened model...")
    stock_model = AutoModelForCausalLM.from_pretrained(resolved, dtype=torch.bfloat16, device_map="cuda").eval()
    patched_model = AutoModelForCausalLM.from_pretrained(resolved, dtype=torch.bfloat16, device_map="cuda").eval()

    layers, _, _ = garak_mod.get_layers_and_head(patched_model)
    target_layer = len(layers) - 2
    if args.cartridge_json is not None:
        manifest = json.loads(args.cartridge_json.read_text(encoding="utf-8"))
        garak_mod.mount_permanent_static_garak_slots_in_place(patched_model, manifest, target_layer)
        garak_receipt: dict[str, Any] = {"cartridge_json": str(args.cartridge_json)}
    else:
        bundle = json.loads((ROOT / "artifacts" / "garak_splits_bundle.json").read_text(encoding="utf-8"))
        cache_path = resolve_state_cache_path(args.model, args.cache_dir)
        states = garak_mod.capture_states_for_model(patched_model, tok, bundle, target_layer, cache_path=cache_path)
        garak_receipt = garak_mod.compile_and_widen_portia_slots(
            patched_model, tok, states, bundle, target_layer, model_id=args.model
        )
    for m in patched_model.modules():
        assert len(m._forward_hooks) == 0

    print("\n[1/5] Evaluating WikiText-2 Perplexity & Bounded 32-Token Chunked KL...")
    wt2 = eval_wikitext2_ppl_and_kl(stock_model, patched_model, tok)
    print("  WikiText-2:", wt2)

    print("\n[2/5] Evaluating BoolQ (200 True='Yes' + 200 False='No' balanced adversarial check)...")
    from datasets import load_dataset

    boolq_ds = load_dataset("google/boolq", split="validation")
    trues = [r for r in boolq_ds if bool(r["answer"]) is True][:200]
    falses = [r for r in boolq_ds if bool(r["answer"]) is False][:200]
    boolq_items = trues + falses
    boolq_stock = eval_boolq_balanced(stock_model, tok, boolq_items)
    boolq_patch = eval_boolq_balanced(patched_model, tok, boolq_items)
    print("  BoolQ Stock  :", boolq_stock)
    print("  BoolQ Patched:", boolq_patch)

    print("\n[3/5] Evaluating MMLU (570 questions = 10 × all 57 subjects)...")
    mmlu_prompts, mmlu_golds = load_mmlu_stratified(per_subject=10)
    mmlu_stock = eval_abcd_next_token(stock_model, tok, mmlu_prompts, mmlu_golds)
    mmlu_patch = eval_abcd_next_token(patched_model, tok, mmlu_prompts, mmlu_golds)
    print(f"  MMLU (n={len(mmlu_prompts)}): Stock={mmlu_stock}% | Patched={mmlu_patch}% (delta={mmlu_patch - mmlu_stock:+.2f}%)")

    print("\n[4/5] Evaluating ARC-Challenge (n=400 test)...")
    arc_prompts, arc_golds = load_arc_challenge(limit=400)
    arc_stock = eval_abcd_next_token(stock_model, tok, arc_prompts, arc_golds)
    arc_patch = eval_abcd_next_token(patched_model, tok, arc_prompts, arc_golds)
    print(f"  ARC-Challenge (n={len(arc_prompts)}): Stock={arc_stock}% | Patched={arc_patch}% (delta={arc_patch - arc_stock:+.2f}%)")

    print("\n[5/5] Evaluating GSM8K Generative Chain-of-Thought Math (n=200 test)...")
    gsm_ds = list(load_dataset("openai/gsm8k", "main", split="test"))[:200]
    gsm_stock = eval_gsm8k(stock_model, tok, gsm_ds)
    gsm_patch = eval_gsm8k(patched_model, tok, gsm_ds)
    print("  GSM8K Stock  :", gsm_stock)
    print("  GSM8K Patched:", gsm_patch)

    results = {
        "model_id": args.model,
        "garak_compiler_receipt": garak_receipt,
        "wikitext2_ppl_and_kl": wt2,
        "boolq_400_balanced": {
            "stock": boolq_stock,
            "patched": boolq_patch,
            "delta_overall_pct": round(boolq_patch["overall_acc_pct"] - boolq_stock["overall_acc_pct"], 2),
            "delta_true_yes_pct": round(boolq_patch["true_yes_acc_pct"] - boolq_stock["true_yes_acc_pct"], 2),
            "delta_false_no_pct": round(boolq_patch["false_no_acc_pct"] - boolq_stock["false_no_acc_pct"], 2),
        },
        "mmlu_570_stratified": {
            "n": len(mmlu_prompts),
            "stock_acc_pct": mmlu_stock,
            "patched_acc_pct": mmlu_patch,
            "delta_pct": round(mmlu_patch - mmlu_stock, 2),
        },
        "arc_challenge_400": {
            "n": len(arc_prompts),
            "stock_acc_pct": arc_stock,
            "patched_acc_pct": arc_patch,
            "delta_pct": round(arc_patch - arc_stock, 2),
        },
        "gsm8k_200_cot": {
            "stock": gsm_stock,
            "patched": gsm_patch,
            "delta_pct": round(gsm_patch["exact_match_pct"] - gsm_stock["exact_match_pct"], 2),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved public non-Garak benchmark suite receipt to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
