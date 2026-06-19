#!/usr/bin/env python3
"""
Regenerate per-kernel multilingual baselines from `transformers.AutoTokenizer`
for the ObjCTokenizer conformance test target. The Python reference is the
authoritative source of truth; the Obj-C port must produce byte-identical
output to it.

Each baseline file is keyed by the stable input id from `inputs.json` and
holds:
    - input_ids:           [int]    canonical token-id sequence (the parity target)
    - tokens:              [str]    convert_ids_to_tokens(input_ids) — diagnostic
    - decoded_with_special:   str   tokenizer.decode(input_ids)
    - decoded_skip_special:   str   tokenizer.decode(input_ids, skip_special_tokens=True)

Top-level shape:
    {
        "metadata": {
            "model_id":             str,
            "transformers_version": str,
            "generated_at":         str   (ISO-8601 UTC),
            "input_count":          int
        },
        "entries": [
            {
                "id":                    str,
                "input_ids":             [int],
                "tokens":                [str],
                "decoded_with_special":  str,
                "decoded_skip_special":  str
            },
            ...
        ]
    }

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python regenerate_goldens.py              # all kernels
    python regenerate_goldens.py bge_small    # one kernel

Kernel matrix is intentionally larger than the swift-transformers companion:
adds Llama 3 (byte-level BPE + Split + pre-tokenized regex), plus Qwen2.5
and TinyLlama from the swift-transformers PR. Drop a kernel by deleting its
row from FAMILIES; add one with a single-line append + rerun.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# (kernel_slug, hf_model_id) — kernel matrix.
#
# Covers every tokenizer kernel + post-processor surface ObjCTokenizer supports:
#   - bge_small     WordPiece (Bert family)
#   - t5_small      Unigram (SentencePiece)
#   - gpt2          Byte-level BPE
#   - roberta_base  Byte-level BPE + RobertaProcessing
#   - llama_7b      SentencePiece BPE + byte-fallback (legacy Llama 1/2)
#   - llama3        Byte-level BPE + Split + tiktoken-style pre-tokenize regex
#   - qwen2_5       Byte-level BPE (modern vocab/merges)
#   - tinyllama     SentencePiece BPE + byte-fallback (Llama family, no auth gate)
#   - gemma         BPE + Replace-normalizer + Metaspace (Gemma SentencePiece, 262k vocab)
FAMILIES = [
    ("bge_small",     "BAAI/bge-small-en-v1.5"),
    ("t5_small",      "google-t5/t5-small"),
    ("gpt2",          "openai-community/gpt2"),
    ("roberta_base",  "FacebookAI/roberta-base"),
    ("llama_7b",      "huggyllama/llama-7b"),
    ("qwen2_5",       "Qwen/Qwen2.5-0.5B"),
    ("tinyllama",     "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    # Gemma 4 31B — the Apertura inference target. Gated repo, but the same
    # tokenizer ships across every Gemma 4 size; log into the Hub
    # (`huggingface-cli login`) or have the snapshot cached to regenerate.
    ("gemma",         "google/gemma-4-31b-it"),
    # Llama 3 (`meta-llama/Meta-Llama-3-8B`) is an auth-gated repo, so it's
    # excluded from the default matrix. To regenerate against it, log into
    # the Hub (`huggingface-cli login`) and re-add the row.
]

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs.json"
OUTDIR = ROOT / "goldens"


def load_inputs():
    with open(INPUTS) as f:
        return json.load(f)


# Families that must load via the raw `tokenizer.json` path because their
# `tokenizer_config.json` is newer than the pinned transformers, AND whose
# model-specific wrapper injects special tokens the shipped tokenizer.json's
# post-processor omits. Each spec mirrors the wrapper's documented defaults so
# the raw fast tokenizer reproduces `AutoTokenizer` exactly.
#
#   gemma: `GemmaTokenizerFast` defaults add_bos_token=True / add_eos_token=False.
#          Gemma 4's shipped tokenizer.json post-processor is a no-op (empty
#          `special_tokens`) — the wrapper rewrites it at load time to prepend
#          <bos>. Without this, the golden would omit the leading <bos> that
#          every real Gemma forward pass sees. (<bos>=2, <eos>=1 in the 262k vocab.)
RAW_LOAD_WRAPPER = {
    "gemma": {"add_bos": True, "add_eos": False, "bos": "<bos>", "eos": "<eos>"},
}


def _load_fast_from_tokenizer_json(slug: str, model_id: str):
    """Load the fast tokenizer straight from `tokenizer.json`, bypassing the
    model-specific `AutoTokenizer` wrapper.

    Needed when a model ships a `tokenizer_config.json` newer than the pinned
    `transformers` (e.g. Gemma 4 against 4.57.1: its config carries an
    `extra_special_tokens` list the older `GemmaTokenizerFast` mis-parses).

    The shipped `tokenizer.json` carries no post-processor that adds special
    tokens — the wrapper class does that at load time from `add_bos_token` /
    `add_eos_token`. For families in `RAW_LOAD_WRAPPER` we reconstruct that
    post-processor here so the reference matches a real forward pass, and we
    write the *resolved* tokenizer.json (post-processor baked in) beside the
    golden. That resolved file — not the upstream one — is what the Obj-C port
    must bundle, exactly as the bundled `llama-7b.tokenizer.json` bakes in <s>.
    """
    from transformers import PreTrainedTokenizerFast
    from tokenizers.processors import TemplateProcessing
    from huggingface_hub import hf_hub_download

    offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    tok_json = hf_hub_download(
        model_id, "tokenizer.json", local_files_only=bool(offline)
    )
    print(f"[{slug}] AutoTokenizer unavailable — loading tokenizer.json directly "
          f"({tok_json})", flush=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)

    spec = RAW_LOAD_WRAPPER.get(slug)
    if spec:
        bos, eos = spec["bos"], spec["eos"]
        add_bos, add_eos = spec["add_bos"], spec["add_eos"]
        bos_id = tok.convert_tokens_to_ids(bos)
        eos_id = tok.convert_tokens_to_ids(eos)
        # Mirror GemmaTokenizerFast.update_post_processor() string templates.
        single = (f"{bos}:0 " if add_bos else "") + "$A:0" + (f" {eos}:0" if add_eos else "")
        pair = single + (f" {bos}:0" if add_bos else "") + " $B:0" + (f" {eos}:0" if add_eos else "")
        special = []
        if add_bos:
            special.append((bos, bos_id))
        if add_eos:
            special.append((eos, eos_id))
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single=single, pair=pair, special_tokens=special
        )
        OUTDIR.mkdir(parents=True, exist_ok=True)
        resolved = OUTDIR / f"{slug}.tokenizer.json"
        tok.backend_tokenizer.save(str(resolved))
        print(f"[{slug}] wrote resolved tokenizer.json (post-processor baked in) "
              f"-> {resolved}", flush=True)

    return tok


def regenerate(slug: str, model_id: str) -> None:
    from transformers import AutoTokenizer, __version__ as transformers_version

    print(f"[{slug}] loading {model_id} …", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:  # model config newer than pinned transformers
        print(f"[{slug}] AutoTokenizer.from_pretrained failed: {exc}", flush=True)
        tok = _load_fast_from_tokenizer_json(slug, model_id)

    # Parity contract is against the fast (Rust) tokenizer; slow-only models
    # would produce a non-comparable reference.
    if not getattr(tok, "is_fast", False):
        raise SystemExit(
            f"[{slug}] {model_id} is not a fast tokenizer — parity with the "
            "Obj-C port would be undefined. Pick a model that ships tokenizer.json."
        )

    inputs = load_inputs()
    entries = []
    for entry in inputs:
        text = entry["text"]
        ids = tok(text, add_special_tokens=True)["input_ids"]
        tokens = tok.convert_ids_to_tokens(ids)
        decoded_with    = tok.decode(ids, skip_special_tokens=False)
        decoded_without = tok.decode(ids, skip_special_tokens=True)
        entries.append({
            "id":                   entry["id"],
            "input_ids":            ids,
            "tokens":               tokens,
            "decoded_with_special": decoded_with,
            "decoded_skip_special": decoded_without,
        })

    payload = {
        "metadata": {
            "model_id":             model_id,
            "transformers_version": transformers_version,
            "generated_at":         datetime.datetime.now(datetime.timezone.utc)
                                        .replace(microsecond=0).isoformat(),
            "input_count":          len(entries),
        },
        "entries": entries,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{slug}_multilingual.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[{slug}] wrote {out_path} ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kernels",
        nargs="*",
        help="Subset of kernel slugs to regenerate (default: all).",
    )
    args = parser.parse_args()

    known = {slug: model_id for slug, model_id in FAMILIES}
    targets = args.kernels or list(known.keys())
    for slug in targets:
        if slug not in known:
            print(f"unknown kernel slug: {slug} (known: {sorted(known)})", file=sys.stderr)
            sys.exit(2)
    for slug in targets:
        regenerate(slug, known[slug])


if __name__ == "__main__":
    main()
