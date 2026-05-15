#!/usr/bin/env python3
"""
Regenerate per-family multilingual golden JSONs from corpus_multilingual.jsonl
against HuggingFace `transformers.AutoTokenizer` references.

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install 'transformers==4.57.1' 'tokenizers>=0.20'
    python regenerate_goldens.py            # all 5 families
    python regenerate_goldens.py t5_small   # one family

Output is written to ./goldens/<family>_multilingual.json. Each record is
{"text": "...", "ids": [int, ...]} — the same shape this directory ships.

Pinning `transformers==4.57.1` is intentional: that's the reference the
swift-transformers conformance tests were calibrated against. Newer or older
versions may produce subtly different IDs, especially around the WordPiece
NFD step that landed in `transformers` v5.
"""

import json
import os
import sys

FAMILIES = {
    "bge_small":    ("BAAI/bge-small-en-v1.5",   True),
    "gpt2":         ("gpt2",                     True),
    "t5_small":     ("google-t5/t5-small",       True),
    "llama_7b":     ("huggyllama/llama-7b",      True),
    "roberta_base": ("FacebookAI/roberta-base",  True),
}

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus_multilingual.jsonl")
OUTDIR = os.path.join(ROOT, "goldens")


def load_corpus():
    with open(CORPUS) as f:
        return [json.loads(line)["text"] for line in f]


def regenerate(family: str, model_id: str, add_special_tokens: bool):
    from transformers import AutoTokenizer
    print(f"[{family}] loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    texts = load_corpus()
    out = []
    for text in texts:
        ids = tok(text, add_special_tokens=add_special_tokens)["input_ids"]
        out.append({"text": text, "ids": ids})
    out_path = os.path.join(OUTDIR, f"{family}_multilingual.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[{family}] wrote {out_path} ({len(out)} cases)")


def main():
    targets = sys.argv[1:] or sorted(FAMILIES.keys())
    os.makedirs(OUTDIR, exist_ok=True)
    for family in targets:
        if family not in FAMILIES:
            print(f"unknown family: {family} (known: {sorted(FAMILIES)})", file=sys.stderr)
            sys.exit(2)
        model_id, ast = FAMILIES[family]
        regenerate(family, model_id, ast)


if __name__ == "__main__":
    main()
