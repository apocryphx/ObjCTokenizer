#!/usr/bin/env python3
"""
Regenerate the {text, ids} goldens in ObjCTokenizerTests/Resources/ that the
older byte-identity test classes read:

    <family>_golden.json               corpus.txt (585 lines)
                                       -> OCTGoldenCorpusTests
    <family>_lineending_golden.json    corpus_lineending.json (18 texts)
                                       -> OCTLineEndingGoldenTests
    <family>_multilingual_golden.json  corpus_multilingual.jsonl (83 lines)
                                       -> OCTMultilingualGoldenTests

Each file is a JSON list of {"text", "ids"}, ids = encode(text,
add_special_tokens=True).ids. These were first produced by a since-removed
`make golden` pipeline using `transformers.AutoTokenizer`; the Rust
`tokenizers` library reproduces every one of them byte for byte from the
bundled tokenizer.json files, so no network access or Hub login is needed
(Meta-Llama-3 is gated on the Hub).

Usage:
    python -m venv .venv && .venv/bin/pip install tokenizers
    .venv/bin/python regenerate_bundled_goldens.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE = ROOT / "Conformance"
RESOURCES = ROOT / "ObjCTokenizerTests" / "Resources"

FAMILIES = {
    "bge_small": "bge-small-en-v1.5.tokenizer.json",
    "gpt2": "gpt2.tokenizer.json",
    "t5_small": "t5-small.tokenizer.json",
    "llama_7b": "llama-7b.tokenizer.json",
    "roberta_base": "roberta-base.tokenizer.json",
    "llama3": "llama3.tokenizer.json",
}

CORPORA = {
    # suffix: (texts, families)
    "golden": (
        [line for line in (RESOURCES / "corpus.txt").read_text().split("\n") if line],
        ["bge_small", "gpt2", "t5_small", "llama_7b", "roberta_base"],
    ),
    "lineending_golden": (
        json.loads((CONFORMANCE / "corpus_lineending.json").read_text()),
        ["bge_small", "gpt2", "t5_small", "llama_7b", "roberta_base"],
    ),
    "multilingual_golden": (
        [json.loads(line)["text"]
         for line in (CONFORMANCE / "corpus_multilingual.jsonl").read_text().splitlines() if line],
        ["bge_small", "gpt2", "t5_small", "llama_7b", "roberta_base", "llama3"],
    ),
}


def main():
    for suffix, (texts, families) in CORPORA.items():
        for family in families:
            tok = Tokenizer.from_file(str(RESOURCES / FAMILIES[family]))
            out = [{"text": t, "ids": tok.encode(t, add_special_tokens=True).ids} for t in texts]
            path = RESOURCES / f"{family}_{suffix}.json"
            path.write_text(json.dumps(out, ensure_ascii=False))
            print(f"{path.name}: {len(out)} records")


if __name__ == "__main__":
    main()
