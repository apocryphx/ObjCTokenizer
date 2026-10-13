#!/usr/bin/env python3
"""
Regenerate the goldens that pin two swift-transformers fixes ported after
tag 1.3.2:

  - huggingface/swift-transformers#397  Honor `ignore_merges` in BPE
  - huggingface/swift-transformers#395  Match on code points in Replace,
                                         Metaspace (pre-tokenizer + decoder)

Unlike regenerate_goldens.py this needs no network access and no
`transformers`: it loads the tokenizer.json files already bundled in
ObjCTokenizerTests/Resources/ with the Rust `tokenizers` library directly.

Output: ObjCTokenizerTests/Resources/<family>_upstream_fixes_golden.json,
a JSON list of {"text", "ids", "decoded"} where
    ids     = encode(text, add_special_tokens=True).ids
    decoded = decode(ids, skip_special_tokens=True)

Usage:
    python -m venv .venv && .venv/bin/pip install tokenizers
    .venv/bin/python generate_upstream_fix_goldens.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "ObjCTokenizerTests" / "Resources"

# Space (or U+2581) immediately followed by a combining mark: NSString's
# default (non-literal) search treats the pair as one composed character
# sequence and will not match the bare space inside it.
COMBINING_MARK_TEXTS = [
    "hello world",
    "a ́b",
    " ́",
    "́",
    "x  ́y",
    "café au lait",
    "e ̈ o",
    "word ́",
    "▁́x",
    "x ▁́",
    "▁",
    "▁▁a",
    "a▁b ▁c",
    "नमस्ते  ्क",
    "ก ัก",
    "1 ️⃣",
    "ש ָל",
    "a ́ ́ ́ b",
]

# Sentences around words that Llama 3's merges cannot rebuild (the
# vocab-entry sweep below covers each word in isolation).
LLAMA3_SENTENCES = [
    "Türkiye'de 85 milyon insan yaşıyor.",
    "Ayrıca toplumsal bir sorun olarak görülüyor, özellikle gençler arasında.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Например, в октябре качество воздуха было хорошим.",
    "Smlouva obsahuje důležité informace, obvykle v příloze.",
    "Hello world, nothing special here.",
]


def load(name):
    return Tokenizer.from_file(str(RESOURCES / name))


def golden(tok, texts):
    out = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True).ids
        out.append({"text": text, "ids": ids,
                    "decoded": tok.decode(ids, skip_special_tokens=True)})
    return out


def llama3_ignore_merges_texts(path):
    """Every vocab entry whose ids change when ignore_merges is turned off."""
    raw = json.loads(path.read_text())
    on = Tokenizer.from_str(json.dumps(raw))
    raw["model"]["ignore_merges"] = False
    off = Tokenizer.from_str(json.dumps(raw))
    base = len(raw["model"]["vocab"])
    texts = [on.decode([i]) for i in range(base)]
    a = on.encode_batch(texts, add_special_tokens=False)
    b = off.encode_batch(texts, add_special_tokens=False)
    return [t for t, x, y in zip(texts, a, b) if x.ids != y.ids]


def write(name, entries):
    path = RESOURCES / f"{name}_upstream_fixes_golden.json"
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n")
    print(f"{path.name}: {len(entries)} records")


def main():
    llama3_path = RESOURCES / "llama3.tokenizer.json"
    words = llama3_ignore_merges_texts(llama3_path)
    write("llama3", golden(load(llama3_path.name), LLAMA3_SENTENCES + words))

    for family, file in [
        ("llama_7b", "llama-7b.tokenizer.json"),
        ("tinyllama", "tinyllama-1.1b.tokenizer.json"),
        ("gemma", "gemma-4-31b.tokenizer.json"),
    ]:
        write(family, golden(load(file), COMBINING_MARK_TEXTS))

    # T5's Precompiled charsmap rewrites "▁́" to a single space as
    # one entry; OCTPrecompiledNormalizer (like swift-transformers) only
    # approximates the charsmap scalar by scalar. Out of scope for #395.
    t5_texts = [t for t in COMBINING_MARK_TEXTS if "▁́" not in t]
    write("t5_small", golden(load("t5-small.tokenizer.json"), t5_texts))


if __name__ == "__main__":
    main()
