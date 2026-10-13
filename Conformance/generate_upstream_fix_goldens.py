#!/usr/bin/env python3
"""
Regenerate the goldens that pin swift-transformers fixes ported after
tag 1.3.2:

  - huggingface/swift-transformers#397  Honor `ignore_merges` in BPE
  - huggingface/swift-transformers#395  Match on code points in Replace,
                                         Metaspace (pre-tokenizer + decoder)
  - huggingface/swift-transformers#388  Unigram honors byte_fallback
  - huggingface/swift-transformers#390  BPE byte_fallback / unk edge cases

plus unknown-token fusion per word and T5's Precompiled normalizer, where
swift-transformers itself still differs from the Rust reference, and the
Llama 3 multilingual golden (the model is gated on the Hub).

Unlike regenerate_goldens.py this needs no network access and no
`transformers`: it loads the tokenizer.json files already bundled in
ObjCTokenizerTests/Resources/ with the Rust `tokenizers` library directly.

Output: ObjCTokenizerTests/Resources/<family>_upstream_fixes_golden.json,
a JSON list of {"text", "ids", "decoded"}, and
<family>_{byte_fallback,unk_fusion}_golden.json, lists of {"variant", "patch", "cases"}
with the same case shape (see BYTE_FALLBACK_VARIANTS), where
    ids     = encode(text, add_special_tokens=True).ids
    decoded = decode(ids, skip_special_tokens=True)

Usage:
    python -m venv .venv && .venv/bin/pip install tokenizers
    .venv/bin/python generate_upstream_fix_goldens.py
"""

import json
import unicodedata
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


# Byte-fallback variants (swift-transformers #388, #390). None of the
# bundled tokenizers exercises these paths as shipped, so each variant
# patches `model` in a copy of a bundled tokenizer.json. The patch is
# written into the golden so the Obj-C test applies the same one:
#   set                 model fields to overwrite
#   append_byte_pieces  [lo, hi]: append Unigram pieces <0xlo>..<0xhi>, score 0
#   remove_vocab        BPE vocab keys to delete
BYTE_FALLBACK_TEXTS = ["hello", "a\U0001F99Cb", "ꙮ x", "\U0001F99C\U0001F600 ꙮꙮ",
                       "日本語", "x 😀 y", "Привет", "naïve café",
                       # Rust merge_word does not flush a pending <unk> when a
                       # later char falls back to bytes: bytes come first.
                       "\U0001F99C\uA66E\U0001F99Cb", "\U0001F99C\U0001F600\uA66E", "x\U0001F99C\U0001F600"]

BYTE_FALLBACK_VARIANTS = [
    ("t5_small", "t5-small.tokenizer.json", [
        ("unigram_all_bytes", {"set": {"byte_fallback": True}, "append_byte_pieces": [0, 255]}),
        ("unigram_ascii_bytes", {"set": {"byte_fallback": True}, "append_byte_pieces": [0, 127]}),
        ("unigram_flag_only", {"set": {"byte_fallback": True}}),
    ]),
    ("llama_7b", "llama-7b.tokenizer.json", [
        ("bpe_missing_byte", {"remove_vocab": ["<0xF0>"]}),
        ("bpe_no_byte_fallback", {"set": {"byte_fallback": False}}),
        ("bpe_no_unk", {"set": {"byte_fallback": False, "unk_token": None}}),
        ("bpe_missing_byte_no_fuse", {"set": {"fuse_unk": False}, "remove_vocab": ["<0xF0>"]}),
    ]),
]


def patched(file, patch):
    raw = json.loads((RESOURCES / file).read_text())
    model = raw["model"]
    model.update(patch.get("set", {}))
    if "append_byte_pieces" in patch:
        lo, hi = patch["append_byte_pieces"]
        model["vocab"] += [[f"<0x{b:02X}>", 0.0] for b in range(lo, hi + 1)]
    for key in patch.get("remove_vocab", []):
        del model["vocab"][key]
    if "pre_tokenizer" in patch:
        raw["pre_tokenizer"] = patch["pre_tokenizer"]
    return Tokenizer.from_str(json.dumps(raw))


# Unknown-token fusion happens per pre-tokenized word (inside the model), not
# across words: with a pre-tokenizer that isolates every character, adjacent
# unknown characters stay separate <unk> tokens. Patch keys as above, plus
#   pre_tokenizer       replaces the top-level pre_tokenizer
SPLIT_EACH_CHAR = {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": False}
UNK_FUSION_TEXTS = ["\U0001F99C\U0001F600", "x\U0001F99C\U0001F600 \uA66E!", "\U0001F99C \U0001F600",
                    "\uA66E\uA66E", "a\U0001F99Cb\U0001F600c", "\U0001F99C\U0001F600\uA66E\uA66E\uA66E"]
UNK_FUSION_VARIANTS = [
    ("t5_small", "t5-small.tokenizer.json", [
        ("unigram_as_shipped", {}),
        ("unigram_split_each_char", {"pre_tokenizer": SPLIT_EACH_CHAR}),
    ]),
    ("llama_7b", "llama-7b.tokenizer.json", [
        ("bpe_fuse_unk", {"set": {"byte_fallback": False}}),
        ("bpe_fuse_unk_split_each_char", {"set": {"byte_fallback": False}, "pre_tokenizer": SPLIT_EACH_CHAR}),
        ("bpe_no_fuse_unk", {"set": {"byte_fallback": False, "fuse_unk": False}}),
    ]),
]


def write_precompiled_sweep():
    """T5's Precompiled normalizer over every code point.

    "range" cases normalize the 256 code points lo..hi-1 (surrogates
    skipped) concatenated; "expected" is null when the output is unchanged.
    "cps" cases put `prefix` before each listed code point, to exercise the
    whole-grapheme rule on base + combining-mark clusters.
    """
    norm = load("t5-small.tokenizer.json").normalizer
    cases = []
    for lo in range(0, 0x110000, 256):
        if 0xD800 <= lo < 0xE000:
            continue
        text = "".join(chr(c) for c in range(lo, lo + 256))
        out = norm.normalize_str(text)
        cases.append({"range": [lo, lo + 256], "expected": None if out == text else out})
    marks = [c for c in range(0x110000)
             if not 0xD800 <= c < 0xE000 and unicodedata.category(chr(c)) in ("Mn", "Mc", "Me")]
    marks += [0x200C, 0x200D, 0xFE0F, 0x20E3, 0x1F3FB]
    for prefix in ["\u2581", " ", "a", "\u3042", "\r"]:
        for i in range(0, len(marks), 64):
            cps = marks[i:i + 64]
            text = "".join(prefix + chr(c) for c in cps)
            cases.append({"prefix": prefix, "cps": cps, "expected": norm.normalize_str(text)})
    path = RESOURCES / "t5_small_precompiled_golden.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{path.name}: {len(cases)} cases")


def write_variants(suffix, families, texts):
    for family, file, variants in families:
        out = [{"variant": name, "patch": patch, "cases": golden(patched(file, patch), texts)}
               for name, patch in variants]
        path = RESOURCES / f"{family}_{suffix}_golden.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(f"{path.name}: {len(out)} variants")


def write_llama3_multilingual():
    """llama3_multilingual_golden.json for OCTMultilingualGoldenTests.

    Same {text, ids} shape as the other *_multilingual_golden.json files,
    over Conformance/corpus_multilingual.jsonl. Meta-Llama-3 is gated on the
    Hub, so this uses the bundled tokenizer.json; Rust `tokenizers`
    reproduces the existing transformers-generated goldens for the other
    families exactly.
    """
    tok = load("llama3.tokenizer.json")
    texts = [json.loads(line)["text"]
             for line in (ROOT / "Conformance" / "corpus_multilingual.jsonl").read_text().splitlines() if line]
    out = [{"text": t, "ids": tok.encode(t, add_special_tokens=True).ids} for t in texts]
    path = RESOURCES / "llama3_multilingual_golden.json"
    path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"{path.name}: {len(out)} records")


def main():
    write_variants("byte_fallback", BYTE_FALLBACK_VARIANTS, BYTE_FALLBACK_TEXTS)
    write_variants("unk_fusion", UNK_FUSION_VARIANTS, UNK_FUSION_TEXTS)

    llama3_path = RESOURCES / "llama3.tokenizer.json"
    words = llama3_ignore_merges_texts(llama3_path)
    write("llama3", golden(load(llama3_path.name), LLAMA3_SENTENCES + words))

    for family, file in [
        ("llama_7b", "llama-7b.tokenizer.json"),
        ("tinyllama", "tinyllama-1.1b.tokenizer.json"),
        ("gemma", "gemma-4-31b.tokenizer.json"),
    ]:
        write(family, golden(load(file), COMBINING_MARK_TEXTS))

    write("t5_small", golden(load("t5-small.tokenizer.json"), COMBINING_MARK_TEXTS))
    write_precompiled_sweep()
    write_llama3_multilingual()


if __name__ == "__main__":
    main()
