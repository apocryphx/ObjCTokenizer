# Multilingual conformance corpus for HuggingFace tokenizer ports

83-line multilingual corpus + per-family golden tokenizations against
`transformers==4.57.1`. Used to verify that ports of HuggingFace's tokenizer
pipeline (Swift, Obj-C, Rust, JS, …) produce byte-identical output to the
reference Python `AutoTokenizer`.

Originally assembled while filing
[swift-transformers#352](https://github.com/huggingface/swift-transformers/issues/352).

## Layout

- `corpus_multilingual.jsonl` — 83 records, one per line, each `{"text": "..."}`.
  JSONL preserves embedded ZWJ sequences, variation selectors, multi-line
  code samples, and control characters losslessly.
- `goldens/<family>_multilingual.json` — array of `{"text": "...", "ids": [int, ...]}`
  records. The `text` column is identical across families; the `ids` column
  is the output of `AutoTokenizer.from_pretrained(model_id)(text, add_special_tokens=True)`
  on `transformers==4.57.1`.
- `regenerate_goldens.py` — regenerate the goldens from `corpus_multilingual.jsonl`
  against a pinned `transformers` version. See the docstring for usage.

## Families covered

| Family       | HuggingFace model           | Tokenizer kernel       |
|--------------|-----------------------------|------------------------|
| `bge_small`  | `BAAI/bge-small-en-v1.5`    | WordPiece              |
| `gpt2`       | `gpt2`                      | BPE byte-level         |
| `t5_small`   | `google-t5/t5-small`        | Unigram (SentencePiece)|
| `llama_7b`   | `huggyllama/llama-7b`       | BPE byte-fallback      |
| `roberta_base`| `FacebookAI/roberta-base`  | BPE byte-level + RobertaProcessing |

## Per-line independence

Each line of the corpus is its own test case — no inter-line state. A
divergence on a single line tells you exactly which axis broke (CJK,
combining marks, surrogate pairs, ZWJ sequences, programming code, …).

## Bug-to-line map

For [swift-transformers#352](https://github.com/huggingface/swift-transformers/issues/352)'s four bugs:

| Bug | Description                                            | Corpus case |
|-----|--------------------------------------------------------|-------------|
| 1   | BertNormalizer missing Hangul NFD decomposition        | case 12 (`안녕하세요, 세계! 토크나이저 테스트입니다.`) |
| 2   | BasicTokenizer doesn't strip Japanese voiced-kana marks | case 10 (`こんにちは、世界。トークナイザーのテストです。`) |
| 3   | Unigram iterates by grapheme cluster, not scalar       | case 35 (`Keycaps: 1️⃣ 2️⃣ 3️⃣ — flag: 🇯🇵 🇩🇪 🇺🇸.`) |
| 4   | BPE byte-fallback fires on combining-mark contexts     | case 19 (`สวัสดีชาวโลก! นี่คือการทดสอบเครื่องตัดคำ`) |

## Reproduction

```sh
python -m venv .venv
source .venv/bin/activate
pip install 'transformers==4.57.1' 'tokenizers>=0.20'
python regenerate_goldens.py
```

Each golden takes a few seconds. The 5 families together take under a minute
on a warm HF cache.

## Using as a regression bar

The expected use is to load each `tokenizer.json` into your port, encode every
text in `corpus_multilingual.jsonl`, and diff the resulting IDs against the
corresponding golden. Any divergence is a port bug.

Reference example in Objective-C: see [`ObjCTokenizerTests/OCTMultilingualGoldenTests.m`](../ObjCTokenizerTests/OCTMultilingualGoldenTests.m)
in this repo, which does exactly this against five tokenizer families.

## License / attribution

The corpus is derivative-free original test material; reuse freely. The
golden IDs are deterministic outputs of upstream HuggingFace tokenizers
under their respective licenses.
