# Multilingual conformance corpus for HuggingFace tokenizer ports

83-line multilingual corpus + per-family golden tokenizations against
`transformers==4.57.1`. Used to verify that ports of HuggingFace's tokenizer
pipeline (Swift, Obj-C, Rust, JS, …) produce byte-identical output to the
reference Python `AutoTokenizer`.

Originally assembled while filing
[swift-transformers#352](https://github.com/huggingface/swift-transformers/issues/352);
extended afterwards as the shared conformance bed for both `ObjCTokenizer`
and the swift-transformers
[`MultilingualConformanceTests`](https://github.com/huggingface/swift-transformers/pull/360) target.

## Layout

- `inputs.json` — 83 records, each `{id, category, text}`. Stable ids keep
  baselines aligned across regenerations; categories let a divergence message
  cite the broken axis directly (`japanese-voiced-kana`, `emoji-keycap`,
  `thai-combining-marks`, `devanagari`, …).
- `corpus_multilingual.jsonl` — legacy shape, `{"text": ...}` per line, no
  ids/categories. Kept for the existing
  [`ObjCTokenizerTests/OCTMultilingualGoldenTests.m`](../ObjCTokenizerTests/OCTMultilingualGoldenTests.m)
  test target; new work should consume `inputs.json`.
- `goldens/<family>_multilingual.json` — one per kernel, top-level shape:
    ```jsonc
    {
      "metadata": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "transformers_version": "4.57.1",
        "generated_at": "2026-05-16T00:19:58+00:00",
        "input_count": 83
      },
      "entries": [
        {
          "id": "japanese-voiced-kana-greeting",
          "input_ids": [...],
          "tokens": ["[CLS]", "こ", "##ん", …],
          "decoded_with_special": "[CLS] こんにちは ...",
          "decoded_skip_special": "こんにちは ..."
        },
        …
      ]
    }
    ```
  `tokens` is `convert_ids_to_tokens(input_ids)` for diagnostic windowed
  diffs. The two decoded fields are forward-compatible material for a
  decoder-side parity test; current Obj-C tests only consume `input_ids`.
- `regenerate_goldens.py` — regenerate the per-kernel baselines from
  `inputs.json` against the pinned `transformers` version. See the
  docstring for usage.

## Families covered

| Family         | HuggingFace model                          | Tokenizer kernel                              |
|----------------|--------------------------------------------|-----------------------------------------------|
| `bge_small`    | `BAAI/bge-small-en-v1.5`                   | WordPiece                                     |
| `t5_small`     | `google-t5/t5-small`                       | Unigram (SentencePiece)                       |
| `gpt2`         | `openai-community/gpt2`                    | Byte-level BPE                                |
| `roberta_base` | `FacebookAI/roberta-base`                  | Byte-level BPE + RobertaProcessing            |
| `llama_7b`     | `huggyllama/llama-7b`                      | SentencePiece BPE + byte-fallback (legacy)    |
| `qwen2_5`      | `Qwen/Qwen2.5-0.5B`                        | Byte-level BPE (modern vocab/merges)          |
| `tinyllama`    | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`       | SentencePiece BPE + byte-fallback (no auth)   |
| `gemma`        | `google/gemma-4-31b-it`                    | BPE + Replace-normalizer + Metaspace (262k)   |

Llama 3 (`meta-llama/Meta-Llama-3-8B`) is auth-gated and not in the default
matrix; log into the Hub and add a row to regenerate against it.

### Gemma 4 — raw-load path

`google/gemma-4-31b-it` is the Apertura inference target. Its
`tokenizer_config.json` is newer than the pinned `transformers==4.57.1`, so
`GemmaTokenizerFast` mis-parses it (`extra_special_tokens` is a list, not a
dict). `regenerate_goldens.py` falls back to loading the `tokenizer.json`
directly with `PreTrainedTokenizerFast` and reconstructs the wrapper's
post-processor (`add_bos_token=True`, `add_eos_token=False` → prepend `<bos>`,
id 2). Without that step the golden would omit the leading `<bos>` every real
Gemma forward pass emits.

Because Gemma's shipped `tokenizer.json` post-processor is a no-op (the wrapper
adds `<bos>` at load time), the regeneration also writes a **resolved**
`goldens/gemma.tokenizer.json` with the post-processor baked in — exactly as the
bundled `llama-7b.tokenizer.json` bakes in `<s>`. That resolved file (not the
upstream one) is what the Obj-C port bundles as
`ObjCTokenizerTests/Resources/gemma-4-31b.tokenizer.json`. Regenerating Gemma
also needs `protobuf` + `sentencepiece` on the path.

## Per-line independence

Each line of the corpus is its own test case — no inter-line state. A
divergence on a single line attributes the failure to a specific axis. The
`category` field on each entry groups inputs by axis so the failure message
cites the broken axis directly.

## Bug-to-input map

For [swift-transformers#352](https://github.com/huggingface/swift-transformers/issues/352)'s four bugs:

| Bug | Description                                            | Stable input id                       |
|-----|--------------------------------------------------------|---------------------------------------|
| 1   | BertNormalizer missing Hangul NFD decomposition        | `hangul-syllables-greeting`           |
| 2   | BasicTokenizer doesn't strip Japanese voiced-kana marks | `japanese-voiced-kana-greeting`       |
| 3   | Unigram iterates by grapheme cluster, not scalar       | `emoji-keycap-and-flags`              |
| 4   | BPE byte-fallback fires on combining-mark contexts     | `thai-combining-marks-greeting`       |

## Reproduction

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # transformers==4.57.1
python regenerate_goldens.py
```

Each baseline takes a few seconds. The 8 families together take under a
minute on a warm HF cache.

## Using as a regression bar

The expected use is to load each `tokenizer.json` into your port, encode
every text in `inputs.json`, and diff the resulting IDs against the
corresponding baseline's `input_ids`. Any divergence is a port bug.

Reference implementations:
- Obj-C: [`ObjCTokenizerTests/OCTMultilingualConformanceTests.m`](../ObjCTokenizerTests/OCTMultilingualConformanceTests.m).
  Currently passes byte-identity on all 7 kernels including the 3 new
  bug clusters that swift-transformers surfaces (Metaspace leading-
  whitespace runs, T5 TM/VS-16 segmentation, Qwen2.5 BPE merge ordering).
- Swift: [`Tests/TokenizersTests/MultilingualConformanceTests.swift`](https://github.com/huggingface/swift-transformers/pull/360)
  (swift-transformers PR #360). Lands with `expectedDivergences` covering
  the in-flight fixes (#354/#355/#356) and the 3 bug clusters above.

## License / attribution

The corpus is derivative-free original test material; reuse freely. The
golden IDs are deterministic outputs of upstream HuggingFace tokenizers
under their respective licenses.
