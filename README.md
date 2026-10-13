# ObjCTokenizer

Pure-Foundation Objective-C port of HuggingFace's
[swift-transformers](https://github.com/huggingface/swift-transformers)
Tokenizers module, packaged as a macOS framework. Loads any
`tokenizer.json` (WordPiece, Unigram, BPE) and produces byte-identical
output to HuggingFace's reference implementation (Python `transformers`,
backed by the Rust `tokenizers` library).

**License:** MIT (this project) over Apache 2.0 (`swift-transformers`, Rust
`tokenizers` and `spm_precompiled` attribution in [NOTICE](./NOTICE)).

**Dependencies:** Foundation, plus the system `libicucore` for Unicode
character properties, and a vendored copy of
[yyjson](https://github.com/ibireme/yyjson) 0.12.0 (MIT) for JSON parsing.
No `Hub` (downloading is out of scope — ship `tokenizer.json` as a bundle
resource alongside your app).

## Quick start

```objc
@import ObjCTokenizer;

NSError *err = nil;
NSURL *url = [NSBundle.mainBundle URLForResource:@"tokenizer" withExtension:@"json"];
OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:url error:&err];
NSArray<NSNumber *> *ids = [tok encode:@"Hello, world." error:&err];
NSString *text = [tok decode:ids error:&err];
```

Power-user API with padding / truncation / attention masks:

```objc
OCTEncodeOptions *opt = [OCTEncodeOptions new];
opt.maxLength = 512;
opt.padding = OCTPaddingMaxLength;
opt.addSpecialTokens = YES;
OCTEncoding *enc = [tok encodeAsEncoding:text options:opt error:&err];
// enc.ids, enc.attentionMask, enc.tokenTypeIds, enc.offsets
```

A single `OCTTokenizer` instance is immutable after init and safe to call
from multiple threads.

Loading fails with an `NSError` in `OCTTokenizerErrorDomain` when the JSON
is malformed (`OCTTokenizerErrorInvalidJSON`), has no usable model or vocab
(`…MissingModel`, `…MissingVocab`, `…UnsupportedModel`), or contains a
`{"Regex": …}` pattern that `NSRegularExpression` cannot compile
(`…InvalidRegex`).

## Build

The project is a stand-alone Xcode workspace — open `ObjCTokenizer.xcodeproj`
in Xcode and Cmd+B / Cmd+U, or from the command line:

```sh
xcodebuild -project ObjCTokenizer.xcodeproj -scheme ObjCTokenizer build
xcodebuild -project ObjCTokenizer.xcodeproj -scheme ObjCTokenizer test
```

The product is `ObjCTokenizer.framework` with module support
(`@import ObjCTokenizer;` works), `@rpath` install name, and 16 public
headers (umbrella + 15 `OCT*.h`).

Deployment target is macOS 26.4. The framework links Foundation,
CoreFoundation and `libicucore`; yyjson is compiled in.

## Status

Ported from `swift-transformers` release tag 1.3.2, plus the upstream
Tokenizers fixes up to commit `2590d23` (October 13, 2026) — the list is in
[NOTICE](./NOTICE).

Every bundled family is verified byte-identical against HuggingFace
reference output by golden tests:

| Family       | Model kernel                                | Corpus | Line endings | Multilingual | Conformance | Upstream fixes |
| ------------ | ------------------------------------------- | :----: | :----------: | :----------: | :---------: | :------------: |
| BGE-small    | WordPiece                                   | ✅     | ✅           | ✅           | ✅          |                |
| T5-small     | Unigram + Precompiled normalizer            | ✅     | ✅           | ✅           | ✅          | ✅             |
| GPT-2        | Byte-level BPE                              | ✅     | ✅           | ✅           | ✅          |                |
| RoBERTa-base | Byte-level BPE + RobertaProcessing          | ✅     | ✅           | ✅           | ✅          |                |
| Llama-7B     | SentencePiece BPE + byte fallback           | ✅     | ✅           | ✅           | ✅          | ✅             |
| TinyLlama    | SentencePiece BPE + byte fallback           |        |              |              | ✅          | ✅             |
| Llama 3      | Byte-level BPE + Split + `ignore_merges`    |        |              | ✅           |             | ✅             |
| Qwen2.5      | Byte-level BPE                              |        |              |              | ✅          |                |
| Gemma 4 31B  | BPE + byte fallback + Replace normalizer    |        |              |              | ✅          | ✅             |

| Column         | Test class                         | Inputs                                                             |
| -------------- | ---------------------------------- | ------------------------------------------------------------------ |
| Corpus         | `OCTGoldenCorpusTests`             | `corpus.txt`, 585 paragraphs of *War and Peace* (Garnett translation) |
| Line endings   | `OCTLineEndingGoldenTests`         | 18 texts mixing LF / CRLF / CR with multilingual axes              |
| Multilingual   | `OCTMultilingualGoldenTests`       | 83 lines: CJK, RTL, Indic, emoji, code, IPA                        |
| Conformance    | `OCTMultilingualConformanceTests`  | the same 83 inputs, categorised and stable-keyed (`Conformance/`)  |
| Upstream fixes | `OCTUpstreamFixesGoldenTests`      | targeted cases for each ported fix, byte fallback and unknown-token variants, and T5's normalizer over every Unicode code point |

Unit tests cover each pipeline component (normalizers, pre-tokenizers,
model kernels, post-processors, decoders).

## Differences from swift-transformers

The Rust `tokenizers` library is the reference. Where `swift-transformers`
still differs from it, ObjCTokenizer follows Rust:

- **BPE unknown characters** are resolved before merging, as in Rust's
  `merge_word`: byte-fallback pieces, else the unk token (fused when
  `fuse_unk`), else dropped. With no unk token, `"a🦜b"` therefore merges to
  `▁ab`; Swift resolves after merging and returns `▁a`, `b`.
- **Unknown-token fusion** happens per pre-tokenized word inside the model.
  Swift fuses consecutive unknowns across words.
- **Precompiled normalizer** (T5, XLM-R, mBART, …) decodes the real
  SentencePiece `precompiled_charsmap`, including Rust's whole-grapheme
  replacement rule, with UAX #29 grapheme clusters computed from ICU
  properties. Swift approximates the charsmap with NFKC.
- **Invalid regexes** are a load error in the Replace normalizer and decoder
  as well as the Split pre-tokenizer.

Not ported: the `Hub`, `Generation` and `Models` modules, and chat-template
rendering (Jinja). Non-normalized added tokens in already-rendered text,
such as Gemma 4's chat-turn markers, are split out before normalization,
as in HuggingFace.

## Layout

- [`ObjCTokenizer/`](ObjCTokenizer/) — framework sources. Public
  `OCT*.{h,m}` plus `Internal/` for symbols that aren't part of the API.
- [`ObjCTokenizer/ObjCTokenizer.h`](ObjCTokenizer/ObjCTokenizer.h)
  — umbrella header. Imports every public header via `<ObjCTokenizer/X.h>`.
- [`ObjCTokenizerTests/`](ObjCTokenizerTests/) — XCTest bundle.
- [`ObjCTokenizerTests/Resources/`](ObjCTokenizerTests/Resources/) —
  the nine bundled `tokenizer.json` files, golden files, and the shared
  `corpus.txt`.
- [`Conformance/`](Conformance/) — test corpora and the scripts that
  regenerate the golden files (see below and
  [`Conformance/README.md`](Conformance/README.md)).
- [`ObjCTokenizer.xcodeproj/`](ObjCTokenizer.xcodeproj/) — Xcode 16+
  project using File System Synchronized groups (`PBXFileSystemSynchronizedRootGroup`),
  so adding a new `OCT*.{h,m}` file picks up automatically. Public-header
  status is declared in the `publicHeaders` exception set inside
  `project.pbxproj`.

## Regenerating golden files

The byte-identity tests pin against snapshots of HuggingFace's reference
output, checked into
[`ObjCTokenizerTests/Resources/`](ObjCTokenizerTests/Resources/) and read at
test time via `[NSBundle bundleForClass:]`. Three scripts in
[`Conformance/`](Conformance/) regenerate them:

| Script                            | Reference                               | Writes                                                                                     |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------ |
| `regenerate_bundled_goldens.py`   | Rust `tokenizers`, offline              | `*_golden.json`, `*_lineending_golden.json`, `*_multilingual_golden.json`                  |
| `generate_upstream_fix_goldens.py`| Rust `tokenizers`, offline              | `*_upstream_fixes_golden.json`, `*_byte_fallback_golden.json`, `*_unk_fusion_golden.json`, `t5_small_precompiled_golden.json` |
| `regenerate_goldens.py`           | `transformers==4.57.1`, downloads from the Hub | `Conformance/goldens/<family>_multilingual.json`; copy each to `Resources/<family>_multilingual_v2.json` |

The two offline scripts read only the bundled `tokenizer.json` files and
need just the `tokenizers` package (the goldens were generated with 0.22.2):

```sh
python3 -m venv .venv
.venv/bin/pip install tokenizers
.venv/bin/python Conformance/regenerate_bundled_goldens.py
.venv/bin/python Conformance/generate_upstream_fix_goldens.py
```

`regenerate_bundled_goldens.py` reproduces the original
`transformers`-generated files byte for byte. Regeneration is a one-off
data pipeline rather than an Xcode build phase — it is rare and shouldn't
gate every build.

## Adding a new tokenizer family

1. Drop the `tokenizer.json` into [`ObjCTokenizerTests/Resources/`](ObjCTokenizerTests/Resources/).
   It will be auto-bundled into the `.xctest` by FSSync.
2. Add the family to `FAMILIES` and the corpora it should run on in
   [`Conformance/regenerate_bundled_goldens.py`](Conformance/regenerate_bundled_goldens.py)
   (or to the conformance matrix in `regenerate_goldens.py`), and run it.
3. Add a test method for it in
   [`ObjCTokenizerTests/OCTGoldenCorpusTests.m`](ObjCTokenizerTests/OCTGoldenCorpusTests.m)
   and/or [`OCTMultilingualGoldenTests.m`](ObjCTokenizerTests/OCTMultilingualGoldenTests.m).
4. If it diverges, the port is wrong — never the test.

## Discipline

- **Port first, refactor later.** The Swift original is well-engineered.
  Avoid "improvements" while porting — they create divergence bugs. Where
  Swift itself differs from the Rust reference, follow Rust and record it
  under [Differences from swift-transformers](#differences-from-swift-transformers).
- **Byte-identical or it's a port bug.** A single diverging record means
  the port is wrong, not the test.
- **Code points, not grapheme clusters.** HuggingFace matches on Unicode
  scalars. NSString's default search and replace are non-literal (they
  respect composed character sequences), so pass `NSLiteralSearch` (or use
  `OCTReplaceLiteral`) and iterate with `OCTEnumerateUnicodeScalars`. Where
  graphemes do matter, don't rely on
  `NSStringEnumerationByComposedCharacterSequences` — it is not UAX #29 exact.
- **Foundation primitives over the obvious splitter.** Use
  `enumerateSubstringsInRange:options:NSStringEnumerationByLines` instead
  of `componentsSeparatedByString:@"\n"`; prefer `NSData` byte iteration
  to building intermediate `NSString`s when the algorithm is byte-level.
