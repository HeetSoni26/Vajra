# Section 3 — Tokenizer Plan

## 3.1 Design goals

The tokenizer must be lossless, reproducible, efficient for English prose, and strong on code, math, and technical text. It must not depend on an existing pretrained tokenizer. The packaged tokenizer must load through HuggingFace `AutoTokenizer` and support future chat special tokens without checkpoint-breaking ID changes.

Primary targets:

| Metric | Target |
|---|---:|
| English compression | ≥ 4.2 characters/token |
| Python code fertility | ≤ 1.4 tokens/word |
| Math/LaTeX fertility | ≤ 2.0 tokens/symbol unit |
| Round-trip fidelity | 100% |
| Unknown-token failures | 0 through byte fallback |
| Load latency | < 200 ms on normal workstation |

## 3.2 Algorithm selection

Use byte-level BPE with HuggingFace `tokenizers`. Byte-level BPE is selected because it gives deterministic training, strong inference support, fast Rust implementation, and lossless handling of arbitrary Unicode bytes. Unigram and WordPiece are not selected because they do not provide a meaningful advantage for this project and are less aligned with common decoder-only deployment stacks.

Implementation entry points:

- `configs/tokenizer.yaml`
- `tokenizer/collect_corpus.py`
- `tokenizer/train.py`
- `tokenizer/evaluate.py`
- `tokenizer/analyze.py`

## 3.3 Training corpus composition

Target tokenizer corpus size: 10–50 GB. The tokenizer corpus is a curated sample of the pretraining corpus, not the full pretraining corpus.

| Domain | Target ratio | Purpose |
|---|---:|---|
| English web / educational text | 40% | high prose compression |
| Python code | 20% | strong indentation, identifiers, operators |
| Other code | 10% | JS, C/C++, Rust, SQL, Bash coverage |
| Scientific / LaTeX text | 10% | equations, citations, technical vocabulary |
| Math datasets | 5% | symbolic notation and proof text |
| Books | 5% | long-form prose |
| Technical docs / Q&A | 5% | markdown, APIs, command lines |
| Minor multilingual sample | 5% | robust Unicode behavior |

## 3.4 Vocabulary size

Use a 65,536-token vocabulary. This is large enough for code and technical text while remaining practical for 1B–2B parameter models. The power-of-two size is convenient for hardware-friendly embedding dimensions and leaves room for reserved special-token IDs.

## 3.5 Normalization and pre-tokenization

Use NFC normalization. Do not lowercase. Do not use aggressive Unicode compatibility normalization because it can damage mathematical notation and code semantics. Use byte-level pre-tokenization with no prefix-space convention change.

Recommended normalization policy:

```text
Unicode: NFC
Case: preserved
Whitespace: normalize only clearly pathological repeated whitespace during corpus preparation
Byte fallback: enabled
Unknown token: defined but expected to be unused
```

## 3.6 Special tokens

Reserve IDs 0–63 for current and future special tokens.

| Token | Purpose |
|---|---|
| `<|pad|>` | padding |
| `<|bos|>` | beginning of sequence |
| `<|eos|>` | end of sequence |
| `<|unk|>` | unknown fallback marker; byte fallback should avoid actual use |
| `<|sep|>` | separator between packed documents |
| `<|sys|>` | system prompt marker |
| `<|user|>` | user turn marker |
| `<|assistant|>` | assistant turn marker |
| `<|code|>` / `<|endcode|>` | optional code block delimiters |
| `<|math|>` / `<|endmath|>` | optional math block delimiters |

## 3.7 Training command

```bash
python tokenizer/train.py --config configs/tokenizer.yaml
```

The config specifies input glob, vocabulary size, minimum frequency, output directory, and special tokens.

## 3.8 Evaluation protocol

Evaluate on held-out English, Python, JavaScript, C++, Rust, SQL, LaTeX, math, Unicode stress tests, and markdown. Required tests:

1. Encode/decode round trip for every sample.
2. Compression comparison against GPT-2, Mistral, and LLaMA-family tokenizers.
3. Fertility measurement for code and math.
4. Token frequency coverage: >99% of vocab should appear at least 100 times in the tokenizer training corpus.
5. Load and save compatibility with `PreTrainedTokenizerFast`.

## 3.9 Release artifacts

The final tokenizer directory must contain:

- `tokenizer.json`
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`
- tokenizer evaluation report
- exact corpus manifest and random seed
