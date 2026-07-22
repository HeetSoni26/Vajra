# Phase 1 — Tokenizer

## Included files

- `configs/tokenizer.yaml`
- `tokenizer/collect_corpus.py`
- `tokenizer/train.py`
- `tokenizer/evaluate.py`
- `tokenizer/analyze.py`

## Next implementation tasks

1. Replace smoke-test corpus collection with licensed source sampling.
2. Add compression/fertility evaluation against GPT-2, Mistral, and LLaMA-family tokenizers.
3. Freeze `tokenizer/v1.0` artifacts after validation.
