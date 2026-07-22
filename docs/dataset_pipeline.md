# Dataset Ingestion & Preprocessing Pipeline

This document describes the design, architecture, processing stages, configuration options, expected output artifacts, and scaling guidelines for the dataset pipeline in `vajra-lm`.

## Architecture Overview

The dataset pipeline is designed to ingest raw, heterogeneous text datasets (JSONL, TXT, Markdown, CSV), apply text normalization, filter low-quality documents, deduplicate exact content, tokenize documents into streaming ID arrays, and pack tokens into memory-mapped binary files (`train.bin`, `val.bin`, `test.bin`) with `np.uint32` encoding.

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Raw Documents  │ ──> │ Ingestion & Clean   │ ──> │ Deduplication        │
│ (.jsonl, .txt)  │     │ (NFC, Quality, LEN) │     │ (SHA256 text hashes) │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Dataset Manifest│ <── │ Binary Builder      │ <── │ Tokenizer Engine     │
│ & Verification  │     │ (train/val/test.bin)│     │ (AutoTokenizer BPE)  │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

## Processing Stages

### 1. Ingestion (`dataset/ingest.py`)
- **Supported Formats**: `.jsonl`, `.txt`, `.md`, `.csv`.
- **Directory Traversal**: Recursive folder scanning using `Path.rglob()`.
- **Fault Tolerance**: UTF-8 replacement on invalid characters, skipping malformed JSON lines or unreadable files cleanly.
- **Metadata**: Each document retains its `doc_id` and `source_file`.

### 2. Cleaning & Filtering (`dataset/processing/`)
- **Normalization** (`normalize.py`): Applies Unicode NFC normalization, strips non-printable control characters (`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]`), collapses horizontal spaces, and restricts consecutive newlines to maximum 2.
- **Quality Filter** (`quality_filter.py`): Rejects empty documents, enforces minimum/maximum word counts (`min_words=5`, `max_words=100000`), minimum alphanumeric character ratio (`min_alnum_ratio=0.50`), and maximum repetition ratio.
- **Deduplication** (`deduplication.py`): Tracks SHA256 document hashes to prevent duplicate content across multi-file sources.

### 3. Tokenization & Binary Memmap Packing (`dataset/tokenize_dataset.py` & `dataset/builder.py`)
- **Tokenizer**: Loads the trained ByteLevel BPE tokenizer (`tokenizer/v1.0`) via `AutoTokenizer`.
- **EOS Token**: Automatically appends `<|eos|>` token after each document.
- **Memmap Generation**: Packs token streams into `np.uint32` binary files:
  - `data/tokenized/train.bin` (90%)
  - `data/tokenized/val.bin` (5%)
  - `data/tokenized/test.bin` (5%)
- **Checksums**: Calculates SHA256 hashes of generated `.bin` files for reproducibility.

### 4. Resume & Manifest System (`dataset/run_pipeline.py`)
- **State File**: `pipeline_state.json` tracks completed pipeline stages. Re-running `python dataset/run_pipeline.py` skips completed stages unless `--force_rebuild` is passed.
- **Dataset Manifest**: Writes `dataset_manifest.json` recording timestamp, pipeline version, raw input counts, cleaned doc count, total token count, split breakdown, sequence length, checksums, and performance metrics.

## Configuration Guide (`configs/data/preprocessing.yaml`)

```yaml
raw_dir: data/raw
processed_dir: data/processed
tokenized_dir: data/tokenized
tokenizer_path: tokenizer/v1.0

language:
  keep_language: en
  min_confidence: 0.65

quality:
  min_words: 5
  max_words: 100000
  max_char_word_ratio: 10.0
  min_alnum_ratio: 0.60
  max_repetition_ratio: 0.30

dedup:
  shingle_size: 13
  minhash_threshold: 0.80

packing:
  sequence_length: 4096
  separator_token: "<|sep|>"
```

## Scaling Guidelines (1 GB to 1 TB Datasets)

1. **Chunked Memmap Appending**: For multi-gigabyte or terabyte corpora, replace in-memory token list aggregation with progressive chunk writing (`mode="r+"` on `np.memmap`) in fixed 100MB buffers.
2. **MinHash / LSH Deduplication**: For multi-terabyte web scale data (e.g. FineWeb/SlimPajama), upgrade exact SHA256 deduplication to MinHash LSH fuzzy deduplication using `datasketch` or PySpark.
3. **Multi-Processing Tokenization**: Distribute document tokenization across CPU workers using `multiprocessing.Pool` or Ray.
