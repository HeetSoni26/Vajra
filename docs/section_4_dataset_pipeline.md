# Section 4 — Dataset Pipeline

## 4.1 Objective

Build a reproducible pipeline that converts licensed raw sources into a clean, deduplicated, quality-filtered, tokenized pretraining corpus. The target is 50B–200B tokens depending on whether the 1B or 2B model is trained.

Implementation entry points:

- `configs/data/dataset_mix.yaml`
- `configs/data/preprocessing.yaml`
- `dataset/run_pipeline.py`
- `dataset/processing/*`
- `dataset/tokenize_dataset.py`
- `dataset/verify_dataset.py`
- `dataset/manifests/v1.0_manifest.json`

## 4.2 Source and license audit

Every dataset must be recorded with source URL, license, acquisition date, checksum, intended use, and exclusion rules. Do not start full preprocessing until the license audit is complete.

Recommended source categories:

| Domain | Examples | Required checks |
|---|---|---|
| Web / educational | FineWeb, FineWeb-Edu, Dolma subsets | crawl license and filtering rules |
| Code | The Stack v2, StarCoder-style open code | repository license and PII filters |
| Math | OpenWebMath, MATH-style datasets | license and benchmark contamination |
| Science | arXiv, S2ORC | terms of use and paper licenses |
| Books | Project Gutenberg and permissive sources | public-domain status |
| Wikipedia | Wikimedia dumps | CC-BY-SA attribution obligations |
| Technical Q&A | Stack Overflow / docs | license compatibility and attribution |

## 4.3 Target domain mix

Default 100B-token mix:

| Domain | Ratio | Tokens at 100B |
|---|---:|---:|
| Web / English | 35% | 35B |
| Code | 25% | 25B |
| Math | 10% | 10B |
| Scientific papers | 10% | 10B |
| Books | 8% | 8B |
| Wikipedia | 5% | 5B |
| Technical docs / Q&A | 5% | 5B |
| Other / multilingual | 2% | 2B |

The mix is configured in `configs/data/dataset_mix.yaml` and should be treated as versioned experimental state.

## 4.4 Processing stages

The required pipeline is:

```text
raw download
  → extraction / format normalization
  → language identification
  → Unicode and whitespace normalization
  → quality filtering
  → toxicity / safety filtering
  → exact deduplication
  → near deduplication
  → document scoring
  → benchmark contamination removal
  → tokenization
  → sequence packing
  → shard writing
  → manifest generation
```

## 4.5 Filtering rules

Default quality thresholds are stored in `configs/data/preprocessing.yaml`.

Core filters:

- word count between 50 and 100,000 for prose documents
- character/word ratio ≤ 10
- alphanumeric ratio ≥ 0.60 for prose
- repeated n-gram ratio ≤ 0.30
- English language confidence ≥ 0.65 for web text
- code bypasses prose language filters but uses code-specific license and quality filters

## 4.6 Deduplication

Use exact hash deduplication first, then near-deduplication.

- Exact dedup: hash normalized full document text.
- Near dedup: 13-gram shingles, MinHash signatures, LSH threshold around 0.80.
- Cross-domain dedup is required to prevent the same web page or code snippet from appearing in multiple source categories.

## 4.7 Benchmark contamination checks

Before tokenization, remove documents with high n-gram overlap against evaluation sets. At minimum cover MMLU, ARC, HellaSwag, WinoGrande, PIQA, TruthfulQA, GSM8K, MATH, HumanEval, and MBPP. Store contamination-removal statistics in the dataset manifest.

## 4.8 Tokenization and packing

After filtering and scoring, tokenize with the frozen tokenizer from Section 3. Pack documents to the model context length and insert `<|sep|>` between documents. Write memory-mapped binary shards for training throughput.

Recommended shard size: around 100M tokens per shard. Store SHA256 checksums for every shard.

## 4.9 Manifest requirements

Each dataset release must include:

- version
- source list and licenses
- raw bytes per source
- filter removal rates
- exact/near duplicate removal rates
- final tokens by domain
- shard checksums
- tokenizer version
- contamination-check summary
- creation command and commit hash

## 4.10 Validation criteria

- Final token count meets the training target.
- Domain mix is within configured tolerance.
- Duplicate rate is near zero after MinHash deduplication.
- Benchmark contamination removal has run and is documented.
- A 125M debug model trained on filtered data outperforms one trained on unfiltered data.
