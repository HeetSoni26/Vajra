# Section 8 — Repository Structure

## 8.1 Objective

Define a maintainable repository layout for a production-grade open-source language-model project. The repository must separate model code, data pipeline code, training infrastructure, evaluation, inference, deployment, documentation, tests, and release artifacts.

## 8.2 Top-level layout

```text
vajra-lm/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── environment.yml
├── Dockerfile
├── Dockerfile.serve
├── docker-compose.yml
├── requirements-serve.txt
├── configs/
├── tokenizer/
├── dataset/
├── model/
├── training/
├── evaluation/
├── inference/
├── api/
├── sdk/
├── deployment/
├── scripts/
├── tests/
├── docs/
├── notebooks/
├── model_card/
└── .github/
```

## 8.3 Configuration directory

`configs/` contains all versioned runtime choices. No major training behavior should be hardcoded in Python.

```text
configs/
├── tokenizer.yaml
├── model/
│   ├── model_1b.yaml
│   ├── model_2b.yaml
│   └── debug_125m.yaml
├── training/
│   ├── pretrain_1b.yaml
│   ├── pretrain_2b.yaml
│   ├── debug_125m.yaml
│   ├── sft.yaml
│   └── dpo.yaml
├── data/
│   ├── dataset_mix.yaml
│   └── preprocessing.yaml
├── deepspeed/
│   ├── zero2.json
│   └── zero3.json
├── eval/
│   └── benchmarks.yaml
└── deployment/
    ├── serve.yaml
    └── quantization.yaml
```

## 8.4 Tokenizer directory

`tokenizer/` owns Phase 1 tokenizer collection, training, analysis, and packaging.

Required files:

- `collect_corpus.py` — creates or samples tokenizer training data
- `train.py` — trains byte-level BPE tokenizer
- `evaluate.py` — computes compression, fertility, and round-trip metrics
- `analyze.py` — reports vocabulary coverage and token distributions
- `v1.0/` — frozen tokenizer artifacts after training

## 8.5 Dataset directory

`dataset/` owns Phase 2 preprocessing.

```text
dataset/
├── download/
├── processing/
├── tokenize_dataset.py
├── verify_dataset.py
├── run_pipeline.py
└── manifests/
```

The `download/` directory contains source-specific acquisition code. The `processing/` directory contains one module per pipeline stage: extraction, normalization, language filtering, quality filtering, toxicity filtering, deduplication, scoring, and contamination removal.

## 8.6 Model directory

`model/` contains only architecture and generation code.

```text
model/
├── config.py
├── model.py
├── attention.py
├── feedforward.py
├── norm.py
├── rope.py
└── generation.py
```

This separation makes the architecture importable by tests, training, inference, and conversion tools.

## 8.7 Training directory

`training/` contains pretraining, SFT, DPO, shared trainer logic, data loading, optimizer/scheduler utilities, metrics, checkpointing, and launch scripts.

Training code must remain checkpoint-resumable and should never assume local absolute paths.

## 8.8 Evaluation directory

`evaluation/` stores benchmark wrappers, baseline comparison tools, and raw result artifacts.

Rules:

- Raw JSON results go under `evaluation/results/`.
- Published tables must be reproducible from raw results.
- Evaluation commands must record checkpoint ID, tokenizer ID, and harness version.

## 8.9 Inference, API, SDK, and deployment directories

- `inference/` — local generation, chat, batch inference, and model conversion.
- `api/` — OpenAI-compatible FastAPI endpoints.
- `sdk/` — Python client package for the API.
- `deployment/` — HuggingFace upload, Ollama Modelfile, Gradio demo, and release helpers.

## 8.10 Tests directory

Tests should cover:

- model configuration and forward pass
- tokenizer invariants
- dataset manifest and packing assumptions
- training utilities
- API importability
- documentation completeness

Heavy GPU integration tests should be separated from CPU CI tests.

## 8.11 Repository hygiene rules

Do not commit:

- raw datasets
- tokenized dataset shards
- model checkpoints
- W&B logs
- API keys or `.env` files
- generated GGUF files
- cloud-provider credentials

Commit:

- configs
- code
- tests
- documentation
- small synthetic fixtures
- manifests without secrets

## 8.12 Validation criteria

The repository structure is complete when:

- `pytest` passes on CPU-only CI.
- `make test`, `make lint`, and `make smoke` are defined.
- every phase has a clear entry point.
- generated large artifacts are excluded by `.gitignore`.
- release-critical docs are present before public launch.
