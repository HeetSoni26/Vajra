# Section 13 — Required Software & Tools

## 13.1 Objective

Define the software stack required to train, evaluate, optimize, and deploy the 1B–2B parameter foundation language model. Versions should be pinned before full-scale training and recorded in every run manifest.

## 13.2 Core frameworks

| Software | Recommended version | Purpose |
|---|---|---|
| Python | 3.11.x | primary implementation language |
| PyTorch | 2.3.x or newer validated version | model implementation and training |
| CUDA | 12.1.x or provider-supported equivalent | GPU compute runtime |
| cuDNN | 9.x where compatible | neural-network kernels |
| NCCL | 2.20.x or provider-supported equivalent | distributed GPU communication |

The scaffold provides `environment.yml`, `pyproject.toml`, and Dockerfiles. Final training images should be built and frozen before the main run.

## 13.3 Training stack

| Software | Purpose |
|---|---|
| DeepSpeed | ZeRO optimizer, distributed launch, optimizer sharding |
| HuggingFace Accelerate | optional distributed abstraction and checkpoint utilities |
| FlashAttention | memory-efficient attention kernels on supported GPUs |
| HuggingFace Transformers | export, AutoClasses, SFT/DPO compatibility |
| HuggingFace Tokenizers | fast byte-level BPE tokenizer training/loading |
| HuggingFace Datasets | dataset loading and JSON/Parquet streaming |
| TRL | SFT and DPO training workflows |
| safetensors | safe checkpoint serialization |

## 13.4 Data pipeline stack

| Software | Purpose |
|---|---|
| datatrove | scalable LLM data processing pipelines |
| trafilatura / resiliparse | HTML-to-text extraction |
| ftfy | Unicode repair |
| fastText language ID | language filtering |
| datasketch | MinHash and LSH near-deduplication |
| KenLM | optional n-gram perplexity quality filter |
| Detoxify or replacement classifier | toxicity filtering |
| pyarrow / Parquet | efficient intermediate storage |
| zstandard | compressed JSONL/stream storage |
| NumPy memmap | high-throughput tokenized training shards |

## 13.5 Evaluation stack

| Software | Purpose |
|---|---|
| lm-evaluation-harness | MMLU, ARC, HellaSwag, WinoGrande, PIQA, TruthfulQA, GSM8K, MATH |
| bigcode-evaluation-harness | HumanEval and MBPP |
| FastChat / MT-Bench tooling | instruction-following judge workflow |
| pandas | result aggregation |
| matplotlib / seaborn | static plots |

Evaluation environments must be pinned separately from training if needed. Record exact harness commit hashes.

## 13.6 Inference and deployment stack

| Software | Purpose |
|---|---|
| vLLM | high-throughput GPU serving |
| llama.cpp | GGUF conversion and local inference |
| AWQ | activation-aware weight quantization |
| FastAPI | REST API service |
| Uvicorn | ASGI server |
| Docker | containerized serving and reproducible environments |
| Gradio | demo application |
| HuggingFace Hub | checkpoint and model-card hosting |
| Ollama | local model packaging |
| httpx | SDK client HTTP transport |

## 13.7 Development tools

| Software | Purpose |
|---|---|
| pytest | tests |
| ruff | linting and formatting |
| mypy | optional static typing checks |
| pre-commit | local quality gates |
| GitHub Actions | CI |
| Weights & Biases | experiment tracking |
| TensorBoard | optional local metric visualization |

## 13.8 Environment files in this repository

| File | Purpose |
|---|---|
| `environment.yml` | Conda environment for research/development |
| `pyproject.toml` | package metadata and Python dependencies |
| `Dockerfile` | development/training container scaffold |
| `Dockerfile.serve` | inference API serving container |
| `requirements-serve.txt` | serving dependencies |
| `docker-compose.yml` | local container orchestration scaffold |

## 13.9 Version pinning policy

Before a full pretraining run:

1. freeze the Docker image digest
2. record `pip freeze` or `conda env export`
3. record CUDA, driver, NCCL, and GPU model
4. record Git commit hash
5. record dataset manifest and tokenizer version
6. store the exact DeepSpeed and PyTorch versions in the training run metadata

## 13.10 Validation criteria

The software stack is ready when:

- CPU CI passes tests.
- GPU smoke test imports PyTorch and detects CUDA.
- FlashAttention or the chosen attention backend runs on the target GPU.
- DeepSpeed launches the debug config.
- W&B or local metric logging records a dummy run.
- Docker serving image builds and exposes a health route.
