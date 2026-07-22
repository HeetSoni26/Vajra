# Section 12 — Required Knowledge & Skills

## 12.1 Objective

List the knowledge areas required to execute the project responsibly. A solo researcher does not need to be an expert in every area before starting, but must understand each enough to debug failures and make informed tradeoffs.

## 12.2 Deep learning fundamentals

Required knowledge:

- backpropagation and automatic differentiation
- cross-entropy language-modeling loss
- AdamW optimizer
- learning-rate schedules
- gradient clipping
- regularization and overfitting
- numerical precision and stability

Why it matters: training failures often look like infrastructure problems but originate from optimization settings, unstable gradients, or bad loss scaling.

## 12.3 Transformer architecture

Required knowledge:

- causal self-attention
- multi-head attention and grouped-query attention
- KV cache behavior
- residual streams
- normalization placement
- feed-forward blocks
- positional encodings
- autoregressive generation

The model implementation in this scaffold maps these concepts to `model/attention.py`, `model/model.py`, `model/rope.py`, `model/norm.py`, and `model/feedforward.py`.

## 12.4 Modern LLM techniques

Required knowledge:

- RoPE and context-extension methods
- RMSNorm
- SwiGLU
- GQA/MQA/MHA tradeoffs
- FlashAttention and memory-efficient attention
- BF16 training
- checkpointing and activation recomputation
- tokenizer/model compatibility

## 12.5 Scaling laws and compute planning

Required knowledge:

- tokens-per-parameter tradeoffs
- Chinchilla-style compute estimates
- FLOP estimation: `6 × parameters × tokens`
- model FLOP utilization
- cost per billion tokens
- early stopping based on validation trajectory

## 12.6 Tokenization

Required knowledge:

- BPE merge training
- byte-level tokenization
- vocabulary-size tradeoffs
- special-token design
- code and math tokenization challenges
- round-trip fidelity tests

## 12.7 Distributed training

Required knowledge:

- data parallelism
- gradient accumulation
- DeepSpeed ZeRO stages
- FSDP basics
- NCCL communication
- checkpoint sharding
- GPU memory accounting
- failure recovery after preemption

## 12.8 Data engineering at scale

Required knowledge:

- streaming dataset processing
- Parquet/JSONL/Zstandard tradeoffs
- web extraction quality
- language identification
- exact and near deduplication
- MinHash/LSH basics
- toxicity and PII filtering
- benchmark contamination removal
- dataset manifests and checksums

## 12.9 Instruction tuning and alignment

Required knowledge:

- chat templates
- supervised fine-tuning
- completion-only loss
- preference datasets
- DPO objective and beta parameter
- catastrophic forgetting
- alignment regressions and safety evaluation

## 12.10 Evaluation and benchmarking

Required knowledge:

- few-shot prompting
- exact-match versus multiple-choice metrics
- benchmark contamination risks
- lm-evaluation-harness
- code benchmark sandboxing
- statistical variance in small benchmarks
- baseline comparison methodology

## 12.11 Inference optimization

Required knowledge:

- autoregressive decoding
- KV-cache memory
- quantization formats: GGUF, GPTQ, AWQ
- vLLM PagedAttention
- throughput versus latency
- batch sizing
- CPU versus GPU inference tradeoffs

## 12.12 MLOps and release engineering

Required knowledge:

- experiment tracking
- reproducible configs
- artifact versioning
- CI and tests
- Docker images
- API serving
- Python packaging
- HuggingFace Hub release workflows
- model cards and technical reports

## 12.13 Software engineering skills

Required skills:

- Python and PyTorch
- Linux shell usage
- Git workflows
- Docker basics
- YAML/JSON configuration
- test writing with pytest
- linting and formatting
- profiling and debugging

## 12.14 Suggested learning path

1. Implement and train a tiny decoder-only transformer.
2. Train a custom tokenizer and inspect failures.
3. Build a small deduplicated dataset pipeline.
4. Run a 10M–100M token debug training job.
5. Add checkpoint restore and evaluation.
6. Scale to the 125M debug config.
7. Only then launch 1B+ pretraining.

## 12.15 Team role mapping

| Role | Core responsibilities |
|---|---|
| Research lead | architecture, scaling, ablations, benchmark interpretation |
| Data engineer | dataset acquisition, filtering, dedup, manifests |
| Training engineer | distributed runs, checkpoints, monitoring, cost control |
| Evaluation lead | benchmark harnesses, contamination checks, reports |
| Deployment engineer | HF export, GGUF, API, Docker, SDK |
| Documentation owner | README, model card, technical report, release notes |

A solo researcher must cover all roles, so phase gates and checklists are essential.
