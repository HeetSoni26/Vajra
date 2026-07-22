# Section 1 — Project Phases & Milestones

This section defines the complete execution plan from repository initialization through public release. Each phase has a concrete objective, deliverables, dependencies, and validation criteria. The critical path is Phase 2 → Phase 3 → Phase 4; later release work can be prepared while pretraining runs.

## Phase 0 — Project Vajra & Environment Setup

**Objective:** Establish a reproducible development environment, repository layout, tooling, and research baseline.

**Deliverables:** repository scaffold, pinned environment files, Dockerfile, CI, W&B setup instructions, ADR template, README skeleton, literature review plan.

**Dependencies:** none.

**Validation criteria:** tests run, CUDA is available on the target GPU instance, FlashAttention benchmark runs, W&B dummy metric logs, and contributors can reproduce the environment.

## Phase 1 — Tokenizer Design & Training

**Objective:** Train and validate a byte-level BPE tokenizer optimized for English, code, math, and technical text.

**Deliverables:** tokenizer corpus, BPE training script, tokenizer artifacts, HuggingFace-compatible tokenizer, compression/fertility evaluation, round-trip tests.

**Dependencies:** Phase 0.

**Validation criteria:** high English compression, efficient code/math tokenization, zero unknown-token failures through byte fallback, and lossless encode/decode on edge cases.

## Phase 2 — Dataset Pipeline & Preprocessing

**Objective:** Build a reproducible data pipeline producing a clean, deduplicated, tokenized pretraining corpus.

**Deliverables:** source/license audit, downloaders, extraction, normalization, language and quality filters, toxicity filtering, exact and near deduplication, benchmark contamination checks, tokenized shards, manifest.

**Dependencies:** Phase 1 tokenizer.

**Validation criteria:** target token count reached, deduplication verified, domain mix documented, contamination checks complete, and a debug model performs better on filtered than unfiltered data.

## Phase 3 — Model Architecture Implementation

**Objective:** Implement the decoder-only transformer and verify training-loop correctness at small scale.

**Deliverables:** model configs for 1B and 2B variants, RoPE, RMSNorm, SwiGLU, GQA attention, tied embeddings, parameter counter, forward/gradient tests, checkpoint utilities, 125M debug model config.

**Dependencies:** Phase 2 tokenized debug data.

**Validation criteria:** parameter counts match formulas, forward shapes are correct, all parameters receive gradients, toy overfit loss decreases, checkpoint save/load is lossless.

## Phase 4 — Pretraining

**Objective:** Train the base model on the full corpus with stable monitoring and recovery.

**Deliverables:** pretraining configs, launch scripts, DeepSpeed ZeRO configs, W&B logging, checkpoint cadence, loss curves, validation perplexity, milestone evaluations.

**Dependencies:** Phase 3 and full tokenized dataset.

**Validation criteria:** smooth loss convergence, no unrecovered spikes, validation perplexity improves, generations become coherent, and early benchmarks exceed random baselines.

## Phase 5 — Instruction Tuning & Alignment

**Objective:** Fine-tune the base model for instruction following and preference alignment.

**Deliverables:** SFT dataset, SFT script/config, SFT checkpoint, DPO dataset, DPO script/config, aligned checkpoint, chat template.

**Dependencies:** Phase 4 base checkpoint.

**Validation criteria:** instruction adherence improves, DPO checkpoint is preferred over SFT on a fixed comparison set, MT-Bench target is reached, and base capabilities are not catastrophically forgotten.

## Phase 6 — Evaluation, Optimization & Quantization

**Objective:** Run final benchmarks, compare against baselines, and produce efficient inference artifacts.

**Deliverables:** benchmark report, comparison tables, GGUF quantizations, AWQ/vLLM-compatible exports, latency and throughput measurements, final model card results.

**Dependencies:** Phase 5 aligned checkpoint.

**Validation criteria:** all benchmark runs are reproducible, quantization degradation is within tolerance, serving throughput is documented, and baseline comparisons are complete.

## Phase 7 — Deployment & Public Release

**Objective:** Release the model, code, documentation, API, SDK, and technical report.

**Deliverables:** HuggingFace Hub upload, GGUF artifacts, Ollama Modelfile, FastAPI server, Dockerfile, Python SDK, Gradio demo, technical report, final README and docs.

**Dependencies:** Phase 6 final artifacts.

**Validation criteria:** model loads in one line through HuggingFace, Ollama path works, Docker API serves completions, docs are complete, and release assets are publicly accessible.
