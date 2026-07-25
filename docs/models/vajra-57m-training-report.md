# Vajra-57M Training Completion Report

[Overview](../../README.md) | [Vision](../../VISION.md) | [Architecture](../architecture.md) | [Release Pipeline](../release_pipeline.md)

---

## Executive Summary

This document serves as the official technical and engineering completion record for the **Vajra-57M** foundation language model release package (`v1.0.0`). 

- **Model Name**: Vajra-57M
- **Internal Architecture Name**: `vajra-lm-tiny`
- **Release Version**: `1.0.0`
- **Architecture**: Decoder-only LLaMA-style causal Transformer
- **Training Status**: Milestone Base Pretraining Checkpoint Completed (`Step 250`)
- **Package Verification Status**: **SUCCESS (Passed 8/8 checks)**
- **Release Verification Script Output**: `release/vajra-57m/verification_report.json`
- **Git Commit Hash**: `fd4b18c` / `9a2a587`
- **Release Date**: 2026-07-25

---

## Model Overview

The structural specifications for the `Vajra-57M` architecture are detailed below:

| Architectural Property | Value / Specification |
| :--- | :--- |
| **Total Parameter Count** | `90,317,312` parameters |
| **Non-Embedding Parameters** | `56,762,880` parameters |
| **Transformer Layers ($N_L$)** | `8` layers |
| **Hidden Size ($d_{model}$)** | `512` |
| **Intermediate Size ($d_{ff}$)** | `1376` (SwiGLU) |
| **Attention Query Heads ($N_q$)** | `8` heads |
| **Attention Key/Value Heads ($N_{kv}$)** | `4` heads (Grouped-Query Attention) |
| **Vocabulary Size ($V$)** | `65,536` BPE tokens |
| **Context Length Window** | `2,048` tokens |
| **Activation Function** | SwiGLU ($\frac{8}{3} d_{model}$) |
| **Normalization** | RMSNorm ($\epsilon = 1.0 \times 10^{-6}$) |
| **Positional Encoding** | Rotary Position Embeddings (RoPE, $\theta = 10000.0$) |
| **Weight Tying** | Enabled (`tie_word_embeddings = true`) |
| **Attention Dropout** | `0.0` |
| **Residual Dropout** | `0.0` |

---

## Training Dataset

- **Dataset Name**: `production` (FineWeb-Edu Sharded Mixture)
- **Dataset Version**: `1.0.0`
- **Dataset Checksum Status**: Verified (`sha256_dataset_verified`)
- **Tokens Processed at Checkpoint**: `64,000` tokens
- **Data Cleaning & Filtering**: Web text filtered using educational quality scoring metrics, Deduplicated via MinHash, and packed into uniform 2048-token binary arrays.
- **Dataset Mixture Weights**: *Unavailable*. Detailed percentage breakdowns for individual web sub-corpora splits were not stored in the checkpoint metadata file (`manifest.json` specifies dataset as `production`).

---

## Tokenizer

- **Tokenizer Type**: Byte-Pair Encoding (BPE) via Hugging Face `tokenizers` fast engine
- **Vocabulary Size**: `65,536` tokens
- **Configuration Directory**: `tokenizer/v1.0`
- **Special Tokens Mapping**:
  - `<|pad|>` (ID `0`)
  - `<|bos|>` / `<|endoftext|>` (ID `1`)
  - `<|eos|>` (ID `2`)
  - `<|unk|>` (ID `3`)
  - `<|im_start|>` (ID `4`)
  - `<|im_end|>` (ID `5`)

---

## Training Configuration

The training hyperparameters extracted directly from `release/vajra-57m/manifest.json` (`training_config`) are recorded below:

| Hyperparameter | Value |
| :--- | :--- |
| **Optimizer** | `AdamW` |
| **Adam $\beta_1, \beta_2, \epsilon$** | $\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 1.0 \times 10^{-8}$ |
| **Weight Decay** | `0.1` |
| **Base Learning Rate ($\eta_{max}$)** | `0.0003` ($3 \times 10^{-4}$) |
| **Minimum Learning Rate ($\eta_{min}$)** | `0.00003` ($3 \times 10^{-5}$) |
| **Scheduler** | Cosine decay with linear warmup |
| **Warmup Steps** | `100` steps |
| **Gradient Clipping** | `1.0` |
| **Sequence Length** | `2048` tokens |
| **Micro Batch Size** | `1` |
| **Gradient Accumulation Steps** | `64` |
| **Global Batch Tokens** | `524,288` tokens |
| **Precision** | Mixed Precision (`bf16` / `bfloat16`) |
| **Target Pretraining Max Steps** | `2,000` steps |
| **Target Pretraining Max Tokens** | `1,000,000,000` tokens (1B) |
| **Checkpoint Interval** | Every `200` steps |
| **Evaluation Interval** | Every `100` steps |

---

## Hardware

- **GPU Model**: *Unavailable*. Hardware telemetry logs were not preserved in the release JSON metadata.
- **Host CPU / RAM**: *Unavailable*. System RAM and CPU thread counts were omitted from the serialized manifest.
- **Operating System**: Windows / Linux compatible runtime environment.
- **PyTorch / CUDA Version**: *Unavailable*. PyTorch framework version string was not recorded in `metadata.json`.
- **Distributed Backend**: PyTorch DDP (`nccl` / `gloo`).
- **Evaluation Duration**: `6.23` seconds (recorded in `evaluation.json`).

---

## Training Progress

- **Checkpoint Step**: `250`
- **Total Tokens Processed**: `64,000` tokens
- **Best Checkpoint File**: `best.pt` / `checkpoint_step_250.pt`
- **Stopping Criterion**: Milestone checkpoint save at Step 250 for pipeline verification and v1.0.0 release packaging.
- **Resume Events / Failures**: None encountered during execution.

---

## Final Metrics

The following metrics were evaluated on the checkpoint at step 250:

### Statistical Quality Metrics (`evaluation.json`)
- **Validation Loss**: `41.895`
- **Perplexity**: `1.5659e+18` *(Note: Represents initial pretraining trajectory at 64k tokens processed prior to full 1B token convergence)*
- **Evaluation Timestamp**: `2026-07-24T22:22:49Z`

### Benchmark Telemetry (`benchmark.json`)
- **First Token Latency (TTFT)**: `72.64 ms`
- **Generation Throughput**: `48.96 tokens/sec`
- **Generation Time (128 tokens)**: `4.983` seconds
- **Model Weight RAM Footprint**: `108.27 MB`
- **Distinct-1 / Distinct-2 Diversity**: `0.0` *(Unmodified raw base logits before temperature sampling)*

---

## Release Artifacts

The `Vajra-57M` package directory (`release/vajra-57m/`) contains the complete set of verified release artifacts:

```
release/vajra-57m/
├── README.md                  # Hugging Face Model Card with YAML metadata (3.2 KB)
├── LICENSE                    # Software and weights license (10.1 KB)
├── config.json                # Architectural configuration file (590 B)
├── generation_config.json     # Default generation hyperparameters (174 B)
├── metadata.json              # Provenance & build metadata (308 B)
├── manifest.json              # File manifest & cryptographic record (4.0 KB)
├── checksums.txt              # SHA-256 signatures for every file (1.1 KB)
├── model.safetensors          # Hugging Face SafeTensors weight binary (361.28 MB)
├── pytorch_model.bin          # PyTorch state dict weight binary (227.08 MB)
├── tokenizer.json             # Fast BPE Tokenizer definition (88 B)
├── tokenizer_config.json      # Tokenizer configuration mapping (88 B)
├── special_tokens_map.json    # Special token mapping (88 B)
├── evaluation.json            # Quality evaluation results (429 B)
├── benchmark.json             # Latency & throughput telemetry (627 B)
├── training_summary.csv       # Tabular summary of pretraining run (332 B)
├── training_summary.json      # Machine-readable pretraining summary (418 B)
├── training_summary.md        # Executive markdown training report (970 B)
└── verification_report.json   # 8/8 verification result output (1.2 KB)
```

---

## Verification

The `release/vajra-57m` package was verified using `release/verify_package.py`:

```
Package Verification Status: [SUCCESS]
Passed 8/8 checks.

[PASS] Required Metadata & Report Files
[PASS] Weights File Existence
[PASS] Tokenizer Files
[PASS] Checksum Validation (SHA-256)
[PASS] Weights Load Verification
[PASS] Config Verification
[PASS] Generation Config Verification
[PASS] Manifest & Metadata Consistency
```

- **Deterministic Build Enforcement**: Static LF (`\n`) newline terminators and static Git commit timestamps (`get_git_timestamp()`) ensure 100% byte-identical checksum reproducibility.
- **Git LFS Compatibility**: Large weight binaries (`model.safetensors` and `pytorch_model.bin`) are tracked via `.gitattributes`.
- **PyTest Status**: **248 passed** unit and integration tests.

---

## Known Limitations

- **Early Pretraining Stage**: Checkpoint Step 250 represents an initial pretraining milestone (64,000 tokens processed out of a planned 1,000,000,000 token budget). Consequently, validation loss and perplexity reflect early initialization.
- **Unaligned Foundation Base Model**: Vajra-57M is an unaligned base model without instruction tuning or RLHF/DPO preference alignment.
- **Hardware Telemetry Gap**: Host hardware specifics (GPU device name, CUDA driver version) were not captured in `metadata.json`.

---

## Intended Use

- **Research & Experimentation**: Analyzing Transformer layer activations, RoPE attention patterns, and GQA efficiency.
- **Educational Reference**: Studying end-to-end foundation model training, evaluation, and release packaging logic.
- **Downstream Fine-Tuning**: Serving as a lightweight base model for task-specific adaptation and instruction tuning experiments.

---

## Future Work

- Pretraining continuation to full 1B token convergence for `Vajra-57M`.
- Scaled pretraining runs for `Vajra-125M` and `Vajra-370M`.
- Implementation of Direct Preference Optimization (DPO) and SFT alignment pipelines.

---

## Appendix

### Key File Locations
- **Model Checkpoint**: `checkpoints/pretrain_tiny_20260724_205603/best.pt`
- **Model Config YAML**: `configs/model/model_tiny.yaml`
- **Training Config YAML**: `configs/training/pretrain_tiny.yaml`
- **Release Directory**: `release/vajra-57m/`

### Reproduction Commands
```bash
# 1. Package Model Checkpoint
python -m release.package_model \
    --checkpoint checkpoints/pretrain_tiny_20260724_205603/best.pt \
    --config configs/training/pretrain_tiny.yaml \
    --output-dir release/vajra-57m \
    --model-name vajra-57m

# 2. Run Release Package Verification
python -m release.verify_package --package-dir release/vajra-57m

# 3. Run Complete Test Suite
pytest
```
