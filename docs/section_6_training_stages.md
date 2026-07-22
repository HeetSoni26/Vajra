# Section 6 — Training Stages

## 6.1 Stage overview

The project uses six operational training stages:

1. tokenizer training
2. debug model training
3. foundation pretraining
4. supervised fine-tuning
5. preference optimization
6. quantization and inference optimization

Stages 1–3 are mandatory before any aligned release. Stages 4–6 produce the instruct and deployable variants.

## 6.2 Stage 1 — Tokenizer training

Inputs:

- curated 10–50 GB tokenizer corpus
- `configs/tokenizer.yaml`

Command:

```bash
python tokenizer/train.py --config configs/tokenizer.yaml
python tokenizer/evaluate.py --tokenizer_dir tokenizer/v1.0
```

Exit criteria:

- round-trip fidelity is 100%
- compression/fertility targets are met
- tokenizer artifacts are frozen and versioned

## 6.3 Stage 2 — Debug model training

Purpose: validate the full stack before spending major compute.

Inputs:

- small tokenized dataset
- `configs/model/debug_125m.yaml`
- `configs/training/debug_125m.yaml`

Command:

```bash
bash training/launch/launch_debug_125m.sh
```

Exit criteria:

- loss decreases
- checkpoint restore works
- validation perplexity improves
- generated samples become non-random
- no data-loader or memory bottleneck is observed

## 6.4 Stage 3 — Vajra pretraining

Purpose: train the base model from random initialization.

Commands:

```bash
bash training/launch/launch_pretrain_1b.sh
bash training/launch/launch_pretrain_2b.sh
```

Run only one target at a time unless the compute budget supports parallel experiments. Evaluate at milestone token counts and preserve milestone checkpoints.

Exit criteria:

- training loss is smooth
- validation perplexity improves
- milestone benchmarks exceed random baselines
- final checkpoint exports to HuggingFace format

## 6.5 Stage 4 — Supervised fine-tuning

Purpose: convert the base model into an instruction-following model.

Input format: JSONL with a `text` field containing the rendered chat transcript.

Command:

```bash
python training/sft.py --config configs/training/sft.yaml
```

Default training settings:

- LR 2e-5
- 2 epochs
- max sequence length 2048
- completion-only loss when implemented in the final data collator
- BF16

Exit criteria:

- held-out prompt adherence ≥ 80% by manual rubric
- no severe benchmark regression
- chat template is documented and saved

## 6.6 Stage 5 — Direct Preference Optimization

Purpose: improve response preference quality without full RLHF complexity.

Input format: JSONL preference records with prompt, chosen response, and rejected response.

Command:

```bash
python training/dpo.py --config configs/training/dpo.yaml
```

Default settings:

- LR 5e-7
- beta 0.1
- 1 epoch
- frozen reference model initialized from the SFT checkpoint

Exit criteria:

- DPO model wins against SFT on a fixed preference set
- MT-Bench improves or remains stable
- factuality and safety regressions are reviewed

## 6.7 Stage 6 — Quantization and optimization

Purpose: generate deployable artifacts for local and server inference.

Commands:

```bash
python inference/convert/to_gguf.py --llama_cpp /path/to/llama.cpp --model checkpoints/final/hf
python inference/convert/quantize_gguf.py --quantizer /path/to/llama-quantize --type Q4_K_M
python inference/convert/to_awq.py --model_path checkpoints/final/hf
```

Exit criteria:

- GGUF Q4_K_M, Q5_K_M, and Q8_0 are produced
- quantized perplexity degradation is measured
- vLLM/HF/llama.cpp throughput is documented

## 6.8 Milestone evaluation cadence

| Milestone | Evaluation |
|---|---|
| 1B tokens | HellaSwag, PIQA sanity check |
| 5B tokens | add ARC and WinoGrande |
| 10B tokens | full primary suite |
| 25B tokens | full primary suite and qualitative generations |
| 50B tokens | continue/stop decision |
| 100B+ tokens | final base model suite |
| after SFT | primary suite and instruction eval |
| after DPO | primary suite, MT-Bench, preference eval |

## 6.9 Training records

Each run must save:

- config snapshot
- git commit hash
- dataset manifest ID
- tokenizer version
- random seeds
- checkpoint list
- W&B or local run ID
- known incidents and mitigations
