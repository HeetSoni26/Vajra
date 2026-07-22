# Section 7 — Evaluation Pipeline

## 7.1 Objective

Measure base and aligned model quality reproducibly across knowledge, reasoning, commonsense, math, code, truthfulness, and instruction-following tasks. Evaluation must be versioned and repeatable from checked-in configs.

Implementation entry points:

- `configs/eval/benchmarks.yaml`
- `evaluation/run_lm_eval.py`
- `evaluation/run_codegen_eval.py`
- `evaluation/compare_baselines.py`
- `evaluation/results/`

## 7.2 Primary benchmark suite

| Benchmark | Task | Shots | Metric |
|---|---|---:|---|
| MMLU | broad knowledge | 5 | accuracy |
| ARC-Challenge | science QA | 25 | accuracy |
| HellaSwag | commonsense NLI | 10 | normalized accuracy |
| WinoGrande | coreference | 5 | accuracy |
| PIQA | physical reasoning | 0 | accuracy |
| TruthfulQA | truthfulness | 0 | MC accuracy |
| GSM8K | grade-school math | 8 | exact match |
| MATH | competition math | 4 | exact match |
| HumanEval | Python code generation | 0 | pass@1 |
| MBPP | Python code generation | 3 | pass@1 |

## 7.3 Secondary evaluations

Run these for final release candidates:

- BIG-Bench Hard for reasoning
- DROP for reading comprehension
- LAMBADA for language modeling
- NaturalQuestions for open-domain QA
- MT-Bench for instruction-following quality
- fixed internal prompt suite for qualitative regression checks
- safety and refusal-behavior checks after alignment

## 7.4 lm-evaluation-harness workflow

Dry-run command construction:

```bash
python evaluation/run_lm_eval.py \
  --config configs/eval/benchmarks.yaml \
  --model_path checkpoints/final/hf \
  --dry_run
```

Actual run:

```bash
python evaluation/run_lm_eval.py \
  --config configs/eval/benchmarks.yaml \
  --model_path checkpoints/final/hf \
  --output_path evaluation/results/final_base
```

Every published number must include model checkpoint hash, tokenizer version, dataset manifest, evaluation harness version, shot count, and decoding parameters.

## 7.5 Code evaluation workflow

HumanEval and MBPP require careful sandboxing because generated code is executed.

```bash
python evaluation/run_codegen_eval.py \
  --model_path checkpoints/final/hf \
  --task humaneval
```

Record pass@1, sample count, temperature, top-p, and execution environment.

## 7.6 Baseline comparison

Compare against public models in the same parameter class:

| Model | Why included |
|---|---|
| TinyLlama-1.1B | direct 1B-scale open baseline |
| OLMo-1B | transparent dataset/training reference |
| SmolLM-1.7B | strong small-model reference |
| Phi-1.5 | small model with strong reasoning/code emphasis |

Use `evaluation/compare_baselines.py` to generate markdown tables from recorded CSV results.

## 7.7 Result interpretation

| Symptom | Likely cause | Action |
|---|---|---|
| MMLU near random | model not learning or eval formatting bug | inspect loss, tokenizer, prompts |
| HellaSwag weak at 10B tokens | poor web/data quality | inspect dataset filters |
| GSM8K weak after full pretraining | insufficient math or no reasoning format | increase math mix or add continued training |
| HumanEval weak | code underrepresented or formatting damaged | inspect code tokenization and data extraction |
| TruthfulQA drops after SFT | sycophancy introduced | reduce SFT epochs or rebalance alignment data |
| Benchmarks regress after DPO | preference over-optimization | lower LR, beta, or epochs |

## 7.8 Reporting format

Every evaluation report should contain:

- model name and checkpoint path
- parameter count
- training tokens seen
- tokenizer version
- benchmark table
- baseline comparison table
- confidence intervals where available
- decoding settings
- known caveats
- links to raw result JSON files

## 7.9 Release gates

A release candidate can proceed to Phase 7 deployment only if:

- primary benchmark suite completes without harness errors
- code benchmarks are run in a controlled execution sandbox
- aligned model is compared against base and SFT checkpoints
- quantized model perplexity degradation is measured
- model card contains final evaluation numbers and limitations
