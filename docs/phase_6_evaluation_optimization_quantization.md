# Phase 6 — Evaluation, Optimization & Quantization

## Objective

Benchmark the base and aligned models, compare against public baselines, and export optimized inference formats.

## Deliverables included

- `evaluation/run_lm_eval.py`
- `evaluation/run_codegen_eval.py`
- `evaluation/compare_baselines.py`
- `inference/convert/to_gguf.py`
- `inference/convert/quantize_gguf.py`
- `inference/convert/to_awq.py`
- `configs/deployment/quantization.yaml`

## Benchmark suite

Primary tasks: MMLU, ARC-Challenge, HellaSwag, WinoGrande, PIQA, TruthfulQA, GSM8K, MATH.

Code tasks: HumanEval and MBPP through the BigCode evaluation harness.

## Validation criteria

- lm-eval-harness commands are reproducible from checked-in config.
- Quantized GGUF models are produced for Q4_K_M, Q5_K_M, and Q8_0.
- Q4_K_M perplexity degradation is less than 1.5 PPL versus BF16.
- Throughput and latency are measured for HuggingFace, vLLM, and llama.cpp paths.
