# Vajra Roadmap

This document outlines the version roadmap for the Vajra framework and model scaling.

## ✅ Vajra Framework v1.0.0 Released (Current)

The base infrastructure, tools, LLM engine, and autonomous multi-agent platform have been fully implemented, validated, and released as an open-source project.

## ↓ Vajra-370M Training

The next major milestone is to train the first production model, **Vajra-370M**, on the validated `Vajra-LM` engine.
- Distributed training on high-quality code and language datasets.
- Hyperparameter tuning and continuous evaluation.

## ↓ Evaluation

Comprehensive evaluation of Vajra-370M on established benchmarks (HumanEval, MBPP, MMLU) and integrated Vajra-Agent tasks.

## ↓ Improvements

Based on the 370M evaluation results, apply architectural, data pipeline, and system improvements to both the engine and agent reasoning logic.

## ↓ Vajra-1B Training

Scale up the model to 1B parameters (**Vajra-1B**) using the refined dataset and hyperparameter schedules.

## ↓ Evaluation

Rigorous evaluation of Vajra-1B against state-of-the-art 1B-3B models.
- Deep integration testing as the core `VajraReasoner` for complex multi-agent coding workflows.

## ↓ Future Scaling

- Scaling to Vajra-3B and beyond.
- Advanced RLHF and DPO fine-tuning using agent execution trace logs.
- Interactive Web UI and native GitHub/GitLab CI integrations.
