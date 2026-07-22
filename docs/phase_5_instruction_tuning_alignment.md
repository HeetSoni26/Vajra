# Phase 5 — Instruction Tuning & Alignment

## Objective

Convert the pretrained base model into a chat-capable assistant using supervised fine-tuning followed by Direct Preference Optimization.

## Deliverables included

- `configs/training/sft.yaml`
- `configs/training/dpo.yaml`
- `training/sft.py`
- `training/dpo.py`
- chat template guidance in deployment docs

## Required datasets

SFT data should be normalized into JSONL records with a `text` field containing the fully rendered chat conversation. DPO data should use the common preference format containing prompt, chosen, and rejected responses.

Recommended sources:

| Dataset | Purpose | Notes |
|---|---|---|
| OpenHermes-style instruction data | general instruction following | filter aggressively for quality |
| MetaMathQA-style data | mathematical reasoning | preserve equations and solution steps |
| WizardCoder-style data | code instructions | preserve markdown/code fences |
| UltraFeedback / HH-RLHF-style data | preference optimization | use after SFT only |

## Validation criteria

- SFT checkpoint loads through HuggingFace `AutoModelForCausalLM`.
- Held-out instruction prompts show at least 80% adherence in manual review.
- DPO checkpoint is preferred over the SFT checkpoint on a fixed preference set.
- Core base-model benchmarks degrade by less than 5% after alignment.
