# Vajra Evaluation Framework

The Vajra Evaluation Framework provides a suite of tools for evaluating, comparing, and interacting with native Vajra checkpoints. This avoids the overhead of converting checkpoints to HuggingFace format for intermediate evaluation.

## Tools Overview

All scripts are located in the `evaluation/` directory and use the `evaluations/` directory for storing outputs.

### 1. `evaluate.py`
Runs a native forward pass over the validation dataset to compute `validation_loss` and `perplexity`.
Saves detailed metadata (tokens seen, git hash, timing) to `evaluations/checkpoint_NAME/metrics.json`.

**Usage:**
```bash
python -m evaluation.evaluate \
    --checkpoint checkpoints/pretrain_tiny_xyz/latest.pt \
    --config configs/training/pretrain_tiny.yaml \
    --batch-size 4
```

### 2. `generate.py`
Leverages the production `InferenceEngine` for native text generation with KV cache support and sampling parameters.
Can stream output to console or save samples to the evaluation directory.

**Usage:**
```bash
python -m evaluation.generate \
    --checkpoint checkpoints/pretrain_tiny_xyz/latest.pt \
    --config configs/training/pretrain_tiny.yaml \
    --prompt "Artificial intelligence is" \
    --max-new-tokens 128 \
    --temperature 0.8 \
    --stream
```

### 3. `compare_checkpoints.py`
Scans the `evaluations/` directory to aggregate `metrics.json` outputs.
Produces a unified leaderboard in Terminal, JSON, CSV, and Markdown formats.

**Usage:**
```bash
python -m evaluation.compare_checkpoints
```

### 4. `evaluate_all.py`
End-to-end evaluation orchestrator. Scans an experiment directory for `checkpoint_step_XYZ.pt` files, automatically evaluates them, generates a text sample, and updates the leaderboard.

**Usage:**
```bash
python -m evaluation.evaluate_all \
    --experiment-dir checkpoints/pretrain_tiny_xyz \
    --config configs/training/pretrain_tiny.yaml
```

## Output Structure

The framework generates a standardized output layout for downstream analysis:

```
evaluations/
├── checkpoint_step_250/
│   ├── metrics.json
│   └── samples.txt
├── checkpoint_step_500/
│   ├── metrics.json
│   └── samples.txt
├── leaderboard.json
├── leaderboard.csv
└── leaderboard.md
```
