# ADR 0001: Core Infrastructure & Tokenizer Architecture

## Status
Accepted

## Context
During Sprint 2, the vajra-lm project required reusable infrastructure utilities (config loading, logging, seed management, device detection, checkpoint management) and a fully validated ByteLevel BPE tokenizer subsystem.

## Decisions

### 1. Centralized Utilities (`utils/` Package)
* **Decision**: Group general environment detection (`set_seed`, `get_device`, `get_memory_info`), logging (`setup_logger`, `create_experiment_dir`), file IO (`read_json`, `read_yaml`), and configuration loading into a root `utils` package.
* **Rationale**: Prevents code duplication across training, evaluation, and dataset scripts while establishing unified logging and seed handling.

### 2. Configuration System (`utils/config.py`)
* **Decision**: Implement a hierarchical configuration parser supporting YAML loading, recursive dictionary merging, and dot-notation CLI overrides (e.g. `model.hidden_size=512`).
* **Rationale**: Enables experiments to override specific hyperparameters from bash scripts without modifying YAML files directly on disk.

### 3. Checkpoint Management (`training/checkpoint.py`)
* **Decision**: Extend existing `save_checkpoint` / `load_checkpoint` with a stateful `CheckpointManager` class.
* **Rationale**: `CheckpointManager` tracks the top `max_to_keep` checkpoints, maintains `latest.pt` and `best.pt` pointers based on target metrics (e.g. `val_loss`), and preserves full metadata (`step`, `tokens_seen`, `metrics`).

### 4. Tokenizer Serialization & Fast Integration (`tokenizer/train.py`)
* **Decision**: Explicitly export `tokenizer.json` via `tokenizer.save(str(out / "tokenizer.json"))` before instantiating `PreTrainedTokenizerFast`.
* **Rationale**: Hugging Face's `PreTrainedTokenizerFast` requires `tokenizer.json` to serialize fast tokenizers correctly. This resolves loading issues in `AutoTokenizer.from_pretrained()`.

## Consequences
* All top-level packages (`model`, `training`, `utils`, etc.) import cleanly.
* Tokenizer quality can be benchmarked directly against GPT-2 and Llama tokenizers.
* Training scripts gain structured logging, seed reproducibility, and automated checkpoint retention.
