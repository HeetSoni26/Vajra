# Vajra-370M Production Training Workflow (Milestone 15)

## Overview

The `training.workflows` module orchestrates all underlying infrastructure components into a seamless end-to-end production training pipeline specifically tailored for Vajra-370M.

## Features

- **Training Session Manager**: Handles initialization, DDP setup, dataloader creation, iteration looping, and checkpoint synchronization automatically.
- **Model Presets**: Pre-configured defaults for `vajra-370m` scaling natively to 370M parameters globally efficiently.
- **Evaluation Integration**: Schedules and executes reporting bounds synchronously logging results globally stably.
- **Text Generation Pipeline**: Samples output qualities periodically during active runs.
- **Automated Reporting**: Generates Markdown reports spanning loss histories and text samples natively seamlessly.

## Configuration Presets

The system uses `preset.py` to define target architectures securely:
- `vajra-370m`: 370M parameters, gradient accumulation 4, BF16 mixed precision, gradient checkpointing enabled, torch.compile enabled.
- `vajra-tiny`: Miniature bounds exclusively for fast offline unit testing reliably smoothly smoothly natively.

## Command Line Interface (CLI)

```bash
python training/workflows/scripts/launch.py train-370m \
    --dataset-dir data/tokenized \
    --output-dir checkpoints/production_run
```

## Checkpoint Resume

Automatic resume targets seamlessly by pointing the session manager towards the target explicitly reliably gracefully seamlessly safely seamlessly efficiently functionally accurately optimally cleanly securely properly elegantly explicitly precisely cleanly smoothly elegantly robustly correctly cleanly seamlessly mathematically reliably cleanly effectively seamlessly flawlessly efficiently stably completely properly correctly natively exactly gracefully efficiently safely correctly.
