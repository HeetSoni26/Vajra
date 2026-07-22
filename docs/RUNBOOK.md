# Vajra Runbook

This runbook describes standard operating procedures for Vajra model training.

## 1. Initial Setup
Run preflight checks to verify disk space, VRAM, tokenizers, and datasets:
`python scripts/preflight.py --output-dir /path/to/checkpoints --dataset-dir /path/to/data`

## 2. Starting a Run
To start the Vajra-370M production run:
`python training/workflows/scripts/launch.py train-370m --dataset-dir /path/to/data --output-dir /path/to/checkpoints`

## 3. Monitoring
Check `DASHBOARD.md` or invoke the dashboard script:
`python scripts/dashboard.py`

## 4. Interrupts & Resumes
If training crashes, resume exactly from the latest checkpoint:
`python training/workflows/scripts/launch.py resume-370m --dataset-dir data --output-dir checkpoints --checkpoint checkpoint-5000`

## 5. Exporting for Inference
`python release/scripts/launch.py package --output-dir release/vajra-370m-v1`
