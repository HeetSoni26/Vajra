# Vajra Production Training Orchestrator Guide

## Overview

The `TrainingOrchestrator` is the single coordinator for all aspects of a production training run. It replaces the scattered control flow previously in `pretrain.py` with a clean, autonomous lifecycle manager.

Users run:

```bash
python -m training.pretrain --config configs/training/pretrain_tiny.yaml
```

and the Orchestrator handles:
- Resuming from local or cloud checkpoints
- Periodic (step and time-based) checkpoint saves
- Background cloud uploads
- Health monitoring (GPU, disk, loss spikes)
- Watchdog freeze detection with emergency saves
- Signal handling (SIGINT, SIGTERM) for graceful shutdown
- ETA and progress reporting
- Experiment registry and lifecycle state transitions

---

## Architecture

```
TrainingOrchestrator
├── Trainer               (gradient computation only)
├── CheckpointManager     (local saves / rotation)
├── ResumeManager         (checkpoint discovery and restoration)
├── CloudSyncManager      (background uploads, remote resume)
├── HealthMonitor         (GPU/disk/loss/grad health checks)
├── ExperimentManager     (lifecycle state machine + JSON registry)
├── Watchdog              (freeze detection, emergency saves)
└── ETAEngine             (throughput, ETA, progress %)
```

---

## Lifecycle States

```
INITIALIZING → RESTORING → TRAINING → VALIDATING → CHECKPOINTING → UPLOADING
                                ↘ COMPLETED
                                ↘ FAILED
                                ↘ INTERRUPTED
```

States are persisted to `exp_dir/experiment_registry.json` for audit and debugging.

---

## Health Monitoring

| Check | Warning Condition |
|---|---|
| GPU memory | Reserved > 90% of total VRAM |
| Disk space | Free < `min_disk_free_gb` (default: 2 GB) |
| Loss spike | Current loss > 3x rolling average |
| NaN/Inf loss | Immediate warning; Trainer will abort |
| Gradient norm | > 50 (configurable) |

---

## Watchdog

A background daemon thread monitors for training freezes. If no heartbeat arrives within `watchdog_timeout_seconds` (default: 300s), it:
1. Logs a diagnostic error
2. Saves an emergency checkpoint
3. Triggers a background cloud upload

---

## Checkpointing Policy

| Trigger | Behaviour |
|---|---|
| Every `save_every_steps` | Step-based checkpoint |
| Every `time_checkpoint_every_minutes` | Time-based checkpoint |
| SIGINT / SIGTERM | Emergency checkpoint + upload before exit |
| Watchdog trigger | Emergency checkpoint |
| End of training | Final checkpoint |

---

## Experiment Registry

Each experiment directory contains `experiment_registry.json`:

```json
{
  "experiment_id": "exp_20260724_080000",
  "state": "COMPLETED",
  "checkpoint_history": [...],
  "resume_history": [...],
  "provider_history": [...],
  "training_summary": {...}
}
```

---

## Configuration (`configs/training/orchestration.yaml`)

| Key | Default | Description |
|---|---|---|
| `checkpoint_every_steps` | 50 | Step-based save interval |
| `time_checkpoint_every_minutes` | 15.0 | Time-based save interval |
| `watchdog_timeout_seconds` | 300.0 | Freeze detection timeout |
| `min_disk_free_gb` | 2.0 | Disk warning threshold |
| `gpu_mem_warn_pct` | 90.0 | VRAM warning % |
| `loss_spike_factor` | 3.0 | Loss spike multiplier |
| `grad_norm_warn_threshold` | 50.0 | Gradient norm warning |
| `retention.keep_last_n` | 5 | Number of step checkpoints to keep |

## Validation Report & Known Fixes

During real-world validation on Lightning AI T4 instances, the training orchestrator, health monitor, and cloud sync manager behaved flawlessly across network disconnects and interruptions. 

### Fixed Production Bugs

1. **Exploding Gradient Abort**: Previously, if gradient norm exceeded the threshold, `Trainer` would raise an error and abort training immediately. This has been updated to clip the gradient, record the pre-clipped norm in the metrics, log a warning, and continue training seamlessly.
2. **Tokens per Second Computation**: Previously `tokens_per_sec` could result in a `KeyError` or display `0.0`. It is now computed using actual elapsed wall-clock time in `Trainer` and safely populated for all consumers (ETA, Logger, Orchestrator). Safe defaults have also been added to other time metrics to ensure `KeyError`s are never raised in the logging pipeline.

### Recovery Workflows

- **Lightning / Kaggle / Colab**: If an instance resets, simply re-run the same `pretrain.py` command. Vajra will discover the incomplete experiment directory (if persistent) or download the latest checkpoint from Hugging Face if local state is missing, resuming perfectly with identical optimizer, scheduler, and RNG state.
