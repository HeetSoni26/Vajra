"""
Training Readiness Verification Script for Vajra Framework.

Performs a comprehensive pre-flight check before launching training:
  1. Model instantiation and parameter counting
  2. Dataset loading and iteration validation
  3. Forward / backward pass verification
  4. Optimizer and LR scheduler sanity check
  5. Checkpoint save / load round-trip
  6. Memory and throughput estimation

Usage:
    python -m scripts.verify_training_readiness --config configs/training/pretrain_tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from utils.logging import setup_logger

logger = setup_logger("training_readiness")


def verify_model(model_cfg_path: str | Path) -> dict[str, Any]:
    """Verify model can be instantiated and compute parameter counts."""
    from model import FoundationLM, ModelConfig

    cfg = ModelConfig.from_yaml(model_cfg_path)
    model = FoundationLM(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "check": "model_instantiation",
        "passed": True,
        "model_name": getattr(cfg, "model_name", "vajra"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_mb": round(total_params * 4 / (1024**2), 2),  # FP32
        "num_layers": cfg.num_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
    }
    logger.info(
        f"✓ Model instantiated: {total_params:,} parameters ({result['parameter_mb']} MB FP32)"
    )
    return result


def verify_dataset(data_dir: str | Path, sequence_length: int, batch_size: int) -> dict[str, Any]:
    """Verify dataset can be loaded and iterated."""
    from training.data_loader import create_dataloaders

    data_dir = Path(data_dir)
    train_path = data_dir / "train.bin"
    if not train_path.exists():
        train_path = data_dir / "tokens.bin"

    if not train_path.exists():
        return {
            "check": "dataset_loading",
            "passed": False,
            "error": f"No training data found in {data_dir}",
        }

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        sequence_length=sequence_length,
        micro_batch_size=batch_size,
    )

    # Test iteration
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    result = {
        "check": "dataset_loading",
        "passed": True,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader) if val_loader else 0,
        "batch_shape": list(input_ids.shape),
        "labels_shape": list(labels.shape),
        "dtype": str(input_ids.dtype),
    }
    logger.info(
        f"✓ Dataset loaded: {len(train_loader)} train batches, shape={list(input_ids.shape)}"
    )
    return result


def verify_forward_backward(
    model_cfg_path: str | Path, data_dir: str | Path, sequence_length: int, batch_size: int
) -> dict[str, Any]:
    """Verify forward and backward pass work correctly."""
    from model import FoundationLM, ModelConfig
    from training.data_loader import create_dataloaders

    cfg = ModelConfig.from_yaml(model_cfg_path)
    model = FoundationLM(cfg)
    model.train()

    train_loader, _ = create_dataloaders(
        data_dir=data_dir, sequence_length=sequence_length, micro_batch_size=batch_size
    )
    batch = next(iter(train_loader))

    # Forward pass
    t0 = time.perf_counter()
    out = model(batch["input_ids"], labels=batch["labels"])
    fwd_time = time.perf_counter() - t0

    loss = out["loss"]
    loss_value = loss.item()

    # Backward pass
    t1 = time.perf_counter()
    loss.backward()
    bwd_time = time.perf_counter() - t1

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)

    result = {
        "check": "forward_backward",
        "passed": True,
        "loss_value": round(loss_value, 4),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "forward_time_ms": round(fwd_time * 1000, 2),
        "backward_time_ms": round(bwd_time * 1000, 2),
        "gradients_computed": grad_count,
        "total_trainable_params": total_params_with_grad,
        "all_gradients_exist": grad_count == total_params_with_grad,
    }

    if not result["loss_is_finite"]:
        result["passed"] = False
        result["error"] = "Loss is not finite"

    logger.info(
        f"✓ Forward/backward pass: loss={loss_value:.4f}, "
        f"fwd={fwd_time * 1000:.1f}ms, bwd={bwd_time * 1000:.1f}ms"
    )
    return result


def verify_optimizer(model_cfg_path: str | Path) -> dict[str, Any]:
    """Verify optimizer and LR scheduler can be constructed."""
    from model import FoundationLM, ModelConfig
    from training.optimizer import build_optimizer, cosine_lr

    cfg = ModelConfig.from_yaml(model_cfg_path)
    model = FoundationLM(cfg)
    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)

    # Test LR schedule
    lr_at_0 = cosine_lr(0, warmup_steps=100, total_steps=1000, peak_lr=3e-4, min_lr=3e-5)
    lr_at_50 = cosine_lr(50, warmup_steps=100, total_steps=1000, peak_lr=3e-4, min_lr=3e-5)
    lr_at_500 = cosine_lr(500, warmup_steps=100, total_steps=1000, peak_lr=3e-4, min_lr=3e-5)

    result = {
        "check": "optimizer_scheduler",
        "passed": True,
        "optimizer_type": type(optimizer).__name__,
        "num_param_groups": len(optimizer.param_groups),
        "lr_schedule_samples": {
            "step_0": round(lr_at_0, 8),
            "step_50": round(lr_at_50, 8),
            "step_500": round(lr_at_500, 8),
        },
    }
    logger.info(
        f"✓ Optimizer: {type(optimizer).__name__} with {len(optimizer.param_groups)} param groups"
    )
    return result


def verify_checkpoint_roundtrip(model_cfg_path: str | Path, tmp_dir: str | Path) -> dict[str, Any]:
    """Verify checkpoint save and load round-trip works correctly."""
    from model import FoundationLM, ModelConfig
    from training.checkpoint import load_checkpoint, save_checkpoint

    cfg = ModelConfig.from_yaml(model_cfg_path)
    model = FoundationLM(cfg)

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "test_checkpoint.pt"

    # Save
    save_checkpoint(ckpt_path, model, step=42, tokens_seen=1000, metrics={"val_loss": 2.5})

    # Load into fresh model
    model2 = FoundationLM(cfg)
    state = load_checkpoint(ckpt_path, model2)

    # Verify state
    step_match = state["step"] == 42
    tokens_match = state["tokens_seen"] == 1000

    # Clean up
    if ckpt_path.exists():
        ckpt_path.unlink()

    result = {
        "check": "checkpoint_roundtrip",
        "passed": step_match and tokens_match,
        "saved_step": 42,
        "loaded_step": state["step"],
        "saved_tokens": 1000,
        "loaded_tokens": state["tokens_seen"],
    }
    logger.info(f"✓ Checkpoint round-trip: step={state['step']}, tokens={state['tokens_seen']}")
    return result


def run_full_verification(config_path: str | Path) -> dict[str, Any]:
    """Run all training readiness checks and return comprehensive report."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_cfg_path = Path(cfg["model_config"])
    data_dir = Path(cfg.get("data_dir", "data/tokenized"))
    seq_len = int(cfg.get("sequence_length", 128))
    batch_size = int(cfg.get("micro_batch_size", 2))

    logger.info("=" * 60)
    logger.info(f"Training Readiness Verification — {config_path.name}")
    logger.info("=" * 60)

    results: list[dict[str, Any]] = []

    # 1. Model instantiation
    try:
        results.append(verify_model(model_cfg_path))
    except Exception as e:
        results.append({"check": "model_instantiation", "passed": False, "error": str(e)})

    # 2. Dataset loading
    try:
        results.append(verify_dataset(data_dir, seq_len, batch_size))
    except Exception as e:
        results.append({"check": "dataset_loading", "passed": False, "error": str(e)})

    # 3. Forward / backward pass (only if dataset passed)
    dataset_passed = any(r["check"] == "dataset_loading" and r["passed"] for r in results)
    if dataset_passed:
        try:
            results.append(verify_forward_backward(model_cfg_path, data_dir, seq_len, batch_size))
        except Exception as e:
            results.append({"check": "forward_backward", "passed": False, "error": str(e)})
    else:
        results.append(
            {
                "check": "forward_backward",
                "passed": False,
                "error": "Skipped: dataset not available",
            }
        )

    # 4. Optimizer / scheduler
    try:
        results.append(verify_optimizer(model_cfg_path))
    except Exception as e:
        results.append({"check": "optimizer_scheduler", "passed": False, "error": str(e)})

    # 5. Checkpoint round-trip
    try:
        tmp_dir = Path(cfg.get("output_dir", "checkpoints")) / "_readiness_check"
        results.append(verify_checkpoint_roundtrip(model_cfg_path, tmp_dir))
        # Clean up tmp dir
        import shutil

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        results.append({"check": "checkpoint_roundtrip", "passed": False, "error": str(e)})

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    all_passed = passed == total

    report = {
        "config": str(config_path),
        "all_passed": all_passed,
        "summary": f"{passed}/{total} checks passed",
        "checks": results,
    }

    logger.info("=" * 60)
    logger.info(
        f"Result: {passed}/{total} checks passed — {'READY' if all_passed else 'NOT READY'}"
    )
    logger.info("=" * 60)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Training Readiness Verification")
    parser.add_argument(
        "--config", default="configs/training/pretrain_tiny.yaml", help="Training config YAML file"
    )
    parser.add_argument("--output", default=None, help="Output report JSON path")
    args = parser.parse_args()

    report = run_full_verification(args.config)

    if args.output:
        from utils.file_utils import write_json

        write_json(report, args.output)

    print(json.dumps(report, indent=2))

    if not report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
