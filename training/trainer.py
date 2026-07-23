from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker, perplexity
from training.optimizer import cosine_lr
from utils.logging import setup_logger

logger = setup_logger("trainer")


def resolve_precision_and_scaler(precision_mode: str, device: str | torch.device | None = None) -> tuple[torch.dtype, bool, Any]:
    """Resolve AMP precision dtype, autocast enabled state, and GradScaler based on hardware capabilities."""
    mode = precision_mode.lower()

    if device is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    else:
        dev_str = str(device).lower()
        if "cuda" in dev_str:
            device_type = "cuda"
        elif "mps" in dev_str:
            device_type = "mps"
        else:
            device_type = "cpu"

    scaler_device = "cuda" if device_type == "cuda" else "cpu"

    if mode == "bf16":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, torch.amp.GradScaler(scaler_device, enabled=False)
        elif device_type == "mps":
            return torch.bfloat16, True, torch.amp.GradScaler(scaler_device, enabled=False)
        else:
            logger.warning("BF16 precision requested but hardware does not support BF16. Falling back to FP32.")
            return torch.float32, False, torch.amp.GradScaler(scaler_device, enabled=False)

    elif mode == "fp16":
        if device_type == "cuda":
            return torch.float16, True, torch.amp.GradScaler("cuda", enabled=True)
        elif device_type == "mps":
            return torch.float16, True, torch.amp.GradScaler("cpu", enabled=False)
        else:
            logger.warning("FP16 precision requested but CUDA is unavailable. Falling back to FP32.")
            return torch.float32, False, torch.amp.GradScaler("cpu", enabled=False)

    # FP32 default
    return torch.float32, False, torch.amp.GradScaler(scaler_device, enabled=False)



class Trainer:
    """Production-grade single-node (multi-GPU DDP) training engine supporting AMP (BF16/FP16/FP32) and step profiling."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        peak_lr: float = 3.0e-4,
        min_lr: float = 3.0e-5,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        device: torch.device | str = "cpu",
        checkpoint_dir: str | Path = "checkpoints",
        precision: str = "fp32",
        is_distributed: bool = False,
        local_rank: int = 0,
        rank: int = 0,
        max_grad_norm_threshold: float = 100.0,
    ) -> None:
        self.device = torch.device(device)
        self.raw_model = model.to(self.device)
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.rank = rank

        # Wrap model in DDP if distributed
        if is_distributed and torch.distributed.is_initialized():
            device_ids = [local_rank] if self.device.type == "cuda" else None
            self.model = torch.nn.parallel.DistributedDataParallel(self.raw_model, device_ids=device_ids)
        else:
            self.model = self.raw_model

        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_grad_norm_threshold = max_grad_norm_threshold

        # AMP Mixed Precision Setup
        self.amp_dtype, self.amp_enabled, self.scaler = resolve_precision_and_scaler(precision)
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"

        # Checkpoint Manager (rank 0 only saves checkpoints)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir, metric_name="val_loss", mode="min")
        self.metrics_tracker = MetricsTracker()
        self.accumulated_loss = 0.0

    def get_current_lr(self, step: int) -> float:
        """Calculate cosine decay learning rate with linear warmup."""
        return cosine_lr(
            step=step,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            peak_lr=self.peak_lr,
            min_lr=self.min_lr,
        )

    def train_step(self, batch: dict[str, torch.Tensor], step: int, is_accum_step: bool) -> dict[str, Any] | None:
        """Execute a single forward/backward pass with AMP and early failure checks."""
        self.model.train()
        step_start_t = time.time()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with AMP autocast
        fwd_start_t = time.time()
        with torch.amp.autocast(device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            out = self.model(input_ids, labels=labels)
            raw_loss = out["loss"]

        # ----------------------------------------------------
        # Early Failure Check 1: Loss NaN / Inf
        # ----------------------------------------------------
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            raise ValueError(f"[EARLY FAILURE] Training aborted at rank {self.rank}, step {step}: Loss is NaN/Inf ({raw_loss.item()}).")

        loss = raw_loss / self.grad_accum_steps
        fwd_time = time.time() - fwd_start_t

        # Backward pass with GradScaler
        bwd_start_t = time.time()
        self.scaler.scale(loss).backward()
        bwd_time = time.time() - bwd_start_t

        self.accumulated_loss += raw_loss.item() / self.grad_accum_steps

        # Perform optimizer step only on accumulation boundaries
        if is_accum_step:
            opt_start_t = time.time()

            # Unscale gradients for clipping if scaler is enabled
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            # Early Failure Check 2: Exploding / NaN Gradients
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))

            if math.isnan(grad_norm) or math.isinf(grad_norm):
                raise ValueError(f"[EARLY FAILURE] Training aborted at rank {self.rank}, step {step}: Gradient norm is NaN/Inf.")

            if grad_norm > self.max_grad_norm_threshold:
                raise ValueError(
                    f"[EARLY FAILURE] Training aborted at rank {self.rank}, step {step}: Exploding gradient norm ({grad_norm:.2f} > {self.max_grad_norm_threshold})."
                )

            # Update learning rate
            current_lr = self.get_current_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            # Step optimizer with scaler
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            opt_time = time.time() - opt_start_t

            # Calculate throughput, timing, and memory
            batch_size, seq_len = input_ids.shape
            tokens_in_step = batch_size * seq_len * self.grad_accum_steps
            samples_in_step = batch_size * self.grad_accum_steps

            metrics = self.metrics_tracker.record_step(
                step=step,
                loss=self.accumulated_loss,
                lr=current_lr,
                tokens_in_step=tokens_in_step,
                samples_in_step=samples_in_step,
                grad_norm=grad_norm,
            )

            step_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            metrics["loss"] = step_loss
            
            # GPU Memory Monitoring
            if torch.cuda.is_available():
                metrics["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                metrics["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            
            # ETA Calculation
            steps_remaining = self.total_steps - step
            if "tokens_per_sec" in metrics and metrics["tokens_per_sec"] > 0:
                time_per_step = tokens_in_step / metrics["tokens_per_sec"]
                metrics["eta_hours"] = (steps_remaining * time_per_step) / 3600.0

            metrics["timing"] = {
                "step_time_ms": round((time.time() - step_start_t) * 1000, 2),
                "forward_time_ms": round(fwd_time * 1000, 2),
                "backward_time_ms": round(bwd_time * 1000, 2),
                "optimizer_time_ms": round(opt_time * 1000, 2),
            }
            return metrics

        return None

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Execute deterministic evaluation loop on validation dataset."""
        self.model.eval()
        total_val_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.amp.autocast(device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    out = self.model(input_ids, labels=labels)
                total_val_loss += float(out["loss"].item())
                total_batches += 1

        avg_val_loss = total_val_loss / max(1, total_batches)

        # Reduce validation loss across DDP ranks if distributed
        if self.is_distributed and torch.distributed.is_initialized():
            val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
            torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = float((val_loss_tensor / torch.distributed.get_world_size()).item())

        val_ppl = perplexity(avg_val_loss)

        if self.rank == 0:
            logger.info(f"[EVALUATION] Val Loss: {avg_val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

        return {"val_loss": round(avg_val_loss, 4), "val_perplexity": round(val_ppl, 2)}
