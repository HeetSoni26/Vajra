import time
import torch
import math
from typing import Optional
from pathlib import Path

from model.modeling import VajraForCausalLM
from training.config import TrainingConfig
from training.data.loader import create_dataloader
from training.optim.optimizers import create_optimizer
from training.optim.schedulers import create_scheduler
from training.metrics.tracker import MetricsTracker
from training.checkpoints.manager import TrainingCheckpointManager


class TrainingEngine:
    def __init__(self, model: VajraForCausalLM, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)

        self.tracker = MetricsTracker(self.config.output_dir, self.config.logging_steps)
        self.checkpoint_manager = TrainingCheckpointManager(
            self.config.output_dir, self.config.save_total_limit
        )

        self.global_step = 0
        self.tokens_processed = 0
        self.samples_processed = 0

        # AMP
        device_type = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        scaler_device = "cuda" if device_type == "cuda" else "cpu"
        scaler_enabled = (self.config.mixed_precision == "fp16") and (device_type == "cuda")
        self.scaler = torch.amp.GradScaler(device=scaler_device, enabled=scaler_enabled)
        self.dtype = (
            torch.bfloat16
            if self.config.mixed_precision == "bf16"
            else (torch.float16 if self.config.mixed_precision == "fp16" else torch.float32)
        )

    def train(self, resume_from_checkpoint: Optional[str | Path] = None):
        if resume_from_checkpoint:
            state = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint, self.model, self.optimizer, self.scheduler
            )
            self.global_step = state.get("step", 0)
            self.tokens_processed = state.get("tokens_processed", 0)
            self.samples_processed = state.get("samples_processed", 0)
            print(f"Resumed training from step {self.global_step}")

        dataloader = create_dataloader(
            self.config.dataset_dir,
            batch_size=self.config.batch_size,
            sequence_length=self.config.max_sequence_length,
        )

        self.model.train()
        data_iter = iter(dataloader)

        print("Starting training loop...")

        while self.global_step < self.config.max_steps:
            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Dataset exhausted, loop or abort (usually we loop epochs, but here we just rebuild iter)
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch = batch.to(self.device, non_blocking=True)

                # Input and Labels (shift happens inside VajraForCausalLM if labels are provided)
                input_ids = batch
                labels = batch

                # Forward pass with AMP
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.config.mixed_precision != "none",
                ):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                self.tokens_processed += input_ids.numel()
                self.samples_processed += input_ids.size(0)

            # Unscale and clip grads
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1

            # Metrics
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = (
                self.config.batch_size
                * self.config.max_sequence_length
                * self.config.gradient_accumulation_steps
            ) / dt

            if math.isnan(accumulated_loss) or math.isinf(accumulated_loss):
                print(f"Warning: NaN or Inf loss detected at step {self.global_step}!")

            metrics = {
                "loss": accumulated_loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "tokens_processed": self.tokens_processed,
                "samples_processed": self.samples_processed,
                "tokens_per_sec": tokens_per_sec,
            }

            self.tracker.log(self.global_step, metrics)

            # Checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.global_step, self.model, self.optimizer, self.scheduler, metrics
                )

        self.tracker.close()
        print("Training complete.")
