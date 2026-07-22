import time
import math
import torch
from pathlib import Path
from typing import Optional

from model.modeling import VajraForCausalLM
from training.config import TrainingConfig
from training.ddp.config import DDPConfig
from training.ddp.init import (
    cleanup,
    barrier,
)
from training.ddp.wrapper import wrap_model_ddp, unwrap_model
from training.ddp.dataloader import create_distributed_dataloader
from training.ddp.metrics import aggregate_metrics
from training.optim.optimizers import create_optimizer
from training.optim.schedulers import create_scheduler
from training.metrics.tracker import MetricsTracker
from training.checkpoints.manager import TrainingCheckpointManager


class DDPTrainingEngine:
    """
    Extends the single-GPU TrainingEngine for single-node multi-GPU DDP training.

    Usage (via torchrun / mp.spawn):
        engine = DDPTrainingEngine(model, train_config, ddp_config, rank, world_size)
        engine.train()
    """

    def __init__(
        self,
        model: VajraForCausalLM,
        config: TrainingConfig,
        ddp_config: DDPConfig,
        rank: int,
        world_size: int,
    ):
        self.config = config
        self.ddp_config = ddp_config
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        # Determine device
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        # Wrap model in DDP
        self.ddp_model = wrap_model_ddp(model, ddp_config, self.device)

        # Optimizer and scheduler operate on the *underlying* module parameters
        bare_model = unwrap_model(self.ddp_model)
        self.optimizer = create_optimizer(bare_model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)

        # AMP scaler
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=(config.mixed_precision == "fp16")
        )
        self.dtype = (
            torch.bfloat16
            if config.mixed_precision == "bf16"
            else (
                torch.float16 if config.mixed_precision == "fp16" else torch.float32
            )
        )

        # Only rank-0 writes logs / checkpoints
        if self.is_main:
            self.tracker = MetricsTracker(config.output_dir, config.logging_steps)
            self.checkpoint_manager = TrainingCheckpointManager(
                config.output_dir, config.save_total_limit
            )
        else:
            self.tracker = None
            self.checkpoint_manager = None

        self.global_step = 0
        self.tokens_processed = 0
        self.samples_processed = 0

    # ------------------------------------------------------------------
    def train(self, resume_from_checkpoint: Optional[str | Path] = None, epoch: int = 0):
        dataloader = create_distributed_dataloader(
            dataset_dir=self.config.dataset_dir,
            batch_size=self.config.batch_size,
            sequence_length=self.config.max_sequence_length,
            rank=self.rank,
            world_size=self.world_size,
            epoch=epoch,
            seed=self.config.seed,
        )

        self.ddp_model.train()

        if self.is_main:
            print(
                f"[DDP] Rank 0 of {self.world_size} — "
                f"device {self.device} — "
                f"effective batch {self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps}"
            )

        data_iter = iter(dataloader)

        while self.global_step < self.config.max_steps:
            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch = batch.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.config.mixed_precision != "none",
                ):
                    outputs = self.ddp_model(input_ids=batch, labels=batch)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                self.tokens_processed += batch.numel()
                self.samples_processed += batch.size(0)

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ddp_model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_rank = (
                self.config.batch_size
                * self.config.max_sequence_length
                * self.config.gradient_accumulation_steps
            )
            local_metrics = {
                "loss": accumulated_loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
                "tokens_processed": self.tokens_processed,
                "samples_processed": self.samples_processed,
                "tokens_per_sec": tokens_per_rank / dt,
            }

            global_metrics = aggregate_metrics(local_metrics)

            if self.is_main:
                if math.isnan(global_metrics.get("loss", 0)):
                    print(f"[DDP] NaN loss at step {self.global_step}!")
                self.tracker.log(self.global_step, global_metrics)

            if self.is_main and self.global_step % self.config.save_steps == 0:
                barrier()  # ensure all ranks finished before writing
                self.checkpoint_manager.save_checkpoint(
                    self.global_step,
                    unwrap_model(self.ddp_model),
                    self.optimizer,
                    self.scheduler,
                    global_metrics,
                )
            else:
                barrier()

        if self.is_main:
            self.tracker.close()
        cleanup()
