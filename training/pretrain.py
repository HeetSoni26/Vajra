from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from model import FoundationLM, ModelConfig
from training.cloud.sync_manager import CloudSyncManager
from training.data_loader import create_dataloaders
from training.optimizer import build_optimizer
from training.resume import ResumeManager
from training.trainer import Trainer
from training.training_logger import TrainingLogger
from utils.environment import get_device, get_git_hash, set_seed
from utils.file_utils import read_json, write_json
from utils.logging import create_experiment_dir, setup_logger

logger = setup_logger("pretrain_runner")


def setup_ddp_environment() -> tuple[bool, int, int, int]:
    """Discover environment variables and initialize torch.distributed process group if running under torchrun."""
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_distributed:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return True, world_size, rank, local_rank

    return False, 1, 0, 0


def cleanup_ddp_environment(is_distributed: bool) -> None:
    """Cleanly destroy PyTorch process group if distributed."""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def compute_model_summary(model: FoundationLM, model_cfg: ModelConfig) -> dict[str, Any]:
    """Compute structural parameter counts, FLOPs, and VRAM estimates."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    flops_per_token = 6 * total_params
    est_vram_mb = round((total_params * 16) / (1024**2), 2)

    return {
        "model_name": getattr(model_cfg, "model_name", "vajra-lm"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "flops_per_token": flops_per_token,
        "estimated_vram_mb": est_vram_mb,
        "model_depth_layers": model_cfg.num_layers,
        "hidden_size": model_cfg.hidden_size,
        "vocabulary_size": model_cfg.vocab_size,
    }


def generate_training_curves(history: list[dict[str, Any]], exp_dir: Path) -> None:
    """Generate loss_curve.png and learning_rate_curve.png if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in history if "step" in h]
        train_losses = [h["train_loss"] for h in history if "train_loss" in h]
        lrs = [h["learning_rate"] for h in history if "learning_rate" in h]

        if steps and train_losses:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, train_losses, label="Train Loss", color="#1f77b4")
            val_steps = [h["step"] for h in history if h.get("val_loss") is not None]
            val_losses = [h["val_loss"] for h in history if h.get("val_loss") is not None]
            if val_steps and val_losses:
                plt.plot(val_steps, val_losses, "ro-", label="Val Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Pretraining Loss Curve")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(exp_dir / "loss_curve.png", dpi=150)
            plt.close()

        if steps and lrs:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, lrs, label="Learning Rate", color="#ff7f0e")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(exp_dir / "learning_rate_curve.png", dpi=150)
            plt.close()
    except Exception as e:
        logger.warning(f"Could not generate PNG plots (matplotlib missing or error): {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation LM Pretraining Engine")
    parser.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint if available"
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # DDP Setup
    is_distributed, world_size, rank, local_rank = setup_ddp_environment()

    try:
        # Synchronize seed per rank
        set_seed(args.seed + rank)

        config_path = Path(args.config)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        # Load Model Config
        model_cfg_path = Path(cfg["model_config"])
        model_cfg = ModelConfig.from_yaml(model_cfg_path)
        model = FoundationLM(model_cfg)

        device = get_device()
        if rank == 0:
            logger.info(
                f"Using compute device: {device} | Distributed: {is_distributed} (World Size: {world_size})"
            )

        base_out_dir = Path(cfg.get("output_dir", "checkpoints/run"))

        # Resume or Create Experiment Directory
        resume_state = None
        if args.resume:
            resume_manager = ResumeManager(base_dir=base_out_dir.parent)
            exp_dir, resume_state = resume_manager.find_latest_valid_experiment(
                prefix=base_out_dir.name + "_"
            )
        else:
            exp_dir = create_experiment_dir(base_out_dir.parent, name=base_out_dir.name, config=cfg)

        # Model Summary & Reproducibility Metadata (Rank 0 saves artifacts)
        model_summary = compute_model_summary(model, model_cfg)

        if rank == 0 and not args.resume:
            write_json(model_summary, exp_dir / "model_summary.json")

            data_dir = Path(cfg.get("data_dir", "data/tokenized"))
            manifest_file = data_dir / "dataset_manifest.json"
            manifest_info = read_json(manifest_file) if manifest_file.exists() else {}

            exp_metadata = {
                "world_size": world_size,
                "random_seed": args.seed,
                "git_commit_hash": get_git_hash(),
                "tokenizer_version": cfg.get("tokenizer_path", "tokenizer/v1.0"),
                "dataset_manifest_checksums": manifest_info.get("checksums", {}),
                "full_config": cfg,
                "model_config": model_summary,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(exp_metadata, exp_dir / "experiment_metadata.json")

        # Optimizer & Trainer Setup
        lr = float(cfg["learning_rate"])
        weight_decay = float(cfg["weight_decay"])
        betas = (float(cfg["adam_beta1"]), float(cfg["adam_beta2"]))
        eps = float(cfg.get("adam_epsilon", 1e-8))

        optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

        sequence_length = int(cfg["sequence_length"])
        micro_batch_size = int(cfg["micro_batch_size"])
        grad_accum_steps = int(cfg.get("gradient_accumulation_steps", 1))

        data_dir = Path(cfg.get("data_dir", "data/tokenized"))
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            grad_clip=float(cfg.get("grad_clip", 1.0)),
            grad_accum_steps=grad_accum_steps,
            peak_lr=lr,
            min_lr=float(cfg.get("min_learning_rate", lr * 0.1)),
            warmup_steps=int(cfg.get("warmup_steps", 100)),
            total_steps=int(cfg.get("max_steps", 1000)),
            device=device,
            checkpoint_dir=exp_dir,
            precision=str(cfg.get("precision", "fp32")),
            is_distributed=is_distributed,
            local_rank=local_rank,
            rank=rank,
        )

        start_step = 0
        tokens_seen = 0

        # Setup Unified Logger
        training_logger = TrainingLogger(
            log_dir=exp_dir,
            run_name=getattr(model_cfg, "model_name", "vajra-run"),
            use_tensorboard=True,
            use_wandb=True,
            wandb_project=cfg.get("wandb_project"),
            wandb_config={**cfg, **model_summary},
            rank=rank,
        )

        # Resume from checkpoint if requested
        if args.resume and resume_state is not None:
            resume_manager = ResumeManager(base_dir=base_out_dir.parent)
            start_step, tokens_seen = resume_manager.restore_state(
                state=resume_state,
                model=trainer.raw_model,
                optimizer=optimizer,
                scaler=trainer.scaler,
                device=device,
            )
            # Update trainer's checkpoint manager to know about the resumed state
            trainer.checkpoint_manager.checkpoint_dir = exp_dir
            # Sync scaler state correctly if using AMP (ResumeManager restores it if present)

        # Initialize CloudSyncManager for background uploads
        cloud_sync = CloudSyncManager()

        max_steps = int(cfg.get("max_steps", 50))
        save_every = int(cfg.get("save_every_steps", 10))
        eval_every = int(cfg.get("eval_every_steps", 10))

        if rank == 0:
            logger.info(
                f"Starting Pretraining: {model_summary['total_parameters']:,} parameters | max_steps={max_steps}"
            )

        start_time = time.time()
        global_step = start_step
        micro_step = 0

        train_iter = iter(train_loader)
        history: list[dict[str, Any]] = []

        while global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            micro_step += 1
            is_accum_boundary = micro_step % grad_accum_steps == 0
            step_metrics = trainer.train_step(
                batch, step=global_step, is_accum_step=is_accum_boundary
            )

            if is_accum_boundary and step_metrics is not None:
                global_step += 1
                tokens_seen += step_metrics["tokens_processed"] * world_size

                # Periodic Validation Loop
                if val_loader is not None and global_step % eval_every == 0:
                    val_stats = trainer.evaluate(val_loader)
                    step_metrics.update(val_stats)

                # Periodic Checkpoint Saving (Rank 0 only)
                if rank == 0 and (global_step % save_every == 0 or global_step == max_steps):
                    ckpt_path = trainer.checkpoint_manager.save(
                        step=global_step,
                        model=trainer.raw_model,
                        optimizer=optimizer,
                        tokens_seen=tokens_seen,
                        metrics={"val_loss": step_metrics.get("val_loss", step_metrics["loss"])},
                    )
                    training_logger.log_checkpoint(global_step, ckpt_path.name)
                    logger.info(f"Checkpoint saved at step {global_step} -> {ckpt_path.name}")

                    # Trigger background sync to cloud
                    cloud_sync.sync_experiment(exp_dir)

                if is_distributed:
                    dist.barrier()

                history.append(step_metrics)

                if rank == 0:
                    training_logger.log_step(global_step, step_metrics)
                    logger.info(
                        f"Step {global_step:4d}/{max_steps} | Loss: {step_metrics['loss']:.4f} | "
                        f"LR: {step_metrics['learning_rate']:.2e} | Tokens/sec: {step_metrics['tokens_per_sec']:,}"
                    )

        total_training_time = round(time.time() - start_time, 2)

        if rank == 0:
            summary = {
                "world_size": world_size,
                "total_training_time_s": total_training_time,
                "max_steps": max_steps,
                "completed_steps": global_step,
                "tokens_processed": tokens_seen,
                "final_train_loss": history[-1]["loss"] if history else None,
                "final_val_loss": history[-1].get("val_loss") if history else None,
            }
            training_logger.log_summary(summary)
            generate_training_curves(history, exp_dir)

            logger.info(f"Pretraining Run Complete! Total Time: {total_training_time}s")

    finally:
        if "training_logger" in locals():
            training_logger.close()
        cleanup_ddp_environment(is_distributed)


if __name__ == "__main__":
    main()
