"""Model evaluation framework for perplexity, cross-entropy, and memmap dataset benchmarks."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from model import FoundationLM, ModelConfig
from training.checkpoint import load_checkpoint
from training.data_loader import MemmapTokenDataset
from utils.environment import get_device, get_memory_info
from utils.file_utils import write_json
from utils.logging import setup_logger

logger = setup_logger("evaluator")


class ModelEvaluator:
    """Evaluate a FoundationLM model on memmap binary datasets."""

    def __init__(
        self,
        model: FoundationLM,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
    ) -> "ModelEvaluator":
        """Construct evaluator from a training YAML config."""
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        model_cfg = ModelConfig.from_yaml(cfg["model_config"])
        model = FoundationLM(model_cfg)

        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                load_checkpoint(ckpt, model)
                logger.info(f"Loaded checkpoint: {ckpt}")

        device = get_device()
        return cls(model, device)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_path: str | Path,
        sequence_length: int = 128,
        batch_size: int = 4,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate perplexity and cross-entropy on a memmap binary file.

        Args:
            data_path: Path to a ``.bin`` memmap uint32 token file.
            sequence_length: Context window for evaluation.
            batch_size: Micro-batch size.
            max_batches: Limit evaluation to N batches (for quick benchmarks).

        Returns:
            Dict with loss, perplexity, and throughput metrics.
        """
        dataset = MemmapTokenDataset(data_path, sequence_length)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        total_loss = 0.0
        total_tokens = 0
        total_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=(self.dtype != torch.float32),
            ):
                out = self.model(input_ids, labels=labels)

            loss_val = float(out["loss"].item())
            batch_tokens = input_ids.numel()

            total_loss += loss_val * batch_tokens
            total_tokens += batch_tokens
            total_batches += 1

        elapsed = max(0.001, time.time() - start_time)
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20.0))  # Clamp to avoid overflow

        mem_info = get_memory_info()

        result = {
            "dataset": str(data_path),
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "total_tokens": total_tokens,
            "avg_cross_entropy": round(avg_loss, 6),
            "perplexity": round(perplexity, 4),
            "bits_per_character": round(avg_loss / math.log(2), 6),
            "tokens_per_sec": round(total_tokens / elapsed, 2),
            "evaluation_time_s": round(elapsed, 2),
            "memory_info": mem_info,
        }

        logger.info(
            f"Evaluation on {Path(data_path).name}: "
            f"Loss={avg_loss:.4f} | PPL={perplexity:.2f} | "
            f"Tokens/sec={result['tokens_per_sec']:,.0f}"
        )
        return result

    def evaluate_all(
        self,
        data_dir: str | Path,
        sequence_length: int = 128,
        batch_size: int = 4,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate on all available splits (val.bin, test.bin)."""
        data_dir = Path(data_dir)
        report: dict[str, Any] = {"splits": {}}

        for split_name in ["val.bin", "test.bin", "train.bin"]:
            split_path = data_dir / split_name
            if split_path.exists() and split_path.stat().st_size > 0:
                result = self.evaluate_dataset(
                    split_path, sequence_length, batch_size, max_batches
                )
                report["splits"][split_name] = result

        report["model_info"] = {
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_layers,
            "vocab_size": self.model.config.vocab_size,
        }

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FoundationLM on memmap datasets")
    parser.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt file")
    parser.add_argument("--data_dir", default="data/tokenized")
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()

    evaluator = ModelEvaluator.from_config(args.config, args.checkpoint)
    report = evaluator.evaluate_all(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    write_json(report, args.output)
    logger.info(f"Evaluation report saved to {args.output}")


if __name__ == "__main__":
    main()
