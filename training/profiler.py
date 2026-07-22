from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from model import FoundationLM, ModelConfig
from training.data_loader import create_dataloaders
from training.optimizer import build_optimizer
from training.trainer import Trainer
from utils.environment import get_device, get_memory_info, set_seed
from utils.file_utils import ensure_dir, write_json
from utils.logging import setup_logger

logger = setup_logger("profiler")


def run_benchmark(
    config_path: str | Path = "configs/training/pretrain_tiny.yaml",
    num_benchmark_steps: int = 15,
    precision: str = "fp32",
    use_gradient_checkpointing: bool = False,
    output_report: str | Path = "benchmark_report.json",
) -> dict[str, Any]:
    """Run performance profiling benchmark and generate benchmark_report.json."""
    set_seed(1337)
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    model_cfg = ModelConfig.from_yaml(cfg["model_config"])
    # Dynamic override for gradient checkpointing benchmark
    if use_gradient_checkpointing:
        object.__setattr__(model_cfg, "use_gradient_checkpointing", True)

    model = FoundationLM(model_cfg)
    device = get_device()

    lr = float(cfg["learning_rate"])
    optimizer = build_optimizer(model, lr=lr, weight_decay=float(cfg["weight_decay"]))

    seq_len = int(cfg["sequence_length"])
    batch_size = int(cfg["micro_batch_size"])
    data_dir = Path(cfg.get("data_dir", "data/tokenized"))

    train_loader, _ = create_dataloaders(data_dir, seq_len, batch_size)
    trainer = Trainer(model, optimizer, device=device, precision=precision)

    # Warmup step
    train_iter = iter(train_loader)
    batch = next(train_iter)
    trainer.train_step(batch, step=0, is_accum_step=True)

    # Benchmark loop
    step_times: list[float] = []
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    opt_times: list[float] = []
    dl_times: list[float] = []

    start_bench_time = time.time()

    for step in range(1, num_benchmark_steps + 1):
        dl_start = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        dl_time = time.time() - dl_start
        dl_times.append(dl_time)

        step_res = trainer.train_step(batch, step=step, is_accum_step=True)
        if step_res and "timing" in step_res:
            timing = step_res["timing"]
            step_times.append(timing["step_time_ms"])
            fwd_times.append(timing["forward_time_ms"])
            bwd_times.append(timing["backward_time_ms"])
            opt_times.append(timing["optimizer_time_ms"])

    total_bench_duration = max(0.001, time.time() - start_bench_time)
    total_tokens = num_benchmark_steps * batch_size * seq_len
    tokens_per_sec = round(total_tokens / total_bench_duration, 2)
    samples_per_sec = round((num_benchmark_steps * batch_size) / total_bench_duration, 2)

    # System memory reporting
    mem_info = get_memory_info()
    peak_vram_mb = mem_info.get("vram_allocated_mb", 0.0)
    peak_ram_mb = mem_info.get("ram_used_mb", 0.0)

    report = {
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "device_information": str(device),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "precision_mode": precision,
        "gradient_checkpointing_enabled": use_gradient_checkpointing,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
        "step_time_ms": round(float(sum(step_times) / max(1, len(step_times))), 2),
        "dataloader_time_ms": round(float((sum(dl_times) / max(1, len(dl_times))) * 1000), 2),
        "forward_time_ms": round(float(sum(fwd_times) / max(1, len(fwd_times))), 2),
        "backward_time_ms": round(float(sum(bwd_times) / max(1, len(bwd_times))), 2),
        "optimizer_time_ms": round(float(sum(opt_times) / max(1, len(opt_times))), 2),
        "peak_gpu_memory_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
        "throughput_scaling_efficiency": 1.00,  # 100% baseline for single process
    }

    report_path = Path(output_report)
    ensure_dir(report_path.parent)
    write_json(report, report_path)

    logger.info(f"Performance profiling complete! Report written to: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance Profiling & Benchmark Engine")
    parser.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--output_report", default="benchmark_report.json")
    args = parser.parse_args()

    report = run_benchmark(
        config_path=args.config,
        precision=args.precision,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        output_report=args.output_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
