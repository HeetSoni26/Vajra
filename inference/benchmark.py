"""Generation performance benchmarking — KV cache vs no cache, sampling strategies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from inference.engine import InferenceEngine, GenerationConfig
from utils.environment import get_memory_info
from utils.file_utils import write_json
from utils.logging import setup_logger

logger = setup_logger("generation_benchmark")


def _bench_single(
    engine: InferenceEngine,
    prompt: str,
    gen_cfg: GenerationConfig,
    label: str,
    num_runs: int = 3,
) -> dict[str, Any]:
    """Benchmark a single generation configuration."""
    times = []
    first_token_times = []
    output_lens = []

    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.perf_counter()
        # Use streaming to measure first-token latency
        first_token_t = None
        token_count = 0
        for _tok in engine.generate_stream(prompt, gen_cfg):
            if first_token_t is None:
                first_token_t = time.perf_counter() - t0
            token_count += 1
        total_t = time.perf_counter() - t0

        times.append(total_t)
        first_token_times.append(first_token_t or total_t)
        output_lens.append(token_count)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_lens) / len(output_lens)
    avg_first = sum(first_token_times) / len(first_token_times)

    return {
        "label": label,
        "runs": num_runs,
        "first_token_latency_ms": round(avg_first * 1000, 2),
        "average_total_time_ms": round(avg_time * 1000, 2),
        "average_tokens_generated": round(avg_tokens, 1),
        "tokens_per_sec": round(avg_tokens / max(avg_time, 0.001), 2),
    }


def run_generation_benchmark(
    config_path: str | Path = "configs/training/pretrain_tiny.yaml",
    output_path: str | Path = "generation_benchmark.json",
) -> dict[str, Any]:
    """Run full generation benchmark suite."""
    engine = InferenceEngine.from_config(config_path)
    prompt = "The meaning of life is"
    max_tokens = 32

    benchmarks: list[dict[str, Any]] = []

    # 1. Greedy — KV Cache ON
    benchmarks.append(_bench_single(engine, prompt, GenerationConfig(
        max_new_tokens=max_tokens, temperature=0.0, do_sample=False, use_kv_cache=True,
    ), "greedy_kv_cache_on"))

    # 2. Greedy — KV Cache OFF
    benchmarks.append(_bench_single(engine, prompt, GenerationConfig(
        max_new_tokens=max_tokens, temperature=0.0, do_sample=False, use_kv_cache=False,
    ), "greedy_kv_cache_off"))

    # 3. Top-K (k=50, T=0.8)
    benchmarks.append(_bench_single(engine, prompt, GenerationConfig(
        max_new_tokens=max_tokens, temperature=0.8, top_k=50, do_sample=True, use_kv_cache=True, seed=42,
    ), "top_k_50"))

    # 4. Top-P (p=0.9, T=0.8)
    benchmarks.append(_bench_single(engine, prompt, GenerationConfig(
        max_new_tokens=max_tokens, temperature=0.8, top_p=0.9, do_sample=True, use_kv_cache=True, seed=42,
    ), "top_p_0.9"))

    # 5. Temperature sweep
    for temp in [0.3, 0.7, 1.0, 1.5]:
        benchmarks.append(_bench_single(engine, prompt, GenerationConfig(
            max_new_tokens=max_tokens, temperature=temp, do_sample=True, use_kv_cache=True, seed=42,
        ), f"temperature_{temp}"))

    # KV Cache speedup
    kv_on = next(b for b in benchmarks if b["label"] == "greedy_kv_cache_on")
    kv_off = next(b for b in benchmarks if b["label"] == "greedy_kv_cache_off")
    kv_speedup = round(kv_off["average_total_time_ms"] / max(kv_on["average_total_time_ms"], 0.01), 2)

    mem_info = get_memory_info()

    report = {
        "device": str(engine.device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "model_parameters": sum(p.numel() for p in engine.model.parameters()),
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "kv_cache_speedup": f"{kv_speedup}x",
        "peak_ram_mb": mem_info.get("ram_used_mb", 0),
        "peak_vram_mb": mem_info.get("cuda_max_allocated_mb", 0),
        "benchmarks": benchmarks,
    }

    write_json(report, output_path)
    logger.info(f"Generation benchmark saved to {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation Performance Benchmark")
    parser.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    parser.add_argument("--output", default="generation_benchmark.json")
    args = parser.parse_args()

    import json
    report = run_generation_benchmark(args.config, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
