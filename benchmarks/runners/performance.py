import time
from pathlib import Path
from typing import Any

import torch

from inference.engine import GenerationConfig, InferenceEngine
from utils.environment import get_memory_info


def run_performance_benchmark(
    checkpoint_path: Path, config_path: Path
) -> dict[str, Any]:
    """Run performance benchmarks including latency, throughput, memory, and model size."""
    results: dict[str, Any] = {}

    # 1. Loading Time & Model Size
    t0 = time.perf_counter()
    engine = InferenceEngine.from_config(config_path, str(checkpoint_path))
    loading_time = time.perf_counter() - t0

    model_parameters = sum(p.numel() for p in engine.model.parameters())
    # Approximation of size in MB assuming FP32/FP16 (fallback to FP32)
    # Using 4 bytes per param for FP32 as baseline
    element_size = 4
    for p in engine.model.parameters():
        element_size = p.element_size()
        break
        
    model_size_mb = (model_parameters * element_size) / (1024 * 1024)

    results["loading_time_sec"] = round(loading_time, 4)
    results["model_size_mb"] = round(model_size_mb, 2)
    results["model_parameters"] = model_parameters

    # 2. Memory Usage (Peak RAM/VRAM)
    mem_info = get_memory_info()
    results["memory_ram_mb"] = mem_info.get("ram_used_mb", 0)
    results["memory_vram_mb"] = mem_info.get("cuda_max_allocated_mb", 0)

    # 3. Inference Latency, Tokens/sec, Throughput
    prompt = "The meaning of life is"
    gen_cfg = GenerationConfig(
        max_new_tokens=64, temperature=0.0, do_sample=False, use_kv_cache=True
    )

    times = []
    first_token_times = []
    output_lens = []

    for _ in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        first_token_t = None
        token_count = 0
        
        for _tok in engine.generate_stream(prompt, gen_cfg):
            if first_token_t is None:
                first_token_t = time.perf_counter() - t_start
            token_count += 1
            
        total_t = time.perf_counter() - t_start

        times.append(total_t)
        first_token_times.append(first_token_t or total_t)
        output_lens.append(token_count)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_lens) / len(output_lens)
    avg_first = sum(first_token_times) / len(first_token_times)
    
    tokens_per_sec = avg_tokens / max(avg_time, 0.001)

    results["inference_latency_first_token_ms"] = round(avg_first * 1000, 2)
    results["inference_latency_total_ms"] = round(avg_time * 1000, 2)
    results["tokens_per_sec"] = round(tokens_per_sec, 2)
    results["throughput_tokens_per_sec"] = round(tokens_per_sec, 2)  # Alias for clarity

    return results
