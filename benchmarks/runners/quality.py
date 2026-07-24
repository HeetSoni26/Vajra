import json
import time
from pathlib import Path
from typing import Any

from inference.engine import GenerationConfig, InferenceEngine


def compute_diversity_metrics(text: str) -> dict[str, float]:
    """Compute Distinct-1, Distinct-2, and repetition rate for a text."""
    tokens = text.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}

    unigrams = set(tokens)
    bigrams = set(zip(tokens[:-1], tokens[1:]))

    distinct_1 = len(unigrams) / total_tokens
    distinct_2 = len(bigrams) / max(1, (total_tokens - 1))
    repetition_rate = 1.0 - distinct_1

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_rate": round(repetition_rate, 4),
    }


def run_quality_benchmark(
    checkpoint_path: Path, config_path: Path, eval_dir: Path
) -> dict[str, Any]:
    """Run quality benchmarks including fetching loss/perplexity and generation quality."""
    results: dict[str, Any] = {}

    # 1. Fetch Validation Loss and Perplexity from evaluation framework
    # Assumes evaluate_all.py or evaluate.py has already run for this checkpoint
    step = checkpoint_path.stem.split("_")[-1]
    metrics_file = eval_dir / f"checkpoint_{step}" / "metrics.json"

    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as f:
            eval_metrics = json.load(f)
            results["validation_loss"] = eval_metrics.get("validation_loss", None)
            results["perplexity"] = eval_metrics.get("perplexity", None)
    else:
        results["validation_loss"] = None
        results["perplexity"] = None

    # 2. Text Generation Quality
    engine = InferenceEngine.from_config(config_path, str(checkpoint_path))
    prompt = "Artificial intelligence will transform the future by"
    gen_cfg = GenerationConfig(
        max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True, seed=42
    )

    t0 = time.perf_counter()
    tokens_generated = 0
    generated_text = ""

    for token in engine.generate_stream(prompt, gen_cfg):
        generated_text += token
        tokens_generated += 1

    generation_time = time.perf_counter() - t0

    diversity = compute_diversity_metrics(generated_text)

    results.update(diversity)
    results["average_generated_length"] = tokens_generated
    results["generation_time_sec"] = round(generation_time, 4)
    results["tokens_generated"] = tokens_generated
    results["generated_sample"] = generated_text

    return results
