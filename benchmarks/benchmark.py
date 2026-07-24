import argparse
import csv
import json
import time
from pathlib import Path

from benchmarks.runners.performance import run_performance_benchmark
from benchmarks.runners.quality import run_quality_benchmark
from utils.logging import setup_logger

logger = setup_logger("benchmark")


def save_reports(out_dir: Path, ckpt_name: str, results: dict) -> None:
    """Save benchmark.json, benchmark.csv, and benchmark.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / "benchmark.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = out_dir / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in results.items():
            if k != "generated_sample":
                writer.writerow([k, v])

    # Save MD
    md_path = out_dir / "benchmark.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Benchmark Report for {ckpt_name}\n\n")
        f.write(f"**Generated:** {results.get('timestamp')}\n\n")
        
        f.write("## Quality Metrics\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Validation Loss | {results.get('validation_loss')} |\n")
        f.write(f"| Perplexity | {results.get('perplexity')} |\n")
        f.write(f"| Distinct-1 | {results.get('distinct_1')} |\n")
        f.write(f"| Distinct-2 | {results.get('distinct_2')} |\n")
        f.write(f"| Repetition Rate | {results.get('repetition_rate')} |\n")
        f.write(f"| Avg Gen Length | {results.get('average_generated_length')} |\n")
        f.write(f"| Gen Time (s) | {results.get('generation_time_sec')} |\n\n")
        
        f.write("## Performance Metrics\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Inference Latency (First Token ms) | {results.get('inference_latency_first_token_ms')} |\n")
        f.write(f"| Tokens / Sec | {results.get('tokens_per_sec')} |\n")
        f.write(f"| Loading Time (s) | {results.get('loading_time_sec')} |\n")
        f.write(f"| Model Size (MB) | {results.get('model_size_mb')} |\n")
        f.write(f"| Memory RAM (MB) | {results.get('memory_ram_mb')} |\n")
        f.write(f"| Memory VRAM (MB) | {results.get('memory_vram_mb')} |\n")
        f.write(f"| Model Parameters | {results.get('model_parameters')} |\n\n")
        
        f.write("## Sample Generation\n")
        f.write("```text\n")
        f.write(str(results.get("generated_sample", "")))
        f.write("\n```\n")

    logger.info(f"Saved benchmark reports to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vajra benchmark suite.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--eval-dir", default="evaluations", help="Directory containing evaluation metrics")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    eval_dir = Path(args.eval_dir)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Starting benchmark for {checkpoint_path.name}")
    
    # Run runners
    logger.info("Running Quality benchmarks...")
    quality_results = run_quality_benchmark(checkpoint_path, config_path, eval_dir)
    
    logger.info("Running Performance benchmarks...")
    performance_results = run_performance_benchmark(checkpoint_path, config_path)
    
    # Merge results
    results = {
        "checkpoint": checkpoint_path.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    results.update(quality_results)
    results.update(performance_results)
    
    # Determine step and output directory
    step = checkpoint_path.stem.split("_")[-1]
    out_dir = Path("benchmarks") / "reports" / f"checkpoint_{step}"
    
    # Save outputs
    save_reports(out_dir, f"checkpoint_{step}", results)
    logger.info("Benchmarking complete.")


if __name__ == "__main__":
    main()
