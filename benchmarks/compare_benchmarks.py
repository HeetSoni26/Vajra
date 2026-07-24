import csv
import json
from pathlib import Path

from utils.logging import setup_logger

logger = setup_logger("compare_benchmarks")


def generate_comparison_report(reports_dir: Path) -> None:
    """Scan benchmark.json files and generate comparison reports."""
    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        return

    results = []
    for benchmark_file in reports_dir.glob("*/benchmark.json"):
        try:
            data = json.loads(benchmark_file.read_text(encoding="utf-8"))
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {benchmark_file}: {e}")

    if not results:
        logger.error("No benchmark results found.")
        return

    # Sort by step, then by loss
    def _get_step(data: dict) -> int:
        name = data.get("checkpoint", "")
        try:
            return int(name.replace(".pt", "").split("_")[-1])
        except ValueError:
            return 0

    results.sort(key=lambda x: _get_step(x))

    # 1. Save comparison.json
    json_path = reports_dir / "comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Save comparison.csv
    csv_path = reports_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        headers = [
            "Checkpoint", "Validation Loss", "Perplexity", 
            "Distinct-1", "Tokens/Sec", "Inference Latency (ms)", "Memory RAM (MB)"
        ]
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([
                r.get("checkpoint"),
                r.get("validation_loss"),
                r.get("perplexity"),
                r.get("distinct_1"),
                r.get("tokens_per_sec"),
                r.get("inference_latency_first_token_ms"),
                r.get("memory_ram_mb"),
            ])

    # 3. Save comparison.md
    md_path = reports_dir / "comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Vajra Benchmark Comparison\n\n")
        f.write("| Checkpoint | Val Loss | Perplexity | Tokens/Sec | Latency (ms) | Distinct-1 |\n")
        f.write("|------------|----------|------------|------------|--------------|------------|\n")
        for r in results:
            ckpt = r.get("checkpoint", "-")
            loss = r.get("validation_loss", "-")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            ppl = r.get("perplexity", "-")
            if isinstance(ppl, float):
                ppl = f"{ppl:.4f}"
            tps = r.get("tokens_per_sec", "-")
            lat = r.get("inference_latency_first_token_ms", "-")
            d1 = r.get("distinct_1", "-")
            f.write(f"| {ckpt} | {loss} | {ppl} | {tps} | {lat} | {d1} |\n")

    logger.info("Comparison reports generated successfully.")


if __name__ == "__main__":
    import sys
    reports_dir_arg = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/reports")
    generate_comparison_report(reports_dir_arg)
