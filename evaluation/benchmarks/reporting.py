import csv
import json
from pathlib import Path
from typing import Any


class BenchmarkReporter:
    """Generates benchmark reports in various formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, run_name: str, results: dict[str, dict[str, Any]]):
        path = self.output_dir / f"{run_name}_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path

    def generate_csv(self, run_name: str, results: dict[str, dict[str, Any]]):
        path = self.output_dir / f"{run_name}_results.csv"
        if not results:
            return path

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Benchmark", "Metric", "Value"])
            for bench, metrics in results.items():
                for k, v in metrics.items():
                    writer.writerow([bench, k, v])
        return path

    def generate_markdown(self, run_name: str, results: dict[str, dict[str, Any]]):
        path = self.output_dir / f"{run_name}_report.md"

        lines = [f"# Benchmark Report: {run_name}", ""]
        lines.append("| Benchmark | Accuracy / Primary Metric | Latency | Tokens/sec |")
        lines.append("|---|---|---|---|")

        for bench, metrics in results.items():
            primary = metrics.get(
                "accuracy", metrics.get("exact_match", metrics.get("pass@1", 0.0))
            )
            lat = metrics.get("latency", 0.0)
            tps = metrics.get("tokens_per_sec", 0.0)
            lines.append(f"| {bench} | {primary:.4f} | {lat:.2f}s | {tps:.2f} |")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def generate_leaderboard(self, historical_runs: dict[str, dict[str, dict[str, Any]]]):
        """Generates a leaderboard from historical runs."""
        path = self.output_dir / "leaderboard.md"

        lines = ["# Vajra Model Leaderboard", ""]
        lines.append("| Run Name | Average Score |")
        lines.append("|---|---|")

        averages = []
        for run, results in historical_runs.items():
            scores = [
                m.get("accuracy", m.get("exact_match", m.get("pass@1", 0.0)))
                for m in results.values()
            ]
            avg = sum(scores) / len(scores) if scores else 0
            averages.append((run, avg))

        averages.sort(key=lambda x: x[1], reverse=True)

        for run, avg in averages:
            lines.append(f"| {run} | {avg:.4f} |")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
