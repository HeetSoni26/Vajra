import json
from pathlib import Path
from typing import Any


class MetricsHistory:
    """
    Manages historical metrics tracking, moving averages, best/min/max values.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.history_file = self.run_dir / "metrics_history.jsonl"
        self.history: list[dict[str, Any]] = []
        self.best_metrics: dict[str, Any] = {}

    def log_metrics(self, step: int, metrics: dict[str, float]):
        entry = {"step": step, **metrics}
        self.history.append(entry)

        # Track minimum loss and maximum throughput
        if "loss" in metrics:
            if (
                "min_loss" not in self.best_metrics
                or metrics["loss"] < self.best_metrics["min_loss"]
            ):
                self.best_metrics["min_loss"] = metrics["loss"]
                self.best_metrics["min_loss_step"] = step

        if "throughput" in metrics or "tokens_per_sec" in metrics:
            t = metrics.get("throughput", metrics.get("tokens_per_sec", 0))
            if "max_throughput" not in self.best_metrics or t > self.best_metrics["max_throughput"]:
                self.best_metrics["max_throughput"] = t

        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_steps_logged": len(self.history),
            "best": self.best_metrics,
            "latest": self.history[-1] if self.history else {},
        }
