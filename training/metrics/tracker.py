import csv
import json
import time
from pathlib import Path
from typing import Any


class MetricsTracker:
    """
    Tracks and logs training metrics to console, JSON, CSV.
    Optionally logs to TensorBoard if available.
    """

    def __init__(self, output_dir: str | Path = "logs", log_interval: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.csv_path = self.output_dir / "training_log.csv"
        self.json_path = self.output_dir / "training_log.jsonl"

        # Init CSV header
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "loss",
                        "learning_rate",
                        "grad_norm",
                        "tokens_processed",
                        "samples_processed",
                        "tokens_per_sec",
                        "step_time",
                        "epoch",
                    ]
                )

        self.start_time = time.time()
        self.last_log_time = self.start_time

        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        except ImportError:
            pass

    def log(self, step: int, metrics: dict[str, Any]):
        if step % self.log_interval != 0 and step != 1:
            return

        current_time = time.time()
        step_time = current_time - self.last_log_time
        self.last_log_time = current_time

        metrics["step_time"] = step_time

        # Log to Console
        print(
            f"Step {step} | Loss: {metrics.get('loss', 0.0):.4f} | LR: {metrics.get('learning_rate', 0.0):.2e} | "
            f"Tokens/s: {metrics.get('tokens_per_sec', 0.0):.2f}"
        )

        # Log to JSONL
        with open(self.json_path, "a") as f:
            log_entry = {"step": step, **metrics}
            f.write(json.dumps(log_entry) + "\n")

        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    metrics.get("loss", ""),
                    metrics.get("learning_rate", ""),
                    metrics.get("grad_norm", ""),
                    metrics.get("tokens_processed", ""),
                    metrics.get("samples_processed", ""),
                    metrics.get("tokens_per_sec", ""),
                    metrics.get("step_time", ""),
                    metrics.get("epoch", ""),
                ]
            )

        # Log to TensorBoard
        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_in_step: int,
        samples_in_step: int,
        grad_norm: float = 0.0,
        tokens_per_sec: float = 0.0,
        val_loss: float | None = None,
    ) -> dict[str, Any]:
        metrics = {
            "loss": loss,
            "learning_rate": lr,
            "tokens_processed": tokens_in_step,
            "samples_processed": samples_in_step,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "val_loss": val_loss,
        }
        self.log(step, metrics)
        return metrics

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
