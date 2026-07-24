from pathlib import Path
from datetime import datetime


class TrainingDashboard:
    """Generates a Markdown dashboard tracking training progress."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.dashboard_file = self.output_dir / "DASHBOARD.md"

    def update(self, metrics: dict):
        content = f"""# Vajra Training Dashboard
_Last Updated: {datetime.now().isoformat()}_

## Global Progress
- **Hardware**: {metrics.get("hardware", "A100 x8")}
- **Dataset**: {metrics.get("dataset", "Vajra-370M-Production-Corpus")}
- **Total Tokens processed**: {metrics.get("tokens_processed", 0):,}
- **ETA**: {metrics.get("eta", "Unknown")}

## Current Step Metrics
- **Step**: {metrics.get("step", 0)}
- **Loss**: {metrics.get("loss", 0.0):.4f}
- **Throughput (Tokens/s)**: {metrics.get("throughput", 0):.2f}
- **VRAM Usage**: {metrics.get("vram_usage", "0GB")}

## Checkpoints
- **Latest**: {metrics.get("latest_checkpoint", "None")}
- **Best**: {metrics.get("best_checkpoint", "None")}
"""
        with open(self.dashboard_file, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == "__main__":
    dash = TrainingDashboard(".")
    dash.update(
        {
            "hardware": "RTX 4090 x2",
            "dataset": "Code-Corpus",
            "tokens_processed": 10500234,
            "eta": "12 hours 4 mins",
            "step": 2500,
            "loss": 2.145,
            "throughput": 4500.5,
            "vram_usage": "22.4 GB",
            "latest_checkpoint": "checkpoint-2000",
            "best_checkpoint": "checkpoint-1000",
        }
    )
    print("Dashboard generated.")
