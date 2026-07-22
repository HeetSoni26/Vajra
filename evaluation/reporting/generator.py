import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

class ReportGenerator:
    """
    Generates evaluation reports in JSON, CSV, and Markdown formats.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, model_id: str, metrics: Dict[str, float], hardware_info: Dict[str, str] = None) -> Path:
        timestamp = datetime.now().isoformat()
        
        report_data = {
            "model_id": model_id,
            "timestamp": timestamp,
            "metrics": metrics,
            "hardware": hardware_info or {}
        }
        
        # JSON
        json_path = self.output_dir / f"{model_id}_eval.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        # CSV
        csv_path = self.output_dir / f"{model_id}_eval.csv"
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model_id", "timestamp"] + list(metrics.keys()))
            writer.writerow([model_id, timestamp] + list(metrics.values()))
            
        # Markdown
        md_path = self.output_dir / f"{model_id}_eval.md"
        with open(md_path, "w") as f:
            f.write(f"# Evaluation Report: {model_id}\n")
            f.write(f"**Date**: {timestamp}\n\n")
            f.write("## Metrics\n\n")
            for k, v in metrics.items():
                f.write(f"- **{k}**: {v:.4f}\n")
                
        return json_path
