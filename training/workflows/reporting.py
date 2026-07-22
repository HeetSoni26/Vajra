import json
from pathlib import Path
from typing import List, Dict, Any

class TrainingReportGenerator:
    """Generates Markdown reports summarising training progress."""
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, step: int, metrics_history: List[Dict[str, Any]], samples: List[str]):
        """Generates a markdown report for the given step."""
        report_path = self.output_dir / f"training_report_step_{step}.md"
        
        loss_val = metrics_history[-1].get("loss", "N/A") if metrics_history else "N/A"
        
        lines = [
            f"# Vajra Training Report (Step {step})",
            "",
            "## Metrics Snapshot",
            f"- **Loss**: {loss_val}",
            ""
        ]
        
        if metrics_history:
            lines.append("## Recent History")
            lines.append("```json")
            lines.append(json.dumps(metrics_history[-min(5, len(metrics_history)):], indent=2))
            lines.append("```")
            lines.append("")
            
        if samples:
            lines.append("## Generated Samples")
            for i, sample in enumerate(samples):
                lines.append(f"### Sample {i+1}")
                lines.append(f"{sample}")
                lines.append("")
                
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
