import csv
import json
from pathlib import Path
from typing import Any


class ExportManager:
    @staticmethod
    def export_summary(run_metadata: dict[str, Any], output_path: Path, format: str = "json"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(run_metadata, f, indent=2)
        elif format == "csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for k, v in run_metadata.items():
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v)
                    writer.writerow([k, v])
        elif format == "md":
            with open(output_path, "w") as f:
                f.write(f"# Run Summary: {run_metadata.get('run_name')}\n")
                f.write(f"**Status**: {run_metadata.get('status')}\n")
                f.write(f"**Duration**: {run_metadata.get('duration', 0):.2f}s\n\n")
                f.write("## Metrics\n```json\n")
                f.write(json.dumps(run_metadata.get("metrics_summary", {}), indent=2))
                f.write("\n```\n")
        else:
            raise ValueError(f"Unknown export format {format}")
