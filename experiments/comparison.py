from typing import Dict, Any
import json
from pathlib import Path


class ComparisonEngine:
    @staticmethod
    def compare_runs(run_a_dir: Path, run_b_dir: Path) -> Dict[str, Any]:
        with open(run_a_dir / "run_metadata.json") as f:
            meta_a = json.load(f)
        with open(run_b_dir / "run_metadata.json") as f:
            meta_b = json.load(f)

        # Extract best metrics
        best_a = meta_a.get("metrics_summary", {}).get("best", {})
        best_b = meta_b.get("metrics_summary", {}).get("best", {})

        metric_diffs = {}
        for k, va in best_a.items():
            if k in best_b:
                vb = best_b[k]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != 0:
                    diff = vb - va
                    pct = (diff / va) * 100
                    metric_diffs[k] = {"run_a": va, "run_b": vb, "diff": diff, "pct_change": pct}

        return {
            "run_a": meta_a["run_name"],
            "run_b": meta_b["run_name"],
            "metric_comparisons": metric_diffs,
        }
