from typing import Any


class ComparisonEngine:
    """
    Compares two evaluation reports.
    """

    @staticmethod
    def compare(report_a: dict[str, float], report_b: dict[str, float]) -> dict[str, Any]:
        """
        Compare metrics between model A (baseline) and model B.
        """
        comparison = {}
        for metric in report_a:
            if metric in report_b:
                val_a = report_a[metric]
                val_b = report_b[metric]

                # Assume lower is better for loss/ppl/bpb, higher is better for acc/throughput
                lower_is_better = metric in ["loss", "perplexity", "bpb"]

                if val_a == 0:
                    pct_change = 0.0
                else:
                    pct_change = ((val_b - val_a) / val_a) * 100.0

                improvement = -pct_change if lower_is_better else pct_change

                comparison[metric] = {
                    "baseline": val_a,
                    "target": val_b,
                    "diff": val_b - val_a,
                    "improvement_pct": improvement,
                    "regression": improvement < 0,
                }

        return comparison
