from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a benchmark comparison markdown table from CSV.")
    parser.add_argument("--csv", default="evaluation/results/baselines.csv")
    parser.add_argument("--output", default="evaluation/results/comparison.md")
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model,params,tokens,mmlu,arc_c,hellaswag,winogrande,piqa,gsm8k,humaneval\n")
    rows = list(csv.DictReader(path.open()))
    headers = rows[0].keys() if rows else ["model", "params", "tokens", "mmlu", "arc_c", "hellaswag", "winogrande", "piqa", "gsm8k", "humaneval"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
