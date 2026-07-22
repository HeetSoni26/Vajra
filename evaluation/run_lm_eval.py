from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for lm-evaluation-harness.")
    parser.add_argument("--config", default="configs/eval/benchmarks.yaml")
    parser.add_argument("--model_path", default="checkpoints/final/hf")
    parser.add_argument("--output_path", default="evaluation/results/lm_eval")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = list(cfg.get("primary", {}).keys())
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model_path},dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--batch_size", "8",
        "--output_path", args.output_path,
        "--log_samples",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
