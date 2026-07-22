from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for bigcode-evaluation-harness.")
    parser.add_argument("--model_path", default="checkpoints/final/hf")
    parser.add_argument("--task", default="humaneval")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    cmd = [
        "python", "main.py",
        "--model", args.model_path,
        "--tasks", args.task,
        "--n_samples", "20",
        "--batch_size", "10",
        "--temperature", "0.2",
        "--allow_code_execution",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
