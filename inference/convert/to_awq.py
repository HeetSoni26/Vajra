from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ quantization wrapper.")
    parser.add_argument("--model_path", default="checkpoints/final/hf")
    parser.add_argument("--output_path", default="models/awq-4bit")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    cmd = [
        "python",
        "-m",
        "awq.quantize",
        "--model_path",
        args.model_path,
        "--output_path",
        args.output_path,
        "--w_bit",
        "4",
        "--q_group_size",
        "128",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
