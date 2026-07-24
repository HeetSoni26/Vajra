from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate/export a HuggingFace-compatible checkpoint directory."
    )
    parser.add_argument("--model_dir", default="checkpoints/final/hf")
    args = parser.parse_args()
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (Path(args.model_dir) / name).exists()]
    if missing:
        raise SystemExit(f"Missing HuggingFace files: {missing}")
    print({"status": "ok", "model_dir": args.model_dir})


if __name__ == "__main__":
    main()
