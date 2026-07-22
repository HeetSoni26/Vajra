from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny local tokenizer corpus for smoke tests.")
    parser.add_argument("--output_dir", default="data/tokenizer_corpus")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"text": "Transformers use attention, residual streams, RMSNorm, and SwiGLU blocks."},
        {"text": "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"},
        {"text": "For all x \\in \\mathbb{R}, x^2 \\ge 0."},
    ]
    with (out / "sample.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
