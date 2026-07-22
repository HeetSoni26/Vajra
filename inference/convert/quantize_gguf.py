from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize GGUF with llama.cpp quantizer.")
    parser.add_argument("--quantizer", required=True, help="Path to llama-quantize binary")
    parser.add_argument("--input", default="models/gguf/model-f16.gguf")
    parser.add_argument("--output", default="models/gguf/model-Q4_K_M.gguf")
    parser.add_argument("--type", default="Q4_K_M")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.quantizer, args.input, args.output, args.type]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
