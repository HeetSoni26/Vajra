from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to GGUF using llama.cpp.")
    parser.add_argument("--llama_cpp", required=True, help="Path to llama.cpp checkout")
    parser.add_argument("--model", default="checkpoints/final/hf")
    parser.add_argument("--outfile", default="models/gguf/model-f16.gguf")
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python", str(Path(args.llama_cpp) / "convert_hf_to_gguf.py"), args.model, "--outfile", args.outfile, "--outtype", args.outtype]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
