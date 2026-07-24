from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from utils.file_utils import write_json
from utils.logging import setup_logger

logger = setup_logger("dataset_verifier")


def verify_dataset(
    tokenized_dir: str | Path = "data/tokenized",
    tokenizer_path: str | Path = "tokenizer/v1.0",
) -> dict[str, Any]:
    """Verify generated binary dataset files and generate a validation report."""
    tokenized_dir = Path(tokenized_dir)
    report_path = tokenized_dir / "dataset_statistics.json"

    splits = ["train.bin", "val.bin", "test.bin"]
    split_stats: dict[str, Any] = {}
    total_tokens = 0

    all_sample_tokens: list[int] = []

    for split in splits:
        file_path = tokenized_dir / split
        if not file_path.exists():
            continue

        arr = np.memmap(file_path, dtype=np.uint32, mode="r")
        count = int(arr.shape[0])
        total_tokens += count
        split_stats[split] = {
            "token_count": count,
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024**2), 2),
        }

        if count > 0:
            sample_size = min(1000, count)
            all_sample_tokens.extend(arr[:sample_size].tolist())

    # Tokenizer vocabulary coverage
    unique_tokens = len(set(all_sample_tokens))
    vocab_size = 0
    special_token_stats: dict[str, int] = {}

    if Path(tokenizer_path).exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer.get_vocab()))

            for token_name, token_id in [
                ("eos", getattr(tokenizer, "eos_token_id", 2)),
                ("bos", getattr(tokenizer, "bos_token_id", 1)),
                ("pad", getattr(tokenizer, "pad_token_id", 0)),
            ]:
                if token_id is not None:
                    special_token_stats[f"{token_name}_token_count"] = all_sample_tokens.count(
                        token_id
                    )
        except Exception as e:
            logger.warning(f"Could not load tokenizer at {tokenizer_path}: {e}")

    coverage_pct = round((unique_tokens / max(1, vocab_size)) * 100, 2) if vocab_size > 0 else 0.0

    report = {
        "tokenized_dir": str(tokenized_dir),
        "total_tokens": total_tokens,
        "splits": split_stats,
        "sample_unique_tokens": unique_tokens,
        "tokenizer_vocab_size": vocab_size,
        "sample_vocab_coverage_pct": coverage_pct,
        "special_tokens_sample_freq": special_token_stats,
    }

    write_json(report, report_path)
    logger.info(f"Dataset verification complete! Report saved to {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Verification Utility")
    parser.add_argument("--tokenized_dir", default="data/tokenized")
    parser.add_argument("--tokenizer", default="tokenizer/v1.0")
    args = parser.parse_args()

    report = verify_dataset(args.tokenized_dir, args.tokenizer)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
