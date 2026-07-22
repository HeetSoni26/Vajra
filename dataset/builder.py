from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from utils.file_utils import ensure_dir, write_json
from utils.logging import setup_logger

logger = setup_logger("dataset_builder")


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class BinaryDatasetBuilder:
    """High-throughput binary dataset builder creating memmap token arrays with train/val/test splits."""

    def __init__(
        self,
        output_dir: str | Path = "data/tokenized",
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        sequence_length: int = 4096,
        eos_token_id: int = 2,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id

    def build_binary_dataset(
        self,
        token_stream: list[int],
        metadata_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a flat token stream into train.bin, val.bin, and test.bin memmap files."""
        total_tokens = len(token_stream)
        if total_tokens == 0:
            raise ValueError("Token stream is empty. Cannot build binary dataset.")

        start_time = time.time()

        # Calculate split indices
        val_count = int(total_tokens * self.val_ratio)
        test_count = int(total_tokens * self.test_ratio)
        train_count = total_tokens - val_count - test_count

        train_tokens = np.array(token_stream[:train_count], dtype=np.uint32)
        val_tokens = np.array(token_stream[train_count : train_count + val_count], dtype=np.uint32)
        test_tokens = np.array(token_stream[train_count + val_count :], dtype=np.uint32)

        # Write binary files
        train_path = self.output_dir / "train.bin"
        val_path = self.output_dir / "val.bin"
        test_path = self.output_dir / "test.bin"

        train_tokens.tofile(train_path)
        val_tokens.tofile(val_path)
        test_tokens.tofile(test_path)

        write_duration = max(0.001, time.time() - start_time)
        bytes_written = train_path.stat().st_size + val_path.stat().st_size + test_path.stat().st_size
        write_speed_mb = round((bytes_written / (1024**2)) / write_duration, 2)

        # Compute checksums
        checksums = {
            "train.bin": compute_file_hash(train_path),
            "val.bin": compute_file_hash(val_path),
            "test.bin": compute_file_hash(test_path),
        }

        split_stats = {
            "total_tokens": total_tokens,
            "train_tokens": int(train_tokens.size),
            "val_tokens": int(val_tokens.size),
            "test_tokens": int(test_tokens.size),
            "sequence_length": self.sequence_length,
            "num_train_sequences": int(train_tokens.size // self.sequence_length),
            "write_speed_mb_s": write_speed_mb,
            "checksums": checksums,
        }

        # Save metadata.json
        meta = {
            "dtype": "uint32",
            "sequence_length": self.sequence_length,
            "splits": {
                "train": {"file": "train.bin", "count": int(train_tokens.size)},
                "val": {"file": "val.bin", "count": int(val_tokens.size)},
                "test": {"file": "test.bin", "count": int(test_tokens.size)},
            },
            "extra_info": metadata_info or {},
        }
        write_json(meta, self.output_dir / "metadata.json")

        logger.info(f"Built binary dataset: {total_tokens:,} tokens ({write_speed_mb} MB/s)")
        return split_stats
