import hashlib
import random
import time
from collections.abc import Iterable
from pathlib import Path
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


class StreamingBinaryWriter:
    """Writes tokens to a binary file incrementally with buffering."""

    def __init__(self, path: Path, dtype: np.dtype = np.uint32, buffer_size: int = 1024 * 1024):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.buffer_size = buffer_size

        self.file = path.open("wb")
        self.buffer = []
        self.tokens_written = 0

    def write(self, tokens: list[int]):
        self.buffer.extend(tokens)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer:
            arr = np.array(self.buffer, dtype=self.dtype)
            arr.tofile(self.file)
            self.tokens_written += len(self.buffer)
            self.buffer.clear()
            self.file.flush()

    def close(self):
        self.flush()
        self.file.close()


class BinaryDatasetBuilder:
    """High-throughput streaming binary dataset builder."""

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

    def build_from_stream(
        self,
        document_token_stream: Iterable[list[int]],
        metadata_info: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Convert a stream of tokenized documents into train, val, and test splits."""
        start_time = time.time()

        rng = random.Random(seed)

        writers = {
            "train": StreamingBinaryWriter(self.output_dir / "train.bin"),
            "val": StreamingBinaryWriter(self.output_dir / "val.bin"),
            "test": StreamingBinaryWriter(self.output_dir / "test.bin"),
        }

        total_tokens = 0

        try:
            for tokens in document_token_stream:
                if not tokens:
                    continue
                r = rng.random()
                if r < self.test_ratio:
                    split = "test"
                elif r < self.test_ratio + self.val_ratio:
                    split = "val"
                else:
                    split = "train"

                writers[split].write(tokens)
                total_tokens += len(tokens)
        finally:
            for w in writers.values():
                w.close()

        if total_tokens == 0:
            raise ValueError("Token stream is empty. Cannot build binary dataset.")

        write_duration = max(0.001, time.time() - start_time)

        train_path = self.output_dir / "train.bin"
        val_path = self.output_dir / "val.bin"
        test_path = self.output_dir / "test.bin"

        bytes_written = (
            train_path.stat().st_size + val_path.stat().st_size + test_path.stat().st_size
        )
        write_speed_mb = round((bytes_written / (1024**2)) / write_duration, 2)

        # Compute checksums
        checksums = {
            "train.bin": compute_file_hash(train_path),
            "val.bin": compute_file_hash(val_path),
            "test.bin": compute_file_hash(test_path),
        }

        train_tokens = writers["train"].tokens_written
        val_tokens = writers["val"].tokens_written
        test_tokens = writers["test"].tokens_written

        split_stats = {
            "total_tokens": total_tokens,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "test_tokens": test_tokens,
            "sequence_length": self.sequence_length,
            "num_train_sequences": train_tokens // self.sequence_length,
            "write_speed_mb_s": write_speed_mb,
            "checksums": checksums,
        }

        # Save metadata.json
        meta = {
            "dtype": "uint32",
            "sequence_length": self.sequence_length,
            "splits": {
                "train": {"file": "train.bin", "count": train_tokens},
                "val": {"file": "val.bin", "count": val_tokens},
                "test": {"file": "test.bin", "count": test_tokens},
            },
            "extra_info": metadata_info or {},
        }
        write_json(meta, self.output_dir / "metadata.json")

        logger.info(f"Built binary dataset: {total_tokens:,} tokens ({write_speed_mb} MB/s)")
        return split_stats

    def build_binary_dataset(
        self,
        token_stream: list[int],
        metadata_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Backward compatibility for existing pipeline."""
        start_time = time.time()
        total_tokens = len(token_stream)
        if total_tokens == 0:
            raise ValueError("Token stream is empty. Cannot build binary dataset.")

        val_count = int(total_tokens * self.val_ratio)
        test_count = int(total_tokens * self.test_ratio)
        train_count = total_tokens - val_count - test_count

        writers = {
            "train": StreamingBinaryWriter(self.output_dir / "train.bin"),
            "val": StreamingBinaryWriter(self.output_dir / "val.bin"),
            "test": StreamingBinaryWriter(self.output_dir / "test.bin"),
        }

        writers["train"].write(token_stream[:train_count])
        writers["val"].write(token_stream[train_count : train_count + val_count])
        writers["test"].write(token_stream[train_count + val_count :])

        for w in writers.values():
            w.close()

        write_duration = max(0.001, time.time() - start_time)

        train_path = self.output_dir / "train.bin"
        val_path = self.output_dir / "val.bin"
        test_path = self.output_dir / "test.bin"

        bytes_written = (
            train_path.stat().st_size + val_path.stat().st_size + test_path.stat().st_size
        )
        write_speed_mb = round((bytes_written / (1024**2)) / write_duration, 2)

        checksums = {
            "train.bin": compute_file_hash(train_path),
            "val.bin": compute_file_hash(val_path),
            "test.bin": compute_file_hash(test_path),
        }

        train_tokens_count = writers["train"].tokens_written
        val_tokens_count = writers["val"].tokens_written
        test_tokens_count = writers["test"].tokens_written

        split_stats = {
            "total_tokens": total_tokens,
            "train_tokens": train_tokens_count,
            "val_tokens": val_tokens_count,
            "test_tokens": test_tokens_count,
            "sequence_length": self.sequence_length,
            "num_train_sequences": train_tokens_count // self.sequence_length,
            "write_speed_mb_s": write_speed_mb,
            "checksums": checksums,
        }

        meta = {
            "dtype": "uint32",
            "sequence_length": self.sequence_length,
            "splits": {
                "train": {"file": "train.bin", "count": train_tokens_count},
                "val": {"file": "val.bin", "count": val_tokens_count},
                "test": {"file": "test.bin", "count": test_tokens_count},
            },
            "extra_info": metadata_info or {},
        }
        write_json(meta, self.output_dir / "metadata.json")

        logger.info(f"Built binary dataset: {total_tokens:,} tokens ({write_speed_mb} MB/s)")
        return split_stats
