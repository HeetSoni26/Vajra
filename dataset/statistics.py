"""
Dataset statistics and validation module for Vajra Framework.

Provides comprehensive analysis of tokenized binary datasets including:
  • Token distribution analysis
  • Sequence length statistics
  • Domain coverage metrics
  • Data integrity verification
  • Training readiness checks
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from utils.file_utils import read_json, write_json
from utils.logging import setup_logger

logger = setup_logger("dataset_statistics")


class DatasetStatistics:
    """Comprehensive dataset statistics and validation engine."""

    def __init__(self, tokenized_dir: str | Path, vocab_size: int = 32000) -> None:
        self.tokenized_dir = Path(tokenized_dir)
        self.vocab_size = vocab_size

    def compute_statistics(self) -> dict[str, Any]:
        """Compute comprehensive statistics across all splits."""
        stats: dict[str, Any] = {
            "tokenized_dir": str(self.tokenized_dir),
            "splits": {},
            "aggregate": {},
        }

        total_tokens = 0
        all_token_ids: list[int] = []

        for split_name in ["train", "val", "test"]:
            file_path = self.tokenized_dir / f"{split_name}.bin"
            if not file_path.exists():
                continue

            split_stats = self._compute_split_stats(file_path, split_name)
            stats["splits"][split_name] = split_stats
            total_tokens += split_stats["token_count"]

            # Collect sample tokens for aggregate analysis
            arr = np.memmap(file_path, dtype=np.uint32, mode="r")
            sample_size = min(100000, len(arr))
            all_token_ids.extend(arr[:sample_size].tolist())

        # Aggregate statistics
        stats["aggregate"] = self._compute_aggregate_stats(all_token_ids, total_tokens)

        return stats

    def _compute_split_stats(self, file_path: Path, split_name: str) -> dict[str, Any]:
        """Compute statistics for a single split file."""
        arr = np.memmap(file_path, dtype=np.uint32, mode="r")
        token_count = int(arr.shape[0])
        file_size = file_path.stat().st_size

        # Token value distribution
        sample_size = min(100000, token_count)
        sample = arr[:sample_size]
        unique_tokens = len(np.unique(sample))

        # Compute hash
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        return {
            "file": str(file_path.name),
            "token_count": token_count,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024**2), 2),
            "unique_tokens_in_sample": unique_tokens,
            "min_token_id": int(sample.min()),
            "max_token_id": int(sample.max()),
            "mean_token_id": round(float(sample.mean()), 2),
            "sha256": file_hash,
        }

    def _compute_aggregate_stats(
        self, sample_tokens: list[int], total_tokens: int
    ) -> dict[str, Any]:
        """Compute aggregate statistics from sampled tokens."""
        if not sample_tokens:
            return {"total_tokens": 0, "vocab_coverage_pct": 0.0}

        unique = set(sample_tokens)
        coverage_pct = round((len(unique) / max(1, self.vocab_size)) * 100, 2)

        # Token frequency distribution (top-20)
        from collections import Counter

        freq = Counter(sample_tokens)
        top_tokens = freq.most_common(20)

        return {
            "total_tokens": total_tokens,
            "sample_size": len(sample_tokens),
            "unique_tokens_in_sample": len(unique),
            "vocab_size": self.vocab_size,
            "vocab_coverage_pct": coverage_pct,
            "top_20_tokens": [{"token_id": t, "count": c} for t, c in top_tokens],
            "tokens_per_billion": round(total_tokens / 1_000_000_000, 4),
        }

    def validate_integrity(self) -> dict[str, Any]:
        """Run integrity checks on the tokenized dataset."""
        checks: list[dict[str, Any]] = []
        all_passed = True

        # Check 1: Required files exist
        for split in ["train.bin"]:
            path = self.tokenized_dir / split
            passed = path.exists() and path.stat().st_size > 0
            checks.append(
                {
                    "check": f"{split}_exists",
                    "passed": passed,
                    "detail": f"{'Found' if passed else 'MISSING'}: {split}",
                }
            )
            if not passed:
                all_passed = False

        # Check 2: Token IDs are within vocab range
        for split in ["train.bin", "val.bin", "test.bin"]:
            path = self.tokenized_dir / split
            if not path.exists():
                continue
            arr = np.memmap(path, dtype=np.uint32, mode="r")
            sample_size = min(50000, len(arr))
            sample = arr[:sample_size]
            max_id = int(sample.max())
            passed = max_id < self.vocab_size
            checks.append(
                {
                    "check": f"{split}_vocab_range",
                    "passed": passed,
                    "detail": f"Max token ID: {max_id}, vocab_size: {self.vocab_size}",
                }
            )
            if not passed:
                all_passed = False

        # Check 3: No all-zero files
        for split in ["train.bin", "val.bin", "test.bin"]:
            path = self.tokenized_dir / split
            if not path.exists():
                continue
            arr = np.memmap(path, dtype=np.uint32, mode="r")
            sample = arr[: min(1000, len(arr))]
            has_nonzero = bool(np.any(sample > 0))
            checks.append(
                {
                    "check": f"{split}_non_zero",
                    "passed": has_nonzero,
                    "detail": f"{'Has non-zero tokens' if has_nonzero else 'ALL ZEROS — likely corrupt'}",
                }
            )
            if not has_nonzero:
                all_passed = False

        # Check 4: Metadata consistency
        meta_path = self.tokenized_dir / "metadata.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            train_meta_count = meta.get("splits", {}).get("train", {}).get("count", 0)
            train_path = self.tokenized_dir / "train.bin"
            if train_path.exists():
                actual_count = len(np.memmap(train_path, dtype=np.uint32, mode="r"))
                passed = train_meta_count == actual_count
                checks.append(
                    {
                        "check": "metadata_consistency",
                        "passed": passed,
                        "detail": f"metadata={train_meta_count}, actual={actual_count}",
                    }
                )
                if not passed:
                    all_passed = False

        return {
            "all_passed": all_passed,
            "checks": checks,
            "num_checks": len(checks),
            "num_passed": sum(1 for c in checks if c["passed"]),
        }

    def generate_report(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Generate complete statistics and validation report."""
        stats = self.compute_statistics()
        integrity = self.validate_integrity()

        report = {
            "statistics": stats,
            "integrity_validation": integrity,
        }

        if output_path is None:
            output_path = self.tokenized_dir / "dataset_report.json"
        write_json(report, output_path)
        logger.info(f"Dataset report written to {output_path}")

        return report
