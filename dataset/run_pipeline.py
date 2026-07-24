from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from dataset.builder import BinaryDatasetBuilder
from dataset.ingest import DataIngestor
from dataset.processing.deduplication import Deduplicator
from dataset.processing.normalize import normalize_text
from dataset.processing.quality_filter import QualityFilter
from dataset.tokenize_dataset import DatasetTokenizer
from utils.file_utils import ensure_dir, read_json, write_json
from utils.logging import setup_logger

logger = setup_logger("pipeline_runner")


def get_peak_ram_mb() -> float:
    """Get peak RAM memory usage of current process."""
    try:
        import psutil

        process = psutil.Process()
        return round(process.memory_info().rss / (1024**2), 2)
    except Exception:
        return 0.0


class DatasetPipelineRunner:
    """Production orchestrator with stage-based resume capability, performance metrics, and manifest generation."""

    def __init__(self, config_path: str | Path = "configs/data/preprocessing.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = (
            read_json(self.config_path)
            if self.config_path.suffix == ".json"
            else self._load_yaml(self.config_path)
        )

        self.raw_dir = ensure_dir(self.config.get("raw_dir", "data/raw"))
        self.processed_dir = ensure_dir(self.config.get("processed_dir", "data/processed"))
        self.tokenized_dir = ensure_dir(self.config.get("tokenized_dir", "data/tokenized"))
        self.tokenizer_path = self.config.get("tokenizer_path", "tokenizer/v1.0")

        self.state_file = self.tokenized_dir / "pipeline_state.json"
        self.manifest_file = self.tokenized_dir / "dataset_manifest.json"

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_state(self) -> dict[str, Any]:
        """Load pipeline state for resume capability."""
        if self.state_file.exists():
            return read_json(self.state_file)
        return {"completed_stages": [], "stage_data": {}}

    def save_state(self, state: dict[str, Any]) -> None:
        """Save pipeline state."""
        write_json(state, self.state_file)

    def run(self, force_rebuild: bool = False) -> dict[str, Any]:
        """Run dataset pipeline end-to-end with resume support."""
        pipeline_start_time = time.time()
        state = (
            self.load_state() if not force_rebuild else {"completed_stages": [], "stage_data": {}}
        )

        # ----------------------------------------------------
        # STAGE 1: Ingestion, Cleaning & Deduplication
        # ----------------------------------------------------
        cleaned_docs: list[dict[str, Any]] = []
        if "ingest_clean" in state["completed_stages"] and not force_rebuild:
            logger.info("Resuming: Stage 1 (Ingest & Clean) already completed.")
            cleaned_file = self.processed_dir / "cleaned_documents.jsonl"
            if cleaned_file.exists():
                for line in cleaned_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        cleaned_docs.append(json.loads(line))
            stage1_metrics = state["stage_data"].get("stage1", {})
        else:
            logger.info("Executing Stage 1: Ingestion, Cleaning & Deduplication...")
            s1_start = time.time()

            ingestor = DataIngestor(self.raw_dir)
            files = ingestor.discover_files()

            quality = self.config.get("quality", {})
            quality_filter = QualityFilter(
                min_words=int(quality.get("min_words", 10)),
                max_words=int(quality.get("max_words", 100000)),
                min_alnum_ratio=float(quality.get("min_alnum_ratio", 0.50)),
                max_repetition_ratio=float(quality.get("max_repetition_ratio", 0.30)),
            )
            deduplicator = Deduplicator()

            raw_doc_count = 0
            removed_count = 0
            duplicate_count = 0

            for doc in ingestor.stream_documents():
                raw_doc_count += 1
                norm_text = normalize_text(doc.get("text", ""))

                valid, reason = quality_filter.is_valid(norm_text)
                if not valid:
                    removed_count += 1
                    continue

                if deduplicator.is_duplicate(norm_text):
                    duplicate_count += 1
                    continue

                cleaned_docs.append(
                    {
                        "doc_id": doc["doc_id"],
                        "text": norm_text,
                        "source_file": doc["source_file"],
                    }
                )

            s1_duration = max(0.001, time.time() - s1_start)
            stage1_metrics = {
                "raw_files_count": len(files),
                "raw_doc_count": raw_doc_count,
                "cleaned_doc_count": len(cleaned_docs),
                "removed_doc_count": removed_count,
                "duplicate_count": duplicate_count,
                "ingest_clean_duration_s": round(s1_duration, 2),
                "ingest_speed_docs_s": round(raw_doc_count / s1_duration, 2),
            }

            # Save intermediate cleaned file
            cleaned_file = self.processed_dir / "cleaned_documents.jsonl"
            with cleaned_file.open("w", encoding="utf-8") as f:
                for d in cleaned_docs:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            state["completed_stages"].append("ingest_clean")
            state["stage_data"]["stage1"] = stage1_metrics
            self.save_state(state)

        # ----------------------------------------------------
        # STAGE 2: Tokenization & Binary Memmap Building
        # ----------------------------------------------------
        if "tokenize_build" in state["completed_stages"] and not force_rebuild:
            logger.info("Resuming: Stage 2 (Tokenization & Binary Building) already completed.")
            stage2_metrics = state["stage_data"].get("stage2", {})
        else:
            logger.info("Executing Stage 2: Tokenization & Binary Dataset Building...")
            s2_start = time.time()

            # Ensure tokenizer is available or fallback
            tok_dir = (
                self.tokenizer_path if Path(self.tokenizer_path).exists() else "tokenizer/v1.0"
            )
            tokenizer_engine = DatasetTokenizer(tok_dir)

            tokens, tok_metrics = tokenizer_engine.tokenize_documents(cleaned_docs)

            # Build binary files
            packing = self.config.get("packing", {})
            builder = BinaryDatasetBuilder(
                output_dir=self.tokenized_dir,
                val_ratio=0.05,
                test_ratio=0.05,
                sequence_length=int(packing.get("sequence_length", 4096)),
            )

            split_stats = builder.build_binary_dataset(
                tokens,
                metadata_info={
                    "tokenizer_path": str(tok_dir),
                    "cleaned_documents": len(cleaned_docs),
                },
            )

            s2_duration = max(0.001, time.time() - s2_start)
            stage2_metrics = {
                "tokenization_duration_s": round(s2_duration, 2),
                "tokens_per_sec": tok_metrics["tokens_per_sec"],
                "split_stats": split_stats,
            }

            state["completed_stages"].append("tokenize_build")
            state["stage_data"]["stage2"] = stage2_metrics
            self.save_state(state)

        # ----------------------------------------------------
        # STAGE 3: Dataset Manifest Generation
        # ----------------------------------------------------
        total_runtime = max(0.001, time.time() - pipeline_start_time)
        peak_ram_mb = get_peak_ram_mb()

        split_info = stage2_metrics.get("split_stats", {})
        checksums = split_info.get("checksums", {})

        manifest = {
            "dataset_name": "vajra-lm-corpus",
            "creation_timestamp": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "tokenizer_version": str(self.tokenizer_path),
            "source_directories": [str(self.raw_dir)],
            "input_files_count": stage1_metrics.get("raw_files_count", 0),
            "raw_documents_count": stage1_metrics.get("raw_doc_count", 0),
            "cleaned_documents": stage1_metrics.get("cleaned_doc_count", 0),
            "removed_documents": stage1_metrics.get("removed_doc_count", 0),
            "duplicate_count": stage1_metrics.get("duplicate_count", 0),
            "total_tokens": split_info.get("total_tokens", 0),
            "split_sizes": {
                "train": split_info.get("train_tokens", 0),
                "val": split_info.get("val_tokens", 0),
                "test": split_info.get("test_tokens", 0),
            },
            "sequence_length": split_info.get("sequence_length", 4096),
            "checksums": checksums,
            "performance_metrics": {
                "ingest_speed_docs_s": stage1_metrics.get("ingest_speed_docs_s", 0),
                "tokenization_speed_tokens_s": stage2_metrics.get("tokens_per_sec", 0),
                "binary_write_speed_mb_s": split_info.get("write_speed_mb_s", 0),
                "peak_ram_mb": peak_ram_mb,
                "total_runtime_s": round(total_runtime, 2),
            },
        }

        write_json(manifest, self.manifest_file)
        logger.info(f"Dataset Pipeline Complete! Manifest generated at: {self.manifest_file}")

        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Pipeline Runner")
    parser.add_argument("--config", default="configs/data/preprocessing.yaml")
    parser.add_argument(
        "--force_rebuild", action="store_true", help="Ignore pipeline_state.json and rebuild"
    )
    args = parser.parse_args()

    runner = DatasetPipelineRunner(args.config)
    manifest = runner.run(force_rebuild=args.force_rebuild)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
