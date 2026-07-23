"""
Dataset preparation script for Vajra Framework.

Orchestrates the full data preparation pipeline:
  1. Generate synthetic corpus (for development/testing)
  2. Run the full dataset pipeline (ingest → clean → tokenize → build binary)
  3. Compute dataset statistics and validation report
  4. Write manifest with checksums

Usage:
    # Development: generate synthetic data and prepare for training
    python -m scripts.prepare_dataset --synthetic --num_docs 200

    # Production: prepare real data from raw_dir
    python -m scripts.prepare_dataset --config configs/data/preprocessing.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.file_utils import ensure_dir, write_json
from utils.logging import setup_logger

logger = setup_logger("prepare_dataset")


def prepare_synthetic(
    output_dir: str | Path = "data",
    num_docs: int = 200,
    sequence_length: int = 128,
    vocab_size: int = 297,
    seed: int = 42,
) -> dict:
    """Generate synthetic corpus and build training-ready binary files.

    Designed for CI and development testing — no real data download required.
    """
    from dataset.sources.synthetic import write_synthetic_corpus
    from dataset.ingest import DataIngestor
    from dataset.processing.normalize import normalize_text
    from dataset.processing.quality_filter import QualityFilter
    from dataset.processing.deduplication import Deduplicator
    from dataset.builder import BinaryDatasetBuilder
    from dataset.statistics import DatasetStatistics

    output_dir = Path(output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "processed")
    tokenized_dir = ensure_dir(output_dir / "tokenized")

    logger.info(f"Generating {num_docs} synthetic documents...")
    corpus_stats = write_synthetic_corpus(
        output_dir=raw_dir,
        num_documents=num_docs,
        seed=seed,
    )
    logger.info(f"Synthetic corpus: {corpus_stats}")

    # Ingest
    ingestor = DataIngestor(raw_dir)
    quality_filter = QualityFilter(min_words=5, max_words=100000, min_alnum_ratio=0.40)
    deduplicator = Deduplicator()

    cleaned_docs = []
    for doc in ingestor.stream_documents():
        norm_text = normalize_text(doc.get("text", ""))
        valid, _ = quality_filter.is_valid(norm_text)
        if valid and not deduplicator.is_duplicate(norm_text):
            cleaned_docs.append({"doc_id": doc["doc_id"], "text": norm_text})

    logger.info(f"After cleaning: {len(cleaned_docs)} documents")

    # Simple character-level tokenization for synthetic data (no real tokenizer required)
    # Produces token IDs in range [0, vocab_size)
    all_tokens: list[int] = []
    for doc in cleaned_docs:
        text = doc["text"]
        # Convert characters to token IDs bounded by vocab_size
        token_ids = [ord(c) % vocab_size for c in text]
        token_ids.append(2)  # EOS token
        all_tokens.extend(token_ids)

    if len(all_tokens) < sequence_length + 2:
        # Pad to ensure at least one full sequence
        all_tokens.extend([0] * (sequence_length + 2 - len(all_tokens)))

    logger.info(f"Total tokens: {len(all_tokens):,}")

    # Build binary dataset
    builder = BinaryDatasetBuilder(
        output_dir=tokenized_dir,
        val_ratio=0.10,
        test_ratio=0.10,
        sequence_length=sequence_length,
    )
    split_stats = builder.build_binary_dataset(
        all_tokens,
        metadata_info={
            "tokenizer": "synthetic_char",
            "vocab_size": vocab_size,
            "cleaned_documents": len(cleaned_docs),
        },
    )

    # Statistics report
    stats_engine = DatasetStatistics(tokenized_dir, vocab_size=vocab_size)
    report = stats_engine.generate_report()

    result = {
        "corpus_stats": corpus_stats,
        "cleaned_docs": len(cleaned_docs),
        "split_stats": split_stats,
        "integrity": report["integrity_validation"],
        "tokenized_dir": str(tokenized_dir),
    }

    write_json(result, tokenized_dir / "preparation_summary.json")
    logger.info(f"Dataset preparation complete → {tokenized_dir}")
    return result


def prepare_from_config(
    config_path: str | Path = "configs/data/preprocessing.yaml",
    force_rebuild: bool = False,
) -> dict:
    """Run the full production dataset pipeline from config."""
    from dataset.run_pipeline import DatasetPipelineRunner
    from dataset.statistics import DatasetStatistics

    runner = DatasetPipelineRunner(config_path)
    manifest = runner.run(force_rebuild=force_rebuild)

    # Load vocab size from config
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tokenized_dir = Path(cfg.get("tokenized_dir", "data/tokenized"))

    stats_engine = DatasetStatistics(tokenized_dir, vocab_size=102400)
    stats_engine.generate_report()

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Vajra Dataset Preparation")
    parser.add_argument("--config", default="configs/data/preprocessing.yaml",
                        help="Preprocessing config YAML")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of downloading real data")
    parser.add_argument("--num_docs", type=int, default=200,
                        help="Number of synthetic documents to generate")
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Sequence length for binary dataset chunks")
    parser.add_argument("--vocab_size", type=int, default=297,
                        help="Vocab size for synthetic tokenization")
    parser.add_argument("--output_dir", default="data",
                        help="Output directory for synthetic data")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild ignoring existing pipeline state")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        result = prepare_synthetic(
            output_dir=args.output_dir,
            num_docs=args.num_docs,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
    else:
        result = prepare_from_config(
            config_path=args.config,
            force_rebuild=args.force_rebuild,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
