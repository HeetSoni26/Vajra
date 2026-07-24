from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset.builder import BinaryDatasetBuilder
from dataset.ingest import DataIngestor
from dataset.processing.deduplication import Deduplicator
from dataset.processing.normalize import normalize_text
from dataset.processing.quality_filter import QualityFilter
from dataset.run_pipeline import DatasetPipelineRunner
from dataset.verify_dataset import verify_dataset


def test_ingest_normalize_and_clean(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    jsonl_file = raw_dir / "docs.jsonl"
    jsonl_file.write_text(
        '{"text": "  Transformers use attention mechanisms and SwiGLU  "}\n'
        '{"text": "  Transformers use attention mechanisms and SwiGLU  "}\n'  # duplicate
        '{"text": "short"}\n'  # quality filter drop
    )

    ingestor = DataIngestor(raw_dir)
    docs = list(ingestor.stream_documents())
    assert len(docs) == 3

    quality_filter = QualityFilter(min_words=5, max_words=100)
    deduplicator = Deduplicator()

    cleaned_docs = []
    for d in docs:
        norm = normalize_text(d["text"])
        valid, reason = quality_filter.is_valid(norm)
        if valid and not deduplicator.is_duplicate(norm):
            cleaned_docs.append(norm)

    assert len(cleaned_docs) == 1
    assert "SwiGLU" in cleaned_docs[0]


def test_binary_dataset_builder(tmp_path: Path):
    out_dir = tmp_path / "tokenized"
    builder = BinaryDatasetBuilder(
        output_dir=out_dir, val_ratio=0.10, test_ratio=0.10, sequence_length=32
    )
    sample_tokens = list(range(100))

    stats = builder.build_binary_dataset(sample_tokens)

    assert (out_dir / "train.bin").exists()
    assert (out_dir / "val.bin").exists()
    assert (out_dir / "test.bin").exists()
    assert (out_dir / "metadata.json").exists()

    arr = np.memmap(out_dir / "train.bin", dtype=np.uint32, mode="r")
    assert len(arr) == 80
    assert stats["checksums"]["train.bin"] != ""


def test_pipeline_end_to_end_and_resume(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    tokenized_dir = tmp_path / "tokenized"

    raw_dir.mkdir(parents=True)
    sample_file = raw_dir / "sample.jsonl"
    sample_file.write_text(
        '{"text": "Deep learning models require scalable dataset pipelines for training."}\n'
        '{"text": "Binary memmap format enables high speed random access to token arrays."}\n'
    )

    cfg_file = tmp_path / "config.json"
    config = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "tokenized_dir": str(tokenized_dir),
        "tokenizer_path": "tokenizer/v1.0",
        "quality": {"min_words": 3},
        "packing": {"sequence_length": 128},
    }
    cfg_file.write_text(json.dumps(config))

    runner = DatasetPipelineRunner(cfg_file)
    manifest = runner.run(force_rebuild=True)

    assert manifest["cleaned_documents"] == 2
    assert (tokenized_dir / "dataset_manifest.json").exists()

    # Test Resume
    resume_manifest = runner.run(force_rebuild=False)
    assert resume_manifest["cleaned_documents"] == 2

    # Verification Report Test
    report = verify_dataset(tokenized_dir, "tokenizer/v1.0")
    assert report["total_tokens"] > 0
    assert (tokenized_dir / "dataset_statistics.json").exists()
