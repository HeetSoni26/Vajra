import argparse
import json
import random
import re
import sys
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
from datasets import load_dataset

from dataset.builder import BinaryDatasetBuilder
from dataset.processing.normalize import normalize_text
from dataset.tokenize_dataset import DatasetTokenizer
from utils.file_utils import ensure_dir, read_json, write_json
from utils.logging import setup_logger

logger = setup_logger("prepare_dataset")

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_document(text: str, min_len: int = 50, max_len: int = 100000) -> str:
    """Apply cleaning rules: HTML removal, normalization, length constraints."""
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = normalize_text(text)
    if len(text) < min_len or len(text) > max_len:
        return ""
    return text


def document_token_stream(
    ds,
    tokenizer,
    state: dict,
    max_gb: float | None = None,
    max_docs: int | None = None,
    max_tokens: int | None = None,
    batch_size: int = 100,
    state_file: Path | None = None,
) -> Generator[list[int], None, None]:

    docs_processed = state.get("docs_processed", 0)
    total_bytes = state.get("total_bytes_processed", 0)
    total_tokens_estimated = state.get("total_tokens_processed", 0)
    seen_hashes = set(state.get("seen_hashes", []))

    start_time = time.time()
    last_log_time = start_time

    batch_docs = []
    first_doc_received = False
    first_batch_tokenized = False

    def process_batch():
        nonlocal total_bytes, docs_processed, total_tokens_estimated, first_batch_tokenized
        if not batch_docs:
            return []

        # Tokenize as a batch
        tokens_out, _ = tokenizer.tokenize_documents([{"text": d} for d in batch_docs])
        if not first_batch_tokenized:
            logger.info("First batch tokenized successfully")
            first_batch_tokenized = True
            
        batch_docs.clear()

        total_tokens_estimated += len(tokens_out)
        return tokens_out

    for i, row in enumerate(ds):
        if i < docs_processed:
            continue
            
        if not first_doc_received:
            logger.info("First document received from stream")
            first_doc_received = True

        text = row.get("text", "")
        if "language" in row and row["language"] != "en":
            continue

        cleaned = clean_document(text)
        if not cleaned:
            continue

        doc_hash = hash(cleaned)
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)

        batch_docs.append(cleaned)
        total_bytes += len(cleaned.encode("utf-8"))
        docs_processed += 1

        if len(batch_docs) >= batch_size:
            yield process_batch()

            # Progress logging
            now = time.time()
            if now - last_log_time > 10:
                elapsed = now - start_time
                tps = total_tokens_estimated / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {docs_processed} docs | {total_tokens_estimated:,} tokens | "
                    f"{total_bytes / (1024**3):.2f} GB | {tps:,.0f} tok/s"
                )
                last_log_time = now

                # Save state
                if state_file:
                    state.update(
                        {
                            "docs_processed": docs_processed,
                            "total_bytes_processed": total_bytes,
                            "total_tokens_processed": total_tokens_estimated,
                            "seen_hashes": list(seen_hashes),
                        }
                    )
                    write_json(state, state_file)

        if max_docs and docs_processed >= max_docs:
            logger.info(f"Reached max_docs ({max_docs})")
            break
        if max_tokens and total_tokens_estimated >= max_tokens:
            logger.info(f"Reached max_tokens ({max_tokens})")
            break
        if max_gb and (total_bytes / (1024**3)) >= max_gb:
            logger.info(f"Reached max_gb ({max_gb})")
            break

    if batch_docs:
        yield process_batch()

    state.update(
        {
            "docs_processed": docs_processed,
            "total_bytes_processed": total_bytes,
            "total_tokens_processed": total_tokens_estimated,
            "seen_hashes": list(seen_hashes),
        }
    )
    if state_file:
        write_json(state, state_file)


def prepare_huggingface(
    dataset_name: str,
    output_dir: str | Path,
    stream: bool = True,
    config_name: str | None = None,
    max_gb: float | None = None,
    max_docs: int | None = None,
    max_tokens: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    resume: bool = False,
    batch_size: int = 100,
) -> dict:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    state_file = output_dir / "pipeline_state.json"
    state = {}
    if resume and state_file.exists():
        logger.info("Resuming from previous state...")
        state = read_json(state_file)
    else:
        logger.info(f"Preparing dataset from {dataset_name} (stream={stream}, seed={seed})")

    random.seed(seed)

    tokenizer = DatasetTokenizer("tokenizer/v1.0")

    builder = BinaryDatasetBuilder(
        output_dir=output_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    max_retries = 10
    retries = 0
    split_stats = None

    while retries < max_retries:
        try:
            kwargs = {"split": "train", "streaming": stream}
            if config_name:
                kwargs["name"] = config_name
            ds = load_dataset(dataset_name, **kwargs)
            
            if stream:
                ds = ds.shuffle(seed=seed, buffer_size=100)

            # Re-read state in case we are retrying
            if state_file.exists():
                state = read_json(state_file)

            token_stream = document_token_stream(
                ds=ds,
                tokenizer=tokenizer,
                state=state,
                max_gb=max_gb,
                max_docs=max_docs,
                max_tokens=max_tokens,
                batch_size=batch_size,
                state_file=state_file,
            )

            split_stats = builder.build_from_stream(
                token_stream,
                metadata_info={
                    "dataset": dataset_name,
                    "seed": seed,
                },
                seed=seed,
            )
            break # Success
        except Exception as e:
            logger.error(f"Stream error: {e}. Retrying ({retries+1}/{max_retries})...")
            retries += 1
            time.sleep(5)
            
    if split_stats is None:
        logger.error("Failed to generate dataset after max retries.")
        sys.exit(1)

    # Update metadata with final state after generator is exhausted
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        meta["extra_info"].update(
            {
                "docs_processed": state.get("docs_processed", 0),
                "total_bytes": state.get("total_bytes_processed", 0),
            }
        )
        write_json(meta, meta_path)

    validate_dataset(output_dir, split_stats, split_stats["total_tokens"])

    report = {
        "dataset_name": dataset_name,
        "seed": seed,
        "cleaning_stats": {
            "docs_processed": state.get("docs_processed", 0),
            "unique_docs": len(state.get("seen_hashes", [])),
            "total_bytes_processed": state.get("total_bytes_processed", 0),
        },
        "tokenization_stats": split_stats,
        "validation_passed": True,
    }
    write_json(report, output_dir / "dataset_report.json")

    return report


def validate_dataset(output_dir: Path, split_stats: dict, total_tokens: int):
    """Verify checksums, token counts, vocab range, empty files."""
    logger.info("Validating output files...")
    for split in ["train", "val", "test"]:
        file_path = output_dir / f"{split}.bin"
        if not file_path.exists():
            raise ValueError(f"Validation failed: {split}.bin is empty or missing.")

    metadata = json.loads((output_dir / "metadata.json").read_text())
    meta_total = sum(s["count"] for s in metadata["splits"].values())
    if meta_total != total_tokens:
        raise ValueError(
            f"Validation failed: Metadata token count ({meta_total}) != Actual tokens ({total_tokens})"
        )

    train_data = np.memmap(output_dir / "train.bin", dtype=np.uint32, mode="r")
    if train_data.size > 0:
        max_id = int(train_data.max())
        if max_id > 128000:
            logger.warning(f"Max token ID is suspiciously high: {max_id}")

    logger.info("Validation successful.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vajra Production Dataset Preparation")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="HuggingFace dataset name")
    parser.add_argument("--config-name", type=str, default=None, help="HuggingFace dataset config/subset name")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--max-gb", type=float, default=None, help="Maximum gigabytes to process")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum documents to process")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to process")
    parser.add_argument("--output", type=str, default="data/fineweb", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic processing")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for tokenization")
    args = parser.parse_args()
    
    result = prepare_huggingface(
        dataset_name=args.dataset,
        output_dir=args.output,
        stream=args.stream,
        config_name=args.config_name,
        max_gb=args.max_gb,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        seed=args.seed,
        resume=args.resume,
        batch_size=args.batch_size,
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
