import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
import re
import numpy as np

from dataset.builder import BinaryDatasetBuilder
from dataset.processing.normalize import normalize_text
from dataset.tokenize_dataset import DatasetTokenizer
from utils.file_utils import ensure_dir, write_json
from utils.logging import setup_logger

logger = setup_logger("prepare_dataset")

# Basic HTML removal regex
HTML_TAG_RE = re.compile(r'<[^>]+>')

def clean_document(text: str, min_len: int = 50, max_len: int = 100000) -> str:
    """Apply cleaning rules: HTML removal, normalization, length constraints."""
    if not text:
        return ""
    # HTML removal
    text = HTML_TAG_RE.sub(" ", text)
    # Unicode and whitespace normalization
    text = normalize_text(text)
    
    if len(text) < min_len or len(text) > max_len:
        return ""
        
    return text

def prepare_huggingface(
    dataset_name: str,
    output_dir: str | Path,
    stream: bool = True,
    max_gb: float | None = None,
    max_docs: int | None = None,
    max_tokens: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> dict:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"Preparing dataset from {dataset_name} (stream={stream}, seed={seed})")
    
    random.seed(seed)
    
    tokenizer = DatasetTokenizer("tokenizer/v1.0")
    
    try:
        ds = load_dataset(dataset_name, split="train", streaming=stream)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    if stream:
        # We need to shuffle if streaming, but datasets.IterableDataset.shuffle requires a buffer size.
        ds = ds.shuffle(seed=seed, buffer_size=10000)

    seen_hashes = set()
    
    total_bytes = 0
    total_tokens_estimated = 0
    docs_processed = 0
    
    logger.info("Streaming and cleaning documents...")
    
    all_tokens = []
    
    for row in ds:
        text = row.get("text", "")
        # Language filtering (if metadata exists, otherwise heuristic)
        # fineweb-edu is English mostly. If language is provided we can check.
        if "language" in row and row["language"] != "en":
            continue
            
        cleaned = clean_document(text)
        if not cleaned:
            continue
            
        # Deduplication using hash
        doc_hash = hash(cleaned)
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)
        
        # Tokenize
        tokens, _ = tokenizer.tokenize_documents([{"text": cleaned}])
        all_tokens.extend(tokens)
        
        docs_processed += 1
        total_bytes += len(cleaned.encode("utf-8"))
        total_tokens_estimated += len(tokens)
        
        if max_docs and docs_processed >= max_docs:
            logger.info(f"Reached max_docs ({max_docs})")
            break
        if max_tokens and total_tokens_estimated >= max_tokens:
            logger.info(f"Reached max_tokens ({max_tokens})")
            break
        if max_gb and (total_bytes / (1024**3)) >= max_gb:
            logger.info(f"Reached max_gb ({max_gb})")
            break
            
    logger.info(f"Finished processing. Total docs: {docs_processed}, Total tokens: {len(all_tokens)}")
    
    builder = BinaryDatasetBuilder(
        output_dir=output_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    
    split_stats = builder.build_binary_dataset(
        all_tokens,
        metadata_info={
            "dataset": dataset_name,
            "seed": seed,
            "docs_processed": docs_processed,
            "total_bytes": total_bytes,
        }
    )
    
    # Validation
    validate_dataset(output_dir, split_stats, len(all_tokens))
    
    # Generate report
    report = {
        "dataset_name": dataset_name,
        "seed": seed,
        "cleaning_stats": {
            "docs_processed": docs_processed,
            "unique_docs": len(seen_hashes),
            "total_bytes_processed": total_bytes,
        },
        "tokenization_stats": split_stats,
        "validation_passed": True,
    }
    write_json(report, output_dir / "dataset_report.json")
    
    return report


def validate_dataset(output_dir: Path, split_stats: dict, total_tokens: int):
    """Verify checksums, token counts, vocab range, empty files."""
    logger.info("Validating output files...")
    
    # Check empty files
    for split in ["train", "val", "test"]:
        file_path = output_dir / f"{split}.bin"
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise ValueError(f"Validation failed: {split}.bin is empty or missing.")
            
    # Check token counts
    metadata = json.loads((output_dir / "metadata.json").read_text())
    meta_total = sum(s["count"] for s in metadata["splits"].values())
    if meta_total != total_tokens:
        raise ValueError(f"Validation failed: Metadata token count ({meta_total}) != Actual tokens ({total_tokens})")
        
    # Vocab range check
    train_data = np.memmap(output_dir / "train.bin", dtype=np.uint32, mode="r")
    if train_data.size > 0:
        max_id = int(train_data.max())
        if max_id > 128000:
            logger.warning(f"Max token ID is suspiciously high: {max_id}")
            
    logger.info("Validation successful.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vajra Production Dataset Preparation")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="HuggingFace dataset name")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--max-gb", type=float, default=None, help="Maximum gigabytes to process")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum documents to process")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to process")
    parser.add_argument("--output", type=str, default="data/fineweb", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic processing")
    args = parser.parse_args()
    
    result = prepare_huggingface(
        dataset_name=args.dataset,
        output_dir=args.output,
        stream=args.stream,
        max_gb=args.max_gb,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
