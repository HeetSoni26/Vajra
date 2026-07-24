from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from transformers import AutoTokenizer

from utils.logging import setup_logger

logger = setup_logger("dataset_tokenize")


class DatasetTokenizer:
    """High-performance batch tokenization engine for document streams."""

    def __init__(self, tokenizer_dir: str | Path = "tokenizer/v1.0") -> None:
        path = Path(tokenizer_dir)
        self.tokenizer_dir = path.parent if path.is_file() else path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", 2) or 2

    def tokenize_documents(
        self, documents: list[dict[str, Any]]
    ) -> tuple[list[int], dict[str, Any]]:
        """Tokenize a list of documents, appending EOS token after each document."""
        all_tokens: list[int] = []
        start_time = time.time()
        doc_lengths: list[int] = []

        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
            encoded = self.tokenizer.encode(text)
            encoded.append(self.eos_token_id)
            doc_lengths.append(len(encoded))
            all_tokens.extend(encoded)

        duration = max(0.001, time.time() - start_time)
        tokens_per_sec = round(len(all_tokens) / duration, 2)

        metrics = {
            "num_documents": len(doc_lengths),
            "total_tokens": len(all_tokens),
            "tokens_per_sec": tokens_per_sec,
            "avg_doc_token_length": round(sum(doc_lengths) / max(1, len(doc_lengths)), 2),
            "max_doc_token_length": max(doc_lengths) if doc_lengths else 0,
        }

        logger.info(
            f"Tokenized {len(doc_lengths)} documents -> {len(all_tokens):,} tokens ({tokens_per_sec:,} tokens/sec)"
        )
        return all_tokens, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a text file into tokens.bin.")
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/v1.0")
    parser.add_argument("--output", default="data/tokenized/tokens.bin")
    args = parser.parse_args()

    engine = DatasetTokenizer(args.tokenizer)
    text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    tokens, metrics = engine.tokenize_documents([{"text": text}])

    import numpy as np

    ids = np.array(tokens, dtype=np.uint32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(args.output)
    print(
        {
            "tokens": int(ids.size),
            "output": args.output,
            "tokens_per_sec": metrics["tokens_per_sec"],
        }
    )


if __name__ == "__main__":
    main()
