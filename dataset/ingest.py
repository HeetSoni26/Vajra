from __future__ import annotations

import csv
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

from utils.logging import setup_logger

logger = setup_logger("dataset_ingest")


class DataIngestor:
    """Configurable multi-format text dataset ingestor with recursive scanning and corrupt file handling."""

    SUPPORTED_EXTENSIONS = {".jsonl", ".txt", ".md", ".csv"}

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    def discover_files(self) -> list[Path]:
        """Recursively discover all supported data files."""
        files: list[Path] = []
        if not self.raw_dir.exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
            return files

        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.raw_dir.rglob(f"*{ext}"))
        return sorted(files)

    def ingest_file(self, path: Path) -> Generator[dict[str, Any], None, None]:
        """Ingest documents from a single file safely."""
        suffix = path.suffix.lower()
        doc_idx = 0

        try:
            if suffix == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            text = item.get("text") or item.get("content") or item.get("body") or ""
                            if text:
                                yield {
                                    "doc_id": f"{path.stem}_{line_num}",
                                    "text": text,
                                    "source_file": str(path),
                                }
                                doc_idx += 1
                        except json.JSONDecodeError:
                            continue

            elif suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    yield {
                        "doc_id": f"{path.stem}_1",
                        "text": text,
                        "source_file": str(path),
                    }
                    doc_idx += 1

            elif suffix == ".csv":
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row_num, row in enumerate(reader, 1):
                        text = row.get("text") or row.get("content") or ""
                        if text:
                            yield {
                                "doc_id": f"{path.stem}_{row_num}",
                                "text": text,
                                "source_file": str(path),
                            }
                            doc_idx += 1

        except Exception as exc:
            logger.error(f"Error reading file {path}: {exc}")

    def stream_documents(self) -> Generator[dict[str, Any], None, None]:
        """Stream documents across all discovered raw files."""
        files = self.discover_files()
        logger.info(f"Discovered {len(files)} raw files in {self.raw_dir}")

        for file_path in files:
            yield from self.ingest_file(file_path)
