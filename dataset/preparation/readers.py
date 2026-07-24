import json
import uuid
from typing import Iterable
from pathlib import Path
from dataset.preparation.models import Document
from dataset.utils.logging import logger


class DocumentReader:
    """
    Reads raw dataset files and yields Document objects.
    """

    @staticmethod
    def read_jsonl(filepath: str | Path) -> Iterable[Document]:
        """Reads a JSONL file assuming a 'text' field exists."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text") or data.get("content") or ""

                    # Store everything else as metadata
                    metadata = {k: v for k, v in data.items() if k not in ("text", "content")}
                    doc_id = str(data.get("id", uuid.uuid4()))

                    yield Document(id=doc_id, text=text, metadata=metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON on line {line_num} in {filepath}")

    @staticmethod
    def read_txt(filepath: str | Path) -> Iterable[Document]:
        """Reads a plain text file, treating the entire file or each paragraph as a document."""
        # For simplicity, treat the whole file as one document
        filepath = Path(filepath)
        if not filepath.exists():
            return

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            yield Document(id=str(uuid.uuid4()), text=text, metadata={"source_file": filepath.name})
