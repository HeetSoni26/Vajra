import json
from collections.abc import Iterable
from pathlib import Path

from dataset.preparation.models import Document
from dataset.utils.logging import logger


class DocumentWriter:
    """
    Writes processed Document objects to disk.
    """

    @staticmethod
    def write_jsonl(documents: Iterable[Document], filepath: str | Path) -> None:
        """Writes documents to a JSONL file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in documents:
                data = {"id": doc.id, "text": doc.text, **doc.metadata}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"Successfully wrote {count} documents to {filepath}")
