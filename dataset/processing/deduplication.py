from __future__ import annotations

import hashlib


class Deduplicator:
    """Exact document deduplicator using text hashing."""

    def __init__(self) -> None:
        self.seen_hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if normalized document text has been seen before."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def clear(self) -> None:
        """Reset deduplication state."""
        self.seen_hashes.clear()
