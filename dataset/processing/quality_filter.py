from __future__ import annotations


class QualityFilter:
    """Configurable quality filter for documents."""

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 100000,
        min_alnum_ratio: float = 0.50,
        max_repetition_ratio: float = 0.30,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_alnum_ratio = min_alnum_ratio
        self.max_repetition_ratio = max_repetition_ratio

    def is_valid(self, text: str) -> tuple[bool, str]:
        """Check if document passes quality criteria. Returns (is_valid, reject_reason)."""
        stripped = text.strip()
        if not stripped:
            return False, "empty"

        words = stripped.split()
        num_words = len(words)

        if num_words < self.min_words:
            return False, "too_short"
        if num_words > self.max_words:
            return False, "too_long"

        # Alphanumeric character ratio
        alnum_chars = sum(1 for c in stripped if c.isalnum())
        alnum_ratio = alnum_chars / max(1, len(stripped))
        if alnum_ratio < self.min_alnum_ratio:
            return False, "low_alnum_ratio"

        # Repetition ratio (unique words vs total words)
        unique_words = len(set(words))
        repetition_ratio = 1.0 - (unique_words / max(1, num_words))
        if repetition_ratio > self.max_repetition_ratio:
            return False, "high_repetition"

        return True, "pass"
