from __future__ import annotations

import re
import unicodedata

# Compile regex for control chars and multi-spaces
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Apply standard NFC normalization, whitespace collapsing, and control char cleanup."""
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove non-printable control characters
    text = _CONTROL_CHAR_RE.sub("", text)

    # Normalize horizontal spaces
    text = _MULTI_SPACE_RE.sub(" ", text)

    # Limit consecutive newlines to maximum 2
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()
