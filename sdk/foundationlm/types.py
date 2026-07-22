from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str
