"""Observability module providing execution tracing, replayability, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any


@dataclass
class TraceSpan:
    """Represents a single execution span in the agent lifecycle."""

    name: str
    start_time: float
    end_time: float = 0.0
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionTrace:
    """Structured execution trace logging agent thoughts, tool calls, and memory retrievals."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.spans: list[TraceSpan] = []
        self.start_time = time.time()
        self.end_time: float = 0.0

    def start_span(self, name: str, metadata: dict[str, Any] | None = None) -> TraceSpan:
        span = TraceSpan(name=name, start_time=time.time(), metadata=metadata or {})
        self.spans.append(span)
        return span

    def end_span(self, span: TraceSpan, status: str = "success") -> None:
        span.end_time = time.time()
        span.status = status

    def finish(self) -> None:
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "duration_s": round(self.end_time - self.start_time, 4) if self.end_time else 0.0,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round((s.end_time - s.start_time) * 1000, 2) if s.end_time else 0.0,
                    "status": s.status,
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
