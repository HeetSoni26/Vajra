"""Memory Retention Policies (LRU, Importance, RecentOnly)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vajra_agent.memory.storage.base import VectorRecord


class BaseMemoryPolicy(ABC):
    """Abstract interface for memory eviction and pruning policies."""

    @abstractmethod
    def prune(self, records: list[VectorRecord], max_records: int) -> list[VectorRecord]:
        """Prune record list according to policy rules."""


class LRUPolicy(BaseMemoryPolicy):
    """Prunes least recently used or oldest timestamp records."""

    def prune(self, records: list[VectorRecord], max_records: int) -> list[VectorRecord]:
        if len(records) <= max_records:
            return records
        sorted_recs = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return sorted_recs[:max_records]


class ImportancePolicy(BaseMemoryPolicy):
    """Retains records marked high importance, pruning low importance ones first."""

    def prune(self, records: list[VectorRecord], max_records: int) -> list[VectorRecord]:
        if len(records) <= max_records:
            return records
        sorted_recs = sorted(
            records,
            key=lambda r: (r.metadata.get("importance", 0), r.timestamp),
            reverse=True,
        )
        return sorted_recs[:max_records]


class RecentOnlyPolicy(BaseMemoryPolicy):
    """Retains only the most recent N records."""

    def prune(self, records: list[VectorRecord], max_records: int) -> list[VectorRecord]:
        return records[-max_records:] if len(records) > max_records else records
