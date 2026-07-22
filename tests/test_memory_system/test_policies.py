"""Tests for memory pruning policies (LRU, Importance, RecentOnly)."""

from vajra_agent.memory.policies import ImportancePolicy, LRUPolicy, RecentOnlyPolicy
from vajra_agent.memory.storage import VectorRecord


def test_lru_policy():
    records = [
        VectorRecord(text="old", vector=[], timestamp=100.0),
        VectorRecord(text="new", vector=[], timestamp=200.0),
    ]
    policy = LRUPolicy()
    pruned = policy.prune(records, max_records=1)

    assert len(pruned) == 1
    assert pruned[0].text == "new"


def test_importance_policy():
    records = [
        VectorRecord(text="low", vector=[], metadata={"importance": 1}),
        VectorRecord(text="high", vector=[], metadata={"importance": 10}),
    ]
    policy = ImportancePolicy()
    pruned = policy.prune(records, max_records=1)

    assert len(pruned) == 1
    assert pruned[0].text == "high"


def test_recent_only_policy():
    records = [VectorRecord(text=f"msg_{i}", vector=[]) for i in range(5)]
    policy = RecentOnlyPolicy()
    pruned = policy.prune(records, max_records=2)

    assert len(pruned) == 2
    assert pruned[-1].text == "msg_4"
