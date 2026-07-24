"""TaskGraph DAG representing execution dependencies across agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any


class TaskStatus(str, Enum):
    """Task node execution state in DAG."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    """Represents an individual node in the execution DAG."""

    description: str
    agent_role: str
    dependencies: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:6]}")
    status: TaskStatus = TaskStatus.PENDING
    output: Any = None
    error: str | None = None
    retry_count: int = 0


class TaskGraph:
    """DAG Task Graph managing execution dependencies, parallel task resolution, and retries."""

    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}

    def add_task(
        self,
        description: str,
        agent_role: str,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> TaskNode:
        """Add a task node to the DAG."""
        tid = task_id or f"task_{uuid.uuid4().hex[:6]}"
        node = TaskNode(
            id=tid,
            description=description,
            agent_role=agent_role,
            dependencies=dependencies or [],
        )
        self.nodes[tid] = node
        return node

    def get_ready_tasks(self) -> list[TaskNode]:
        """Find tasks whose dependencies are all COMPLETED and status is PENDING/READY."""
        ready = []
        for node in self.nodes.values():
            if node.status in (TaskStatus.PENDING, TaskStatus.READY):
                deps_met = all(
                    self.nodes[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in node.dependencies
                    if dep_id in self.nodes
                )
                if deps_met:
                    node.status = TaskStatus.READY
                    ready.append(node)
        return ready

    def mark_completed(self, task_id: str, output: Any) -> None:
        """Mark task completed."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.COMPLETED
            self.nodes[task_id].output = output

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task failed."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.FAILED
            self.nodes[task_id].error = error

    def is_complete(self) -> bool:
        """Check if all nodes are COMPLETED or FAILED."""
        return all(
            n.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for n in self.nodes.values()
        )
