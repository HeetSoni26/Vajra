"""Orchestrator coordinating task assignment, DAG execution, retries, and output synthesis across agents."""

from __future__ import annotations

from typing import Any

from vajra_agent.agent.agent import FoundationAgent
from vajra_agent.multi_agent.communication import AgentMessage, MessageType
from vajra_agent.multi_agent.shared_memory import SharedMemory
from vajra_agent.multi_agent.task_graph import TaskGraph, TaskStatus
from utils.logging import setup_logger

logger = setup_logger("multi_agent_orchestrator")


class Orchestrator:
    """Orchestrator supervising multi-agent DAG task execution and message routing."""

    def __init__(self, shared_memory: SharedMemory | None = None) -> None:
        self.shared_memory = shared_memory or SharedMemory()
        self.agents: dict[str, FoundationAgent] = {}
        self.message_history: list[AgentMessage] = []

    def add_agent(self, role: str, agent: FoundationAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[role] = agent
        # Attach shared memory manager to agent
        agent.attach_memory(self.shared_memory.memory_manager)

    def dispatch_message(self, message: AgentMessage) -> None:
        """Record and route structured message between agents."""
        self.message_history.append(message)
        if message.msg_type == MessageType.SHARED_OBSERVATION:
            text = message.payload.get("observation", "")
            self.shared_memory.publish_observation(message.sender_id, text)

    def execute_task_graph(self, task_graph: TaskGraph, max_retries: int = 2) -> dict[str, Any]:
        """Execute task graph DAG by dispatching ready nodes to corresponding specialized agents."""
        outputs: dict[str, Any] = {}

        while not task_graph.is_complete():
            ready_tasks = task_graph.get_ready_tasks()
            if not ready_tasks:
                # Break if no ready tasks available to prevent deadlock
                break

            for task in ready_tasks:
                agent = self.agents.get(task.agent_role)
                if not agent:
                    # Fallback to default agent if specific role not registered
                    agent = next(iter(self.agents.values())) if self.agents else None

                if not agent:
                    task_graph.mark_failed(task.id, f"No agent registered for role '{task.agent_role}'")
                    continue

                task.status = TaskStatus.RUNNING
                logger.info(f"Orchestrator dispatching task '{task.description}' to role '{task.agent_role}'")

                try:
                    res = agent.run(task.description)
                    task_graph.mark_completed(task.id, res.output)
                    outputs[task.id] = res.output

                    # Broadcast progress update message
                    self.dispatch_message(
                        AgentMessage(
                            sender_id=task.agent_role,
                            recipient_id="orchestrator",
                            msg_type=MessageType.PROGRESS_UPDATE,
                            payload={"task_id": task.id, "output": res.output},
                        )
                    )
                except Exception as e:
                    task.retry_count += 1
                    if task.retry_count <= max_retries:
                        logger.warning(f"Task '{task.id}' failed, retrying ({task.retry_count}/{max_retries})...")
                        task.status = TaskStatus.READY
                    else:
                        task_graph.mark_failed(task.id, str(e))

        summary = "\n\n".join([f"### [{node.agent_role}] Task: {node.description}\n{node.output}" for node in task_graph.nodes.values() if node.status == TaskStatus.COMPLETED])
        return {"outputs": outputs, "summary": summary}
