"""MultiAgentEngine top-level manager interface."""

from __future__ import annotations

from vajra_agent.agent.agent import FoundationAgent
from vajra_agent.multi_agent.orchestrator import Orchestrator
from vajra_agent.multi_agent.shared_memory import SharedMemory
from vajra_agent.multi_agent.task_graph import TaskGraph
from vajra_agent.reasoners.mock import MockReasoner
from vajra_agent.schemas.results import AgentResponse
from vajra_agent.specialized.agents import (
    ArchitectAgent,
    CoderAgent,
    PlannerAgent,
    ReviewerAgent,
    TesterAgent,
)


class MultiAgentEngine:
    """Top-level Multi-Agent SDK engine for coordinating specialized agent teams."""

    def __init__(self, shared_memory: SharedMemory | None = None) -> None:
        self.shared_memory = shared_memory or SharedMemory()
        self.orchestrator = Orchestrator(shared_memory=self.shared_memory)

    def add_agent(self, agent: FoundationAgent, role: str | None = None) -> None:
        """Add an agent instance to the multi-agent team."""
        role_name = role or getattr(agent, "role", "WorkerAgent")
        self.orchestrator.add_agent(role_name, agent)

    def setup_default_team(self) -> None:
        """Convenience method setting up default software engineering agent team."""
        self.add_agent(
            ArchitectAgent(MockReasoner(["Architectural design approved."])), role="ArchitectAgent"
        )
        self.add_agent(PlannerAgent(MockReasoner(["Plan generated."])), role="PlannerAgent")
        self.add_agent(CoderAgent(MockReasoner(["Code implemented."])), role="CoderAgent")
        self.add_agent(TesterAgent(MockReasoner(["Unit tests passed."])), role="TesterAgent")
        self.add_agent(
            ReviewerAgent(MockReasoner(["Review passed cleanly."])), role="ReviewerAgent"
        )

    def run(self, prompt: str, task_graph: TaskGraph | None = None) -> AgentResponse:
        """Run multi-agent collaboration for a user prompt."""
        if not self.orchestrator.agents:
            self.setup_default_team()

        if task_graph is None:
            # Build default multi-stage task graph
            task_graph = TaskGraph()
            t1 = task_graph.add_task(
                f"Design architecture for: {prompt}", agent_role="ArchitectAgent"
            )
            t2 = task_graph.add_task(
                f"Decompose plan for: {prompt}", agent_role="PlannerAgent", dependencies=[t1.id]
            )
            t3 = task_graph.add_task(
                f"Implement solution for: {prompt}", agent_role="CoderAgent", dependencies=[t2.id]
            )
            t4 = task_graph.add_task(
                f"Verify and test solution for: {prompt}",
                agent_role="TesterAgent",
                dependencies=[t3.id],
            )
            task_graph.add_task(
                f"Review code and output for: {prompt}",
                agent_role="ReviewerAgent",
                dependencies=[t4.id],
            )

        res_dict = self.orchestrator.execute_task_graph(task_graph)
        summary_out = res_dict.get("summary", "Multi-agent execution complete.")

        return AgentResponse(
            output=summary_out,
            iterations=len(task_graph.nodes),
            tool_calls_count=0,
            conversation=[],
            execution_time_s=0.1,
            metadata={"outputs": res_dict.get("outputs", {})},
        )
