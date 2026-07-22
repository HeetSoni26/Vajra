"""Tests for SpecializedAgent classes."""

from vajra_agent import (
    ArchitectAgent,
    CoderAgent,
    DebuggerAgent,
    DocumentationAgent,
    PlannerAgent,
    RefactorAgent,
    ResearchAgent,
    ReviewerAgent,
    SecurityAgent,
    TesterAgent,
)


def test_specialized_agent_roles():
    agents = [
        ArchitectAgent(),
        PlannerAgent(),
        ResearchAgent(),
        CoderAgent(),
        ReviewerAgent(),
        TesterAgent(),
        DebuggerAgent(),
        DocumentationAgent(),
        SecurityAgent(),
        RefactorAgent(),
    ]
    assert len(agents) == 10
    for agent in agents:
        assert agent.role is not None
        res = agent.run("Perform assigned duty")
        assert res.output is not None
