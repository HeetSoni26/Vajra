"""Tests for Orchestrator and MultiAgentEngine multi-agent execution."""

from vajra_agent import MultiAgentEngine, TaskGraph


def test_multi_agent_engine_execution():
    engine = MultiAgentEngine()
    engine.setup_default_team()

    graph = TaskGraph()
    t1 = graph.add_task("Design architecture", agent_role="ArchitectAgent")
    graph.add_task("Write implementation code", agent_role="CoderAgent", dependencies=[t1.id])

    response = engine.run("Build module", task_graph=graph)

    assert response.output is not None
    assert response.iterations == 2
