"""Tests for TaskGraph DAG dependency resolution."""

from vajra_agent.multi_agent import TaskGraph, TaskStatus


def test_task_graph_dependency_resolution():
    graph = TaskGraph()
    t1 = graph.add_task("Step 1", agent_role="CoderAgent")
    t2 = graph.add_task("Step 2", agent_role="TesterAgent", dependencies=[t1.id])

    ready = graph.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == t1.id

    graph.mark_completed(t1.id, output="Step 1 complete")

    ready_next = graph.get_ready_tasks()
    assert len(ready_next) == 1
    assert ready_next[0].id == t2.id
    assert graph.nodes[t2.id].status == TaskStatus.READY
