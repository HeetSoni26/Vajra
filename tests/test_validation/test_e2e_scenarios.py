"""Validation suite executing end-to-end multi-stage software engineering pipelines."""

from pathlib import Path
from vajra_agent import (
    ArtifactManager,
    JobManager,
    MemoryManager,
    MultiAgentEngine,
    TaskGraph,
)


def test_validate_full_end_to_end_engineering_scenario(tmp_path: Path):
    # 1. Initialize environment & workspace
    (tmp_path / "app.py").write_text("class App:\n    pass\n", encoding="utf-8")

    # 2. Ingest memory & workspace indexing
    mem = MemoryManager()
    mem.index_repository(tmp_path)
    assert mem.project_context is not None

    # 3. Create Artifacts & Jobs
    am = ArtifactManager(artifact_dir=tmp_path / "artifacts")
    am.create_artifact(kind="plan", name="master_plan", content="Full E2E Scenario Plan")

    jm = JobManager()
    job = jm.submit_job("Index workspace job", fn=lambda: "Completed")
    assert job.status.value == "completed"

    # 4. Multi-Agent DAG Execution
    engine = MultiAgentEngine()
    engine.setup_default_team()

    graph = TaskGraph()
    t1 = graph.add_task("Analyze codebase app.py", agent_role="ResearchAgent")
    t2 = graph.add_task("Design refactored API", agent_role="ArchitectAgent", dependencies=[t1.id])
    t3 = graph.add_task("Implement refactored code", agent_role="CoderAgent", dependencies=[t2.id])
    t4 = graph.add_task(
        "Verify implementation with pytest", agent_role="TesterAgent", dependencies=[t3.id]
    )
    graph.add_task("Review final output", agent_role="ReviewerAgent", dependencies=[t4.id])

    res = engine.run("Full E2E Engineering Scenario", task_graph=graph)

    assert res.output is not None
    assert res.iterations == 5
