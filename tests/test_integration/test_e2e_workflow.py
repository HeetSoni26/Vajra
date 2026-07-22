"""End-to-end integration test combining all Vajra-Agent subsystems."""

from pathlib import Path
from vajra_agent import (
    ActionCategory,
    AgentPluginRegistry,
    ArtifactManager,
    CoderAgent,
    JobManager,
    JobStatus,
    MemoryManager,
    MultiAgentEngine,
    PermissionManager,
    PermissionPolicy,
    SaaSBuildWorkflow,
)


def test_full_system_end_to_end_integration(tmp_path: Path):
    # 1. Setup workspace & files
    (tmp_path / "main.py").write_text("def index(): return {'status': 'ok'}\n", encoding="utf-8")

    # 2. PermissionManager
    pm = PermissionManager(default_policy=PermissionPolicy.ALWAYS_ALLOW)
    pm.set_policy(ActionCategory.FILE_DELETE, PermissionPolicy.ALWAYS_DENY)
    assert pm.check_permission(ActionCategory.FILE_DELETE) is False

    # 3. Artifact & Job Managers
    am = ArtifactManager(artifact_dir=tmp_path / "artifacts")
    art = am.create_artifact(kind="plan", name="e2e_plan", content="E2E Plan Content")
    assert art.version == 1

    jm = JobManager()
    job = jm.submit_job("E2E background task", fn=lambda: "E2E Result")
    assert job.status == JobStatus.COMPLETED

    # 4. MemoryManager Repository Indexing
    mem = MemoryManager()
    mem.index_repository(tmp_path)
    assert mem.project_context is not None

    # 5. Plugin Registry
    AgentPluginRegistry.register_agent_plugin("E2ECoder", CoderAgent)
    assert AgentPluginRegistry.get_agent_plugin("E2ECoder") == CoderAgent

    # 6. MultiAgentEngine & SaaS Workflow
    engine = MultiAgentEngine()
    engine.setup_default_team()

    wf = SaaSBuildWorkflow(engine=engine)
    response = wf.execute("Build E2E feature")

    assert response.output is not None
    assert response.iterations > 0
