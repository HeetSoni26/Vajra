"""Tests for ArtifactManager versioning."""

from pathlib import Path
from vajra_agent import ArtifactManager


def test_artifact_manager_versioning(tmp_path: Path):
    am = ArtifactManager(artifact_dir=tmp_path)
    a1 = am.create_artifact(kind="plan", name="task_plan", content="v1 plan")
    a2 = am.create_artifact(kind="plan", name="task_plan", content="v2 plan")

    assert a1.version == 1
    assert a2.version == 2

    latest = am.get_latest_artifact("task_plan")
    assert latest is not None
    assert latest.content == "v2 plan"
