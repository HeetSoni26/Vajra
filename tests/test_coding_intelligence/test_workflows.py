"""Tests for specialized coding workflows."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.workflows import (
    BugFixWorkflow,
    FeatureWorkflow,
    RefactorWorkflow,
    RepoAnalysisWorkflow,
    TestGenerationWorkflow,
)


def test_coding_workflows_execution():
    reasoner = MockReasoner(["Workflow completed successfully."])
    agent = FoundationAgent(reasoner)

    repo_wf = RepoAnalysisWorkflow(agent)
    res_repo = repo_wf.execute(".")
    assert "Workflow completed" in res_repo.output

    bug_wf = BugFixWorkflow(agent)
    res_bug = bug_wf.execute("Fix division by zero", "calc.py")
    assert "Workflow completed" in res_bug.output

    feat_wf = FeatureWorkflow(agent)
    res_feat = feat_wf.execute("Add JWT auth")
    assert "Workflow completed" in res_feat.output

    ref_wf = RefactorWorkflow(agent)
    res_ref = ref_wf.execute("Extract class", "app.py")
    assert "Workflow completed" in res_ref.output

    test_wf = TestGenerationWorkflow(agent)
    res_test = test_wf.execute("model.py")
    assert "Workflow completed" in res_test.output
