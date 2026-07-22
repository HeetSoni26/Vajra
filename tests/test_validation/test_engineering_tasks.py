"""Validation suite executing 10 real-world software engineering tasks."""

from vajra_agent import (
    BugFixWorkflow,
    DocGenerationWorkflow,
    FeatureWorkflow,
    FixFailingTestsWorkflow,
    FoundationAgent,
    MockReasoner,
    RefactorWorkflow,
    RepoAnalysisWorkflow,
    RepoRefactorWorkflow,
    SecurityAuditWorkflow,
    TestGenerationWorkflow,
)


def test_validate_software_engineering_tasks():
    agent = FoundationAgent(MockReasoner(["Workflow execution complete."]))

    # 1. Repo Analysis
    w1 = RepoAnalysisWorkflow(agent)
    res1 = w1.execute(repo_dir=".")
    assert res1.output is not None

    # 2. Bug Fix
    w2 = BugFixWorkflow(agent)
    res2 = w2.execute(bug_description="Fix ZeroDivisionError", file_target="calc.py")
    assert res2.output is not None

    # 3. Feature Generation
    w3 = FeatureWorkflow(agent)
    res3 = w3.execute(feature_description="Implement OAuth2 login")
    assert res3.output is not None

    # 4. Refactoring
    w4 = RefactorWorkflow(agent)
    res4 = w4.execute(target_file="model.py", refactor_goal="Extract FeedForward")
    assert res4.output is not None

    # 5. Test Generation
    w5 = TestGenerationWorkflow(agent)
    res5 = w5.execute(source_file="config.py", test_file="test_config_gen.py")
    assert res5.output is not None

    # 6. Repo Refactor Workflow
    w6 = RepoRefactorWorkflow()
    res6 = w6.execute(module_path="vajra_agent/memory", objective="Extract interfaces")
    assert res6.output is not None

    # 7. Fix Failing Tests
    w7 = FixFailingTestsWorkflow()
    res7 = w7.execute(test_output="AssertionError: expected 1 got 0")
    assert res7.output is not None

    # 8. Doc Generation
    w8 = DocGenerationWorkflow()
    res8 = w8.execute(target_dir="vajra_agent")
    assert res8.output is not None

    # 9. Security Audit
    w9 = SecurityAuditWorkflow()
    res9 = w9.execute(target_dir="api/routes")
    assert res9.output is not None
