"""Tests for multi-agent re-usable workflow executions."""

from vajra_agent import (
    DocGenerationWorkflow,
    FixFailingTestsWorkflow,
    RepoRefactorWorkflow,
    SaaSBuildWorkflow,
    SecurityAuditWorkflow,
)


def test_multi_agent_workflows():
    saas = SaaSBuildWorkflow()
    res_saas = saas.execute("Build User Auth API")
    assert res_saas.output is not None

    refactor = RepoRefactorWorkflow()
    res_ref = refactor.execute("model/model.py", "Extract SwiGLU")
    assert res_ref.output is not None

    fix_test = FixFailingTestsWorkflow()
    res_fix = fix_test.execute("ZeroDivisionError in calc.py line 12")
    assert res_fix.output is not None

    doc_gen = DocGenerationWorkflow()
    res_doc = doc_gen.execute("vajra_agent/memory")
    assert res_doc.output is not None

    sec_audit = SecurityAuditWorkflow()
    res_sec = sec_audit.execute("api/routes")
    assert res_sec.output is not None
