"""Tests for ReflectionEngine and ReflectionResult generation."""

from vajra_agent.reflection import ReflectionEngine
from vajra_agent.schemas import AgentState, ToolExecutionResult
from vajra_agent.verification import VerificationReport


def test_reflection_engine_success_critique():
    state = AgentState()
    state.record_tool_result(ToolExecutionResult(tool_name="file_tool", success=True, output="ok"))
    state.current_iteration = 1

    ver = VerificationReport(passed=True)
    res = ReflectionEngine.reflect(state, ver)

    assert res.task_success
    assert "Optimal tool usage" in res.tool_efficiency


def test_reflection_engine_failure_critique():
    state = AgentState()
    state.record_tool_result(
        ToolExecutionResult(tool_name="file_tool", success=False, output=None, error="File missing")
    )
    state.current_iteration = 1

    ver = VerificationReport(passed=False, failures=["pytest failed"])
    res = ReflectionEngine.reflect(state, ver)

    assert not res.task_success
    assert len(res.improvements_suggested) > 0
