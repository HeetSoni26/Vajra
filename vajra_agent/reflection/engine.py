"""ReflectionEngine reviewing agent execution, tool efficiency, and verification results."""

from __future__ import annotations

from vajra_agent.reflection.models import ReflectionResult
from vajra_agent.schemas.state import AgentState
from vajra_agent.verification.models import VerificationReport


class ReflectionEngine:
    """Self-reflection and post-execution critique engine."""

    @classmethod
    def reflect(
        cls,
        state: AgentState,
        verification: VerificationReport | None = None,
    ) -> ReflectionResult:
        tool_count = len(state.tool_history)
        errors = state.metadata.errors_encountered

        task_success = (errors == 0) and (verification.passed if verification else True)

        critique = (
            f"Processed {state.current_iteration} iteration(s) with {tool_count} tool call(s)."
        )
        efficiency = (
            "Optimal tool usage"
            if errors == 0
            else f"{errors} error(s) encountered during tool executions."
        )

        ver_feedback = "Verification passed cleanly."
        improvements = []

        if verification and not verification.passed:
            ver_feedback = f"Verification failures: {', '.join(verification.failures)}"
            improvements.append("Fix failing verification assertions or syntax errors.")

        if errors > 0:
            improvements.append("Check tool parameter validation before invocation.")

        return ReflectionResult(
            task_success=task_success,
            reasoning_critique=critique,
            tool_efficiency=efficiency,
            verification_feedback=ver_feedback,
            improvements_suggested=improvements,
        )
