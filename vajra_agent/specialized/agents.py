"""Built-in Specialized Agent definitions (Architect, Planner, Research, Coder, Reviewer, Tester, Debugger, Documentation, Security, Refactor)."""

from __future__ import annotations

from vajra_agent.specialized.base import SpecializedAgent


class ArchitectAgent(SpecializedAgent):
    """System Architect agent responsible for high-level technical design and architectural boundaries."""

    role = "ArchitectAgent"


class PlannerAgent(SpecializedAgent):
    """Task Planning agent responsible for step-by-step task breakdown and DAG dependency generation."""

    role = "PlannerAgent"


class ResearchAgent(SpecializedAgent):
    """Repository and Library Research agent responsible for codebase and documentation analysis."""

    role = "ResearchAgent"


class CoderAgent(SpecializedAgent):
    """Software Engineer agent responsible for code generation and file modifications."""

    role = "CoderAgent"


class ReviewerAgent(SpecializedAgent):
    """Code Reviewer agent responsible for quality checks, code style, and architectural compliance."""

    role = "ReviewerAgent"


class TesterAgent(SpecializedAgent):
    """QA & Testing agent responsible for writing unit test suites and running pytest verification."""

    __test__ = False

    role = "TesterAgent"


class DebuggerAgent(SpecializedAgent):
    """Root Cause Debugger agent responsible for diagnosing stack traces and fixing runtime errors."""

    role = "DebuggerAgent"


class DocumentationAgent(SpecializedAgent):
    """Technical Writer agent responsible for docstrings, READMEs, and API documentation."""

    role = "DocumentationAgent"


class SecurityAgent(SpecializedAgent):
    """Application Security agent responsible for vulnerability scanning and security audits."""

    role = "SecurityAgent"


class RefactorAgent(SpecializedAgent):
    """Refactoring Specialist agent responsible for code modernization, module extraction, and cleanup."""

    role = "RefactorAgent"
