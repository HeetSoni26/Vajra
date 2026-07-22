"""Verification Engine models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationReport:
    """Detailed verification report output from ruff, pytest, and mypy checks."""

    passed: bool
    linter_passed: bool = True
    linter_output: str = ""
    tests_passed: bool = True
    tests_output: str = ""
    type_checks_passed: bool = True
    type_checks_output: str = ""
    failures: list[str] = field(default_factory=list)

    def to_observation_text(self) -> str:
        """Format verification report into an agent observation prompt string."""
        if self.passed:
            return "VERIFICATION SUCCESSFUL: All linter checks and test suites passed cleanly."

        summary = ["VERIFICATION FAILURE DETECTED:"]
        if not self.linter_passed:
            summary.append(f"[Ruff Linter Error]\n{self.linter_output}")
        if not self.tests_passed:
            summary.append(f"[Pytest Failure]\n{self.tests_output}")
        if not self.type_checks_passed:
            summary.append(f"[Mypy Type Error]\n{self.type_checks_output}")

        return "\n\n".join(summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "linter_passed": self.linter_passed,
            "tests_passed": self.tests_passed,
            "type_checks_passed": self.type_checks_passed,
            "failures_count": len(self.failures),
        }
