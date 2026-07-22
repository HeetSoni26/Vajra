"""VerificationEngine running linter (ruff), test suite (pytest), and type checks (mypy)."""

from __future__ import annotations

from pathlib import Path
import subprocess
from vajra_agent.verification.models import VerificationReport


class VerificationEngine:
    """Automatic verification pipeline executing ruff, pytest, and mypy checks."""

    @classmethod
    def verify(
        cls,
        cwd: str | Path = ".",
        run_linter: bool = True,
        run_tests: bool = True,
        run_type_checker: bool = False,
        test_target: str | None = None,
    ) -> VerificationReport:
        cwd_path = Path(cwd).resolve()
        failures = []

        lint_ok = True
        lint_out = ""
        if run_linter:
            lint_ok, lint_out = cls._run_cmd(["ruff", "check", "."], cwd_path)
            if not lint_ok:
                failures.append("Linter (ruff) check failed.")

        test_ok = True
        test_out = ""
        if run_tests:
            cmd = ["pytest"]
            if test_target:
                cmd.append(test_target)
            test_ok, test_out = cls._run_cmd(cmd, cwd_path)
            if not test_ok:
                failures.append("Test suite (pytest) failed.")

        type_ok = True
        type_out = ""
        if run_type_checker:
            type_ok, type_out = cls._run_cmd(["mypy", "."], cwd_path)
            if not type_ok:
                failures.append("Type checker (mypy) failed.")

        passed = lint_ok and test_ok and type_ok

        return VerificationReport(
            passed=passed,
            linter_passed=lint_ok,
            linter_output=lint_out,
            tests_passed=test_ok,
            tests_output=test_out,
            type_checks_passed=type_ok,
            type_checks_output=type_out,
            failures=failures,
        )

    @staticmethod
    def _run_cmd(cmd: list[str], cwd: Path) -> tuple[bool, str]:
        try:
            res = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            out = (res.stdout + "\n" + res.stderr).strip()
            return (res.returncode == 0, out)
        except Exception as e:
            return (False, f"Execution failed: {str(e)}")
