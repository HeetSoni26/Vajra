"""Tests for VerificationEngine and VerificationReport formatting."""

from vajra_agent.verification import VerificationEngine, VerificationReport


def test_verification_report_observation_formatting():
    rep_pass = VerificationReport(passed=True)
    assert "VERIFICATION SUCCESSFUL" in rep_pass.to_observation_text()

    rep_fail = VerificationReport(
        passed=False,
        linter_passed=False,
        linter_output="F401 unused import",
        failures=["Linter check failed"],
    )
    obs = rep_fail.to_observation_text()
    assert "VERIFICATION FAILURE DETECTED" in obs
    assert "F401 unused import" in obs


def test_verification_engine_run():
    # Run linter verification on root
    report = VerificationEngine.verify(run_linter=True, run_tests=False, run_type_checker=False)
    assert isinstance(report.passed, bool)
