"""Verification Engine example running ruff linter checks."""

from vajra_agent import VerificationEngine


def main():
    print("Running automatic verification pipeline (Ruff linter)...")
    report = VerificationEngine.verify(run_linter=True, run_tests=False, run_type_checker=False)

    print(f"Passed: {report.passed}")
    print(f"Linter Passed: {report.linter_passed}")
    print("\nObservation Formatting:")
    print(report.to_observation_text())


if __name__ == "__main__":
    main()
