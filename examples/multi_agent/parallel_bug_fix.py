"""Parallel bug fix workflow example."""

from vajra_agent import FixFailingTestsWorkflow


def main():
    wf = FixFailingTestsWorkflow()
    response = wf.execute("ZeroDivisionError: division by zero in model/config.py line 45")

    print("Bug Fix Workflow Output:")
    print(response.output)


if __name__ == "__main__":
    main()
