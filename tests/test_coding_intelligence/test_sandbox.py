"""Tests for PythonSandbox code execution, file generation tracking, and error handling."""

from pathlib import Path

from vajra_agent.sandbox import PythonSandbox


def test_python_sandbox_successful_execution(tmp_path: Path):
    sandbox = PythonSandbox(work_dir=tmp_path)
    code = """
with open('output.txt', 'w') as f:
    f.write('hello sandbox')
print('Sandbox OK')
"""
    res = sandbox.execute(code)
    assert res.success
    assert res.stdout == "Sandbox OK"
    assert "output.txt" in res.files_generated


def test_python_sandbox_exception_capture(tmp_path: Path):
    sandbox = PythonSandbox(work_dir=tmp_path)
    code = "raise ValueError('Custom error')"
    res = sandbox.execute(code)

    assert not res.success
    assert res.error_type == "ValueError"
    assert res.traceback is not None
