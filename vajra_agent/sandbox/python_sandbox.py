"""Enhanced Python Sandbox for isolated code execution."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from vajra_agent.sandbox.models import SandboxResult

BLOCKED_MODULES = {"subprocess", "ctypes", "pty", "socket", "signal"}


def _sandbox_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    base = name.split(".")[0]
    if base in BLOCKED_MODULES:
        raise ImportError(f"Security error: Module '{name}' is restricted in sandbox execution.")
    return __import__(name, globals, locals, fromlist, level)


def _get_sandbox_builtins(work_dir: Path) -> dict[str, Any]:
    import builtins

    safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_")}
    safe.pop("eval", None)
    safe.pop("compile", None)
    safe["__import__"] = _sandbox_import

    # Safe open function bounded to work_dir or relative paths
    raw_open = builtins.open

    def safe_open(file: str | Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = (work_dir / file_path).resolve()
        else:
            file_path = file_path.resolve()
        # Bounded file check
        try:
            file_path.relative_to(work_dir.resolve())
        except ValueError:
            raise PermissionError(
                f"Access denied: Path '{file}' is outside sandbox working directory '{work_dir}'."
            )
        return raw_open(file_path, mode, *args, **kwargs)

    safe["open"] = safe_open
    return safe


class PythonSandbox:
    """Isolated Python code execution environment capturing stdout/stderr, files created, and execution times."""

    def __init__(self, timeout: float = 30.0, work_dir: str | Path | None = None) -> None:
        self.timeout = timeout
        self.work_dir = Path(work_dir).resolve() if work_dir else Path.cwd().resolve()

    def execute(self, code: str, global_vars: dict[str, Any] | None = None) -> SandboxResult:
        start_t = time.perf_counter()
        buffer_out = io.StringIO()
        buffer_err = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = buffer_out
        sys.stderr = buffer_err

        # Record snapshot of existing files in work_dir
        before_files = set()
        if self.work_dir.exists():
            before_files = {
                str(f.relative_to(self.work_dir)) for f in self.work_dir.rglob("*") if f.is_file()
            }

        namespace = global_vars or {}
        namespace["__builtins__"] = _get_sandbox_builtins(self.work_dir)

        success = True
        error_type = None
        tb_str = None
        exit_code = 0

        import os

        old_cwd = Path.cwd()
        if self.work_dir.exists():
            os.chdir(self.work_dir)

        try:
            exec(code, namespace)
        except SystemExit as se:
            exit_code = int(se.code) if isinstance(se.code, int) else 1
            if exit_code != 0:
                success = False
                error_type = "SystemExit"
        except Exception as e:
            success = False
            exit_code = 1
            error_type = type(e).__name__
            tb_str = traceback.format_exc()
            print(tb_str, file=sys.stderr)
        finally:
            os.chdir(old_cwd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

        # Detect files created during execution
        after_files = set()
        if self.work_dir.exists():
            after_files = {
                str(f.relative_to(self.work_dir)) for f in self.work_dir.rglob("*") if f.is_file()
            }

        new_files = sorted(list(after_files - before_files))

        return SandboxResult(
            success=success,
            stdout=buffer_out.getvalue().strip(),
            stderr=buffer_err.getvalue().strip(),
            exit_code=exit_code,
            execution_time_ms=duration_ms,
            files_generated=new_files,
            error_type=error_type,
            traceback=tb_str,
        )
