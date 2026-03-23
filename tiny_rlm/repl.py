"""Sandboxed Python REPL with FINAL() and async llm_query() support.

Adopted patterns from fast-rlm:
- FINAL() function sets a flag + stores result (any Python object)
- llm_query() is async, supports asyncio.gather() for parallel sub-calls
- Context loaded via JSON serialization (safe string embedding)
- Combined namespace for exec() (avoids Python scoping quirks)
- Each sub-agent gets an isolated namespace
"""

import asyncio
import builtins
import io
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    execution_time: float


def _build_safe_builtins() -> dict:
    """Copy real builtins, remove dangerous ones."""
    safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_")}
    for blocked in ("eval", "exec", "compile", "input", "globals", "locals", "breakpoint"):
        safe.pop(blocked, None)
    # Keep __import__ for re, json, asyncio etc.
    safe["__import__"] = __import__
    return safe


class REPLEnv:
    """Sandboxed Python REPL environment for RLM code execution."""

    def __init__(
        self,
        llm_query_fn: Callable | None = None,
    ):
        self.final_result: Any = None
        self.final_set: bool = False

        # Build namespace
        safe_builtins = _build_safe_builtins()
        self._namespace: dict[str, Any] = {"__builtins__": safe_builtins}

        # Inject FINAL() — same pattern as fast-rlm
        def FINAL(x: Any) -> None:
            self.final_result = x
            self.final_set = True

        self._namespace["FINAL"] = FINAL

        # Inject llm_query if provided (root agents only, not leaf)
        if llm_query_fn is not None:
            self._namespace["llm_query"] = llm_query_fn

    def load_context(self, context: str | dict | list) -> None:
        """Store context in namespace. Uses JSON serialization for safe embedding."""
        setup = f"context = {json.dumps(context)}\n"
        self.execute(setup)

    def execute(self, code: str) -> REPLResult:
        """Execute synchronous Python code in the sandbox."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()

        start = time.time()
        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            exec(code, self._namespace, self._namespace)
            stderr_content = stderr_buf.getvalue()
        except Exception as e:
            stderr_content = stderr_buf.getvalue() + f"\nError: {e}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_content,
            execution_time=time.time() - start,
        )

    async def execute_async(self, code: str) -> REPLResult:
        """Execute code that may contain `await` (for llm_query calls).

        Wraps code in an async function to support await syntax,
        similar to how fast-rlm uses Pyodide's runPythonAsync.
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()

        start = time.time()
        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf

            # Wrap in async function to enable await
            indented = textwrap.indent(code, "    ")
            wrapped = f"async def __repl_main__():\n{indented}\n"
            exec(wrapped, self._namespace, self._namespace)
            await self._namespace["__repl_main__"]()

            stderr_content = stderr_buf.getvalue()
        except Exception as e:
            stderr_content = stderr_buf.getvalue() + f"\nError: {e}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_content,
            execution_time=time.time() - start,
        )

    def get_variable(self, name: str) -> Any | None:
        return self._namespace.get(name)
