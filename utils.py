"""Utility helpers for the ExecuCode environment."""

from __future__ import annotations

import ast
import multiprocessing as mp
import os
import pickle
import re
import sys
import threading
import time
import traceback
from queue import Empty
from copy import deepcopy
from typing import Any

try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore[assignment]


_CODE_BLOCK_RE = re.compile(
    r"```(?:\s*(?:python|py))?\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_OPEN_RE = re.compile(r"^```(?:\s*(?:python|py))?\s*$", re.IGNORECASE)
_ALLOWED_MODULES = {"collections", "functools", "itertools", "math", "re", "string"}
_BLOCKED_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "breakpoint",
}
_BLOCKED_ATTRIBUTE_NAMES = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__mro__",
    "__bases__",
    "__dict__",
}
_DEFAULT_MEMORY_LIMIT_MB = int(os.getenv("EXECUCODE_MEMORY_LIMIT_MB", "256"))


def extract_code(message: str) -> str:
    """Extract Python code from markdown or raw agent text."""

    stripped_message = message.strip()
    blocks = _CODE_BLOCK_RE.findall(stripped_message)
    if blocks:
        return blocks[-1].strip()

    # Gracefully handle partially fenced blocks that may miss the final fence.
    if stripped_message.startswith("```"):
        lines = stripped_message.splitlines()
        if lines and _FENCE_OPEN_RE.match(lines[0].strip()):
            remaining_lines = lines[1:]
            if remaining_lines and remaining_lines[-1].strip() == "```":
                remaining_lines = remaining_lines[:-1]
            unfenced = "\n".join(remaining_lines).strip()
            if unfenced:
                return unfenced

    match = re.search(r"(def\s+\w+\s*\(.*)", stripped_message, re.DOTALL)
    if match:
        return match.group(1).strip()

    return stripped_message


def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    root_name = name.split(".", 1)[0]
    if root_name not in _ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed.")
    return __import__(name, globals_, locals_, fromlist, level)


_SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def _queue_safe(value: Any) -> Any:
    """Convert values to queue-safe payloads."""

    try:
        pickle.dumps(value)
        return value
    except Exception:  # noqa: BLE001 - best effort for non-pickleable values
        return repr(value)


def _structured_error(
    *,
    status: str,
    error_type: str,
    message: str,
    line: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error_type": error_type,
        "message": message,
        "line": line,
    }


def _exception_to_error(
    exc: BaseException,
    *,
    status: str,
) -> dict[str, Any]:
    line: int | None = None
    traceback_frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    for frame in reversed(traceback_frames):
        if frame.filename in {"<string>", "<submitted_code>"}:
            line = frame.lineno
            break
    if line is None and traceback_frames:
        line = traceback_frames[-1].lineno

    return _structured_error(
        status=status,
        error_type=type(exc).__name__,
        message=str(exc),
        line=line,
    )


def _apply_resource_limits(timeout: float, memory_limit_mb: int) -> None:
    """Apply best-effort CPU/memory limits for POSIX worker processes."""

    if resource is None:
        return

    setrlimit = getattr(resource, "setrlimit", None)
    if not callable(setrlimit):
        return

    cpu_budget = max(1, int(timeout) + 1)
    rlimit_cpu = getattr(resource, "RLIMIT_CPU", None)
    try:
        if rlimit_cpu is not None:
            setrlimit(rlimit_cpu, (cpu_budget, cpu_budget + 1))
    except Exception:
        pass

    if memory_limit_mb <= 0:
        return

    memory_bytes = memory_limit_mb * 1024 * 1024
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            setrlimit(limit, (memory_bytes, memory_bytes))
        except Exception:
            continue


def _validate_code_safety(code: str) -> dict[str, Any] | None:
    """Reject submissions that use clearly unsafe primitives."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _structured_error(
            status="compilation_error",
            error_type="SyntaxError",
            message=exc.msg,
            line=exc.lineno,
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALL_NAMES:
                return _structured_error(
                    status="policy_error",
                    error_type="SecurityPolicyError",
                    message=f"Call to blocked function `{node.func.id}` is not allowed.",
                    line=getattr(node, "lineno", None),
                )
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRIBUTE_NAMES:
            return _structured_error(
                status="policy_error",
                error_type="SecurityPolicyError",
                message=f"Access to blocked attribute `{node.attr}` is not allowed.",
                line=getattr(node, "lineno", None),
            )

    return None


def _sequence_worker(
    code: str,
    function_name: str,
    input_args_list: list[tuple[Any, ...]],
    timeout: float,
    memory_limit_mb: int,
    result_queue: Any,
) -> None:
    """Worker process for isolated sequence execution."""

    _apply_resource_limits(timeout, memory_limit_mb)
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}

    try:
        compiled = compile(code, "<submitted_code>", "exec")
        exec(compiled, namespace)  # noqa: S102 - intentional graded execution
    except BaseException as exc:  # noqa: BLE001 - surfaced to grader
        result_queue.put(
            {
                "ran": False,
                "results": [],
                "error": _exception_to_error(exc, status="compilation_error"),
            }
        )
        return

    function = namespace.get(function_name)
    if function is None:
        result_queue.put(
            {
                "ran": False,
                "results": [],
                "error": _structured_error(
                    status="compilation_error",
                    error_type="MissingFunctionError",
                    message=f"Function '{function_name}' not found.",
                    line=None,
                ),
            }
        )
        return

    if not callable(function):
        result_queue.put(
            {
                "ran": False,
                "results": [],
                "error": _structured_error(
                    status="compilation_error",
                    error_type="InvalidFunctionError",
                    message=f"'{function_name}' exists but is not callable.",
                    line=None,
                ),
            }
        )
        return

    results: list[dict[str, Any]] = []
    for input_args in input_args_list:
        try:
            call_args = deepcopy(input_args)
        except Exception:  # noqa: BLE001 - fallback for non-copyable values
            call_args = input_args

        started = time.perf_counter()
        peak_memory_kb: int | None = None
        try:
            import tracemalloc

            tracemalloc.start()
            output = function(*call_args)
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            del current_bytes
            peak_memory_kb = int(peak_bytes / 1024)
            tracemalloc.stop()
            error: dict[str, Any] | None = None
            ok = True
        except BaseException as exc:  # noqa: BLE001 - surfaced to grader
            try:
                import tracemalloc

                if tracemalloc.is_tracing():
                    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                    del current_bytes
                    peak_memory_kb = int(peak_bytes / 1024)
                    tracemalloc.stop()
            except Exception:
                pass
            output = None
            error = _exception_to_error(exc, status="runtime_error")
            ok = False

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if ok:
            try:
                output_snapshot = deepcopy(output)
            except Exception:  # noqa: BLE001 - fallback for non-copyable values
                output_snapshot = output
        else:
            output_snapshot = None

        results.append(
            {
                "ok": ok,
                "output": _queue_safe(output_snapshot),
                "error": _queue_safe(error),
                "elapsed_ms": round(elapsed_ms, 3),
                "memory_kb": peak_memory_kb,
            }
        )

    result_queue.put(
        {
            "ran": True,
            "results": results,
            "error": None,
        }
    )


def _run_with_timeout(
    target: Any,
    timeout: float,
) -> tuple[bool, Any, dict[str, Any] | None]:
    result: Any = None
    error: dict[str, Any] | None = None

    def _wrapped() -> None:
        nonlocal result, error
        try:
            result = target()
        except BaseException as exc:  # noqa: BLE001 - surfaced to grader
            error = _exception_to_error(exc, status="runtime_error")

    worker_thread = threading.Thread(target=_wrapped, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout)

    if worker_thread.is_alive():
        return (
            False,
            None,
            _structured_error(
                status="timeout",
                error_type="TimeoutError",
                message="Code execution exceeded time limit.",
                line=None,
            ),
        )
    if error is not None:
        return False, None, error
    return True, result, None


def _safe_exec_sequence_in_process(
    code: str,
    function_name: str,
    input_args_list: list[tuple[Any, ...]],
    timeout: float,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    """Fallback executor for environments where process sandboxing is unavailable."""

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}

    try:
        compiled = compile(code, "<submitted_code>", "exec")
    except SyntaxError as exc:
        return (
            False,
            [],
            _structured_error(
                status="compilation_error",
                error_type="SyntaxError",
                message=exc.msg,
                line=exc.lineno,
            ),
        )

    ok, _, compile_error = _run_with_timeout(lambda: exec(compiled, namespace), timeout)  # noqa: S102
    if not ok:
        if compile_error is not None:
            compile_error["status"] = "compilation_error"
        return False, [], compile_error

    function = namespace.get(function_name)
    if function is None:
        return (
            False,
            [],
            _structured_error(
                status="compilation_error",
                error_type="MissingFunctionError",
                message=f"Function '{function_name}' not found.",
                line=None,
            ),
        )
    if not callable(function):
        return (
            False,
            [],
            _structured_error(
                status="compilation_error",
                error_type="InvalidFunctionError",
                message=f"'{function_name}' exists but is not callable.",
                line=None,
            ),
        )

    results: list[dict[str, Any]] = []
    for input_args in input_args_list:
        try:
            call_args = deepcopy(input_args)
        except Exception:  # noqa: BLE001 - fallback for non-copyable values
            call_args = input_args

        started = time.perf_counter()
        peak_memory_kb: int | None = None
        try:
            import tracemalloc

            tracemalloc.start()
            ok, output, call_error = _run_with_timeout(
                lambda args=call_args: function(*args),
                timeout,
            )
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            del current_bytes
            peak_memory_kb = int(peak_bytes / 1024)
            tracemalloc.stop()
        except Exception:
            ok, output, call_error = _run_with_timeout(
                lambda args=call_args: function(*args),
                timeout,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if ok:
            try:
                output_snapshot = deepcopy(output)
            except Exception:  # noqa: BLE001 - fallback for non-copyable values
                output_snapshot = output
            error = None
        else:
            output_snapshot = None
            error = call_error

        results.append(
            {
                "ok": ok,
                "output": output_snapshot,
                "error": error,
                "elapsed_ms": round(elapsed_ms, 3),
                "memory_kb": peak_memory_kb,
            }
        )

    return True, results, None


def safe_exec(
    code: str,
    function_name: str,
    input_args: tuple[Any, ...],
    timeout: float = 5.0,
) -> tuple[bool, Any, dict[str, Any] | None]:
    """Execute code in a restricted namespace and call the named function."""

    ran, results, run_error = safe_exec_sequence(
        code=code,
        function_name=function_name,
        input_args_list=[input_args],
        timeout=timeout,
    )
    if not ran:
        return False, None, run_error

    result = results[0]
    if not result.get("ok"):
        return False, None, result.get("error")
    return True, result.get("output"), None


def safe_exec_sequence(
    code: str,
    function_name: str,
    input_args_list: list[tuple[Any, ...]],
    timeout: float = 5.0,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    """Execute code once and run the target function for each input in order."""

    policy_error = _validate_code_safety(code)
    if policy_error is not None:
        return False, [], policy_error

    # Windows and interactive runners can be fragile with `spawn`; use a
    # deterministic in-process fallback there.
    main_file = getattr(sys.modules.get("__main__"), "__file__", "")
    is_hf_space = os.getenv("SPACE_ID") is not None
    # Always use the in-process fallback for reliability in web interfaces and docker
    if True:
        return _safe_exec_sequence_in_process(
            code=code,
            function_name=function_name,
            input_args_list=input_args_list,
            timeout=timeout,
        )

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_sequence_worker,
        args=(
            code,
            function_name,
            input_args_list,
            timeout,
            _DEFAULT_MEMORY_LIMIT_MB,
            result_queue,
        ),
        daemon=True,
    )

    worker.start()
    worker.join(timeout + 0.25)

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=0.5)
        return (
            False,
            [],
            _structured_error(
                status="timeout",
                error_type="TimeoutError",
                message="Code execution exceeded time limit.",
                line=None,
            ),
        )

    try:
        payload = result_queue.get_nowait()
    except Empty:
        if worker.exitcode is not None and worker.exitcode != 0:
            return (
                False,
                [],
                _structured_error(
                    status="sandbox_error",
                    error_type="SandboxProcessError",
                    message=f"Sandbox terminated unexpectedly (exit code {worker.exitcode}).",
                    line=None,
                ),
            )
        return (
            False,
            [],
            _structured_error(
                status="sandbox_error",
                error_type="SandboxProcessError",
                message="Sandbox returned no result payload.",
                line=None,
            ),
        )
    finally:
        try:
            result_queue.close()
        except Exception:
            pass

    ran = bool(payload.get("ran"))
    run_error = payload.get("error")
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    return ran, results, run_error


def generate_feedback(
    correctness_score: float,
    performance_score: float,
    quality_score: float,
    total_reward: float,
    test_details: str | list[dict[str, Any]],
    performance_notes: str | list[str],
    quality_notes: str | list[str],
    is_done: bool,
    step_count: int,
    max_attempts: int,
    ai_mentor_tip: str = "",
) -> str:
    """Generates a well-organized, readable feedback string for the agent."""

    solved = bool(is_done and total_reward >= 0.95)
    status_icon = "✅ Solved" if solved else "⚠️ Needs Improvement"

    if isinstance(test_details, list):
        total_tests = len(test_details)
        passed_tests = sum(1 for detail in test_details if detail.get("passed"))
        test_details_text = (
            f"({passed_tests}/{total_tests} tests passed)" if total_tests else ""
        )
    else:
        test_details_text = test_details.strip()

    # FIX: Format each note on its own bullet line instead of joining into one
    # squashed line. This makes feedback legible for both humans and agents.
    if isinstance(performance_notes, list):
        perf_notes = [note.strip() for note in performance_notes if note and note.strip()]
        performance_notes_text = (
            "\n  " + "\n  ".join(f"• {n}" for n in perf_notes) if perf_notes else ""
        )
    else:
        performance_notes_text = f"\n  {performance_notes.strip()}" if performance_notes.strip() else ""

    if isinstance(quality_notes, list):
        qual_notes = [note.strip() for note in quality_notes if note and note.strip()]
        quality_notes_text = (
            "\n  " + "\n  ".join(f"• {n}" for n in qual_notes) if qual_notes else ""
        )
    else:
        quality_notes_text = f"\n  {quality_notes.strip()}" if quality_notes.strip() else ""

    feedback_lines = [
        f"### Evaluation (Attempt {step_count}/{max_attempts})",
        f"**Status:** {status_icon} | **Total Reward:** {total_reward:.3f} / 1.000",
        "",
        "#### Score Breakdown",
        f"* **Correctness:** {correctness_score:.3f} {test_details_text or ''}".strip(),
        f"* **Performance:** {performance_score:.3f}{performance_notes_text}".rstrip(),
        f"* **Quality:** {quality_score:.3f}{quality_notes_text}".rstrip(),
        "",
        "#### Suggested Next Actions",
    ]

    if solved:
        feedback_lines.append("* 🎉 Excellent work! The task is fully optimized and solved.")
    else:
        if correctness_score < 0.95:
            first_error = next((d.get("error") for d in (test_details if isinstance(test_details, list) else []) if not d.get("passed") and d.get("error")), None)
            if first_error:
                error_msg = first_error.get("message", str(first_error)) if isinstance(first_error, dict) else str(first_error)
                feedback_lines.append(f"* Critical Execution/Compilation Error: {error_msg}")
            else:
                failing = [
                    f"  - Test {d['index']}: input={d['input']}, expected={d['expected']}, got={d['actual']}"
                    for d in (test_details if isinstance(test_details, list) else [])
                    if not d.get("passed") and d.get("error") is None
                ][:3]  # show up to 3 failing cases
                feedback_lines.append(
                    "* Fix failing correctness tests first to ensure the logic works."
                )
                feedback_lines.extend(failing)
        elif performance_score < 0.80:
            feedback_lines.append(
                "* Code logic is correct. Optimize performance: eliminate O(n²) loops, "
                "use memoization or hash-based data structures."
            )
        elif quality_score < 0.80:
            feedback_lines.append(
                "* Improve code quality: add a docstring, use descriptive variable names, "
                "and add type hints."
            )
        else:
            feedback_lines.append(
                "* Solid progress. Keep refining edge cases and maintain clean structure."
            )

    if ai_mentor_tip:
        feedback_lines.append("")
        feedback_lines.append("#### 🤖 AI Mentor Final Review")
        feedback_lines.append(ai_mentor_tip)

    return "\n".join(feedback_lines)
