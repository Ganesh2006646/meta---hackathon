"""Utility helpers for the ExecuCode environment."""

from __future__ import annotations

import re
import threading
from copy import deepcopy
from typing import Any


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_ALLOWED_MODULES = {"collections", "functools", "itertools", "math", "re", "string"}


def extract_code(message: str) -> str:
    """Extract Python code from markdown or raw agent text."""

    blocks = _CODE_BLOCK_RE.findall(message)
    if blocks:
        return blocks[-1].strip()

    match = re.search(r"(def\s+\w+\s*\(.*)", message, re.DOTALL)
    if match:
        return match.group(1).strip()

    return message.strip()


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


def _run_with_timeout(
    target: Any,
    timeout: float,
) -> tuple[bool, Any, str | None]:
    result: Any = None
    error: str | None = None

    def _wrapped() -> None:
        nonlocal result, error
        try:
            result = target()
        except Exception as exc:  # noqa: BLE001 - exposed as grading feedback
            error = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_wrapped, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return False, None, "TimeoutError: Code execution exceeded time limit."
    if error is not None:
        return False, None, error
    return True, result, None


def safe_exec(
    code: str,
    function_name: str,
    input_args: tuple[Any, ...],
    timeout: float = 5.0,
) -> tuple[bool, Any, str | None]:
    """Execute code in a restricted namespace and call the named function."""

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}

    def _define_function() -> None:
        exec(code, namespace)  # noqa: S102 - intentional graded execution

    ok, _, error = _run_with_timeout(_define_function, timeout)
    if not ok:
        return False, None, f"Compilation/definition error: {error}"

    function = namespace.get(function_name)
    if function is None:
        return False, None, f"Function '{function_name}' not found."
    if not callable(function):
        return False, None, f"'{function_name}' exists but is not callable."

    ok, result, error = _run_with_timeout(lambda: function(*input_args), timeout)
    if not ok:
        return False, None, f"Runtime error: {error}"
    return True, result, None


def safe_exec_sequence(
    code: str,
    function_name: str,
    input_args_list: list[tuple[Any, ...]],
    timeout: float = 5.0,
) -> tuple[bool, list[tuple[bool, Any, str | None]], str | None]:
    """Execute code once and run the target function for each input in order."""

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}

    def _define_function() -> None:
        exec(code, namespace)  # noqa: S102 - intentional graded execution

    ok, _, error = _run_with_timeout(_define_function, timeout)
    if not ok:
        return False, [], f"Compilation/definition error: {error}"

    function = namespace.get(function_name)
    if function is None:
        return False, [], f"Function '{function_name}' not found."
    if not callable(function):
        return False, [], f"'{function_name}' exists but is not callable."

    results: list[tuple[bool, Any, str | None]] = []
    for input_args in input_args_list:
        ok, result, call_error = _run_with_timeout(
            lambda args=input_args: function(*args),
            timeout,
        )
        if not ok:
            results.append((False, None, f"Runtime error: {call_error}"))
        else:
            try:
                snapshot = deepcopy(result)
            except Exception:  # noqa: BLE001 - fallback for non-copyable objects
                snapshot = result
            results.append((True, snapshot, None))

    return True, results, None


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
) -> str:
    """Generates a well-organized, readable feedback string for the agent."""

    solved = bool(is_done and total_reward >= 0.95)
    status_icon = "Solved" if solved else "Needs Improvement"

    if isinstance(test_details, list):
        total_tests = len(test_details)
        passed_tests = sum(1 for detail in test_details if detail.get("passed"))
        test_details_text = (
            f"({passed_tests}/{total_tests} tests passed)" if total_tests else ""
        )
    else:
        test_details_text = test_details.strip()

    if isinstance(performance_notes, list):
        perf_notes = [note.strip() for note in performance_notes if note and note.strip()]
        performance_notes_text = f"- {' '.join(perf_notes)}" if perf_notes else ""
    else:
        performance_notes_text = performance_notes.strip()

    if isinstance(quality_notes, list):
        qual_notes = [note.strip() for note in quality_notes if note and note.strip()]
        quality_notes_text = f"- {' '.join(qual_notes)}" if qual_notes else ""
    else:
        quality_notes_text = quality_notes.strip()

    feedback_lines = [
        f"### Evaluation (Attempt {step_count}/{max_attempts})",
        f"**Status:** {status_icon} | **Total Reward:** {total_reward:.3f} / 1.000",
        "",
        "#### Score Breakdown",
        f"* **Correctness:** {correctness_score:.3f} {test_details_text or ''}".strip(),
        f"* **Performance:** {performance_score:.3f} {performance_notes_text or ''}".strip(),
        f"* **Quality:** {quality_score:.3f} {quality_notes_text or ''}".strip(),
        "",
        "#### Suggested Next Actions",
    ]

    if solved:
        feedback_lines.append("* Excellent work! The task is fully optimized and solved.")
    else:
        if correctness_score < 0.95:
            feedback_lines.append(
                "* Fix failing correctness tests first to ensure the logic works."
            )
        elif performance_score < 0.95:
            feedback_lines.append(
                "* Code logic is correct, but needs performance optimization (check for O(n^2) loops)."
            )
        elif quality_score < 0.95:
            feedback_lines.append(
                "* Improve code readability (e.g., add docstrings or more descriptive variable names)."
            )
        else:
            feedback_lines.append(
                "* Solid progress. Keep refining edge cases and maintain clean structure."
            )

    return "\n".join(feedback_lines)
