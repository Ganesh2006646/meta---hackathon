"""Utility helpers for the ExecuCode environment."""

from __future__ import annotations

import re
import threading
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


def generate_feedback(
    correctness_score: float,
    performance_score: float,
    quality_score: float,
    total_reward: float,
    test_details: list[dict[str, Any]],
    performance_notes: list[str],
    quality_notes: list[str],
    is_done: bool,
    step_count: int,
    max_attempts: int,
) -> str:
    """Build deterministic text feedback for the agent."""

    passed = sum(1 for detail in test_details if detail["passed"])
    total = len(test_details)
    lines = [
        f"Step {step_count}/{max_attempts} evaluation",
        f"Total reward: {total_reward:.3f} / 1.000",
        f"Correctness: {correctness_score:.3f} ({passed}/{total} tests passed)",
    ]

    for detail in test_details:
        if detail["passed"]:
            continue
        lines.append(
            "Failed test "
            f"{detail['index']}: input={detail['input']!r}, "
            f"expected={detail['expected']!r}, actual={detail['actual']!r}"
        )
        if detail.get("error"):
            lines.append(f"Error: {detail['error']}")

    lines.append(f"Performance: {performance_score:.3f}")
    lines.extend(f"- {note}" for note in performance_notes)

    lines.append(f"Code quality: {quality_score:.3f}")
    lines.extend(f"- {note}" for note in quality_notes)

    if is_done and total_reward >= 0.95:
        lines.append("Status: solved.")
    elif is_done:
        lines.append(f"Status: maximum attempts reached at reward {total_reward:.3f}.")
    else:
        lines.append(f"Status: {max_attempts - step_count} attempts remaining.")

    return "\n".join(lines)
