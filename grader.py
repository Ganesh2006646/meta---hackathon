"""Deterministic grading engine for ExecuCode submissions."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

try:
    from .tasks import Task
    from .utils import safe_exec
except ImportError:
    from tasks import Task
    from utils import safe_exec


@dataclass(frozen=True)
class GradeResult:
    """Full grading result for a code submission."""

    correctness: float
    performance: float
    quality: float
    reward: float
    test_details: list[dict[str, Any]]
    performance_notes: list[str]
    quality_notes: list[str]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _has_nested_loop(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.For, ast.While)):
                    return True
    return False


def _uses_lookup_structure(tree: ast.AST) -> bool:
    lookup_calls = {"set", "dict", "Counter", "defaultdict"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.Dict)):
            return True
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in lookup_calls:
                return True
            if isinstance(function, ast.Attribute) and function.attr in {"add", "get"}:
                return True
    return False


def _uses_builtin_sort(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "sorted":
            return True
        if isinstance(function, ast.Attribute) and function.attr == "sort":
            return True
    return False


def _score_correctness(code: str, task: Task) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    correct = 0

    for index, test_case in enumerate(task.test_cases, start=1):
        ok, actual, error = safe_exec(
            code=code,
            function_name=task.function_name,
            input_args=test_case.input_args,
            timeout=3.0,
        )
        passed = ok and actual == test_case.expected_output
        if passed:
            correct += 1
        details.append(
            {
                "index": index,
                "input": test_case.input_args,
                "expected": test_case.expected_output,
                "actual": actual,
                "error": error,
                "passed": passed,
            }
        )

    return correct / len(task.test_cases), details


def _score_performance(code: str, task: Task) -> tuple[float, list[str]]:
    notes: list[str] = []
    if task.scoring_weights[1] == 0:
        return 1.0, ["Performance is not weighted for this task."]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return 0.0, [f"Could not parse code for performance analysis: {exc.msg}."]

    score = 0.55
    nested_loop = _has_nested_loop(tree)
    uses_lookup = _uses_lookup_structure(tree)
    uses_sort = _uses_builtin_sort(tree)

    optimal_matches = [
        pattern for pattern in task.optimal_patterns if re.search(pattern, code, re.DOTALL)
    ]
    anti_matches = [
        pattern for pattern in task.anti_patterns if re.search(pattern, code, re.DOTALL)
    ]

    if uses_lookup:
        score += 0.3
        notes.append("Uses set/dict-style lookup structures.")
    if uses_sort:
        score += 0.1
        notes.append("Uses built-in sorting rather than manual sorting.")
    if optimal_matches:
        score += min(0.15, 0.05 * len(optimal_matches))
        notes.append(f"Matched {len(optimal_matches)} task-specific optimal pattern(s).")
    if nested_loop:
        score = min(score, 0.35)
        notes.append("Nested loops detected; this suggests O(n^2) behavior.")
    if anti_matches:
        score = min(score, 0.3)
        notes.append(f"Matched {len(anti_matches)} task-specific anti-pattern(s).")

    if not notes:
        notes.append("No strong optimization signal detected.")

    return _clamp(score), notes


def _score_quality(code: str, task: Task) -> tuple[float, list[str]]:
    notes: list[str] = []
    if task.scoring_weights[2] == 0:
        return 1.0, ["Code quality is not weighted for this task."]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return 0.0, [f"Could not parse code for quality analysis: {exc.msg}."]

    score = 1.0
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    target = next((node for node in function_defs if node.name == task.function_name), None)

    if target is None:
        return 0.0, [f"Function `{task.function_name}` was not found."]

    if ast.get_docstring(target):
        notes.append("Function has a docstring.")
    else:
        score -= 0.2
        notes.append("Function is missing a docstring.")

    body_line_count = max(
        1,
        getattr(target, "end_lineno", target.lineno) - target.lineno + 1,
    )
    if body_line_count > 35:
        score -= 0.15
        notes.append("Function is longer than expected for this task.")
    else:
        notes.append("Function length is concise.")

    single_letter_names = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and len(node.id) == 1 and node.id not in {"i", "j"}
    }
    single_letter_arguments = {
        arg.arg
        for arg in target.args.args
        if len(arg.arg) == 1 and arg.arg not in {"i", "j"}
    }
    terse_names = single_letter_names | single_letter_arguments
    if terse_names:
        score -= min(0.25, 0.08 * len(terse_names))
        notes.append(f"Terse variable names detected: {', '.join(sorted(terse_names))}.")
    else:
        notes.append("Variable names are descriptive enough for this task.")

    return _clamp(score), notes


def grade_submission(code: str, task: Task) -> GradeResult:
    """Grade a submission against a task and return all scoring dimensions."""

    correctness, test_details = _score_correctness(code, task)
    performance, performance_notes = _score_performance(code, task)
    quality, quality_notes = _score_quality(code, task)
    c_weight, p_weight, q_weight = task.scoring_weights
    reward = _clamp(
        correctness * c_weight + performance * p_weight + quality * q_weight
    )

    return GradeResult(
        correctness=correctness,
        performance=performance,
        quality=quality,
        reward=reward,
        test_details=test_details,
        performance_notes=performance_notes,
        quality_notes=quality_notes,
    )
