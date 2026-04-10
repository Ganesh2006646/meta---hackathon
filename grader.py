"""Deterministic grading engine for ExecuCode submissions."""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

try:
    from .tasks import Task
    from .utils import safe_exec_sequence
except ImportError:
    from tasks import Task
    from utils import safe_exec_sequence


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


@dataclass(frozen=True)
class _PerformanceSignals:
    loop_count: int
    max_loop_depth: int
    comprehension_count: int
    sort_calls: int
    sort_inside_loop: int
    linear_membership_checks: int
    hash_lookup_usage: int
    nested_structure: bool


def _clamp(value: float) -> float:
    return max(0.001, min(0.999, value))


def _find_target_function(tree: ast.AST, function_name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None


class _PerformanceVisitor(ast.NodeVisitor):
    """Collect complexity-related AST signals from a function body."""

    def __init__(self) -> None:
        self.loop_count = 0   
        self.current_loop_depth = 0
        self.max_loop_depth = 0
        self.comprehension_count = 0
        self.sort_calls = 0
        self.sort_inside_loop = 0
        self.linear_membership_checks = 0
        self.hash_lookup_usage = 0
        self.nested_structure = False
        self._list_like_names: set[str] = set()
        self._hash_like_names: set[str] = set()

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.loop_count += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        if self.current_loop_depth >= 2:
            self.nested_structure = True
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self.loop_count += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        if self.current_loop_depth >= 2:
            self.nested_structure = True
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._visit_comprehension(node)

    def _visit_comprehension(self, node: ast.AST) -> None:
        generators = getattr(node, "generators", [])
        self.comprehension_count += 1
        self.loop_count += len(generators)
        if len(generators) >= 2:
            self.nested_structure = True
            self.max_loop_depth = max(self.max_loop_depth, len(generators))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._register_container_name(target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._register_container_name(node.target.id, node.value)
        self.generic_visit(node)

    def _register_container_name(self, name: str, value: ast.AST) -> None:
        if isinstance(value, (ast.List, ast.Tuple)):
            self._list_like_names.add(name)
            return
        if isinstance(value, (ast.Set, ast.Dict)):
            self._hash_like_names.add(name)
            return
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                if value.func.id in {"list", "tuple"}:
                    self._list_like_names.add(name)
                elif value.func.id in {"set", "dict", "Counter", "defaultdict"}:
                    self._hash_like_names.add(name)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        for op, right in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if isinstance(right, (ast.List, ast.Tuple)):
                self.linear_membership_checks += 1
            elif isinstance(right, (ast.Set, ast.Dict)):
                self.hash_lookup_usage += 1
            elif isinstance(right, ast.Name):
                if right.id in self._list_like_names:
                    self.linear_membership_checks += 1
                elif right.id in self._hash_like_names:
                    self.hash_lookup_usage += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        function = node.func

        if isinstance(function, ast.Name):
            if function.id == "sorted":
                self.sort_calls += 1
                if self.current_loop_depth > 0:
                    self.sort_inside_loop += 1
            elif function.id in {"set", "dict", "Counter", "defaultdict"}:
                self.hash_lookup_usage += 1
        elif isinstance(function, ast.Attribute):
            if function.attr == "sort":
                self.sort_calls += 1
                if self.current_loop_depth > 0:
                    self.sort_inside_loop += 1
            elif function.attr in {"add", "get", "items", "keys", "values"}:
                self.hash_lookup_usage += 1

        self.generic_visit(node)

    def to_signals(self) -> _PerformanceSignals:
        return _PerformanceSignals(
            loop_count=self.loop_count,
            max_loop_depth=max(1, self.max_loop_depth) if self.loop_count else 0,
            comprehension_count=self.comprehension_count,
            sort_calls=self.sort_calls,
            sort_inside_loop=self.sort_inside_loop,
            linear_membership_checks=self.linear_membership_checks,
            hash_lookup_usage=self.hash_lookup_usage,
            nested_structure=self.nested_structure,
        )


def _extract_performance_signals(target: ast.FunctionDef) -> _PerformanceSignals:
    visitor = _PerformanceVisitor()
    visitor.visit(target)
    return visitor.to_signals()


def _snapshot(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:  # noqa: BLE001 - fallback for non-copyable values
        return value


def _score_correctness(code: str, task: Task) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    correct = 0
    input_args_list = [test_case.input_args for test_case in task.test_cases]

    ran, run_results, run_error = safe_exec_sequence(
        code=code,
        function_name=task.function_name,
        input_args_list=input_args_list,
        timeout=3.0,
    )

    if not ran:
        for index, test_case in enumerate(task.test_cases, start=1):
            details.append(
                {
                    "index": index,
                    "input": _snapshot(test_case.input_args),
                    "expected": _snapshot(test_case.expected_output),
                    "actual": None,
                    "error": run_error,
                    "passed": False,
                }
            )
        return 0.0, details

    for index, (test_case, result) in enumerate(
        zip(task.test_cases, run_results),
        start=1,
    ):
        ok, actual, error = result
        passed = ok and actual == test_case.expected_output
        if passed:
            correct += 1
        details.append(
            {
                "index": index,
                "input": _snapshot(test_case.input_args),
                "expected": _snapshot(test_case.expected_output),
                "actual": _snapshot(actual),
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

    target = _find_target_function(tree, task.function_name)
    if target is None:
        return 0.0, [f"Function `{task.function_name}` was not found."]

    signals = _extract_performance_signals(target)
    score = 0.5

    notes.append(
        "Complexity signals: "
        f"loops={signals.loop_count}, "
        f"max_depth={signals.max_loop_depth}, "
        f"linear_membership={signals.linear_membership_checks}, "
        f"hash_usage={signals.hash_lookup_usage}, "
        f"sort_calls={signals.sort_calls}."
    )

    if not signals.nested_structure:
        score += 0.2
        notes.append("No nested loop/comprehension structure detected.")
    else:
        score -= 0.25
        notes.append("Nested loop/comprehension structure detected; quadratic risk is high.")

    if signals.hash_lookup_usage > 0:
        score += 0.2
        notes.append("Uses hash-based lookup patterns (set/dict style).")
    else:
        notes.append("No strong hash-based lookup signal detected.")

    if signals.linear_membership_checks > 0:
        penalty = min(0.2, 0.07 * signals.linear_membership_checks)
        score -= penalty
        notes.append("Linear membership checks (`in list/tuple`) may hurt scalability.")

    if signals.sort_calls > 0 and signals.sort_inside_loop == 0:
        score += 0.08
        notes.append("Sort usage is outside loops.")
    if signals.sort_inside_loop > 0:
        penalty = min(0.2, 0.1 * signals.sort_inside_loop)
        score -= penalty
        notes.append("Sorting inside loops detected, which can amplify runtime.")

    if signals.comprehension_count > 0 and not signals.nested_structure:
        score += 0.05
        notes.append("Comprehension usage is concise and likely efficient.")

    optimal_matches = [
        pattern for pattern in task.optimal_patterns if re.search(pattern, code, re.DOTALL)
    ]
    anti_matches = [
        pattern for pattern in task.anti_patterns if re.search(pattern, code, re.DOTALL)
    ]

    if optimal_matches:
        bonus = min(0.12, 0.04 * len(optimal_matches))
        score += bonus
        notes.append(f"Matched {len(optimal_matches)} task-specific optimal pattern(s).")
    if anti_matches:
        penalty = min(0.25, 0.12 * len(anti_matches))
        score -= penalty
        notes.append(f"Matched {len(anti_matches)} task-specific anti-pattern(s).")

    return _clamp(score), notes


def _score_quality(code: str, task: Task) -> tuple[float, list[str]]:
    notes: list[str] = []
    if task.scoring_weights[2] == 0:
        return 1.0, ["Code quality is not weighted for this task."]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return 0.0, [f"Could not parse code for quality analysis: {exc.msg}."]

    target = _find_target_function(tree, task.function_name)
    if target is None:
        return 0.0, [f"Function `{task.function_name}` was not found."]

    score = 1.0

    has_docstring = ast.get_docstring(target) is not None
    if has_docstring:
        notes.append("Function includes a docstring.")
    else:
        score -= 0.15
        notes.append("Function is missing a docstring.")

    body_line_count = max(
        1,
        getattr(target, "end_lineno", target.lineno) - target.lineno + 1,
    )
    if body_line_count > 70:
        score -= 0.25
        notes.append("Function body is very long; consider smaller logical blocks.")
    elif body_line_count > 45:
        score -= 0.15
        notes.append("Function body is longer than expected for this task.")
    else:
        notes.append("Function length is concise.")

    branch_nodes = sum(
        1
        for node in ast.walk(target)
        if isinstance(
            node,
            (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.IfExp),
        )
    )
    if branch_nodes > 20:
        score -= 0.25
        notes.append("Control flow is highly complex.")
    elif branch_nodes > 12:
        score -= 0.15
        notes.append("Control flow is moderately complex.")
    else:
        notes.append("Control flow complexity is manageable.")

    variable_names: set[str] = set()
    variable_names.update(arg.arg for arg in target.args.args)
    for node in ast.walk(target):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variable_names.add(node.id)

    terse_names = sorted(
        name
        for name in variable_names
        if len(name) == 1 and name not in {"i", "j", "k"}
    )
    if terse_names:
        score -= min(0.24, 0.06 * len(terse_names))
        notes.append(f"Terse variable names detected: {', '.join(terse_names)}.")
    else:
        notes.append("Variable names are descriptive.")

    non_snake_names = sorted(
        name
        for name in variable_names
        if not re.match(r"^[a-z_][a-z0-9_]*$", name)
    )
    if non_snake_names:
        score -= min(0.12, 0.03 * len(non_snake_names))
        notes.append("Some variable names do not follow snake_case style.")

    statement_signatures: dict[str, int] = {}
    for stmt in target.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                continue
        signature = ast.dump(stmt, include_attributes=False)
        statement_signatures[signature] = statement_signatures.get(signature, 0) + 1

    duplicate_count = sum(count - 1 for count in statement_signatures.values() if count > 1)
    if duplicate_count > 0:
        score -= min(0.12, 0.04 * duplicate_count)
        notes.append("Repeated statement blocks detected; consider refactoring duplicates.")

    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(target)):
        score -= 0.1
        notes.append("Global/nonlocal state usage detected.")

    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(target)
    ):
        score -= 0.05
        notes.append("Debug `print` calls detected in final solution.")

    has_type_hints = (
        all(arg.annotation is not None for arg in target.args.args)
        and target.returns is not None
    )
    if has_type_hints:
        score += 0.03
        notes.append("Type hints are present for arguments and return value.")

    return _clamp(score), notes


def grade_submission(code: str, task: Task) -> GradeResult:
    """Grade a submission against a task and return all scoring dimensions."""

    correctness_raw, test_details = _score_correctness(code, task)
    correctness = _clamp(correctness_raw)
    
    performance, performance_notes = _score_performance(code, task)
    quality, quality_notes = _score_quality(code, task)
    
    c_weight, p_weight, q_weight = task.scoring_weights
    reward = _clamp(
        correctness_raw * c_weight + performance * p_weight + quality * q_weight
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
