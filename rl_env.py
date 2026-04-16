"""RL environment wrappers for ExecuCode.

This file contains two environment classes:
- ExecuCodeEnv: HumanEval-backed Gym-compatible environment.
- ExecuCodeRLEnv: existing task-bank wrapper kept for backwards compatibility.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import Any, Iterable, cast

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym  # type: ignore[no-redef]

from datasets import Dataset, DatasetDict, load_dataset

try:
    from .grader import GradeResult, grade_submission
    from .tasks import ALL_TASKS, Task, TestCase, get_task
    from .utils import extract_code, generate_feedback
except ImportError:
    from grader import GradeResult, grade_submission
    from tasks import ALL_TASKS, Task, TestCase, get_task
    from utils import extract_code, generate_feedback

_SCORE_EPSILON = 1e-3
_ACTION_KEYS = (
    "code",
    "message",
    "answer",
    "submission",
    "response",
    "content",
    "text",
)
_HUMANEVAL_DATASET = "openai/openai_humaneval"
_HUMANEVAL_SPLIT = "test"


def _clamp_open_interval(value: float) -> float:
    """Clamp scores to the strict open interval (0, 1)."""

    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, float(value)))


def _action_to_code(action: str | dict[str, Any]) -> str:
    """Normalize an action payload into Python code text."""

    if isinstance(action, str):
        return extract_code(action)

    if isinstance(action, dict):
        for key in _ACTION_KEYS:
            candidate = action.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return extract_code(candidate)
        return ""

    return extract_code(str(action))


@dataclass(frozen=True)
class HumanEvalSample:
    """Normalized subset of HumanEval fields needed by ExecuCode."""

    task_id: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str


class ExecuCodeEnv(gym.Env):  # type: ignore[misc]
    """Gym-style environment that samples tasks from the HumanEval dataset."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_name: str = _HUMANEVAL_DATASET,
        split: str = _HUMANEVAL_SPLIT,
        *,
        seed: int | None = None,
        max_sampling_attempts: int = 64,
        samples: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.max_sampling_attempts = max(1, int(max_sampling_attempts))
        self._rng = random.Random(seed)
        self._samples = self._load_samples(samples=samples)
        if not self._samples:
            raise ValueError("HumanEval dataset is empty.")

        self.current_task: Task | None = None
        self.current_sample: HumanEvalSample | None = None

    def _load_samples(self, samples: Iterable[dict[str, Any]] | None) -> list[HumanEvalSample]:
        def _row_to_dict(raw_row: Any) -> dict[str, Any]:
            if isinstance(raw_row, dict):
                return dict(raw_row)
            if hasattr(raw_row, "items"):
                try:
                    return dict(raw_row.items())
                except Exception:
                    return {}
            return {}

        if samples is not None:
            rows = [_row_to_dict(row) for row in samples]
        else:
            dataset = load_dataset(self.dataset_name)
            if isinstance(dataset, Dataset):
                split_dataset = dataset
            elif isinstance(dataset, DatasetDict):
                if self.split in dataset:
                    split_dataset = dataset[self.split]
                else:
                    split_dataset = dataset[next(iter(dataset.keys()))]
            else:
                split_dataset = dataset[self.split]  # pragma: no cover
            rows = [
                _row_to_dict(raw_row)
                for raw_row in cast(Iterable[Any], split_dataset)
            ]

        parsed_samples: list[HumanEvalSample] = []
        for row in rows:
            task_id = str(row.get("task_id", ""))
            prompt = str(row.get("prompt", ""))
            test = str(row.get("test", ""))
            entry_point = str(row.get("entry_point", ""))
            canonical_solution = str(row.get("canonical_solution", ""))
            if not task_id or not prompt or not test or not entry_point:
                continue

            parsed_samples.append(
                HumanEvalSample(
                    task_id=task_id,
                    prompt=prompt,
                    test=test,
                    entry_point=entry_point,
                    canonical_solution=canonical_solution,
                )
            )

        return parsed_samples

    def _literal_eval_node(self, node: ast.AST) -> tuple[bool, Any]:
        try:
            return True, ast.literal_eval(node)
        except Exception:
            return False, None

    def _call_name(self, call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def _is_target_call(self, call: ast.Call, entry_point: str) -> bool:
        call_name = self._call_name(call)
        return bool(call_name in {"candidate", entry_point})

    def _call_to_args(self, call: ast.Call) -> tuple[Any, ...] | None:
        if call.keywords:
            return None

        parsed_args: list[Any] = []
        for node in call.args:
            ok, value = self._literal_eval_node(node)
            if not ok:
                return None
            parsed_args.append(value)

        return tuple(parsed_args)

    def _build_case_from_call(self, call: ast.Call, expected: Any) -> TestCase | None:
        input_args = self._call_to_args(call)
        if input_args is None:
            return None
        return TestCase(input_args=input_args, expected_output=expected)

    def _build_case_from_call_with_reference(
        self,
        call: ast.Call,
        reference_callable: Any,
    ) -> TestCase | None:
        input_args = self._call_to_args(call)
        if input_args is None:
            return None

        try:
            expected_output = reference_callable(*input_args)
        except Exception:
            return None

        return TestCase(input_args=input_args, expected_output=expected_output)

    def _find_target_call(self, expr: ast.AST, entry_point: str) -> ast.Call | None:
        for node in ast.walk(expr):
            if isinstance(node, ast.Call) and self._is_target_call(node, entry_point):
                return node
        return None

    def _assert_to_case(
        self,
        expr: ast.AST,
        entry_point: str,
        reference_callable: Any | None = None,
    ) -> TestCase | None:
        if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq):
            left = expr.left
            right = expr.comparators[0]

            if isinstance(left, ast.Call) and self._is_target_call(left, entry_point):
                ok, expected = self._literal_eval_node(right)
                if ok:
                    return self._build_case_from_call(left, expected)
                if reference_callable is not None:
                    inferred_case = self._build_case_from_call_with_reference(
                        left,
                        reference_callable,
                    )
                    if inferred_case is not None:
                        return inferred_case

            if isinstance(right, ast.Call) and self._is_target_call(right, entry_point):
                ok, expected = self._literal_eval_node(left)
                if ok:
                    return self._build_case_from_call(right, expected)
                if reference_callable is not None:
                    inferred_case = self._build_case_from_call_with_reference(
                        right,
                        reference_callable,
                    )
                    if inferred_case is not None:
                        return inferred_case

            return None

        if isinstance(expr, ast.Call) and self._is_target_call(expr, entry_point):
            direct_true_case = self._build_case_from_call(expr, True)
            if direct_true_case is not None:
                return direct_true_case
            if reference_callable is not None:
                return self._build_case_from_call_with_reference(expr, reference_callable)
            return None

        if (
            isinstance(expr, ast.UnaryOp)
            and isinstance(expr.op, ast.Not)
            and isinstance(expr.operand, ast.Call)
            and self._is_target_call(expr.operand, entry_point)
        ):
            direct_false_case = self._build_case_from_call(expr.operand, False)
            if direct_false_case is not None:
                return direct_false_case
            if reference_callable is not None:
                return self._build_case_from_call_with_reference(expr.operand, reference_callable)
            return None

        if reference_callable is not None:
            fallback_call = self._find_target_call(expr, entry_point)
            if fallback_call is not None:
                return self._build_case_from_call_with_reference(
                    fallback_call,
                    reference_callable,
                )

        return None

    def _build_reference_callable(self, sample: HumanEvalSample) -> Any:
        prompt = sample.prompt or ""
        canonical = sample.canonical_solution or ""

        # HumanEval usually ships a prompt prefix plus canonical body suffix,
        # while local tests may provide a full function in canonical_solution.
        candidate_sources = [
            f"{prompt}\n{canonical}",
            prompt + canonical,
            canonical,
            prompt,
        ]

        last_error: Exception | None = None
        for source in candidate_sources:
            source_text = source.strip("\n")
            if not source_text:
                continue

            namespace: dict[str, Any] = {}
            try:
                exec(source_text, namespace)  # noqa: S102 - trusted benchmark dataset
            except Exception as exc:  # noqa: BLE001 - try alternate source forms
                last_error = exc
                continue

            candidate = namespace.get(sample.entry_point)
            if callable(candidate):
                return candidate

        raise ValueError(
            f"Reference callable '{sample.entry_point}' was not built for task '{sample.task_id}'."
        ) from last_error

    def _extract_runtime_cases(
        self,
        sample: HumanEvalSample,
        reference_callable: Any,
    ) -> tuple[TestCase, ...]:
        namespace: dict[str, Any] = {}
        exec(sample.test, namespace)  # noqa: S102 - trusted benchmark dataset
        check_fn = namespace.get("check")
        if not callable(check_fn):
            return ()

        captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        seen_keys: set[str] = set()

        def _candidate_proxy(*args: Any, **kwargs: Any) -> Any:
            key = f"args={repr(args)}|kwargs={repr(sorted(kwargs.items()))}"
            if key not in seen_keys:
                seen_keys.add(key)
                captured_calls.append((tuple(args), dict(kwargs)))
            return reference_callable(*args, **kwargs)

        check_fn(_candidate_proxy)

        runtime_cases: list[TestCase] = []
        for args, kwargs in captured_calls:
            if kwargs:
                continue
            try:
                expected_output = reference_callable(*args)
            except Exception:
                continue
            runtime_cases.append(TestCase(input_args=tuple(args), expected_output=expected_output))

        return tuple(runtime_cases)

    def _dedupe_cases(self, cases: Iterable[TestCase]) -> tuple[TestCase, ...]:
        deduped: list[TestCase] = []
        seen: set[str] = set()
        for case in cases:
            key = f"{repr(case.input_args)}=>{repr(case.expected_output)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(case)
        return tuple(deduped)

    def _extract_test_cases(self, sample: HumanEvalSample) -> tuple[TestCase, ...]:
        tree = ast.parse(sample.test)
        reference_callable = self._build_reference_callable(sample)

        check_function: ast.FunctionDef | None = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "check":
                check_function = node
                break

        scope: ast.AST = check_function if check_function is not None else tree
        cases: list[TestCase] = []
        for node in ast.walk(scope):
            if not isinstance(node, ast.Assert):
                continue
            parsed_case = self._assert_to_case(
                node.test,
                sample.entry_point,
                reference_callable,
            )
            if parsed_case is not None:
                cases.append(parsed_case)

        try:
            runtime_cases = self._extract_runtime_cases(sample, reference_callable)
        except Exception:
            runtime_cases = ()

        return self._dedupe_cases([*cases, *runtime_cases])

    def _task_from_sample(self, sample: HumanEvalSample, task_index: int) -> Task:
        test_cases = self._extract_test_cases(sample)
        if not test_cases:
            raise ValueError(
                f"Could not derive deterministic test cases from HumanEval task '{sample.task_id}'."
            )

        return Task(
            task_id=task_index,
            difficulty="humaneval",
            title=sample.task_id,
            description=sample.prompt,
            buggy_code=sample.prompt,
            function_name=sample.entry_point,
            test_cases=test_cases,
            optimal_patterns=(),
            anti_patterns=(),
            reference_solution=sample.canonical_solution,
            scoring_weights=(0.8, 0.1, 0.1),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng.seed(seed)

        last_error: Exception | None = None
        for _ in range(self.max_sampling_attempts):
            sample_index = self._rng.randrange(len(self._samples))
            sample = self._samples[sample_index]
            try:
                task = self._task_from_sample(sample, sample_index)
            except Exception as exc:  # noqa: BLE001 - try another sample
                last_error = exc
                continue

            self.current_sample = sample
            self.current_task = task
            return task.description, {
                "task_id": sample.task_id,
                "entry_point": sample.entry_point,
                "test_case_count": len(task.test_cases),
            }

        raise RuntimeError(
            "Failed to sample a parseable HumanEval task for deterministic grading."
        ) from last_error

    def step(self, action: str | dict[str, Any]) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self.current_task is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        code = _action_to_code(action)
        result = grade_submission(code, self.current_task)
        reward = float(result.reward)
        terminated = reward >= 0.99
        truncated = False

        performance_lines = result.performance_notes or ["No performance notes available."]
        quality_lines = result.quality_notes or ["No quality notes available."]
        performance_feedback = "\n".join(performance_lines)
        quality_feedback = "\n".join(quality_lines)
        observation = (
            "Performance Feedback:\n"
            f"{performance_feedback}\n\n"
            "Quality Feedback:\n"
            f"{quality_feedback}"
        )
        info = {
            "correctness": result.correctness,
            "test_details": result.test_details,
            "function_name": self.current_task.function_name,
        }
        return observation, reward, terminated, truncated, info


@dataclass
class RLEnvState:
    """Mutable state tracked for a single RL episode."""

    task_id: int
    attempts: int
    max_attempts: int
    best_reward: float
    current_code: str
    done: bool


class ExecuCodeRLEnv:
    """Simple RL environment exposing reset() and step() methods."""

    def __init__(
        self,
        *,
        max_attempts: int = 10,
        success_threshold: float = 0.95,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.success_threshold = float(success_threshold)
        self._next_task_id = 0
        initial_task = get_task(0)
        self._task: Task = initial_task
        self.state = RLEnvState(
            task_id=initial_task.task_id,
            attempts=0,
            max_attempts=self.max_attempts,
            best_reward=0.0,
            current_code=initial_task.buggy_code,
            done=False,
        )

    @property
    def task(self) -> Task:
        return self._task

    def reset(
        self,
        *,
        seed: int | None = None,
        task_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if task_id is None:
            if seed is not None:
                selected_task_id = int(seed) % len(ALL_TASKS)
            else:
                selected_task_id = self._next_task_id % len(ALL_TASKS)
                self._next_task_id += 1
        else:
            selected_task_id = int(task_id) % len(ALL_TASKS)

        self._task = get_task(selected_task_id)
        self.state = RLEnvState(
            task_id=self._task.task_id,
            attempts=0,
            max_attempts=self.max_attempts,
            best_reward=0.0,
            current_code=self._task.buggy_code,
            done=False,
        )

        intro_feedback = (
            f"Task {self._task.task_id}: {self._task.title}\n"
            f"Difficulty: {self._task.difficulty}\n\n"
            f"{self._task.description}\n\n"
            "Submit a complete Python function definition."
        )

        observation = self._build_observation(feedback=intro_feedback)
        info = {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "function_name": self._task.function_name,
            "attempts_remaining": self.state.max_attempts,
        }
        return observation, info

    def step(
        self,
        action: str | dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.state.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        code = _action_to_code(action)
        result = grade_submission(code, self._task)

        reward = _clamp_open_interval(result.reward)
        correctness = _clamp_open_interval(result.correctness)
        performance = _clamp_open_interval(result.performance)
        quality = _clamp_open_interval(result.quality)

        next_attempt = self.state.attempts + 1
        terminated = reward >= self.success_threshold
        truncated = next_attempt >= self.state.max_attempts and not terminated
        done = terminated or truncated

        self.state.attempts = next_attempt
        self.state.current_code = code
        self.state.best_reward = max(self.state.best_reward, reward)
        self.state.done = done

        feedback = generate_feedback(
            correctness_score=correctness,
            performance_score=performance,
            quality_score=quality,
            total_reward=reward,
            test_details=result.test_details,
            performance_notes=result.performance_notes,
            quality_notes=result.quality_notes,
            is_done=done,
            step_count=self.state.attempts,
            max_attempts=self.state.max_attempts,
        )

        info = self._build_info(
            result=result,
            correctness=correctness,
            performance=performance,
            quality=quality,
            terminated=terminated,
            truncated=truncated,
        )
        observation = self._build_observation(feedback=feedback, info=info)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        return

    def _build_observation(
        self,
        *,
        feedback: str,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        observation = {
            "task_id": self._task.task_id,
            "title": self._task.title,
            "difficulty": self._task.difficulty,
            "function_name": self._task.function_name,
            "description": self._task.description,
            "current_code": self.state.current_code,
            "attempt": self.state.attempts,
            "max_attempts": self.state.max_attempts,
            "best_reward": self.state.best_reward,
            "done": self.state.done,
            "feedback": feedback,
        }
        if info:
            observation["last_info"] = info
        return observation

    def _build_info(
        self,
        *,
        result: GradeResult,
        correctness: float,
        performance: float,
        quality: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        first_error = next(
            (
                detail.get("error")
                for detail in result.test_details
                if not detail.get("passed") and detail.get("error")
            ),
            None,
        )

        return {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "function_name": self._task.function_name,
            "attempt": self.state.attempts,
            "attempts_remaining": max(0, self.state.max_attempts - self.state.attempts),
            "best_reward": self.state.best_reward,
            "correctness": correctness,
            "performance": performance,
            "quality": quality,
            "performance_notes": list(result.performance_notes),
            "quality_notes": list(result.quality_notes),
            "test_details": result.test_details,
            "last_action_error": first_error,
            "terminated": terminated,
            "truncated": truncated,
        }


__all__ = ["ExecuCodeEnv", "ExecuCodeRLEnv", "RLEnvState", "HumanEvalSample"]
