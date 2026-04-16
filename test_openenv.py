"""Validation script for ExecuCode."""

from __future__ import annotations

from copy import deepcopy
import random

if __package__ in {None, ""}:
    # Allow: `python execucode/test_openenv.py`
    import pathlib
    import sys

    package_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

from execucode.grader import grade_submission
from execucode.models import ExecuCodeAction
from execucode.rl_env import ExecuCodeEnv, ExecuCodeRLEnv
from execucode.server.environment import ExecuCodeEnvironment
from execucode.tasks import ALL_TASKS
from execucode.utils import extract_code


def _assert_reward_range(reward: float) -> None:
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"


def test_all_tasks_with_reference_solutions() -> None:
    env = ExecuCodeEnvironment()
    for task in ALL_TASKS:
        observation = env.reset(task_id=task.task_id)
        assert observation.done is False
        assert task.function_name in observation.echoed_message

        result = env.step(ExecuCodeAction(message=task.reference_solution))
        _assert_reward_range(result.reward)
        assert result.reward >= 0.95, (
            f"Reference solution for task {task.task_id} scored {result.reward}"
        )
        assert result.done is True


def test_package_import_smoke() -> None:
    import execucode

    assert hasattr(execucode, "ExecuCodeEnv")
    assert hasattr(execucode, "ExecuCodeAction")


def test_deterministic_grading() -> None:
    for task in ALL_TASKS:
        first = grade_submission(task.reference_solution, task)
        second = grade_submission(task.reference_solution, task)
        assert first.reward == second.reward
        assert first.correctness == second.correctness
        assert first.performance == second.performance
        assert first.quality == second.quality


def test_grading_does_not_mutate_task_test_inputs() -> None:
    for task in ALL_TASKS:
        before_inputs = deepcopy([case.input_args for case in task.test_cases])
        grade_submission(task.reference_solution, task)
        after_inputs = [case.input_args for case in task.test_cases]
        assert after_inputs == before_inputs


def test_reset_cycles_tasks() -> None:
    env = ExecuCodeEnvironment()
    observed = [env.reset().metadata["task_id"] for _ in range(5)]
    expected = list(range(len(ALL_TASKS))) + [0]
    assert observed == expected[:5]


def test_state_updates() -> None:
    env = ExecuCodeEnvironment()
    env.reset(task_id=0, episode_id="episode-test")
    assert env.state.episode_id == "episode-test"
    assert env.state.step_count == 0

    observation = env.step(ExecuCodeAction(message="def sum_positive(numbers):\n    return 0"))
    _assert_reward_range(observation.reward)
    assert env.state.step_count == 1
    assert env.state.attempts == 1
    assert env.state.best_reward == observation.reward
    assert "last_action_error" in observation.metadata


def test_step_metadata_includes_execution_profile() -> None:
    env = ExecuCodeEnvironment()
    task = ALL_TASKS[0]
    env.reset(task_id=task.task_id)
    observation = env.step(ExecuCodeAction(message=task.reference_solution))
    assert observation.metadata.get("avg_elapsed_ms") is not None
    assert observation.metadata.get("max_elapsed_ms") is not None


def test_extract_code_handles_common_markdown_variants() -> None:
    task = ALL_TASKS[0]
    fenced = f"```Python\n{task.reference_solution}\n```"
    assert extract_code(fenced).strip() == task.reference_solution.strip()

    missing_closing_fence = f"```py\n{task.reference_solution}\n"
    assert extract_code(missing_closing_fence).strip() == task.reference_solution.strip()


def test_rl_wrapper_reset_and_step() -> None:
    env = ExecuCodeRLEnv(max_attempts=3)
    observation, info = env.reset(task_id=0)

    assert observation["task_id"] == 0
    assert observation["attempt"] == 0
    assert info["function_name"] == "append_to_history"

    observation, reward, terminated, truncated, details = env.step(
        ALL_TASKS[0].reference_solution
    )
    _assert_reward_range(reward)
    assert terminated is True
    assert truncated is False
    assert details["correctness"] >= 0.95
    assert observation["done"] is True


def test_rl_wrapper_truncates_when_attempt_limit_reached() -> None:
    env = ExecuCodeRLEnv(max_attempts=1)
    env.reset(task_id=0)

    _, _, terminated, truncated, _ = env.step(
        "def append_to_history(item, history=None):\n    return []"
    )
    assert terminated is False
    assert truncated is True


def test_humaneval_env_parses_multi_arg_and_sanitizes_fenced_code() -> None:
    sample = {
        "task_id": "HumanEval/mini",
        "prompt": "Implement add(a, b).",
        "entry_point": "add",
        "canonical_solution": "def add(a, b):\n    return a + b",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(5, 7) == 12\n"
        ),
    }
    env = ExecuCodeEnv(samples=[sample], seed=1)
    observation, info = env.reset()
    assert isinstance(observation, str)
    assert info["entry_point"] == "add"

    fenced_code = "```python\ndef add(a, b):\n    return a + b\n```"
    _, reward, terminated, truncated, step_info = env.step(fenced_code)
    _assert_reward_range(reward)
    assert step_info["correctness"] >= 0.95
    assert terminated is False
    assert truncated is False


def test_humaneval_env_reset_does_not_touch_global_random_state() -> None:
    sample = {
        "task_id": "HumanEval/random",
        "prompt": "Return x.",
        "entry_point": "identity",
        "canonical_solution": "def identity(x):\n    return x",
        "test": "def check(candidate):\n    assert candidate(7) == 7\n",
    }

    random.seed(2026)
    _ = random.random()
    expected_next = random.random()

    random.seed(2026)
    _ = random.random()
    env = ExecuCodeEnv(samples=[sample], seed=5)
    env.reset(seed=17)
    actual_next = random.random()

    assert actual_next == expected_next


def test_humaneval_env_non_literal_asserts_fallback_to_reference() -> None:
    sample = {
        "task_id": "HumanEval/tolerance",
        "prompt": "import math\n\ndef approx_sin(x: float) -> float:\n",
        "entry_point": "approx_sin",
        "canonical_solution": "    return math.sin(x)\n",
        "test": (
            "import math\n"
            "def check(candidate):\n"
            "    assert abs(candidate(0.0) - 0.0) < 1e-9\n"
            "    assert abs(candidate(0.5) - math.sin(0.5)) < 1e-9\n"
        ),
    }

    env = ExecuCodeEnv(samples=[sample], seed=3)
    _, info = env.reset()
    assert info["test_case_count"] >= 2

    _, reward, terminated, truncated, step_info = env.step(
        "def approx_sin(x: float) -> float:\n    import math\n    return math.sin(x)"
    )
    _assert_reward_range(reward)
    assert step_info["correctness"] >= 0.95
    assert terminated is False
    assert truncated is False


def main() -> None:
    test_package_import_smoke()
    test_all_tasks_with_reference_solutions()
    test_deterministic_grading()
    test_grading_does_not_mutate_task_test_inputs()
    test_reset_cycles_tasks()
    test_state_updates()
    test_step_metadata_includes_execution_profile()
    test_extract_code_handles_common_markdown_variants()
    print("All ExecuCode validation checks passed.")


if __name__ == "__main__":
    main()
