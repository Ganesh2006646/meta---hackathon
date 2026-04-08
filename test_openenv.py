"""Validation script for ExecuCode."""

from __future__ import annotations

if __package__ in {None, ""}:
    # Allow: `python execucode/test_openenv.py`
    import pathlib
    import sys

    package_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

from execucode.grader import grade_submission
from execucode.models import ExecuCodeAction
from execucode.server.environment import ExecuCodeEnvironment
from execucode.tasks import ALL_TASKS


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


def test_deterministic_grading() -> None:
    for task in ALL_TASKS:
        first = grade_submission(task.reference_solution, task)
        second = grade_submission(task.reference_solution, task)
        assert first.reward == second.reward
        assert first.correctness == second.correctness
        assert first.performance == second.performance
        assert first.quality == second.quality


def test_reset_cycles_tasks() -> None:
    env = ExecuCodeEnvironment()
    observed = [env.reset().metadata["task_id"] for _ in range(5)]
    assert observed == [0, 1, 2, 0, 1]


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


def main() -> None:
    test_all_tasks_with_reference_solutions()
    test_deterministic_grading()
    test_reset_cycles_tasks()
    test_state_updates()
    print("All ExecuCode validation checks passed.")


if __name__ == "__main__":
    main()
