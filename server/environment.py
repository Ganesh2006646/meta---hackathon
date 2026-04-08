"""OpenEnv-compatible ExecuCode environment implementation."""

from __future__ import annotations

from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ModuleNotFoundError:
    try:
        # Compatibility path used in some older OpenEnv examples.
        from openenv.core.env_server.environment import Environment
    except ModuleNotFoundError:
        class Environment:
            """Minimal fallback used for local tests without openenv-core."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def _apply_transform(self, observation: object) -> object:
                return observation

            def _reset_rubric(self) -> None:
                return

try:
    from ..grader import grade_submission
    from ..models import ExecuCodeAction, ExecuCodeObservation, ExecuCodeState
    from ..tasks import ALL_TASKS, Task, get_task
    from ..utils import extract_code, generate_feedback
except ImportError:
    from grader import grade_submission
    from models import ExecuCodeAction, ExecuCodeObservation, ExecuCodeState
    from tasks import ALL_TASKS, Task, get_task
    from utils import extract_code, generate_feedback


class ExecuCodeEnvironment(Environment):
    """Conversational code optimization environment."""

    def __init__(self) -> None:
        super().__init__()
        self._state = ExecuCodeState()
        self._task: Task = get_task(0)
        self._next_task_id = 0

    @property
    def state(self) -> ExecuCodeState:
        return self._state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: int | None = None,
        **_: object,
    ) -> ExecuCodeObservation:
        """Start a new episode and return the task prompt."""
        self._reset_rubric()

        if task_id is None:
            if seed is not None:
                selected_task_id = seed % len(ALL_TASKS)
            else:
                selected_task_id = self._next_task_id % len(ALL_TASKS)
                self._next_task_id += 1
        else:
            selected_task_id = task_id % len(ALL_TASKS)

        self._task = get_task(selected_task_id)
        self._state = ExecuCodeState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            current_code=self._task.buggy_code,
            best_reward=0.0,
            attempts=0,
            max_attempts=10,
        )

        prompt = (
            f"Task {self._task.task_id}: {self._task.title}\n"
            f"Difficulty: {self._task.difficulty}\n\n"
            f"{self._task.description}\n\n"
            "Submit a complete Python function definition in your next message."
        )
        observation = ExecuCodeObservation(
            echoed_message=prompt,
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._task.task_id,
                "difficulty": self._task.difficulty,
                "function_name": self._task.function_name,
                "attempts": 0,
            },
        )
        return self._apply_transform(observation)

    def step(
        self,
        action: ExecuCodeAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> ExecuCodeObservation:
        """Evaluate an agent submission and return deterministic feedback."""

        code = extract_code(action.message)
        result = grade_submission(code, self._task)

        next_attempt = self._state.attempts + 1
        next_step = self._state.step_count + 1
        done = result.reward >= 0.95 or next_attempt >= self._state.max_attempts
        best_reward = max(self._state.best_reward, result.reward)

        self._state = self._state.model_copy(
            update={
                "current_code": code,
                "best_reward": best_reward,
                "attempts": next_attempt,
                "step_count": next_step,
            }
        )

        feedback = generate_feedback(
            correctness_score=result.correctness,
            performance_score=result.performance,
            quality_score=result.quality,
            total_reward=result.reward,
            test_details=result.test_details,
            performance_notes=result.performance_notes,
            quality_notes=result.quality_notes,
            is_done=done,
            step_count=next_step,
            max_attempts=self._state.max_attempts,
        )

        observation = ExecuCodeObservation(
            echoed_message=feedback,
            done=done,
            reward=result.reward,
            metadata={
                "task_id": self._task.task_id,
                "difficulty": self._task.difficulty,
                "function_name": self._task.function_name,
                "attempts": next_attempt,
                "best_reward": best_reward,
                "correctness": result.correctness,
                "performance": result.performance,
                "quality": result.quality,
            },
        )
        return self._apply_transform(observation)
