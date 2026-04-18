# Copyright (c) 2026. ExecuCode Environment.

"""
Environment implementation for ExecuCode.

Core loop:
1. reset() -> provides problem description and buggy code.
2. step(action) -> agent provides fixed code, environment grades it and returns feedback.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import Observation

from ..models import ExecuCodeAction, ExecuCodeObservation, ExecuCodeState
from ..tasks import get_task
from ..grader import grade_submission
from ..utils import extract_code, generate_feedback


class ExecuCodeEnvironment(Environment):
    """ExecuCode: Conversational Code Optimization Environment."""

    def __init__(self):
        super().__init__()
        self._state = ExecuCodeState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecuCodeObservation:
        """Reset the environment to a new task."""
        
        # Cycle through tasks based on episode_id hash if task_id not provided
        ep_id = episode_id or str(uuid4())
        if task_id is None:
            # Deterministic task selection from episode_id
            task_id = int(ep_id.replace("-", ""), 16) % 3
            
        task = get_task(task_id)
        
        self._state = ExecuCodeState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            current_code=task.buggy_code,
            best_reward=0.0,
            attempts=0,
            max_attempts=10
        )

        initial_message = (
            f"🚀 **Task: {task.title}** ({task.difficulty.upper()})\n\n"
            f"{task.description}\n\n"
            "Please analyze the code above and provide a fix or optimization."
        )

        return ExecuCodeObservation(
            echoed_message=initial_message,
            reward=0.0,
            done=False,
            metadata={"status": "ready", "task_id": task_id}
        )

    def step(
        self,
        action: ExecuCodeAction,
        **kwargs: Any,
    ) -> ExecuCodeObservation:
        """Execute a step: Grade the agent's submission and provide feedback."""
        
        self._state.step_count += 1
        self._state.attempts += 1
        
        # 1. Extract code from agent message
        code = extract_code(action.message)
        self._state.current_code = code
        
        # 2. Load task
        task = get_task(self._state.task_id)
        
        # 3. Grade submission
        grading_result = grade_submission(code, task)
        
        reward = grading_result["total_reward"]
        
        # Track best reward
        if reward > self._state.best_reward:
            self._state.best_reward = reward
            
        # 4. Generate feedback
        feedback = generate_feedback(
            correctness_score=grading_result["correctness_score"],
            performance_score=grading_result["performance_score"],
            quality_score=grading_result["quality_score"],
            total_reward=reward,
            test_details=grading_result["test_details"],
            performance_notes=grading_result["performance_notes"],
            quality_notes=grading_result["quality_notes"],
            is_done=False, # Temp till we check done condition
            step_count=self._state.step_count,
            max_attempts=self._state.max_attempts
        )
        
        # 5. Check done condition
        # - High reward reached (>= 0.95)
        # - Max attempts reached
        # - All correctness tests pass + high performance (> 0.8)
        is_success = (reward >= 0.95) or (grading_result["correctness_score"] == 1.0 and grading_result["performance_score"] > 0.8)
        is_done = is_success or (self._state.step_count >= self._state.max_attempts)
        
        # Update feedback if done
        if is_done:
            feedback = generate_feedback(
                correctness_score=grading_result["correctness_score"],
                performance_score=grading_result["performance_score"],
                quality_score=grading_result["quality_score"],
                total_reward=reward,
                test_details=grading_result["test_details"],
                performance_notes=grading_result["performance_notes"],
                quality_notes=grading_result["quality_notes"],
                is_done=True,
                step_count=self._state.step_count,
                max_attempts=self._state.max_attempts
            )
            
        return ExecuCodeObservation(
            echoed_message=feedback,
            reward=reward,
            done=is_done,
            metadata={
                "task_id": self._state.task_id,
                "best_reward": self._state.best_reward,
                "success": is_success
            }
        )

    @property
    def state(self) -> ExecuCodeState:
        """Return current environment state."""
        return self._state
