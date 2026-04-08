"""Data models for the ExecuCode environment."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ModuleNotFoundError:
    # Local development fallback. Production uses OpenEnv's Pydantic models
    # when openenv-core is installed.
    class Action(BaseModel):
        """Fallback action base type."""

    class Observation(BaseModel):
        """Fallback observation base type."""

        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        """Fallback state base type."""

        episode_id: str = Field(default_factory=lambda: str(uuid4()))
        step_count: int = 0


class ExecuCodeAction(Action):
    """Action sent by the agent to the environment."""

    message: str = Field(
        ...,
        description="Agent's natural language analysis and proposed code fix.",
    )


class ExecuCodeObservation(Observation):
    """Observation returned by the environment after reset or step."""

    echoed_message: str = Field(
        ...,
        description="Task prompt or grading feedback from the environment.",
    )


class ExecuCodeState(State):
    """Internal state for the current ExecuCode episode."""

    task_id: int = Field(default=0, description="Current task index.")
    current_code: str = Field(default="", description="Latest submitted code.")
    best_reward: float = Field(default=0.0, description="Best episode reward.")
    attempts: int = Field(default=0, description="Number of submissions made.")
    max_attempts: int = Field(default=10, description="Maximum submissions.")
