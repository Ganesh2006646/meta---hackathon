"""Type-safe EnvClient for ExecuCode."""

from __future__ import annotations

from typing import Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ModuleNotFoundError:
    class EnvClient:
        """Fallback placeholder for local imports without openenv-core."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("openenv-core is required to use ExecuCodeEnv.")

        @classmethod
        def __class_getitem__(cls, _item: object) -> type["EnvClient"]:
            return cls

    class StepResult:
        """Fallback type used only for type checking in no-openenv mode."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("openenv-core is required to use ExecuCodeEnv.")

from .models import ExecuCodeAction, ExecuCodeObservation, ExecuCodeState


try:
    _EnvClientBase = EnvClient[ExecuCodeAction, ExecuCodeObservation, ExecuCodeState]
except TypeError:
    # Some openenv-core versions expose a non-parameterized runtime EnvClient.
    _EnvClientBase = EnvClient


class ExecuCodeEnv(_EnvClientBase):
    """Client wrapper with typed action, observation, and state models."""

    def _step_payload(self, action: ExecuCodeAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self,
        payload: dict[str, Any],
    ) -> StepResult[ExecuCodeObservation]:
        observation_payload = payload.get("observation", {})
        observation = ExecuCodeObservation(
            echoed_message=observation_payload.get("echoed_message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=observation_payload.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ExecuCodeState:
        return ExecuCodeState.model_validate(payload)
