"""FastAPI application entry point for ExecuCode."""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

try:
    from ..grader import grade_submission
    from ..models import ExecuCodeAction, ExecuCodeObservation
    from ..tasks import ALL_TASKS, get_task
    from ..utils import extract_code, generate_feedback
    from .environment import ExecuCodeEnvironment
except ImportError:
    from grader import grade_submission
    from models import ExecuCodeAction, ExecuCodeObservation
    from tasks import ALL_TASKS, get_task
    from utils import extract_code, generate_feedback
    from server.environment import ExecuCodeEnvironment

try:
    from openenv.core.env_server.http_server import create_app
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "openenv-core is required to serve ExecuCode. Install dependencies with "
        "`pip install -e .` before running the FastAPI app."
    ) from exc


app = create_app(
    ExecuCodeEnvironment,
    ExecuCodeAction,
    ExecuCodeObservation,
    env_name="execucode",
)

_SCORE_EPSILON = 1e-3


def _clamp_open_interval(score: float) -> float:
    """Clamp any score to the strict open interval (0, 1)."""

    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, float(score)))


def _resolve_task_id(task_id: int | str) -> int:
    """Resolve task identifiers from numeric ids or known task names."""

    if isinstance(task_id, int):
        return task_id % len(ALL_TASKS)

    normalized = str(task_id).strip().lower()
    if normalized.isdigit():
        return int(normalized) % len(ALL_TASKS)

    if normalized.startswith("task_") and normalized[5:].isdigit():
        return int(normalized[5:]) % len(ALL_TASKS)

    for task in ALL_TASKS:
        aliases = {
            str(task.task_id),
            f"task_{task.task_id}",
            task.function_name.lower(),
            task.title.lower(),
            task.title.lower().replace(" ", "_"),
        }
        if normalized in aliases:
            return task.task_id

    raise HTTPException(
        status_code=422,
        detail=f"Unknown task_id '{task_id}'. Supported ids: {[task.task_id for task in ALL_TASKS]}",
    )


class GraderRequest(BaseModel):
    """Flexible grader request model for validator compatibility."""

    task_id: int | str = Field(default=0)
    answer: str | None = None
    submission: str | None = None
    code: str | None = None
    response: str | None = None
    message: str | None = None


class GraderResponse(BaseModel):
    """Score response for a single task submission."""

    task_id: int
    score: float
    breakdown: dict[str, float]
    feedback: str


class BaselineRequest(BaseModel):
    """Optional subset of tasks for baseline scoring."""

    task_ids: list[int | str] | None = None


class BaselineResponse(BaseModel):
    """Baseline score payload across selected tasks."""

    scores: dict[str, float]
    average: float
    model: str


def _extract_submission(payload: GraderRequest) -> str:
    """Return the first non-empty submission field."""

    for candidate in (
        payload.answer,
        payload.submission,
        payload.code,
        payload.response,
        payload.message,
    ):
        if candidate and candidate.strip():
            return candidate

    raise HTTPException(
        status_code=422,
        detail="Missing submission text. Provide one of: answer, submission, code, response, or message.",
    )


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """Expose all tasks with explicit grader wiring for hackathon validators."""

    return [
        {
            "id": task.task_id,
            "task_id": task.task_id,
            "name": task.title,
            "title": task.title,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": 10,
            "function_name": task.function_name,
            "has_grader": True,
            "grader": "/grader",
            "score_range": {"exclusive_min": 0.0, "exclusive_max": 1.0},
        }
        for task in ALL_TASKS
    ]


@app.post("/grader", response_model=GraderResponse)
def grade_task_submission(request: GraderRequest) -> GraderResponse:
    """Grade a task submission and return strict-open-interval scores."""

    task_id = _resolve_task_id(request.task_id)
    task = get_task(task_id)
    raw_submission = _extract_submission(request)
    code = extract_code(raw_submission)
    result = grade_submission(code, task)

    feedback = generate_feedback(
        correctness_score=result.correctness,
        performance_score=result.performance,
        quality_score=result.quality,
        total_reward=result.reward,
        test_details=result.test_details,
        performance_notes=result.performance_notes,
        quality_notes=result.quality_notes,
        is_done=result.reward >= 0.95,
        step_count=1,
        max_attempts=1,
    )

    return GraderResponse(
        task_id=task.task_id,
        score=_clamp_open_interval(result.reward),
        breakdown={
            "correctness": _clamp_open_interval(result.correctness),
            "performance": _clamp_open_interval(result.performance),
            "quality": _clamp_open_interval(result.quality),
        },
        feedback=feedback,
    )


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(request: BaselineRequest | None = None) -> BaselineResponse:
    """Return deterministic baseline scores using reference solutions."""

    task_ids = request.task_ids if request and request.task_ids else list(range(len(ALL_TASKS)))
    resolved_task_ids = [_resolve_task_id(task_id) for task_id in task_ids]

    scores: dict[str, float] = {}
    for task_id in resolved_task_ids:
        task = get_task(task_id)
        result = grade_submission(task.reference_solution, task)
        scores[str(task.task_id)] = _clamp_open_interval(result.reward)

    average = sum(scores.values()) / len(scores) if scores else _clamp_open_interval(0.0)
    return BaselineResponse(
        scores=scores,
        average=_clamp_open_interval(average),
        model="reference_solution",
    )


def main() -> None:
    """CLI entry point for running the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
