"""FastAPI application entry point for ExecuCode."""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
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
    test_details: list[dict[str, Any]] = Field(default_factory=list)


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


@app.get("/", response_class=HTMLResponse)
def serve_homepage() -> HTMLResponse:
    """Serves an interactive HTML dashboard for hackathon judges."""

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>ExecuCode Environment</title>
        <style>
            :root {
                --bg: #edf2f7;
                --panel: #ffffff;
                --ink: #1f2937;
                --muted: #4b5563;
                --primary: #1769e0;
                --primary-hover: #0f4ca8;
                --ok: #1f7a3f;
                --warn: #996500;
                --err: #b3261e;
                --border: #d1d5db;
                --code-bg: #1f2937;
                --shadow: 0 12px 30px rgba(16, 24, 40, 0.12);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                color: var(--ink);
                background:
                    radial-gradient(1200px 600px at -10% -20%, #dbeafe 0%, transparent 60%),
                    radial-gradient(900px 500px at 110% -10%, #d1fae5 0%, transparent 55%),
                    var(--bg);
                min-height: 100vh;
                padding: 32px 16px;
                display: flex;
                justify-content: center;
            }

            .container {
                width: 100%;
                max-width: 900px;
                background: var(--panel);
                border-radius: 14px;
                box-shadow: var(--shadow);
                padding: 28px;
            }

            h1 {
                margin: 0 0 8px;
                color: var(--primary);
            }

            p {
                margin: 10px 0;
                line-height: 1.55;
                color: var(--muted);
            }

            .help-links {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 12px;
            }

            .help-links a {
                color: var(--primary);
                text-decoration: none;
                font-weight: 600;
            }

            .help-links a:hover {
                text-decoration: underline;
            }

            label {
                display: block;
                margin-top: 16px;
                margin-bottom: 6px;
                font-weight: 700;
            }

            select,
            textarea {
                width: 100%;
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
            }

            select {
                background: #fff;
            }

            textarea {
                min-height: 220px;
                resize: vertical;
                font-family: "Courier New", Courier, monospace;
            }

            button {
                margin-top: 14px;
                width: 100%;
                border: none;
                border-radius: 10px;
                padding: 12px 16px;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                color: #fff;
                background: var(--primary);
                transition: background 0.15s ease;
            }

            button:hover:not(:disabled) {
                background: var(--primary-hover);
            }

            button:disabled {
                opacity: 0.7;
                cursor: not-allowed;
            }

            pre {
                margin-top: 10px;
                background: var(--code-bg);
                color: #b9f3c9;
                border-radius: 10px;
                padding: 14px;
                white-space: pre-wrap;
                overflow-x: auto;
                min-height: 120px;
            }

            @media (max-width: 640px) {
                body {
                    padding: 18px 10px;
                }
                .container {
                    padding: 16px;
                }
            }
        </style>
    </head>
    <body>
        <main class="container">
            <h1>ExecuCode Sandbox</h1>
            <p>
                ExecuCode is an execution-aware coding environment that grades submissions
                on <strong>Correctness</strong>, <strong>Performance</strong>, and
                <strong>Code Quality</strong> instead of pass/fail only.
            </p>
            <p>Paste a Python function, run the grader, and review feedback instantly.</p>
            <div class="help-links">
                <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
                <a href="/tasks" target="_blank" rel="noreferrer">List Tasks</a>
            </div>

            <label for="task">Select a coding task</label>
            <select id="task">
                <option value="0">Task 0 (Easy): append_to_history (mutable default)</option>
                <option value="1">Task 1 (Medium): count_paths (memoized DP)</option>
                <option value="2">Task 2 (Hard): chunk_document (RAG chunker)</option>
            </select>

            <label for="codeBox">Python submission</label>
            <textarea id="codeBox" placeholder="# Paste your Python code here...
def my_function():
    pass"></textarea>
            <button id="submitBtn" onclick="submitCode()">Run Grader</button>

            <label for="output">Grading feedback</label>
            <pre id="output">Waiting for submission...</pre>
        </main>

        <script>
            async function submitCode() {
                const output = document.getElementById("output");
                const btn = document.getElementById("submitBtn");
                const taskId = parseInt(document.getElementById("task").value, 10);
                const code = document.getElementById("codeBox").value;

                output.style.color = "var(--warn)";
                output.innerText = "Executing in restricted sandbox... Please wait.";
                btn.disabled = true;
                btn.innerText = "Grading...";

                try {
                    const response = await fetch("/grader", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ task_id: taskId, code: code }),
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        output.style.color = "var(--err)";
                        output.innerText = JSON.stringify(data, null, 2);
                        return;
                    }

                    output.style.color = "var(--ok)";
                    output.innerText = data.feedback ? data.feedback : JSON.stringify(data, null, 2);
                } catch (error) {
                    output.style.color = "var(--err)";
                    output.innerText = "Error connecting to server: " + error;
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Run Grader";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
            "grader": "POST /grader",
            "grader_method": "POST",
            "grader_endpoint": "/grader",
            "score_range": {"exclusive_min": 0.0, "exclusive_max": 1.0},
            "score_min_exclusive": 0.0,
            "score_max_exclusive": 1.0,
        }
        for task in ALL_TASKS
    ]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple liveness endpoint for validators and deployments."""

    return {"status": "ok", "env": "execucode"}


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
        test_details=result.test_details,
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
        "execucode.server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
