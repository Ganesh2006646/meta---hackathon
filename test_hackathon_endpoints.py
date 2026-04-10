"""Regression tests for hackathon task/grader endpoint validation."""

from __future__ import annotations

if __package__ in {None, ""}:
    # Allow: `python execucode/test_hackathon_endpoints.py`
    import pathlib
    import sys

    package_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

from fastapi.testclient import TestClient

from execucode.server.app import app
from execucode.tasks import ALL_TASKS


client = TestClient(app)


def test_tasks_endpoint_exposes_three_or_more_graded_tasks() -> None:
    response = client.get("/tasks")
    assert response.status_code == 200

    tasks = response.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 3

    tasks_with_graders = [task for task in tasks if task.get("grader")]
    assert len(tasks_with_graders) >= 3


def test_health_endpoint_is_available() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") == "ok"


def test_grader_scores_are_strictly_between_zero_and_one() -> None:
    for task in ALL_TASKS:
        response = client.post(
            "/grader",
            json={"task_id": task.task_id, "answer": task.reference_solution},
        )
        assert response.status_code == 200
        payload = response.json()
        assert 0.0 < payload["score"] < 1.0

        for component_score in payload["breakdown"].values():
            assert 0.0 < component_score < 1.0


def test_grader_accepts_markdown_fenced_submission() -> None:
    task = ALL_TASKS[0]
    fenced_submission = f"```Python\n{task.reference_solution}\n```"
    response = client.post(
        "/grader",
        json={"task_id": task.task_id, "code": fenced_submission},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] >= 0.95


def test_pattern_fields_are_tuples() -> None:
    for task in ALL_TASKS:
        assert isinstance(task.optimal_patterns, tuple)
        assert isinstance(task.anti_patterns, tuple)


def test_baseline_scores_are_strictly_between_zero_and_one() -> None:
    response = client.post("/baseline", json={})
    assert response.status_code == 200

    payload = response.json()
    scores = payload["scores"]
    assert len(scores) >= 3
    assert 0.0 < payload["average"] < 1.0
    assert all(0.0 < score < 1.0 for score in scores.values())
