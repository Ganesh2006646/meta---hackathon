---
title: ExecuCode Env
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ExecuCode

ExecuCode is an execution-aware OpenEnv environment for evaluating coding agents on more than final correctness. Each submission is graded on deterministic tests, performance signals, and code quality so agents can improve over multiple attempts instead of optimizing for a single pass/fail outcome.

## What It Does

ExecuCode runs an iterative optimization loop:

1. An agent receives a buggy Python function and task description.
2. The agent submits a full replacement implementation.
3. The environment executes deterministic checks and static heuristics.
4. The agent receives structured feedback, component scores, and a shaped reward.

The reward is always clamped to the strict open interval `(0.001, 0.999)` for validator compatibility.

## Task Suite

| ID | Task | Difficulty | Focus |
|---|---|---|---|
| `0` | Mutable default argument | Easy | Python correctness |
| `1` | Grid path counting | Medium | Dynamic programming and memoization |
| `2` | RAG document chunker | Hard | Text processing, edge cases, readability |
| `3` | Sliding window rate limiter | Hard | API infra logic, edge cases, code quality |

### Reward weights

| Task ID | Correctness | Performance | Quality |
|---|---:|---:|---:|
| `0` | `1.00` | `0.00` | `0.00` |
| `1` | `0.75` | `0.25` | `0.00` |
| `2` | `0.55` | `0.15` | `0.30` |
| `3` | `0.60` | `0.10` | `0.30` |

## Project Layout

- `tasks.py`: immutable task definitions, test cases, reference solutions, scoring weights
- `grader.py`: deterministic grading pipeline and score aggregation
- `utils.py`: code extraction, restricted execution, feedback formatting
- `server/environment.py`: `ExecuCodeEnvironment` implementation used by OpenEnv
- `environment.py`: package-root shim for OpenEnv validator discovery
- `server/app.py`: FastAPI app, dashboard UI, and task/grader endpoints
- `test_openenv.py`: scriptable validation checks for environment behavior
- `test_hackathon_endpoints.py`: endpoint regression coverage
- `openenv.yaml`: environment metadata and endpoint contract

## API Surface

Core endpoints:

- `GET /`: interactive dashboard for trying tasks manually
- `GET /tasks`: task catalog, metadata, and grader wiring
- `GET /health`: liveness endpoint, currently returns `{"status": "healthy"}`
- `POST /grader`: grades one submission and returns score, breakdown, feedback, and test details
- `POST /baseline`: grades the bundled reference solutions deterministically

OpenEnv routes are also exposed through the app integration:

- `POST /reset`
- `POST /step`
- `GET /state`

## Local Development

From the repository root:

```powershell
uv sync --extra test
```

Run the built-in validation script:

```powershell
uv run python test_openenv.py
```

Run the endpoint regression suite:

```powershell
uv run pytest -q
```

Start the local server:

```powershell
uv run uvicorn execucode.server.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## Example Grader Request

```json
{
  "task_id": 0,
  "code": "def append_to_history(item, history=None):\n    if history is None:\n        history = []\n    history.append(item)\n    return history"
}
```

Typical response shape:

```json
{
  "task_id": 0,
  "score": 0.999,
  "breakdown": {
    "correctness": 0.999,
    "performance": 0.999,
    "quality": 0.999
  },
  "feedback": "### Evaluation (Attempt 1/10)\n...",
  "test_details": []
}
```

## Docker

The included Dockerfile uses `python:3.12-slim`, installs `uv`, and installs the project with:

```powershell
uv pip install --system --no-cache .
```

Build and run locally:

```powershell
docker build -t execucode .
docker run --rm -p 8000:8000 execucode
```

## Hugging Face Space

This repo is configured for Docker-based Space deployment. The current Space metadata lives in the frontmatter above, and `openenv.yaml` points the runtime at:

```text
execucode.server.app:app
```

## Inference Loop

The repo also includes a client/inference path for running agent episodes against the environment:

```powershell
python inference.py
```

Useful environment variables:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `EPISODE_TIMEOUT_SECONDS`

## Validation Notes

- Grading is deterministic for the same submission and task.
- The environment cycles through all configured tasks on repeated `reset()` calls.
- Runtime metadata includes score breakdowns plus execution profile fields such as elapsed time and captured errors when available.
- The package-root `environment.py` shim is intentionally present so OpenEnv validation can discover `ExecuCodeEnvironment` correctly.

## Stack

- FastAPI
- OpenEnv
- Pydantic
- Uvicorn
- Python 3.12
