---
title: ExecuCode Env
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ExecuCode: Multi-Objective Code Grading Environment

ExecuCode is an OpenEnv-compatible coding environment that grades Python submissions on three axes:

- Correctness (deterministic test execution)
- Performance (AST and complexity-oriented signals)
- Code quality (static readability and style indicators)

It includes a FastAPI server, interactive dashboard UI, deterministic grader, RL-style wrappers, and Gemini-based autonomous loops.

## Current Status

- Full automated tests are passing.
- Main API endpoints (`/health`, `/tasks`, `/grader`, `/baseline`) are operational.
- Dashboard UI at `/` is operational.
- Single-agent and multi-agent loops are runnable, with optional reference fallback mode.

## Architecture Overview

- `tasks.py`: immutable task catalog, test cases, reference solutions, scoring weights
- `grader.py`: deterministic grading pipeline and score composition
- `utils.py`: fenced-code extraction, sandboxed execution helpers, feedback formatting
- `server/environment.py`: OpenEnv environment implementation
- `server/app.py`: FastAPI app, dashboard UI, API endpoints
- `rl_env.py`:
  - `ExecuCodeEnv`: HumanEval-backed gym-style environment
  - `ExecuCodeRLEnv`: deterministic task-bank RL wrapper
- `agent_loop.py`: single-agent autonomous loop (Gemini SDK)
- `multi_agent_loop.py`: junior/senior chain-of-reflection loop (Gemini SDK)
- `trajectory_logger.py`: JSONL trajectory summary and ASCII reward chart
- `openenv.yaml`: OpenEnv metadata and endpoint contract

## Task Suite

| ID | Task | Difficulty |
|---|---|---|
| `0` | Python Gotcha: Mutable Default Argument | easy |
| `1` | Dynamic Programming: Grid Path Counting | medium |
| `2` | AI Infra: RAG Document Chunker | hard |
| `3` | API Infrastructure: Sliding Window Rate Limiter | hard |

### Score Weights

Each task uses a weighted blend: `(correctness, performance, quality)`.

| Task ID | Correctness | Performance | Quality |
|---|---:|---:|---:|
| `0` | `1.00` | `0.00` | `0.00` |
| `1` | `0.75` | `0.25` | `0.00` |
| `2` | `0.55` | `0.15` | `0.30` |
| `3` | `0.60` | `0.10` | `0.30` |

All exposed score fields are clamped to the strict open interval `(0, 1)`.

## Quickstart

### 1. Install dependencies

From the `execucode` folder:

```powershell
python -m pip install -e ".[test]"
```

### 2. Run tests

```powershell
python -m pytest -q
```

### 3. Run the API server

```powershell
python -m execucode.server.app
```

Or with Uvicorn:

```powershell
uvicorn execucode.server.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## API Endpoints

- `GET /` : interactive dashboard
- `GET /health` : liveness
- `GET /tasks` : task metadata for validators/clients
- `POST /grader` : grade a submission
- `POST /baseline` : deterministic reference baseline scores

### `/grader` accepted submission keys

Request may provide any one of:

- `answer`
- `submission`
- `code`
- `response`
- `message`

## RL Environments

### HumanEval-backed environment

```python
from execucode.rl_env import ExecuCodeEnv

env = ExecuCodeEnv()
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step("def candidate(...): ...")
```

### Deterministic task-bank wrapper

```python
from execucode.rl_env import ExecuCodeRLEnv

env = ExecuCodeRLEnv(max_attempts=10)
observation, info = env.reset(task_id=1)
observation, reward, terminated, truncated, info = env.step("def count_paths(grid): ...")
```

## Autonomous Loops

### Single-agent loop

```powershell
python agent_loop.py
```

### Multi-agent loop

```powershell
python multi_agent_loop.py
```

### Agent environment variables

- `GEMINI_API_KEY` : Gemini API key
- `GEMINI_MODEL` : primary model name (default: `gemini-2.0-flash`)
- `GEMINI_MODELS` : comma-separated model fallback list
- `GEMINI_MAX_RETRIES` : retry count per model
- `GEMINI_RETRY_BACKOFF_SECONDS` : retry backoff base
- `AGENT_ENABLE_REFERENCE_FALLBACK` : `1/0` fallback toggle
- `HUMANEVAL_DATASET` : dataset name (default: `openai/openai_humaneval`)
- `HUMANEVAL_SPLIT` : dataset split (default: `test`)

Notes:

- If Gemini is unavailable or quota-limited, loops can fall back to reference-solution mode when enabled.
- `OPENAI_API_KEY` is optional and only used for the mentor hint path in `/grader`.

## Trajectory Utilities

```powershell
python trajectory_logger.py --input .\trajectories\latest_run.jsonl
```

This prints a summary and ASCII reward chart.

## OpenEnv Integration

OpenEnv metadata lives in `openenv.yaml` and declares:

- runtime app: `execucode.server.app:app`
- environment endpoints (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`, `/health`)
- reward range `[0.001, 0.999]`

## Docker

```powershell
docker build -t execucode .
docker run --rm -p 8000:8000 execucode
```
