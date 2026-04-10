---
title: ExecuCode Env
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ExecuCode

Execution-aware OpenEnv environment for evaluating coding agents on correctness, performance, and code quality with deterministic grading.

## 1. Problem Statement

Most coding benchmarks reward only final correctness. Real engineering work also needs:

- Robust behavior across edge cases
- Efficient algorithms
- Readable, maintainable code

ExecuCode addresses this by evaluating AI-generated code in an iterative environment with explicit multi-objective feedback.

## 2. Solution Description

ExecuCode runs a conversational optimization loop:

1. Agent receives a buggy coding task.
2. Agent submits a full function implementation.
3. Environment executes deterministic tests and static checks.
4. Agent receives structured feedback and reward for the next attempt.

This mirrors how real developers iterate: fix bugs first, then optimize, then polish quality.

## 3. System Architecture

### High-level flow

```text
Agent -> /reset -> Task Prompt
Agent -> /step  -> Grader(correctness + performance + quality) -> feedback + reward
Repeat until solved or max attempts
```

### Components

- `tasks.py`: immutable task definitions (buggy code, reference solution, deterministic tests, weights).
- `grader.py`: correctness execution + static performance/quality scoring.
- `utils.py`: safe code extraction, restricted execution, feedback formatting.
- `server/environment.py`: OpenEnv environment (`reset`, `step`, state tracking).
- `server/app.py`: FastAPI app, judge dashboard UI, compatibility endpoints.

## 4. Environment Design (OpenEnv)

### Action / Observation / State

- `ExecuCodeAction.message`: agent submission text (contains full Python function).
- `ExecuCodeObservation.echoed_message`: task prompt or grader feedback.
- `ExecuCodeObservation.reward`: normalized reward in strict open interval `(0, 1)`.
- `ExecuCodeState`: `task_id`, `current_code`, `best_reward`, `attempts`, `step_count`, `max_attempts`.

### `reset(...)`

- Selects task deterministically (round-robin unless task/seed provided)
- Resets attempts and best reward
- Returns full task prompt with buggy starter code

### `step(...)`

- Extracts code from agent message
- Grades submission
- Updates state
- Returns feedback, reward, done flag, and scoring metadata

Termination:

- `done=True` when reward `>= 0.95` or max attempts reached.

## 5. Task Design

| Task | Difficulty | Core skill tested | Real-world relevance |
|---|---|---|---|
| `append_to_history(item, history=[])` | Easy | Python language gotcha (mutable default argument) | Common production bug source |
| `count_paths(grid)` | Medium | Dynamic programming + memoization | Performance-critical recursion optimization |
| `chunk_document(text, max_chars)` | Hard | Robust text chunking for RAG pipelines | Practical LLM retrieval infra |

Difficulty progression:

- Easy: language correctness
- Medium: algorithmic optimization
- Hard: AI-infrastructure utility with edge-case handling and quality constraints

## 6. Grading System

Metrics:

- Correctness: deterministic unit tests over fixed inputs/outputs
- Performance: AST/regex signals (nested structure, cache usage, anti-patterns)
- Quality: docstrings, naming, complexity/readability heuristics

Deterministic behavior:

- No random scoring
- No nondeterministic model judgments
- Same input submission always produces the same score and feedback

## 7. Reward Function

All component scores are clamped to `(0.001, 0.999)` for validator compatibility.

Per-task weights:

- Task 0: `1.0 * correctness + 0.0 * performance + 0.0 * quality`
- Task 1: `0.75 * correctness + 0.25 * performance + 0.0 * quality`
- Task 2: `0.55 * correctness + 0.15 * performance + 0.30 * quality`

Final reward:

```text
reward = clamp(correctness * c_weight + performance * p_weight + quality * q_weight)
```

## 8. API / Endpoints

Judge-facing and validator endpoints:

- `GET /` -> interactive dashboard
- `GET /tasks` -> task catalog and grader wiring
- `POST /grader` -> grades one submission with score, breakdown, feedback
- `POST /baseline` -> deterministic reference baseline
- `GET /docs` -> Swagger UI

OpenEnv endpoints are exposed by `openenv` app integration in `server/app.py`:

- Standard environment routes include reset/step style operations for agents.

## 9. Demo / UI

The homepage (`/`) is a live interactive dashboard where judges can:

1. Select a task (easy/medium/hard)
2. Paste Python code
3. Click **Run Grader**
4. View multi-objective feedback instantly

Suggested judge demo:

1. Pick Task 1 and submit naive recursion without memoization.
2. Observe reduced performance score and optimization guidance.
3. Add memoization and rerun to see improved reward.

### Screenshot slots

Add these files before final submission:

- `docs/screenshots/dashboard-home.png`
- `docs/screenshots/grader-feedback.png`

Example markdown to render screenshots:

```markdown
![Dashboard Home](docs/screenshots/dashboard-home.png)
![Live Grader Feedback](docs/screenshots/grader-feedback.png)
```

## 10. Example Interaction

Sample request to `/grader`:

```json
{
  "task_id": 0,
  "code": "def append_to_history(item, history=[]):\n    history.append(item)\n    return history"
}
```

Sample response (abridged):

```json
{
  "score": 0.5,
  "breakdown": {
    "correctness": 0.5,
    "performance": 0.999,
    "quality": 0.999
  },
  "feedback": "### Evaluation (Attempt 1/1)\n..."
}
```

## 11. Local Setup

```powershell
cd "D:\meta hackathon"
python -m pip install -e .\execucode
python .\execucode\test_openenv.py
```

Run server:

```powershell
cd "D:\meta hackathon"
uvicorn execucode.server.app:app --host 0.0.0.0 --port 8000
```

## 12. Deployment

Docker build:

```powershell
docker build -f .\execucode\Dockerfile .\execucode
```

The container runs Uvicorn on port `8000`.

Hugging Face Space example:

- `https://ganeshkankatala4-execucode-env.hf.space`

## 13. Inference Loop

Run:

```powershell
python .\execucode\inference.py
```

Environment variables:

- `HF_TOKEN` (required)
- `API_BASE_URL` (optional, default OpenAI-compatible endpoint)
- `MODEL_NAME` (optional)
- `EPISODE_TIMEOUT_SECONDS` (optional)

Logging format:

- `[START]` task metadata
- `[STEP]` per-attempt reward + done flag
- `[END]` episode summary and rewards

## 14. Evaluation and Reproducibility

- Deterministic test cases and static checks
- Stable scoring across runs
- Reference solutions provide reproducible baseline scores via `/baseline`

## 15. Key Features / Highlights

- Execution-aware grading loop, not single-shot scoring
- Multi-objective reward (correctness, performance, quality)
- Realistic difficulty progression
- Built-in web dashboard for judge-friendly live demo

## 16. Limitations

- Sandbox is lightweight and not a hardened security boundary
- Performance scoring uses static signals, not full runtime profiling
- Task set is intentionally compact for hackathon constraints

## 17. Future Work

- Add larger task library and category tags
- Add leaderboard and run history UI
- Add richer static analysis and optional bounded runtime profiling

## 18. Author / Credits

- Author: `Ganeshkankatala4`
- Built for OpenEnv + FastAPI + Hugging Face Space hackathon workflow
