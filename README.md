# ExecuCode

ExecuCode is an OpenEnv-compatible environment for conversational,
execution-aware code optimization. An agent receives a coding task, submits a
natural-language answer containing Python code, and gets deterministic feedback
on correctness, performance, and code quality.

## Environment Design

The environment uses standard OpenEnv action, observation, and state models:

- `ExecuCodeAction.message`: the agent's analysis and proposed Python function.
- `ExecuCodeObservation.echoed_message`: task prompt or grading feedback.
- `ExecuCodeObservation.reward`: final weighted score in `[0.0, 1.0]`.
- `ExecuCodeState`: task id, latest code, best reward, attempts, and step count.

On each step, ExecuCode extracts the final Python code block or raw function
definition, executes it against deterministic tests in a restricted namespace,
scores static performance signals, scores readability, and returns feedback the
agent can use for the next attempt.

## Tasks

ExecuCode ships with three tasks:

- Easy: `sum_positive(numbers)` must sum only positive numbers. Reward is based
  on correctness only.
- Medium: `find_duplicates(numbers)` must preserve behavior while replacing an
  O(n^2) duplicate search. Reward is correctness plus performance.
- Hard: `word_frequency(text)` must fix case and punctuation bugs, replace
  bubble sort, and improve readability. Reward is correctness, performance, and
  quality.

## Reward Function

The default scoring dimensions are:

- Correctness: fraction of deterministic tests passed.
- Performance: AST and regex checks for nested loops, lookup structures, and
  built-in sorting.
- Code quality: docstring presence, concise implementation, and descriptive
  variable names.

Task weights are:

- Easy: `1.0 correctness + 0.0 performance + 0.0 quality`.
- Medium: `0.7 correctness + 0.3 performance + 0.0 quality`.
- Hard: `0.5 correctness + 0.3 performance + 0.2 quality`.

## Local Setup

```powershell
cd "D:\meta hackathon"
python -m pip install -e .\execucode
python .\execucode\test_openenv.py
```

To run the FastAPI server:

```powershell
cd "D:\meta hackathon"
uvicorn execucode.server.app:app --host 0.0.0.0 --port 8000
```

To run the baseline inference loop:

```powershell
python .\execucode\inference.py
```

If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, the script uses an
OpenAI-compatible model endpoint. Otherwise it falls back to built-in reference
solutions so the environment loop can still be validated.

## Deployment

The included `openenv.yaml` points at `execucode.server.app:app`, and
`server/Dockerfile` uses the OpenEnv base image:

```powershell
docker build -f .\execucode\server\Dockerfile .\execucode
```

The container exposes port `8000` and starts Uvicorn with the ExecuCode FastAPI
app.

## Notes

The execution sandbox is intentionally lightweight for hackathon use. It uses a
restricted namespace and per-test timeouts, but it should not be treated as a
security boundary for untrusted production workloads.
