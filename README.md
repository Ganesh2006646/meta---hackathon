---
title: ExecuCode Env
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ExecuCode: Multi-Objective AST Grader

ExecuCode introduces a fundamentally new way to evaluate AI-generated code. Instead of relying solely on binary unit testing (pass/fail), ExecuCode operates as a **Multi-Objective AST Grader** that evaluates submissions across three distinct pillars: Correctness, Performance, and Code Quality. It simulates the exact code review process a senior engineer would employ, scrutinizing not just whether the output is correct, but how efficiently and beautifully it was achieved.

Under the hood, ExecuCode safely executes AI submissions inside an isolated memory-monitored sandbox to deter resource abuse and test state leakage. However, its true power lies in its deep **Abstract Syntax Tree (AST) analysis**. As the code runs, the AST Visitor statically dissects the submission to detect optimal algorithm patterns (like dynamic programming memoization, generator expressions, and hash-based lookups) while violently penalizing nested loops, mutable default arguments, and computational anti-patterns. 

This strict, deterministic feedback loop forces AI agents to engage in iterative optimization. When an agent submits a brute-force approach, ExecuCode rejects it with specific execution metrics and AST-targeted guidance (e.g., "Linear membership checks detected. Optimize with a hash map"). The result is an environment perfectly built to push large language models beyond simply "working" code—demanding production-ready, highly optimal, and Pythonic solutions.

## Interactive Web UI

![ExecuCode Web UI](./screenshot.png)

## Task Suite

| ID | Task | Difficulty | Focus |
|---|---|---|---|
| `0` | Mutable default argument | Easy | Python correctness |
| `1` | Grid path counting | Medium | Dynamic programming and memoization |
| `2` | RAG document chunker | Hard | Text processing, edge cases, readability |
| `3` | Sliding window rate limiter | Extra-Hard | API infra logic, edge cases, code quality |

### Reward weights

| Task ID | Correctness | Performance | Quality |
|---|---:|---:|---:|
| `0` | `1.00` | `0.00` | `0.00` |
| `1` | `0.75` | `0.25` | `0.00` |
| `2` | `0.55` | `0.15` | `0.30` |
| `3` | `0.60` | `0.10` | `0.30` |

## 💡 Judge & Demo Guidelines

When demoing the platform or testing custom agent submissions, follow these rules to avoid triggering the strict sandbox defense systems:

1. **Do Not Rename the Target Function**: The static AST analyzer specifically targets the function signature. You must keep the names identical (e.g., `def count_paths(grid):` for Task 1) or the execution will immediately abort with a `MissingFunctionError`.
2. **Return Data, Don't Print**: The deterministic grader captures raw returns to cross-reference with perfect output shapes. Using `print()` is fine for local debugging, but ensuring a final `return` is mandatory for getting Correctness points.
3. **No Top-Level Tests**: Do not use `if __name__ == '__main__':` or append raw execution loops at the bottom of the snippet. Treat the environment like LeetCode—submit only the requested function and any necessary helpers.
4. **Iterative Multi-Objective Scoring**:
   - *Correctness (Unit Tests)* ensures edge-case compliance (empty states, massive inputs).
   - *Performance (AST Analysis)* strictly penalizes classic `O(n²)` loops in favor of `O(n)` hash-based lookups and memoization caching.
   - *Quality (Static Analysis)* rewards properly integrated docstrings (`"""..."""`) and strict python Type Hints.

## Project Layout

- `tasks.py`: immutable task definitions, test cases, reference solutions, scoring weights
- `grader.py`: deterministic grading pipeline, OpenEnv metrics, and AST analysis
- `utils.py`: code extraction, restricted execution, feedback formatting, AI Mentor integrations
- `server/environment.py`: `ExecuCodeEnvironment` implementation used by OpenEnv
- `environment.py`: package-root shim for OpenEnv validator discovery
- `server/app.py`: FastAPI app, interactive markdown dashboard UI, and task/grader endpoints
- `pyproject.toml`: Dependency tracking and build definitions

## Local Development & Docker

Start the local server with hot-reloading:

```powershell
uv run uvicorn execucode.server.app:app --host 127.0.0.1 --port 8000
```

To run within a fully contained Docker environment:
```powershell
docker build -t execucode .
docker run --rm -p 8000:8000 execucode
```
