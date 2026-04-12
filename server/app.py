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
    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, float(score)))


def _resolve_task_id(task_id: int | str) -> int:
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
    task_id: int | str = Field(default=0)
    answer: str | None = None
    submission: str | None = None
    code: str | None = None
    response: str | None = None
    message: str | None = None


class GraderResponse(BaseModel):
    task_id: int
    score: float
    breakdown: dict[str, float]
    feedback: str
    test_details: list[dict[str, Any]] = Field(default_factory=list)


class BaselineRequest(BaseModel):
    task_ids: list[int | str] | None = None


class BaselineResponse(BaseModel):
    scores: dict[str, float]
    average: float
    model: str


def _extract_submission(payload: GraderRequest) -> str:
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


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ExecuCode — AI Code Optimization Environment</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0a0e17;
      --surface: #111827;
      --surface2: #1a2235;
      --border: #1e2d45;
      --ink: #e2e8f0;
      --muted: #64748b;
      --accent: #38bdf8;
      --accent2: #818cf8;
      --ok: #34d399;
      --warn: #fbbf24;
      --err: #f87171;
      --code-bg: #0d1117;
      --radius: 12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Syne', sans-serif;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
      padding: 32px 16px;
    }
    body::before {
      content: '';
      position: fixed; inset: 0;
      background:
        radial-gradient(ellipse 900px 500px at 10% -10%, rgba(56,189,248,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 700px 400px at 90% 90%, rgba(129,140,248,0.05) 0%, transparent 70%);
      pointer-events: none;
    }
    .shell {
      max-width: 960px;
      margin: 0 auto;
      display: grid;
      gap: 24px;
    }
    header {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 24px 28px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    .logo {
      font-size: 2rem;
      line-height: 1;
    }
    .hd-text h1 {
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hd-text p {
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 4px;
    }
    .nav-pills {
      margin-left: auto;
      display: flex;
      gap: 8px;
    }
    .nav-pills a {
      color: var(--accent);
      text-decoration: none;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 6px 12px;
      border: 1px solid rgba(56,189,248,0.3);
      border-radius: 20px;
      transition: all 0.15s;
    }
    .nav-pills a:hover { background: rgba(56,189,248,0.1); }

    /* Task pills */
    .task-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
    }
    .task-pill {
      padding: 14px 16px;
      background: var(--surface);
      border: 2px solid var(--border);
      border-radius: var(--radius);
      cursor: pointer;
      transition: all 0.15s;
      text-align: left;
    }
    .task-pill:hover { border-color: var(--accent); background: var(--surface2); }
    .task-pill.active { border-color: var(--accent); background: rgba(56,189,248,0.07); }
    .task-pill .badge {
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      padding: 2px 8px;
      border-radius: 20px;
      margin-bottom: 8px;
    }
    .badge.easy   { background: rgba(52,211,153,0.15); color: var(--ok); }
    .badge.medium { background: rgba(251,191,36,0.15);  color: var(--warn); }
    .badge.hard   { background: rgba(248,113,113,0.15); color: var(--err); }
    .task-pill h3 { font-size: 0.85rem; font-weight: 600; line-height: 1.3; }
    .task-pill p  { font-size: 0.75rem; color: var(--muted); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

    /* Editor area */
    .editor-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .editor-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }
    .editor-header span { font-size: 0.8rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
    .editor-header span.lang { color: var(--accent); font-weight: 600; }
    textarea#codeBox {
      width: 100%;
      min-height: 260px;
      background: var(--code-bg);
      color: #c9d1d9;
      border: none;
      outline: none;
      padding: 16px 20px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13.5px;
      line-height: 1.6;
      resize: vertical;
      tab-size: 4;
    }
    .run-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-top: 1px solid var(--border);
    }
    .run-btn {
      display: flex;
      align-items: center;
      gap: 8px;
      background: linear-gradient(135deg, #0ea5e9, #6366f1);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 10px 24px;
      font-family: 'Syne', sans-serif;
      font-size: 0.9rem;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }
    .run-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
    .run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .run-btn .spinner {
      width: 14px; height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
    }
    .run-btn.loading .spinner { display: block; }
    .run-btn.loading .btn-text { opacity: 0.7; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hint { font-size: 0.75rem; color: var(--muted); }

    /* Result panel */
    .result-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      display: none;
    }
    .result-panel.visible { display: block; }
    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }
    .result-header span { font-size: 0.8rem; font-weight: 600; color: var(--muted); }
    .score-badge {
      font-size: 1.1rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }
    .score-badge.good  { color: var(--ok); }
    .score-badge.mid   { color: var(--warn); }
    .score-badge.bad   { color: var(--err); }

    /* Score bars */
    .bars { padding: 16px 20px; display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 100px 1fr 52px; align-items: center; gap: 10px; }
    .bar-label { font-size: 0.8rem; color: var(--muted); }
    .bar-track {
      height: 7px; background: var(--border); border-radius: 4px; overflow: hidden;
    }
    .bar-fill {
      height: 100%; border-radius: 4px;
      transition: width 0.6s cubic-bezier(.4,0,.2,1);
    }
    .bar-fill.c { background: linear-gradient(90deg, #38bdf8, #818cf8); }
    .bar-fill.p { background: linear-gradient(90deg, #34d399, #059669); }
    .bar-fill.q { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .bar-val { font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: var(--ink); text-align: right; }

    /* Feedback text */
    .feedback-body {
      padding: 0 20px 20px;
      font-size: 0.85rem;
      line-height: 1.65;
      color: #94a3b8;
      white-space: pre-wrap;
      font-family: 'JetBrains Mono', monospace;
    }

    /* History */
    .history-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      display: none;
    }
    .history-wrap.visible { display: block; }
    .history-header {
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      font-size: 0.8rem; color: var(--muted);
    }
    .history-list { padding: 8px 12px; display: grid; gap: 6px; }
    .hist-row {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 12px;
      background: var(--surface2);
      border-radius: 8px;
      font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
    }
    .hist-row .task-tag { color: var(--accent2); }
    .hist-row .score-tag { margin-left: auto; font-weight: 600; }

    footer {
      text-align: center;
      font-size: 0.72rem;
      color: var(--muted);
      padding-top: 8px;
    }
    footer a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
<div class="shell">
  <header>
    <div class="logo">⚡</div>
    <div class="hd-text">
      <h1>ExecuCode</h1>
      <p>Multi-objective AI code optimization environment &mdash; Correctness · Performance · Quality</p>
    </div>
    <nav class="nav-pills">
      <a href="/docs" target="_blank">API Docs</a>
      <a href="/tasks" target="_blank">JSON Tasks</a>
      <a href="/health" target="_blank">Health</a>
    </nav>
  </header>

  <div id="taskGrid" class="task-grid"></div>

  <div class="editor-box">
    <div class="editor-header">
      <span class="lang">python</span>
      <span id="editorLabel">Select a task above, then paste your solution</span>
    </div>
    <textarea id="codeBox" spellcheck="false" placeholder="# Paste your Python function here..."></textarea>
    <div class="run-bar">
      <button class="run-btn" id="runBtn" onclick="runGrader()">
        <div class="spinner"></div>
        <span class="btn-text">▶ Run Grader</span>
      </button>
      <span class="hint">Runs in a deterministic sandbox &mdash; correctness + performance + quality</span>
    </div>
  </div>

  <div class="result-panel" id="resultPanel">
    <div class="result-header">
      <span>Grading Result</span>
      <span class="score-badge" id="scoreBadge">—</span>
    </div>
    <div class="bars" id="barsArea"></div>
    <div class="feedback-body" id="feedbackBody"></div>
  </div>

  <div class="history-wrap" id="historyWrap">
    <div class="history-header">📋 Submission History (this session)</div>
    <div class="history-list" id="historyList"></div>
  </div>

  <footer>
    ExecuCode &mdash; OpenEnv Hackathon &nbsp;|&nbsp;
    <a href="/baseline" target="_blank">Baseline Scores</a>
  </footer>
</div>

<script>
const TASKS = [
  { id: 0, difficulty: "easy",   title: "Mutable Default Argument",    fn: "append_to_history" },
  { id: 1, difficulty: "medium", title: "Grid Path Counting (DP)",      fn: "count_paths" },
  { id: 2, difficulty: "hard",   title: "RAG Document Chunker",         fn: "chunk_document" },
  { id: 3, difficulty: "hard",   title: "Sliding Window Rate Limiter",  fn: "is_allowed" },
];

let selectedTask = 0;
const history = [];

function buildTaskGrid() {
  const grid = document.getElementById("taskGrid");
  TASKS.forEach(t => {
    const el = document.createElement("button");
    el.className = "task-pill" + (t.id === 0 ? " active" : "");
    el.id = `pill-${t.id}`;
    el.onclick = () => selectTask(t.id);
    el.innerHTML = `
      <span class="badge ${t.difficulty}">${t.difficulty.toUpperCase()}</span>
      <h3>${t.title}</h3>
      <p>${t.fn}()</p>
    `;
    grid.appendChild(el);
  });
}

function selectTask(id) {
  selectedTask = id;
  document.querySelectorAll(".task-pill").forEach(p => p.classList.remove("active"));
  document.getElementById(`pill-${id}`).classList.add("active");
  document.getElementById("editorLabel").textContent =
    `Task ${id} — ${TASKS[id].fn}()  |  ${TASKS[id].difficulty}`;
}

async function runGrader() {
  const code = document.getElementById("codeBox").value.trim();
  if (!code) { alert("Paste your Python function first."); return; }

  const btn = document.getElementById("runBtn");
  btn.disabled = true; btn.classList.add("loading");

  try {
    const res = await fetch("/grader", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: selectedTask, code }),
    });
    const data = await res.json();
    if (!res.ok) { showError(JSON.stringify(data, null, 2)); return; }
    showResult(data);
    addHistory(selectedTask, data.score);
  } catch(e) {
    showError("Network error: " + e);
  } finally {
    btn.disabled = false; btn.classList.remove("loading");
  }
}

function showResult(data) {
  const panel = document.getElementById("resultPanel");
  panel.classList.add("visible");

  const score = data.score;
  const badge = document.getElementById("scoreBadge");
  badge.textContent = (score * 100).toFixed(1) + "%";
  badge.className = "score-badge " + (score >= 0.9 ? "good" : score >= 0.6 ? "mid" : "bad");

  const bd = data.breakdown || {};
  document.getElementById("barsArea").innerHTML = [
    { label: "Correctness", key: "correctness", cls: "c" },
    { label: "Performance", key: "performance", cls: "p" },
    { label: "Quality",     key: "quality",     cls: "q" },
  ].map(({ label, key, cls }) => {
    const v = (bd[key] || 0);
    return `<div class="bar-row">
      <span class="bar-label">${label}</span>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${(v*100).toFixed(1)}%"></div></div>
      <span class="bar-val">${(v*100).toFixed(1)}%</span>
    </div>`;
  }).join("");

  document.getElementById("feedbackBody").textContent = data.feedback || "";
}

function showError(msg) {
  const panel = document.getElementById("resultPanel");
  panel.classList.add("visible");
  document.getElementById("scoreBadge").textContent = "ERROR";
  document.getElementById("scoreBadge").className = "score-badge bad";
  document.getElementById("barsArea").innerHTML = "";
  document.getElementById("feedbackBody").textContent = msg;
}

function addHistory(taskId, score) {
  history.unshift({ taskId, score, fn: TASKS[taskId].fn, ts: new Date().toLocaleTimeString() });
  document.getElementById("historyWrap").classList.add("visible");
  document.getElementById("historyList").innerHTML = history.slice(0, 8).map(h => `
    <div class="hist-row">
      <span class="task-tag">task_${h.taskId}</span>
      <span>${h.fn}()</span>
      <span>${h.ts}</span>
      <span class="score-tag" style="color:${h.score>=0.9?'var(--ok)':h.score>=0.6?'var(--warn)':'var(--err)'}">${(h.score*100).toFixed(1)}%</span>
    </div>`).join("");
}

buildTaskGrid();
selectTask(0);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def serve_homepage() -> HTMLResponse:
    """Serves the interactive judge dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """Expose all tasks with grader wiring for validators."""
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
    """Liveness endpoint for validators and deployments."""
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
