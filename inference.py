"""Smarter baseline inference loop for ExecuCode."""

from __future__ import annotations

import os
import time

if __package__ in {None, ""}:
    # Allow: `python execucode/inference.py`
    import pathlib
    import sys

    package_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

from openai import OpenAI

from execucode.models import ExecuCodeAction
from execucode.server.environment import ExecuCodeEnvironment
from execucode.tasks import get_task
from execucode.utils import extract_code


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


SYSTEM_PROMPT = """You are an expert Python code optimization agent.
You will iteratively improve a single function based on evaluator feedback.
Rules:
- Always return exactly one complete Python function definition.
- Preserve required function name and output behavior.
- Prioritize correctness first, then performance, then readability.
- Do not include explanations outside the code block.
"""


def _fallback_solution(task_id: int) -> str:
    return get_task(task_id).reference_solution


def _build_user_prompt(
    task_prompt: str,
    function_name: str,
    attempt: int,
    max_attempts: int,
    previous_feedback: str | None,
    previous_submission: str | None,
) -> str:
    lines = [
        f"Attempt {attempt}/{max_attempts}",
        f"Target function name: {function_name}",
        "Task description:",
        task_prompt,
    ]

    if previous_feedback:
        lines.extend(
            [
                "",
                "Latest evaluator feedback:",
                previous_feedback,
            ]
        )
    if previous_submission:
        lines.extend(
            [
                "",
                "Previous submission:",
                "```python",
                previous_submission,
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "Return only the improved function code.",
        ]
    )
    return "\n".join(lines)


def _call_model(
    client: OpenAI | None,
    *,
    task_prompt: str,
    function_name: str,
    attempt: int,
    max_attempts: int,
    previous_feedback: str | None,
    previous_submission: str | None,
) -> str | None:
    if client is None:
        return None

    prompt = _build_user_prompt(
        task_prompt=task_prompt,
        function_name=function_name,
        attempt=attempt,
        max_attempts=max_attempts,
        previous_feedback=previous_feedback,
        previous_submission=previous_submission,
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
    except Exception:
        return None
    return response.choices[0].message.content or None


def _extract_valid_submission(
    raw_message: str | None,
    *,
    function_name: str,
) -> str | None:
    if not raw_message:
        return None
    code = extract_code(raw_message)
    if f"def {function_name}" not in code:
        return None
    return code


def main() -> None:
    start_time = time.monotonic()
    deadline_seconds = 20 * 60
    env = ExecuCodeEnvironment()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    print(
        "[START] "
        f"api_base_url={API_BASE_URL} "
        f"model_name={MODEL_NAME} "
        f"hf_token_set={bool(HF_TOKEN)}"
    )
    try:
        status = "success"
        for task_id in range(3):
            task = get_task(task_id)
            observation = env.reset(task_id=task_id)
            task_prompt = observation.echoed_message
            previous_feedback: str | None = None
            best_submission = _fallback_solution(task_id)
            best_reward = -1.0

            print(
                "[STEP] "
                "phase=reset "
                f"task_id={task_id} "
                "attempt=0 "
                "reward=0.000 "
                "done=False"
            )

            while not observation.done and time.monotonic() - start_time < deadline_seconds:
                attempt = env.state.attempts + 1
                max_attempts = env.state.max_attempts
                raw_message = _call_model(
                    client,
                    task_prompt=task_prompt,
                    function_name=task.function_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    previous_feedback=previous_feedback,
                    previous_submission=best_submission if env.state.attempts > 0 else None,
                )
                model_submission = _extract_valid_submission(
                    raw_message,
                    function_name=task.function_name,
                )

                if model_submission is None:
                    submission = best_submission
                else:
                    submission = model_submission

                observation = env.step(ExecuCodeAction(message=submission))
                reward = float(observation.reward or 0.0)
                if reward >= best_reward:
                    best_reward = reward
                    best_submission = submission

                print(
                    "[STEP] "
                    "phase=step "
                    f"task_id={task_id} "
                    f"attempt={observation.metadata['attempts']} "
                    f"reward={reward:.3f} "
                    f"done={observation.done}"
                )
                previous_feedback = observation.echoed_message

            if not observation.done and time.monotonic() - start_time >= deadline_seconds:
                status = "timeout"
                break

        duration = time.monotonic() - start_time
        print("[END] " f"status={status} " f"duration_s={duration:.2f}")
    except Exception as exc:  # noqa: BLE001 - maintain strict log-only output
        duration = time.monotonic() - start_time
        print(
            "[END] "
            "status=error "
            f"duration_s={duration:.2f} "
            f"error={type(exc).__name__}:{exc}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
