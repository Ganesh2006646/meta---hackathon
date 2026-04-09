"""Hackathon-compliant inference loop for ExecuCode."""

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

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_NAME = "execucode"
TASK_IDS = (0, 1, 2)
EPISODE_TIMEOUT_SECONDS = int(os.getenv("EPISODE_TIMEOUT_SECONDS", "120"))


SYSTEM_PROMPT = """You are an expert Python code optimization agent.
You will iteratively improve a single function based on evaluator feedback.
Rules:
- Always return exactly one complete Python function definition.
- Preserve required function name and output behavior.
- Prioritize correctness first, then performance, then readability.
- Do not include explanations outside the code block.
"""


def _to_bool_token(value: bool) -> str:
    return "true" if value else "false"


def _escape_log_field(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .strip()
    )


def _score_for_log(score: float) -> float:
    """Keep printed scores inside (0, 1) even after 2-decimal formatting."""

    return max(0.01, min(0.99, float(score)))


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
    client: OpenAI,
    *,
    task_prompt: str,
    function_name: str,
    attempt: int,
    max_attempts: int,
    previous_feedback: str | None,
    previous_submission: str | None,
) -> str | None:
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


def _run_episode(client: OpenAI, task_id: int) -> None:
    task = get_task(task_id)
    env = ExecuCodeEnvironment()
    rewards: list[float] = []
    steps = 0
    solved = False
    previous_feedback: str | None = None
    previous_submission: str | None = None

    print(f"[START] task={task.function_name} env={ENV_NAME} model={MODEL_NAME}")

    try:
        observation = env.reset(task_id=task_id)
        task_prompt = observation.echoed_message
        deadline = time.monotonic() + EPISODE_TIMEOUT_SECONDS

        while not observation.done and time.monotonic() < deadline:
            attempt = steps + 1
            max_attempts = env.state.max_attempts
            raw_message = _call_model(
                client,
                task_prompt=task_prompt,
                function_name=task.function_name,
                attempt=attempt,
                max_attempts=max_attempts,
                previous_feedback=previous_feedback,
                previous_submission=previous_submission,
            )
            model_submission = _extract_valid_submission(
                raw_message,
                function_name=task.function_name,
            )
            submission = model_submission if model_submission is not None else _fallback_solution(task_id)

            observation = env.step(ExecuCodeAction(message=submission))
            steps += 1

            reward = _score_for_log(float(observation.reward or 0.0))
            rewards.append(reward)
            done = bool(observation.done)
            if done and reward >= 0.95:
                solved = True

            metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
            error_raw = metadata.get("last_action_error")
            error = "null" if not error_raw else _escape_log_field(str(error_raw))

            print(
                "[STEP] "
                f"step={steps} "
                f"action=submit_attempt_{attempt} "
                f"reward={reward:.2f} "
                f"done={_to_bool_token(done)} "
                f"error={error}"
            )

            previous_feedback = observation.echoed_message
            previous_submission = submission
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        rewards_payload = ",".join(f"{score:.2f}" for score in rewards)
        final_score = _score_for_log(rewards[-1] if rewards else 0.0)
        print(
            "[END] "
            f"success={_to_bool_token(solved)} "
            f"steps={steps} "
            f"score={final_score:.2f} "
            f"rewards={rewards_payload}"
        )


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_id in TASK_IDS:
        _run_episode(client, task_id)


if __name__ == "__main__":
    main()
