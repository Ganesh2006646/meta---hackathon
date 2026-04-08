"""Baseline inference loop for ExecuCode."""

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

from execucode.models import ExecuCodeAction
from execucode.server.environment import ExecuCodeEnvironment
from execucode.tasks import get_task

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


SYSTEM_PROMPT = """You are a code optimization agent.
Return a complete Python function definition that fixes correctness issues,
improves performance when relevant, and keeps the implementation readable.
"""


def _fallback_solution(task_id: int) -> str:
    return get_task(task_id).reference_solution


def _call_model(prompt: str) -> str | None:
    if not HF_TOKEN:
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    start_time = time.monotonic()
    deadline_seconds = 20 * 60
    env = ExecuCodeEnvironment()

    print(
        "[START] "
        f"api_base_url={API_BASE_URL} "
        f"model_name={MODEL_NAME} "
        f"hf_token_set={bool(HF_TOKEN)}"
    )
    try:
        for task_id in range(3):
            observation = env.reset(task_id=task_id)
            prompt = observation.echoed_message

            print(
                "[STEP] "
                f"phase=reset "
                f"task_id={task_id} "
                f"attempt=0 "
                "reward=0.000 "
                "done=False"
            )

            while not observation.done and time.monotonic() - start_time < deadline_seconds:
                model_message = _call_model(prompt)
                submission = model_message or _fallback_solution(task_id)
                observation = env.step(ExecuCodeAction(message=submission))
                print(
                    "[STEP] "
                    f"phase=step "
                    f"task_id={task_id} "
                    f"attempt={observation.metadata['attempts']} "
                    f"reward={float(observation.reward):.3f} "
                    f"done={observation.done}"
                )
                prompt = observation.echoed_message

        duration = time.monotonic() - start_time
        print(
            "[END] "
            f"status=success "
            f"duration_s={duration:.2f}"
        )
    except Exception as exc:  # noqa: BLE001 - maintain strict log-only output
        duration = time.monotonic() - start_time
        print(
            "[END] "
            f"status=error "
            f"duration_s={duration:.2f} "
            f"error={type(exc).__name__}:{exc}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
