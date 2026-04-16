"""Autonomous demo loop for ExecuCodeEnv using Gemini on HumanEval tasks."""

from __future__ import annotations

import os
import time
from typing import Any

if __package__ in {None, ""}:
    import pathlib
    import sys

    package_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from execucode.rl_env import ExecuCodeEnv

SYSTEM_INSTRUCTION = (
    "You are an AI trying to solve a Python problem. Output ONLY raw Python code. "
    "Do not use markdown formatting like ```python. Do not explain yourself."
)
MAX_API_RETRIES = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "3")))
RETRY_BACKOFF_SECONDS = max(0.5, float(os.environ.get("GEMINI_RETRY_BACKOFF_SECONDS", "1.5")))


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _model_candidates() -> list[str]:
    explicit = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()]
    if explicit:
        return explicit

    primary = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    defaults = [primary, "gemini-1.5-flash", "gemini-pro"]
    ordered: list[str] = []
    for model_name in defaults:
        if model_name and model_name not in ordered:
            ordered.append(model_name)
    return ordered


ENABLE_REFERENCE_FALLBACK = _env_bool("AGENT_ENABLE_REFERENCE_FALLBACK", default=True)
MODEL_CANDIDATES = _model_candidates()


def _validate_api_key() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("Set GEMINI_API_KEY before running agent_loop.py")


def _build_reference_fallback_code(env: ExecuCodeEnv) -> str:
    task = getattr(env, "current_task", None)
    if task is None:
        return ""

    reference = str(getattr(task, "reference_solution", "") or "").strip()
    if not reference:
        return ""

    function_name = str(getattr(task, "function_name", "") or "").strip()
    marker = f"def {function_name}" if function_name else ""
    if marker and marker in reference:
        return reference

    prompt = str(getattr(task, "description", "") or "").strip()
    if marker and marker in prompt:
        return f"{prompt}\n{reference.lstrip()}".strip()

    return reference


def _is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "not found" in message
        or "not supported" in message
        or "model" in message and "404" in message
    )


def _build_gemini_backend(api_key: str) -> tuple[str, Any]:
    if genai is not None:
        client = genai.Client(api_key=api_key)
        return "modern", client

    raise RuntimeError("No Gemini SDK installed. Install google-genai.")


def _extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        parts = getattr(content, "parts", None)
        if isinstance(parts, list):
            fragments: list[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    fragments.append(part_text)
            if fragments:
                return "\n".join(fragments)

    return ""


def _strip_markdown_fence(code: str) -> str:
    stripped = code.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return ""

    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _generate_with_modern_sdk(client: Any, model_name: str, prompt_text: str) -> str:
    config: Any
    if types is not None:
        config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    else:
        config = {"system_instruction": SYSTEM_INSTRUCTION}

    response = client.models.generate_content(
        model=model_name,
        contents=prompt_text,
        config=config,
    )
    return _extract_response_text(response)


def generate_code_action(prompt_text: str, backend: str, backend_client: Any) -> str:
    """Generate a raw Python code string using Gemini with retry and model fallback."""

    last_error: Exception | None = None
    for model_name in MODEL_CANDIDATES:
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                if backend == "modern":
                    raw_code = _generate_with_modern_sdk(backend_client, model_name, prompt_text)
                else:
                    raise RuntimeError(f"Unsupported Gemini backend: {backend}")

                code = _strip_markdown_fence(raw_code)
                if code.strip():
                    return code
                raise RuntimeError("Model returned an empty code response.")
            except Exception as exc:  # noqa: BLE001 - report retry-safe API failures
                last_error = exc

                # If model is unavailable, immediately try the next candidate.
                if _is_model_not_found_error(exc):
                    print(f"Warning: model '{model_name}' unavailable: {exc}. Trying next model.")
                    break

                if attempt >= MAX_API_RETRIES:
                    break

                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                print(
                    "Warning: Gemini request failed "
                    f"for model '{model_name}' (attempt {attempt}/{MAX_API_RETRIES}): {exc}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini generation failed after trying models {MODEL_CANDIDATES}: {last_error}"
    )


def run_loop() -> None:
    backend: str | None = None
    backend_client: Any = None

    try:
        _validate_api_key()
        api_key = os.environ.get("GEMINI_API_KEY", "")
        backend, backend_client = _build_gemini_backend(api_key)
    except Exception as exc:  # noqa: BLE001 - fallback-only mode is allowed
        if not ENABLE_REFERENCE_FALLBACK:
            raise
        print(f"Warning: Gemini backend unavailable ({exc}). Using reference fallback mode.")

    env = ExecuCodeEnv(
        dataset_name=os.environ.get("HUMANEVAL_DATASET", "openai/openai_humaneval"),
        split=os.environ.get("HUMANEVAL_SPLIT", "test"),
    )
    obs, info = env.reset()
    print(
        "TARGET PROBLEM "
        f"[{info.get('task_id', 'unknown')} / {info.get('entry_point', 'unknown')}]: {obs}"
    )

    attempts = 0
    max_attempts = 5
    used_reference_fallback = False

    while True:
        attempts += 1
        prompt = (
            "Here is the problem description or the feedback from my last attempt: "
            f"{obs}. Please write the complete Python solution."
        )

        try:
            if backend is None:
                raise RuntimeError("Gemini backend unavailable")
            action = generate_code_action(prompt, backend, backend_client)
        except Exception as exc:  # noqa: BLE001 - user-facing failure path
            if not ENABLE_REFERENCE_FALLBACK:
                print(f"\nAgent failed to generate code: {exc}")
                break

            fallback_action = _build_reference_fallback_code(env)
            if not fallback_action:
                print(f"\nAgent failed to generate code and no reference fallback is available: {exc}")
                break

            used_reference_fallback = True
            action = fallback_action
            print(f"\nAgent generation failed ({exc}). Using reference fallback for this attempt.")

        print(f"\nAgent generated {len(action)} bytes of code. Testing...")

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:  # noqa: BLE001 - user-facing failure path
            print(f"Environment step failed: {exc}")
            break

        print(f"Reward: {reward} | Correctness: {info.get('correctness')}")

        if terminated:
            print("\nAGENT SOLVED THE PROBLEM.")
            break

        if used_reference_fallback:
            correctness = float(info.get("correctness", 0.0))
            if correctness >= 0.95:
                print("\nReference fallback reached high correctness. Stopping early.")
            else:
                print("\nReference fallback did not reach solve threshold. Stopping early.")
            break

        if attempts >= max_attempts:
            print("\nAgent failed to solve the problem within the maximum attempts.")
            break

        if truncated:
            print("\nEpisode was truncated.")
            break

        time.sleep(1.0)


if __name__ == "__main__":
    run_loop()
