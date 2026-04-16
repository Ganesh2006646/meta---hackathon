"""Utilities for logging and visualizing RL trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def write_trajectory_jsonl(entries: list[dict[str, Any]], output_path: str) -> None:
    """Write trajectory entries to JSON Lines format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_trajectory_jsonl(input_path: str) -> list[dict[str, Any]]:
    """Load trajectory entries from JSON Lines format."""

    path = Path(input_path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def summarize_trajectory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics across trajectory records."""

    if not entries:
        return {
            "steps": 0,
            "episodes": 0,
            "first_reward": 0.0,
            "last_reward": 0.0,
            "best_reward": 0.0,
            "mean_reward": 0.0,
            "solved_episodes": [],
        }

    rewards = [_safe_float(entry.get("reward", 0.0)) for entry in entries]
    episode_ids = sorted(
        {
            int(entry.get("episode", 0))
            for entry in entries
            if isinstance(entry.get("episode"), int)
        }
    )

    solved_episodes = sorted(
        {
            int(entry.get("episode", 0))
            for entry in entries
            if bool(entry.get("terminated")) and _safe_float(entry.get("reward", 0.0)) >= 0.95
        }
    )

    return {
        "steps": len(entries),
        "episodes": len(episode_ids) if episode_ids else 1,
        "first_reward": rewards[0],
        "last_reward": rewards[-1],
        "best_reward": max(rewards),
        "mean_reward": mean(rewards),
        "solved_episodes": solved_episodes,
    }


def format_summary(entries: list[dict[str, Any]]) -> str:
    """Render a compact text summary for CLI output."""

    stats = summarize_trajectory(entries)
    if stats["steps"] == 0:
        return "No trajectory records to summarize."

    improvement = stats["last_reward"] - stats["first_reward"]
    solved = stats["solved_episodes"]
    solved_text = ", ".join(str(episode) for episode in solved) if solved else "none"

    return "\n".join(
        [
            "Trajectory Summary",
            f"- Steps: {stats['steps']}",
            f"- Episodes: {stats['episodes']}",
            f"- First reward: {stats['first_reward']:.3f}",
            f"- Last reward: {stats['last_reward']:.3f}",
            f"- Best reward: {stats['best_reward']:.3f}",
            f"- Mean reward: {stats['mean_reward']:.3f}",
            f"- Net improvement: {improvement:+.3f}",
            f"- Solved episodes: {solved_text}",
        ]
    )


def render_reward_chart(entries: list[dict[str, Any]], width: int = 36) -> str:
    """Render an ASCII reward chart where each row is one step."""

    if not entries:
        return "No data for reward chart."

    width = max(10, int(width))
    lines: list[str] = ["Reward Chart (0.000 to 1.000)"]

    for index, entry in enumerate(entries, start=1):
        reward = max(0.0, min(1.0, _safe_float(entry.get("reward", 0.0))))
        filled = int(round(reward * width))
        bar = "#" * filled + "." * (width - filled)
        lines.append(f"{index:03d} [{bar}] {reward:.3f}")

    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Summarize and chart trajectory JSONL logs.")
    parser.add_argument("--input", required=True, help="Path to trajectory JSONL file")
    parser.add_argument("--chart-width", type=int, default=36, help="Chart width in characters")
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Only print summary without ASCII chart",
    )
    args = parser.parse_args()

    entries = load_trajectory_jsonl(args.input)
    print(format_summary(entries))
    if not args.no_chart:
        print()
        print(render_reward_chart(entries, width=args.chart_width))


if __name__ == "__main__":
    _cli()
