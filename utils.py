"""Helper utilities used across the application."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from config import ENV_PATH


def normalize_text(text: str) -> str:
    """Trim and normalize whitespace in a string."""

    return " ".join(text.split())


def truncate(text: str, limit: int = 160) -> str:
    """Shorten text to a readable length while preserving meaning."""

    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def format_plan(plan: Sequence[Any]) -> str:
    """Format a plan into a terminal-friendly string."""

    lines: list[str] = []
    for step in plan:
        lines.append(f"Step {step.id}: {step.name}")
        lines.append(f"  - {step.description}")
    return "\n".join(lines)


def format_result(title: str, value: str) -> str:
    """Create a short, readable block for displaying agent output."""

    return f"{title}:\n{truncate(value)}"


def ensure_parent_dir(path: Path) -> None:
    """Create a parent directory if it does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_api_key(env_path: Path = ENV_PATH) -> str:
    """Read the Gemini API key from environment variables or the local .env file."""

    env_value = os.getenv("GEMINI_API_KEY", "").strip()
    if env_value and env_value.lower() not in {"your_api_key_here", "your_key_here"}:
        return env_value

    if not env_path.exists():
        return ""

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().upper() == "GEMINI_API_KEY":
            cleaned = value.strip().strip('"').strip("'")
            if cleaned.lower() in {"", "your_api_key_here", "your_key_here"}:
                return ""
            return cleaned
    return ""


def write_json(path: Path, payload: Any) -> None:
    """Persist JSON to disk."""

    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON from disk if present."""

    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for display."""

    return f"{seconds:.2f} s"


def render_banner(title: str) -> str:
    """Create a simple terminal banner."""

    border = "=" * 26
    return f"{border} {title} {border}"
