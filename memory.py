"""Runtime memory used to track the agent execution history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MEMORY_PATH
from utils import ensure_parent_dir, read_json, write_json


@dataclass(slots=True)
class ExecutionRecord:
    """A single entry recorded while the workflow runs."""

    timestamp: str
    stage: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class ExecutionMemory:
    """Stores execution events in memory and on disk for persistence."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or MEMORY_PATH
        self.records: list[ExecutionRecord] = []
        self._load()

    def _load(self) -> None:
        """Load previous run history from disk."""

        payload = read_json(self.storage_path)
        if isinstance(payload, list):
            self.records = [
                ExecutionRecord(
                    timestamp=item.get("timestamp", ""),
                    stage=item.get("stage", "unknown"),
                    message=item.get("message", ""),
                    data=item.get("data", {}),
                )
                for item in payload
            ]

    def _save(self) -> None:
        """Persist the current history to disk."""

        ensure_parent_dir(self.storage_path)
        payload = [
            {
                "timestamp": record.timestamp,
                "stage": record.stage,
                "message": record.message,
                "data": record.data,
            }
            for record in self.records
        ]
        write_json(self.storage_path, payload)

    def log(self, *, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        """Persist a new runtime event."""

        self.records.append(
            ExecutionRecord(
                timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
                stage=stage,
                message=message,
                data=data or {},
            )
        )
        self._save()

    def clear(self) -> None:
        """Clear the in-memory history and persist an empty list."""

        self.records.clear()
        self._save()

    def remember_task(self, task: str, summary: str) -> None:
        """Store a completed task summary for future runs."""

        self.log(stage="memory", message="stored task summary", data={"task": task, "summary": summary})

    def get_recent_tasks(self, limit: int = 5) -> list[dict[str, str]]:
        """Return a small list of prior task summaries for context-aware generation."""

        items: list[dict[str, str]] = []
        for record in reversed(self.records):
            if record.stage != "memory" or not record.data:
                continue
            task = str(record.data.get("task", "")).strip()
            summary = str(record.data.get("summary", "")).strip()
            if task:
                items.append({"task": task, "summary": summary})
            if len(items) >= limit:
                break
        return list(reversed(items))

    def get_history(self) -> list[ExecutionRecord]:
        """Return a copy of the recorded history."""

        return list(self.records)

    def render(self) -> str:
        """Serialize the recorded history into a readable string."""

        if not self.records:
            return "No execution history recorded."

        lines = ["Execution history:"]
        for record in self.records:
            lines.append(f"- [{record.timestamp}] {record.stage}: {record.message}")
        return "\n".join(lines)
