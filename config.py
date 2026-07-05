"""Application configuration and constants for AgentOS AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

APP_NAME: Final[str] = "AgentOS AI"
VERSION: Final[str] = "0.1.0"
PROMPT_PREFIX: Final[str] = "Enter a task: "
EXIT_COMMANDS: Final[tuple[str, ...]] = ("quit", "exit", "q")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"
MEMORY_PATH: Final[Path] = BASE_DIR / "memory.json"
LOG_PATH: Final[Path] = BASE_DIR / "logs" / "execution.log"

DEFAULT_MODEL: Final[str] = "gemini-2.0-flash"
GEMINI_API_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def get_model_name() -> str:
    """Return the configured Gemini model name, allowing environment override."""

    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

RESEARCH_AGENT_NAME: Final[str] = "ResearchAgent"
SUMMARIZER_AGENT_NAME: Final[str] = "SummarizerAgent"
REPORT_AGENT_NAME: Final[str] = "ReportAgent"

TOOL_WEB_SEARCH: Final[str] = "web_search"
TOOL_WIKIPEDIA: Final[str] = "wikipedia"
TOOL_CALCULATOR: Final[str] = "calculator"
TOOL_PDF_READER: Final[str] = "pdf_reader"
