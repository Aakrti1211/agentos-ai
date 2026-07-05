"""Prompt templates used by the agents."""

from __future__ import annotations


def research_prompt(task: str) -> str:
    """Prompt for the research stage."""

    return f"Research the topic: {task}"


def summary_prompt(task: str) -> str:
    """Prompt for the summary stage."""

    return f"Summarize the findings for: {task}"


def report_prompt(task: str) -> str:
    """Prompt for the report stage."""

    return f"Create a concise final report for: {task}"
