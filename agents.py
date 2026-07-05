"""Agent implementations for the AgentOS AI workflow."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import DEFAULT_MODEL, GEMINI_API_URL, REPORT_AGENT_NAME, RESEARCH_AGENT_NAME, SUMMARIZER_AGENT_NAME, get_model_name
from memory import ExecutionMemory
from prompts import report_prompt, research_prompt, summary_prompt
from tools import ToolRegistry
from utils import normalize_text, truncate


@dataclass(slots=True)
class AgentResult:
    """Represents the content produced by an agent."""

    name: str
    output: str
    tokens_used: int = 0


class GeminiClient:
    """A small wrapper around the Google Gemini REST API."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or get_model_name() or DEFAULT_MODEL

    def generate(self, prompt: str) -> str:
        """Generate text from the Gemini API with basic retry support."""

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        url = GEMINI_API_URL.format(model=self.model) + f"?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip() or "No response generated."


class BaseAgent:
    """Shared base class for all agents in the workflow."""

    def __init__(
        self,
        name: str,
        memory: ExecutionMemory | None = None,
        llm_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.name = name
        self.memory = memory
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    def _log(self, message: str, data: dict[str, Any] | None = None) -> None:
        if self.memory is not None:
            self.memory.log(stage=self.name, message=message, data=data)

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        """Call the LLM when available, otherwise return a graceful fallback."""

        if self.llm_client is None:
            return "LLM client is unavailable. Using deterministic fallback.", 0

        try:
            response = self.llm_client.generate(prompt)
            return response, max(1, len(response.split()))
        except Exception as exc:
            return f"Fallback response: {exc}", 0

    def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        """Execute the agent's workflow; subclasses override this."""

        raise NotImplementedError


class ResearchAgent(BaseAgent):
    """Gather fresh context for a task using the LLM and a tool when appropriate."""

    def __init__(
        self,
        memory: ExecutionMemory | None = None,
        llm_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(RESEARCH_AGENT_NAME, memory, llm_client, tool_registry)

    def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        tool_output = ""
        tool_name = ""
        if self.tool_registry is not None:
            tool_name = self.tool_registry.choose_for_task(task)
            try:
                tool_output = self.tool_registry.run(tool_name, task).output
            except Exception:
                tool_output = ""

        previous_context = ""
        if self.memory is not None:
            prior_tasks = self.memory.get_recent_tasks(limit=3)
            if prior_tasks:
                previous_context = "\n".join(
                    f"- {item['task']}: {item['summary'][:120]}" for item in prior_tasks
                )

        prompt = research_prompt(task)
        if previous_context:
            prompt = f"{prompt}\n\nPrevious tasks:\n{previous_context}"
        if tool_output:
            prompt = f"{prompt}\n\nTool output:\n{tool_output}"

        response, tokens = self._call_llm(prompt)
        if not response or response.startswith("Fallback response"):
            if tool_output:
                response = f"Research findings from available tools:\n{tool_output}"
            else:
                response = "Research completed using deterministic fallback guidance."
        summary = normalize_text(response)
        self._log("research completed", {"task": task, "tool": tool_name, "context": context})
        return AgentResult(name=self.name, output=summary, tokens_used=tokens)


class SummarizerAgent(BaseAgent):
    """Condense the gathered research into a concise summary."""

    def __init__(
        self,
        memory: ExecutionMemory | None = None,
        llm_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(SUMMARIZER_AGENT_NAME, memory, llm_client, tool_registry)

    def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        prior = context.get("research_output", "")
        prior_tasks = ""
        if self.memory is not None:
            prior_tasks = "\n".join(
                f"- {item['task']}: {item['summary'][:80]}" for item in self.memory.get_recent_tasks(limit=3)
            )
        prompt = f"{summary_prompt(task)}\n\nResearch findings: {prior}"
        if prior_tasks:
            prompt = f"{prompt}\n\nRelated prior tasks:\n{prior_tasks}"
        response, tokens = self._call_llm(prompt)
        summary = normalize_text(response)
        self._log("summary created", {"task": task, "prior": prior})
        return AgentResult(name=self.name, output=summary, tokens_used=tokens)


class ReportAgent(BaseAgent):
    """Produce the final report from the workflow context."""

    def __init__(
        self,
        memory: ExecutionMemory | None = None,
        llm_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(REPORT_AGENT_NAME, memory, llm_client, tool_registry)

    def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        research = context.get("research_output", "No research available")
        summary = context.get("summary_output", "No summary available")
        prompt = f"{report_prompt(task)}\n\nResearch: {research}\n\nSummary: {summary}"
        response, tokens = self._call_llm(prompt)
        if not response or response.startswith("Fallback response"):
            response = (
                f"Task: {task}\n"
                f"Research highlights: {research}\n"
                f"Summary: {summary}"
            )
        report = normalize_text(response)
        self._log("report generated", {"task": task, "report": report})
        return AgentResult(name=self.name, output=report, tokens_used=tokens)
