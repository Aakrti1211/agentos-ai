"""Tool abstractions for the AgentOS AI workflow."""

from __future__ import annotations

import ast
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import TOOL_CALCULATOR, TOOL_PDF_READER, TOOL_WEB_SEARCH, TOOL_WIKIPEDIA


@dataclass(slots=True)
class ToolResult:
    """Represents the result of running a tool."""

    name: str
    output: str


class BaseTool:
    """Base class for all tools."""

    name: str = "base"

    def execute(self, payload: Any) -> str:
        """Run the tool against a payload."""

        raise NotImplementedError


class WebSearchTool(BaseTool):
    """Perform a lightweight web search using DuckDuckGo HTML."""

    name = TOOL_WEB_SEARCH

    def execute(self, payload: Any) -> str:
        query = str(payload)
        address = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        request = urllib.request.Request(
            address,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return "Web search is currently unavailable."

        snippets = re.findall(r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>', html, re.S)
        if not snippets:
            return f"No search results found for: {query}"
        lines = [f"{index + 1}. {re.sub('<.*?>', '', title).strip()}" for index, (_, title) in enumerate(snippets[:3])]
        return "\n".join(lines)


class WikipediaTool(BaseTool):
    """Query the public Wikipedia summary API."""

    name = TOOL_WIKIPEDIA

    def execute(self, payload: Any) -> str:
        topic = str(payload).strip()
        if not topic:
            return "Wikipedia lookup requires a topic."
        address = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        request = urllib.request.Request(address, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return f"Wikipedia lookup failed for {topic}."
        return data.get("extract", "No summary available.")


class CalculatorTool(BaseTool):
    """Safely evaluate simple arithmetic expressions."""

    name = TOOL_CALCULATOR

    def execute(self, payload: Any) -> str:
        expression = str(payload)
        allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError:
            return "Calculator input was invalid."

        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                return "Calculator only supports basic arithmetic."

        try:
            return str(eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}, {}))
        except Exception:
            return "Calculator evaluation failed."


class PDFReaderTool(BaseTool):
    """Read text from a local PDF file when possible."""

    name = TOOL_PDF_READER

    def execute(self, payload: Any) -> str:
        path = Path(str(payload))
        if not path.exists():
            return f"PDF file not found: {path}"

        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                return "PDF support requires the pypdf package."
            try:
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(page for page in pages if page)
                return text or "The PDF did not contain readable text."
            except Exception as exc:
                return f"Unable to read PDF: {exc}"

        return path.read_text(encoding="utf-8", errors="ignore")


class ToolRegistry:
    """A lightweight registry for available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool | Callable[[Any], Any]] = {}

    def register(self, name: str, handler: BaseTool | Callable[[Any], Any]) -> None:
        """Register a new tool handler."""

        self._tools[name] = handler

    def run(self, name: str, payload: Any) -> ToolResult:
        """Execute a registered tool and return its result."""

        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")

        handler = self._tools[name]
        if isinstance(handler, BaseTool):
            output = handler.execute(payload)
        else:
            output = handler(payload)
        return ToolResult(name=name, output=str(output))

    def available_tools(self) -> list[str]:
        """Return the registered tool names."""

        return sorted(self._tools)

    def choose_for_task(self, task: str) -> str:
        """Choose a tool based on the task keywords."""

        lowered = task.lower()
        if any(token in lowered for token in ("calculate", "sum", "total", "+", "-", "*", "/")):
            return TOOL_CALCULATOR
        if any(token in lowered for token in ("pdf", "document", "read file")):
            return TOOL_PDF_READER
        if any(token in lowered for token in ("definition", "wiki", "concept", "explain")):
            return TOOL_WIKIPEDIA
        return TOOL_WEB_SEARCH
