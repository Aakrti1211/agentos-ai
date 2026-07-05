"""Terminal entry point for the AgentOS AI MVP."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from agents import GeminiClient, ReportAgent, ResearchAgent, SummarizerAgent
from config import APP_NAME, EXIT_COMMANDS, LOG_PATH, PROMPT_PREFIX
from memory import ExecutionMemory
from planner import Planner
from tools import CalculatorTool, PDFReaderTool, ToolRegistry, WebSearchTool, WikipediaTool
from utils import ensure_parent_dir, format_duration, format_plan, load_api_key, render_banner


class AgentOSApp:
    """Main application controller for a simple agent workflow."""

    def __init__(self) -> None:
        self.memory = ExecutionMemory()
        self.planner = Planner(self.memory)
        self.logger = self._setup_logger()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register("web_search", WebSearchTool())
        self.tool_registry.register("wikipedia", WikipediaTool())
        self.tool_registry.register("calculator", CalculatorTool())
        self.tool_registry.register("pdf_reader", PDFReaderTool())

        api_key = load_api_key()
        use_live_model = False
        self.llm_client = GeminiClient(api_key=api_key) if api_key and use_live_model else None

        self.research_agent = ResearchAgent(
            memory=self.memory,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
        )
        self.summarizer_agent = SummarizerAgent(
            memory=self.memory,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
        )
        self.report_agent = ReportAgent(
            memory=self.memory,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
        )

    def _setup_logger(self) -> logging.Logger:
        """Create a file logger for each execution run."""

        ensure_parent_dir(LOG_PATH)
        logger = logging.getLogger("agentos")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(LOG_PATH)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
        return logger

    def run(self) -> None:
        """Start the interactive terminal loop."""

        print(render_banner(APP_NAME))
        print("Live model calls are disabled by default to stay within request limits.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                task = input(PROMPT_PREFIX).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not task:
                print("Please enter a task.")
                continue

            if task.lower() in EXIT_COMMANDS:
                print("Goodbye.")
                break

            self.execute_task(task)

    def execute_task(self, task: str) -> None:
        """Run the planning and agent workflow for a single request."""

        started_at = time.perf_counter()
        self.logger.info("Task received: %s", task)
        self.memory.log(stage="app", message="Task received", data={"task": task})

        print()
        print(render_banner(APP_NAME))
        print(f"Goal: {task}")
        print("Planning...")
        plan = self.planner.create_plan(task)
        print(format_plan(plan))
        print()
        print("Executing...")

        context: dict[str, str] = {"task": task}
        total_tokens = 0

        for step in plan:
            step_name = step.name.lower()
            if step_name.startswith("research") or step_name.startswith("understand"):
                result = self.research_agent.execute(task, context)
                context["research_output"] = result.output
                total_tokens += result.tokens_used
                print(f"{self.research_agent.name} ✓ {result.output[:90]}")
            elif step_name.startswith("summar") or step_name.startswith("analy"):
                result = self.summarizer_agent.execute(task, context)
                context["summary_output"] = result.output
                total_tokens += result.tokens_used
                print(f"{self.summarizer_agent.name} ✓ {result.output[:90]}")
            else:
                result = self.report_agent.execute(task, context)
                context["report_output"] = result.output
                total_tokens += result.tokens_used
                print(f"{self.report_agent.name} ✓ {result.output[:90]}")

        elapsed = time.perf_counter() - started_at
        report_output = context.get("report_output", "No report generated.")
        self.memory.remember_task(task, report_output)
        self.logger.info("Task completed | elapsed=%s | tokens=%s", format_duration(elapsed), total_tokens)

        print("\nExecution Time : " + format_duration(elapsed))
        print(f"Tokens Used    : {total_tokens}")
        print("\nFinal Report")
        print("-" * 40)
        print(report_output)
        print()


def main() -> None:
    """Parse optional CLI arguments and run the app."""

    parser = argparse.ArgumentParser(description="Run the AgentOS AI workflow")
    parser.add_argument("task", nargs="?", help="A task to execute without entering interactive mode")
    args = parser.parse_args()

    app = AgentOSApp()
    if args.task:
        app.execute_task(args.task)
    else:
        app.run()


if __name__ == "__main__":
    main()
