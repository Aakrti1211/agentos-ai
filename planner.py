"""Planning layer that turns a user task into a simple execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory import ExecutionMemory


@dataclass(slots=True)
class PlanStep:
    """A single planned action in the agent workflow."""

    id: int
    name: str
    description: str


class Planner:
    """Creates a lightweight plan for the requested workflow."""

    def __init__(self, memory: ExecutionMemory | None = None) -> None:
        self.memory = memory

    def create_plan(self, task: str) -> list[PlanStep]:
        """Generate a three-step plan from the user's task."""

        lowered = task.lower()
        # Create a compact three-stage workflow for most user requests.
        steps: list[PlanStep] = []

        if "research" in lowered:
            steps.append(PlanStep(1, "Research", "Gather background context for the task."))
        else:
            steps.append(PlanStep(1, "Understand", "Clarify the current request and context."))

        if "summar" in lowered or "summary" in lowered:
            steps.append(PlanStep(2, "Summarize", "Condense the gathered findings."))
        else:
            steps.append(PlanStep(2, "Analyze", "Interpret the outcome of the initial step."))

        steps.append(PlanStep(3, "Generate report", "Turn the workflow into a final answer."))

        if self.memory is not None:
            self.memory.log(
                stage="planner",
                message="Plan created",
                data={"task": task, "steps": [step.name for step in steps]},
            )

        return steps

    def describe_plan(self, plan: list[PlanStep]) -> str:
        """Render a plan into a readable terminal-friendly format."""

        lines = ["Planning..."]
        for step in plan:
            lines.append(f"Step {step.id}: {step.name}")
            lines.append(f"  - {step.description}")
        return "\n".join(lines)
