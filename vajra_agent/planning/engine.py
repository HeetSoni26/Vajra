"""PlanningEngine generating structured execution plans for complex tasks."""

from __future__ import annotations

import json
from typing import Any

from vajra_agent.function_calling.parser import FunctionParser
from vajra_agent.planning.models import Plan, PlanStep
from vajra_agent.reasoners.base import BaseReasoner


class PlanningEngine:
    """Task decomposition and structured plan generator."""

    def __init__(self, reasoner: BaseReasoner | None = None) -> None:
        self.reasoner = reasoner

    def create_simple_plan(self, objective: str, steps_desc: list[str]) -> Plan:
        """Create a multi-step plan from a list of step descriptions."""
        steps = []
        prev_id = None
        for desc in steps_desc:
            step = PlanStep(
                description=desc,
                dependencies=[prev_id] if prev_id else [],
            )
            steps.append(step)
            prev_id = step.step_id

        return Plan(objective=objective, steps=steps)

    def generate_plan(self, objective: str, available_tools: list[dict[str, Any]]) -> Plan:
        """Generate structured Plan using reasoning engine or structured heuristic fallback."""
        from vajra_agent.reasoners.mock import MockReasoner

        if self.reasoner is None or isinstance(self.reasoner, MockReasoner):
            # Fallback simple single-step plan
            return self.create_simple_plan(objective, [f"Analyze and execute: {objective}"])

        tool_names = [t["name"] for t in available_tools]
        prompt = (
            f"Decompose the following coding objective into structured execution steps:\n"
            f"Objective: {objective}\n"
            f"Available Tools: {tool_names}\n\n"
            f"Output a JSON object in this format:\n"
            f"```json\n"
            f"{{\n"
            f'  "steps": [\n'
            f'    {{"description": "Step description", "tool_name": "file_tool", "complexity": "low"}}\n'
            f"  ]\n"
            f"}}\n"
            f"```"
        )

        thought = self.reasoner.generate(prompt=prompt)
        raw_json = FunctionParser.extract_json_block(thought)

        if raw_json:
            try:
                data = json.loads(raw_json)
                step_dicts = data.get("steps", [])
                steps = []
                prev_id = None
                for sdict in step_dicts:
                    step = PlanStep(
                        description=sdict.get("description", "Execute step"),
                        tool_name=sdict.get("tool_name"),
                        estimated_complexity=sdict.get("complexity", "low"),
                        dependencies=[prev_id] if prev_id else [],
                    )
                    steps.append(step)
                    prev_id = step.step_id
                if steps:
                    return Plan(objective=objective, steps=steps)
            except Exception:
                pass

        return self.create_simple_plan(objective, [f"Execute task: {objective}"])
