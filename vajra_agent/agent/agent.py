"""FoundationAgent main class implementing the Observe -> Retrieve -> Think -> Plan -> Execute -> Verify -> Reflect -> Store Knowledge loop."""

from __future__ import annotations

import json
import time
from typing import Any

from utils.logging import setup_logger
from vajra_agent.config import AgentConfig
from vajra_agent.events.bus import EventBus, EventListener
from vajra_agent.events.types import (
    AgentFinished,
    AgentStarted,
    IterationFinished,
    IterationStarted,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from vajra_agent.function_calling.parser import FunctionParser
from vajra_agent.memory.manager import MemoryManager
from vajra_agent.planning.engine import PlanningEngine
from vajra_agent.prompts.system import build_system_prompt
from vajra_agent.reasoners.base import BaseReasoner
from vajra_agent.reasoners.foundation import FoundationReasoner
from vajra_agent.reflection.engine import ReflectionEngine
from vajra_agent.registry.registry import ToolRegistry
from vajra_agent.schemas.results import AgentResponse, ToolExecutionResult
from vajra_agent.schemas.state import AgentState
from vajra_agent.tools.base import BaseTool

logger = setup_logger("vajra_agent")


class FoundationAgent:
    """Autonomous AI Engineer reasoning, planning, tool execution, memory retrieval, and self-reflection agent."""

    def __init__(
        self,
        reasoner: BaseReasoner | Any,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
        planner: PlanningEngine | None = None,
        reflector: ReflectionEngine | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.registry = registry or ToolRegistry()
        self.event_bus = EventBus()

        # Wrap raw LLM engines in FoundationReasoner if needed
        if isinstance(reasoner, BaseReasoner):
            self.reasoner = reasoner
        else:
            self.reasoner = FoundationReasoner(reasoner)

        self.planner = planner or PlanningEngine(self.reasoner)
        self.reflector = reflector or ReflectionEngine()
        self.memory = memory

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance with the agent."""
        self.registry.register(tool)

    def attach_memory(self, memory: MemoryManager) -> None:
        """Attach a MemoryManager instance to the agent for long-term memory retrieval."""
        self.memory = memory

    def subscribe(self, listener: EventListener) -> None:
        """Subscribe to agent execution events."""
        self.event_bus.subscribe(listener)

    def _observe(self, state: AgentState) -> str:
        """Observe current conversation state."""
        return state.conversation.format_history_string()

    def _think(self, prompt: str, system_prompt: str) -> str:
        """Query reasoning engine."""
        return self.reasoner.generate(prompt=prompt, system_prompt=system_prompt)

    def run(self, prompt: str) -> AgentResponse:
        """Execute the agent loop for a user query.

        Lifecycle: Observe -> Retrieve Memory -> Retrieve Repository Knowledge -> Think -> Plan -> Execute -> Verify -> Reflect -> Store Knowledge -> Finish
        """
        start_t = time.perf_counter()
        state = AgentState(max_iterations=self.config.max_iterations)
        state.conversation.add_user(prompt)

        self.event_bus.emit(AgentStarted(prompt=prompt))
        if self.config.verbose:
            logger.info(f"Agent started task: '{prompt}'")

        # Initial Task Planning
        plan = self.planner.generate_plan(prompt, self.registry.get_tool_schemas())
        state.plan_steps = [s.description for s in plan.steps]

        final_output = ""

        while state.current_iteration < self.config.max_iterations:
            state.current_iteration += 1
            iteration = state.current_iteration
            self.event_bus.emit(IterationStarted(iteration=iteration))

            # 1. OBSERVE & RETRIEVE MEMORY / KNOWLEDGE
            if self.memory:
                sys_prompt, history_str = self.memory.build_context(
                    state=state,
                    tool_schemas=self.registry.get_tool_schemas(),
                    query=prompt,
                    custom_system_prompt=self.config.system_prompt_override,
                )
            else:
                history_str = self._observe(state)
                sys_prompt = build_system_prompt(
                    self.registry.get_tool_schemas(),
                    custom_prompt=self.config.system_prompt_override,
                )

            # 2. THINK & PLAN
            thought = self._think(prompt=history_str, system_prompt=sys_prompt)
            if self.config.verbose:
                logger.info(f"[Iter {iteration}] Model Thought:\n{thought}")

            # 3. ACT / EXECUTE STEP
            call = FunctionParser.parse(thought)

            if call is None:
                # No function call requested — model has reached final answer
                final_output = thought.strip()
                state.conversation.add_assistant(final_output)

                # 4. REFLECT & STORE KNOWLEDGE
                reflection = self.reflector.reflect(state)
                state.reflection_notes.append(reflection.reasoning_critique)

                if self.memory:
                    self.memory.remember(
                        text=f"Task: '{prompt}' -> Solution: '{final_output[:200]}...'",
                        metadata={"type": "solution", "iterations": iteration},
                    )

                self.event_bus.emit(IterationFinished(iteration=iteration, has_tool_call=False))
                break

            # Function call requested
            self.event_bus.emit(ToolStarted(tool_name=call.tool_name, arguments=call.arguments))
            state.conversation.add_assistant(thought, tool_call=call.to_dict())

            tool = self.registry.get(call.tool_name)
            if tool is None:
                err_msg = f"Tool '{call.tool_name}' not found in registry."
                res = ToolExecutionResult(
                    tool_name=call.tool_name,
                    success=False,
                    output=None,
                    error=err_msg,
                )
                state.record_tool_result(res)
                state.conversation.add_tool_result(call.tool_name, f"Error: {err_msg}")
                self.event_bus.emit(ToolFailed(tool_name=call.tool_name, error=err_msg))

                if self.config.stop_on_error:
                    final_output = f"Execution stopped due to tool error: {err_msg}"
                    break
            else:
                if self.config.verbose:
                    logger.info(
                        f"[Iter {iteration}] Executing tool '{call.tool_name}' with args {call.arguments}"
                    )

                res = tool.run(**call.arguments)
                state.record_tool_result(res)

                if res.success:
                    formatted_out = (
                        json.dumps(res.output, indent=2)
                        if isinstance(res.output, (dict, list))
                        else str(res.output)
                    )
                    state.conversation.add_tool_result(call.tool_name, formatted_out)
                    self.event_bus.emit(
                        ToolFinished(
                            tool_name=call.tool_name,
                            output=res.output,
                            execution_time_ms=res.execution_time_ms,
                        )
                    )
                else:
                    err_text = f"Error executing tool '{call.tool_name}': {res.error}"
                    state.conversation.add_tool_result(call.tool_name, err_text)
                    self.event_bus.emit(ToolFailed(tool_name=call.tool_name, error=res.error or ""))

                    if self.config.stop_on_error:
                        final_output = f"Execution stopped: {res.error}"
                        break

            # 4. REFLECT
            reflection = self.reflector.reflect(state)
            state.reflection_notes.append(reflection.reasoning_critique)
            self.event_bus.emit(IterationFinished(iteration=iteration, has_tool_call=True))

        total_time_s = round(time.perf_counter() - start_t, 2)
        self.event_bus.emit(
            AgentFinished(
                output=final_output,
                iterations=state.current_iteration,
            )
        )

        return AgentResponse(
            output=final_output,
            iterations=state.current_iteration,
            tool_calls_count=state.metadata.total_tool_calls,
            conversation=state.conversation.to_list(),
            execution_time_s=total_time_s,
            metadata={
                "tool_history": [r.to_dict() for r in state.tool_history],
                "plan_steps": state.plan_steps,
                "reflection_notes": state.reflection_notes,
                "errors_encountered": state.metadata.errors_encountered,
            },
        )
