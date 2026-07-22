"""Tests for ContextBuilder prompt assembly."""

from vajra_agent.memory import ContextBuilder
from vajra_agent.schemas import AgentState


def test_context_builder_assembly():
    state = AgentState()
    state.conversation.add_user("Build feature")
    state.plan_steps = ["Step 1", "Step 2"]

    schemas = [{"name": "file_tool", "description": "file tool", "parameters": {}}]

    sys_prompt, history_prompt = ContextBuilder.build_enriched_context(
        state=state,
        tool_schemas=schemas,
    )

    assert "file_tool" in sys_prompt
    assert "Active Task Plan" in history_prompt
    assert "Step 1" in history_prompt
