"""Tests for AgentPluginRegistry dynamic plugin registration."""

from vajra_agent import AgentPluginRegistry, CoderAgent


def test_agent_plugin_registry():
    AgentPluginRegistry.register_agent_plugin("CustomCoderRole", CoderAgent)
    cls_found = AgentPluginRegistry.get_agent_plugin("CustomCoderRole")

    assert cls_found == CoderAgent
    assert "CustomCoderRole" in AgentPluginRegistry.list_plugins()
