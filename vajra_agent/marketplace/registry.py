"""AgentPluginRegistry supporting dynamic third-party agent registration."""

from __future__ import annotations

from vajra_agent.specialized.base import SpecializedAgent


class AgentPluginRegistry:
    """Plugin architecture for external third-party agent discovery and registration."""

    _registered_agents: dict[str, type[SpecializedAgent]] = {}

    @classmethod
    def register_agent_plugin(cls, role_name: str, agent_class: type[SpecializedAgent]) -> None:
        """Register a third-party agent class plugin."""
        cls._registered_agents[role_name] = agent_class

    @classmethod
    def get_agent_plugin(cls, role_name: str) -> type[SpecializedAgent] | None:
        """Retrieve registered agent class plugin by role name."""
        return cls._registered_agents.get(role_name)

    @classmethod
    def list_plugins(cls) -> list[str]:
        """List all registered agent plugin roles."""
        return list(cls._registered_agents.keys())
