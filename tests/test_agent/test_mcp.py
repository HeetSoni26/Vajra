"""Tests for MCP interface abstractions and ToolAdapter."""

from typing import Any

from vajra_agent.mcp import MCPClient, ToolAdapter, Transport


class MockTransport(Transport):
    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send_message(self, message: dict[str, Any]) -> None:
        pass

    def receive_message(self) -> dict[str, Any]:
        return {}


class MockMCPClient(MCPClient):
    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "external_tool", "description": "An MCP tool", "parameters": {}}]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return {"result": f"Executed MCP tool {name} with {arguments}"}


def test_mcp_tool_adapter():
    transport = MockTransport()
    client = MockMCPClient(transport)

    tools = client.list_tools()
    adapter = ToolAdapter(client, tools[0])

    assert adapter.name == "external_tool"
    assert adapter.description == "An MCP tool"

    res = adapter.run(param="val")
    assert res.success
    assert "Executed MCP tool external_tool" in res.output["result"]
