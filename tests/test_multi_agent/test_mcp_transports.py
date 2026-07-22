"""Tests for StdioTransport and SseTransport MCP client implementations."""

from vajra_agent import SseTransport, StdioTransport


def test_mcp_transports():
    stdio = StdioTransport("python", ["--version"])
    stdio.connect()
    res_stdio = stdio.send_request("tools/list")
    assert "tools" in res_stdio["result"]
    stdio.close()

    sse = SseTransport("http://localhost:8080/sse")
    sse.connect()
    res_sse = sse.send_request("tools/list")
    assert "tools" in res_sse["result"]
    sse.close()
