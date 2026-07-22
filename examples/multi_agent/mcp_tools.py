"""MCP transports demonstration."""

from vajra_agent import SseTransport, StdioTransport


def main():
    stdio = StdioTransport("python", ["--version"])
    stdio.connect()
    res_stdio = stdio.send_request("tools/list")
    print("STDIO MCP Tools Discovered:", res_stdio["result"]["tools"])
    stdio.close()

    sse = SseTransport("http://localhost:8080/sse")
    sse.connect()
    res_sse = sse.send_request("tools/list")
    print("SSE MCP Tools Discovered:", res_sse["result"]["tools"])
    sse.close()


if __name__ == "__main__":
    main()
