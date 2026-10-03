"""Record a tool call before the process exits without a response."""

import os
import sys
from pathlib import Path

from mcp.server import MCPServer

server = MCPServer("crash-test")

if canary := os.environ.get("MCP_STDERR_CANARY"):
    print(canary, file=sys.stderr, flush=True)


@server.tool()
def crash() -> str:
    path = Path(os.environ["MCP_CALL_COUNT"])
    count = int(path.read_text()) if path.exists() else 0
    path.write_text(str(count + 1))
    os._exit(0)


@server.tool()
def ping() -> str:
    return "pong"


if __name__ == "__main__":
    server.run("stdio")
