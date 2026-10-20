"""Keep one tool call pending while other calls use the same process."""

import asyncio
import os
from pathlib import Path

from mcp.server import MCPServer

Path(os.environ["MCP_TEST_PID"]).write_text(str(os.getpid()))
server = MCPServer("wait-test")


@server.tool()
async def wait() -> str:
    Path(os.environ["MCP_WAIT_STARTED"]).write_text("started")
    await asyncio.sleep(60)
    return "finished"


@server.tool()
def ping() -> str:
    return "pong"


if __name__ == "__main__":
    server.run("stdio")
