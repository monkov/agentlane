"""Small stdio MCP server used by the harness integration tests."""

from mcp.server import MCPServer

server = MCPServer("agentlane-test")


@server.tool()
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


if __name__ == "__main__":
    server.run("stdio")
