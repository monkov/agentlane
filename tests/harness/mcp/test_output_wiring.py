"""Validate optional MCP result storage configuration."""

from pathlib import Path

import pytest

from agentlane.harness.filesystem import LocalFileSystem
from agentlane.harness.mcp import MCPServer, MCPStdioTransport, MCPToolsShim


@pytest.mark.parametrize("writer_enabled,directory", [(True, None), (False, "results")])
def test_storage_requires_writer_and_directory_together(
    tmp_path: Path, writer_enabled: bool, directory: str | None
) -> None:
    with pytest.raises(ValueError, match="configured together"):
        MCPToolsShim(
            servers=(MCPServer(name="notes", transport=MCPStdioTransport("fixture")),),
            result_writer=LocalFileSystem(tmp_path) if writer_enabled else None,
            result_directory=directory,
        )


@pytest.mark.parametrize(
    "directory", ["", " ", "../results", "/results", "a\\b", "a\x00b"]
)
def test_storage_rejects_invalid_relative_directory(
    tmp_path: Path, directory: str
) -> None:
    with pytest.raises(ValueError):
        MCPToolsShim(
            servers=(MCPServer(name="notes", transport=MCPStdioTransport("fixture")),),
            result_writer=LocalFileSystem(tmp_path),
            result_directory=directory,
        )

    assert not tuple(tmp_path.iterdir())
