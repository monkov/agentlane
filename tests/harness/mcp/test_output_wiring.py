"""Output storage options follow refreshed and inherited MCP tool sources."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agentlane.harness import RunState, Task
from agentlane.harness.filesystem import FileWriter, LocalFileSystem, MountedFileSystem
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPServer,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._client import MCPClientLease
from agentlane.harness.shims import BoundShim, PreparedTurn, ShimBindingContext
from agentlane.models import Tool
from agentlane.models.run import RunContext
from agentlane.runtime import SingleThreadedRuntimeEngine

from ..tools_test_utils import echo_tool


@dataclass
class _Options:
    writer: FileWriter | None
    directory: str | None
    selected: frozenset[tuple[str, str]]


@dataclass
class _Lease(MCPClientLease):
    observed: list[_Options] = field(default_factory=list[_Options])
    released: bool = False

    async def tools(
        self,
        *,
        result_writer: FileWriter | None = None,
        result_directory: str | None = None,
        preserve_tools: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[Tool[Any, Any], ...]:
        assert not self.released
        self.observed.append(_Options(result_writer, result_directory, preserve_tools))

        return echo_tool("notes__read"), echo_tool("notes__write")

    async def release(self) -> None:
        self.released = True


class _Manager(MCPClientManager):
    def __init__(self) -> None:
        super().__init__()
        self.leases: list[_Lease] = []

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        del server, context
        lease = _Lease()
        self.leases.append(lease)

        return lease


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
async def test_storage_options_survive_refresh_and_restricted_inheritance(
    tmp_path: Path, mounted: bool
) -> None:
    manager = _Manager()
    selected = frozenset({("notes", "read")})
    writer = (
        MountedFileSystem({"results": LocalFileSystem(tmp_path)})
        if mounted
        else LocalFileSystem(tmp_path)
    )
    expected_directory = "/results/tool-results" if mounted else "results/tool-results"
    shim = MCPToolsShim(
        servers=(MCPServer(name="notes", transport=MCPStdioTransport("fixture")),),
        client_manager=manager,
        result_writer=writer,
        result_directory="results/./sessions/../tool-results",
        preserve_tools=selected,
    )
    binding = ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))
    state = RunState(instructions=None, history=[], responses=[])
    context = RunContext[object](context=None)
    parent = await shim.bind(binding)
    child: BoundShim | None = None
    try:
        await parent.on_run_start(state, context)
        for _ in range(2):
            await parent.prepare_turn(
                PreparedTurn(run_state=state, tools=None, model_args=None)
            )

        # A child keeps the writer and directory without restoring excluded tools.
        inherited = parent.inherit_tools(frozenset({"notes__read"}))
        assert inherited is not None and len(inherited) == 1
        child = await inherited[0].bind(binding)
        await child.on_run_start(state, context)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await child.prepare_turn(turn)

        assert turn.tools is not None
        assert {tool.name for tool in turn.tools.normalized_tools} == {"notes__read"}
        assert [len(lease.observed) for lease in manager.leases] == [3, 2]
        for lease in manager.leases:
            assert all(options.writer is writer for options in lease.observed)
            assert all(
                options.directory == expected_directory for options in lease.observed
            )
            assert all(options.selected == selected for options in lease.observed)
    finally:
        if child is not None:
            await child.on_run_end(None, context)
        await parent.on_run_end(None, context)
        await manager.aclose()

    assert all(lease.released for lease in manager.leases)


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
