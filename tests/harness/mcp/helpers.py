"""Shared lifecycle support for MCP tests."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
import uvicorn
from starlette.types import ASGIApp

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPError,
    MCPServer,
    MCPToolsShim,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._client import MCPClientLease, MCPClientManager
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.mcp._shim import (
    _BoundMCPToolsShim,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.shims import BoundShim, ShimBindingContext


class _RunTestManager(MCPClientManager):
    """Give each bound run its own lifetime over shared fake source behavior."""

    def __init__(self, source: MCPClientManager) -> None:
        super().__init__()
        self._source = source
        self._leases: list[MCPClientLease] = []

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        lease = await self._source._acquire(server, context)
        self._leases.append(lease)
        return lease

    async def aclose(self) -> None:
        leases, self._leases = self._leases, []
        try:
            await asyncio.gather(*(lease.release() for lease in leases))
        finally:
            await super().aclose()


class ManagedMCPToolsShim(MCPToolsShim):
    """Bind fake source behavior without adding manager injection to the API."""

    def __init__(
        self,
        *,
        servers: tuple[MCPServer, ...],
        client_manager: MCPClientManager | None = None,
        max_concurrent_discoveries: int = 8,
    ) -> None:
        super().__init__(
            servers=servers, max_concurrent_discoveries=max_concurrent_discoveries
        )
        self._test_manager = client_manager

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        bound = await super().bind(context)
        if isinstance(bound, _BoundMCPToolsShim) and self._test_manager is not None:
            bound.manager = _RunTestManager(self._test_manager)
        return bound


async def acquire_lease(
    manager: MCPClientManager,
    server: MCPServer,
    context: MCPAuthorizationContext,
) -> MCPClientLease:
    """Access the internal lease boundary in connection and transport tests."""
    return await manager._acquire(  # pyright: ignore[reportPrivateUsage]
        server, context
    )


class ConnectionInstaller:
    """Replace SDK startup while keeping the connection owner's lifecycle."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    def __call__(self, prepare: Callable[[MCPConnection], Awaitable[None]]) -> None:
        async def run_connection(connection: MCPConnection) -> None:
            try:
                await prepare(connection)
                connection.ready.set_result(None)
                await connection.stop_event.wait()
            except BaseException as exc:
                connection.failed = True
                if not connection.ready.done():
                    error = (
                        MCPError("Cancelled")
                        if isinstance(exc, asyncio.CancelledError)
                        else exc
                    )
                    connection.ready.set_exception(error)
                    connection.ready.exception()
            finally:
                connection.closing = True

        self._monkeypatch.setattr(mcp_client, "_run_connection", run_connection)


@asynccontextmanager
async def http_server(app: ASGIApp, port: int) -> AsyncIterator[None]:
    """Run a local ASGI peer with bounded startup and shutdown."""
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", lifespan="on"
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError("The local MCP HTTP peer stopped at startup.")
                await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
