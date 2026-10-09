"""Own independent MCP connections for one bound harness run."""

import asyncio
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol, Self

from agentlane.models import Tool, ToolFailure
from agentlane.runtime import CancellationToken

from ._auth import MCPAuthorizationState
from ._catalog import get_catalog
from ._connection import MCPConnection, close_connection
from ._connection import run_connection as _run_connection
from ._errors import (
    MCPAuthorizationError,
    MCPDiscoveryError,
    MCPError,
    MCPShutdownTimeoutError,
)
from ._operation import mcp_operation
from ._sdk import exception_kind
from ._tools import call_tool, native_tool, tool_failure
from ._types import MCPAuthorizationContext, MCPCatalog, MCPServer


@dataclass(slots=True, eq=False)
class _ClientEntry:
    server: MCPServer
    context: MCPAuthorizationContext
    authorization: MCPAuthorizationState = field(default_factory=MCPAuthorizationState)
    connection: MCPConnection | None = None
    connection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    discovery_tasks: set[asyncio.Task[MCPCatalog]] = field(
        default_factory=set[asyncio.Task[MCPCatalog]]
    )
    released: bool = False


class _LeaseManager(Protocol):
    @property
    def closed(self) -> bool: ...

    async def _connection(self, entry: _ClientEntry) -> MCPConnection: ...

    async def _release(self, entry: _ClientEntry) -> None: ...


class MCPClientLease:
    """Access one source connection owned by the current harness run."""

    def __init__(self, manager: _LeaseManager, entry: _ClientEntry) -> None:
        self._manager = manager
        self._entry = entry
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def server(self) -> MCPServer:
        return self._entry.server

    async def tools(self) -> tuple[Tool[Any, Any], ...]:
        if self._released:
            raise MCPError("MCP connection lease has been released.")
        if self._manager.closed:
            raise MCPError("MCP client manager is closed.")

        discovery = asyncio.create_task(
            self._tools(), name=f"agentlane-mcp-discovery-{self.server.name}"
        )
        self._entry.discovery_tasks.add(discovery)
        try:
            catalog = await discovery
            if catalog.authorization_generation != self._entry.authorization.generation:
                raise MCPAuthorizationError(
                    "MCP authorization changed during discovery."
                )

            async def dispatch(
                name: str, arguments: dict[str, Any], token: CancellationToken
            ) -> str | ToolFailure:
                return await self._call_tool(
                    catalog.authorization_generation, name, arguments, token
                )

            return tuple(
                native_tool(self.server, item, dispatch) for item in catalog.tools
            )
        finally:
            self._entry.discovery_tasks.discard(discovery)

    async def _tools(self) -> MCPCatalog:
        try:
            # Validate credentials before reconnecting and before discovery.
            # A provider failure must invalidate already-exposed tool bindings.
            with mcp_operation():
                try:
                    await asyncio.wait_for(
                        self._entry.authorization.get_token(
                            self.server, self._entry.context
                        ),
                        timeout=self.server.discovery_timeout_seconds,
                    )
                except TimeoutError:
                    self._entry.authorization.reject()
                    raise MCPAuthorizationError(
                        "MCP authorization timed out."
                    ) from None
            connection = (
                await self._manager._connection(  # pyright: ignore[reportPrivateUsage]
                    self._entry
                )
            )
            return await get_catalog(connection)
        except MCPAuthorizationError:
            raise
        except Exception as exc:
            if isinstance(exc, MCPDiscoveryError) and not exc.retryable:
                raise
            raise MCPDiscoveryError(
                f"Could not discover tools from MCP server {self.server.name!r}.",
                failure_kind=exception_kind(exc),
            ) from None

    async def _call_tool(
        self,
        generation: int,
        name: str,
        arguments: dict[str, Any],
        token: CancellationToken,
    ) -> str | ToolFailure:
        if token.is_cancelled:
            return tool_failure("MCP tool call was cancelled.", "cancelled")
        if self._released or self._manager.closed:
            return tool_failure("MCP connection is unavailable.", "mcp_transport")
        if generation != self._entry.authorization.generation:
            return tool_failure(
                "MCP authorization changed after discovery.", "mcp_authorization"
            )

        deadline = asyncio.get_running_loop().time() + self.server.tool_timeout_seconds
        # Resolve the live connection before dispatch. A request that may have
        # reached the server is never replayed after a transport failure.
        try:
            opening = asyncio.create_task(
                self._manager._connection(  # pyright: ignore[reportPrivateUsage]
                    self._entry
                )
            )
            token.link_future(opening)
            connection = await asyncio.wait_for(
                opening, timeout=self.server.tool_timeout_seconds
            )
        except asyncio.CancelledError:
            if token.is_cancelled:
                return tool_failure("MCP tool call was cancelled.", "cancelled")
            raise
        except Exception as exc:
            kind = exception_kind(exc)
            return tool_failure(
                f"MCP {kind} failure.",
                "timeout" if kind == "timeout" else f"mcp_{kind}",
            )
        if generation != self._entry.authorization.generation:
            return tool_failure(
                "MCP authorization changed after discovery.", "mcp_authorization"
            )
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return tool_failure("MCP timeout failure.", "timeout")
        return await call_tool(
            connection,
            name,
            arguments,
            token,
            lambda: self._manager.closed,
            remaining,
        )

    async def release(self) -> None:
        if self._release_task is None:
            self._released = True
            self._release_task = asyncio.create_task(
                self._manager._release(  # pyright: ignore[reportPrivateUsage]
                    self._entry
                ),
                name=f"agentlane-mcp-release-{self.server.name}",
            )
        await asyncio.shield(self._release_task)


class MCPClientManager:
    """Internal owner of independent source connections in one harness run."""

    def __init__(self, *, shutdown_timeout_seconds: float = 10.0) -> None:
        if not isfinite(shutdown_timeout_seconds) or shutdown_timeout_seconds <= 0:
            raise ValueError(
                "MCP shutdown timeout must be finite and greater than zero."
            )
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._entries: set[_ClientEntry] = set()
        self._closing: dict[_ClientEntry, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        await self.aclose()

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        async with self._lock:
            if self._closed:
                raise MCPError("MCP client manager is closed.")
            entry = _ClientEntry(server=server, context=context)
            self._entries.add(entry)

        lease = MCPClientLease(self, entry)
        try:
            await self._connection(entry)
        except BaseException as exc:
            try:
                await lease.release()
            except BaseException:
                exc.add_note("MCP connection cleanup also failed.")
            raise
        return lease

    async def _connection(self, entry: _ClientEntry) -> MCPConnection:
        async with entry.connection_lock:
            if self._closed or entry.released:
                raise MCPError("MCP connection is closed.")
            connection = entry.connection
            if connection is not None and (
                connection.failed
                or connection.closing
                or (connection.owner_task is not None and connection.owner_task.done())
            ):
                await close_connection(connection)
                connection = None
            if connection is None:
                if self._closed or entry.released:
                    raise MCPError("MCP connection is closed.")
                connection = MCPConnection(
                    server=entry.server,
                    context=entry.context,
                    ready=asyncio.get_running_loop().create_future(),
                    authorization=entry.authorization,
                    secrets=entry.authorization.secrets,
                    shutdown_timeout_seconds=self._shutdown_timeout_seconds,
                )
                entry.connection = connection
                connection.owner_task = asyncio.create_task(
                    _run_connection(connection),
                    name=f"agentlane-mcp-{entry.server.name}",
                )
            await asyncio.shield(connection.ready)
            if self._closed or entry.released or connection.closing:
                raise MCPError("MCP connection closed while opening.")
            return connection

    async def aclose(self) -> None:
        """Close all run sources concurrently without interrupting SDK reaping."""
        async with self._lock:
            self._closed = True
            for entry in tuple(self._entries):
                self._start_close_locked(entry)
            closing = tuple(self._closing.values())
        await self._wait_for_close(closing)

    async def _release(self, entry: _ClientEntry) -> None:
        async with self._lock:
            if entry.released and entry not in self._closing:
                return
            closing = self._start_close_locked(entry)
        await self._wait_for_close((closing,))

    async def _wait_for_close(self, closing: tuple[asyncio.Task[None], ...]) -> None:
        if not closing:
            return
        completed, pending = await asyncio.wait(
            closing, timeout=self._shutdown_timeout_seconds
        )
        if pending:
            raise MCPShutdownTimeoutError(
                "MCP shutdown timed out; connection cleanup is still in progress."
            )
        await asyncio.gather(*completed)

    def _start_close_locked(self, entry: _ClientEntry) -> asyncio.Task[None]:
        if entry in self._closing:
            return self._closing[entry]
        entry.released = True
        self._entries.discard(entry)
        task = asyncio.create_task(
            self._dispose_entry(entry), name=f"agentlane-mcp-close-{entry.server.name}"
        )
        self._closing[entry] = task
        return task

    async def _dispose_entry(self, entry: _ClientEntry) -> None:
        try:
            discoveries = tuple(entry.discovery_tasks)
            for task in discoveries:
                task.cancel()
            try:
                if entry.connection is not None:
                    await close_connection(entry.connection)
            finally:
                await asyncio.gather(*discoveries, return_exceptions=True)
        finally:
            async with self._lock:
                self._closing.pop(entry, None)
