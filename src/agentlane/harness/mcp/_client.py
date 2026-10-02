"""Bounded identity-isolated connection pool and per-run MCP leases."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Self

import structlog

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
    MCPPoolCapacityError,
    MCPShutdownTimeoutError,
)
from ._operation import mcp_operation
from ._sdk import exception_kind
from ._tools import call_tool, native_tool, tool_failure
from ._types import (
    MCPAuthorizationContext,
    MCPCatalog,
    MCPClientLimits,
    MCPServer,
    MCPStdioTransport,
    MCPStreamableHTTPTransport,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MCPConnectionKey:
    server_name: str
    transport: tuple[object, ...]
    authorization_context_key: str
    authorization_provider_id: int | None
    policy: tuple[object, ...]


@dataclass(slots=True, eq=False)
class _PoolEntry:
    key: MCPConnectionKey
    server: MCPServer
    context: MCPAuthorizationContext
    authorization: MCPAuthorizationState = field(default_factory=MCPAuthorizationState)
    connection: MCPConnection | None = None
    connection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    discovery_tasks: set[asyncio.Task[MCPCatalog]] = field(
        default_factory=set[asyncio.Task[MCPCatalog]]
    )
    leases: int = 0
    calls: int = 0
    idle_since: float = 0
    idle_task: asyncio.Task[None] | None = None


class _LeaseManager(Protocol):
    @property
    def closed(self) -> bool: ...

    async def _connection(self, entry: _PoolEntry) -> MCPConnection: ...

    async def _release(self, entry: _PoolEntry) -> None: ...

    async def _call_finished(self, entry: _PoolEntry) -> None: ...


class MCPClientLease:
    """One run's access to a managed connection and last good catalog."""

    def __init__(self, manager: _LeaseManager, entry: _PoolEntry) -> None:
        self._manager = manager
        self._entry = entry
        self._released = False
        self._release_task: asyncio.Task[None] | None = None
        self._catalog: MCPCatalog | None = None

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
                self._catalog = None
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
            await self._manager._call_finished(  # pyright: ignore[reportPrivateUsage]
                self._entry
            )

    async def _tools(self) -> MCPCatalog:
        try:
            # Validate credentials even when reconnecting. A failed open must
            # never restore a catalog from a previous authorization generation.
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
            catalog = await get_catalog(connection)
        except MCPAuthorizationError:
            self._catalog = None
            raise
        except Exception as exc:
            if isinstance(exc, MCPDiscoveryError) and not exc.retryable:
                self._catalog = None
                raise
            failure_kind = exception_kind(exc)
            if (
                self._catalog is None
                or self._manager.closed
                or self._catalog.authorization_generation
                != self._entry.authorization.generation
                or failure_kind not in {"transport", "timeout"}
            ):
                raise MCPDiscoveryError(
                    f"Could not discover tools from MCP server {self.server.name!r}.",
                    failure_kind=failure_kind,
                ) from None
            logger.warning(
                "mcp_catalog_refresh_failed",
                server=self.server.name,
                status="stale",
                failure_kind=failure_kind,
            )
            return self._catalog

        self._catalog = catalog
        return catalog

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

        self._entry.calls += 1
        deadline = asyncio.get_running_loop().time() + self.server.tool_timeout_seconds
        try:
            # Resolve the live generation once before dispatch. Never replay a
            # call after its request may have reached a remote side-effect.
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
        finally:
            self._entry.calls -= 1
            await self._manager._call_finished(  # pyright: ignore[reportPrivateUsage]
                self._entry
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
    """Own a finite identity- and policy-isolated pool across harness runs."""

    def __init__(self, limits: MCPClientLimits | None = None) -> None:
        self._limits = limits or MCPClientLimits()
        self._connections: dict[MCPConnectionKey, _PoolEntry] = {}
        self._retiring: dict[_PoolEntry, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether application shutdown has started."""
        return self._closed

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        await self.aclose()

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        key = _connection_key(server, context)
        while True:
            retirement: asyncio.Task[None] | None = None
            async with self._lock:
                if self._closed:
                    raise MCPError("MCP client manager is closed.")
                entry = self._connections.get(key)
                if entry is None:
                    if (
                        len(self._connections) + len(self._retiring)
                        >= self._limits.max_connections
                    ):
                        idle = [
                            candidate
                            for candidate in self._connections.values()
                            if _is_idle(candidate)
                        ]
                        if not idle:
                            raise MCPPoolCapacityError()
                        retirement = self._retire_locked(
                            min(idle, key=lambda candidate: candidate.idle_since)
                        )
                    else:
                        entry = _PoolEntry(key=key, server=server, context=context)
                        self._connections[key] = entry
                if entry is not None:
                    entry.leases += 1
                    self._cancel_idle_locked(entry)
                    break
            if retirement is not None:
                # A closing transport still occupies capacity. Do not start its
                # replacement until the owner has released all SDK resources.
                await asyncio.shield(retirement)

        lease = MCPClientLease(self, entry)
        try:
            await self._connection(entry)
        except BaseException:
            await lease.release()
            raise
        return lease

    async def _connection(self, entry: _PoolEntry) -> MCPConnection:
        async with entry.connection_lock:
            if self._closed:
                raise MCPError("MCP client manager is closed.")
            connection = entry.connection
            if connection is not None and (
                connection.failed
                or connection.closing
                or (connection.owner_task is not None and connection.owner_task.done())
            ):
                await close_connection(connection)
                connection = None
            if connection is None:
                if self._closed:
                    raise MCPError("MCP client manager is closed.")
                connection = MCPConnection(
                    server=entry.server,
                    context=entry.context,
                    ready=asyncio.get_running_loop().create_future(),
                    authorization=entry.authorization,
                    secrets=entry.authorization.secrets,
                    shutdown_timeout_seconds=self._limits.shutdown_timeout_seconds,
                )
                entry.connection = connection
                connection.owner_task = asyncio.create_task(
                    _run_connection(connection),
                    name=f"agentlane-mcp-{entry.server.name}",
                )
            await asyncio.shield(connection.ready)
            if self._closed or connection.closing:
                raise MCPError("MCP client manager closed while opening a connection.")
            return connection

    async def aclose(self) -> None:
        """Close connections or report a deadline while owned cleanup continues."""
        async with self._lock:
            self._closed = True
            for entry in tuple(self._connections.values()):
                self._retire_locked(entry)
            retirements = tuple(self._retiring.values())
        if not retirements:
            return

        completed, pending = await asyncio.wait(
            retirements, timeout=self._limits.shutdown_timeout_seconds
        )
        if pending:
            # Do not cancel SDK process reaping to satisfy the caller's deadline.
            # A later aclose() observes the same owned cleanup tasks.
            raise MCPShutdownTimeoutError(
                "MCP shutdown timed out; connection cleanup is still in progress."
            )
        await asyncio.gather(*completed)

    async def _release(self, entry: _PoolEntry) -> None:
        retirement: asyncio.Task[None] | None = None
        async with self._lock:
            entry.leases -= 1
            if entry.leases == 0 and self._connections.get(entry.key) is entry:
                entry.idle_since = time.monotonic()
                connection = entry.connection
                if _is_idle(entry) and (
                    connection is None
                    or not connection.ready.done()
                    or connection.failed
                ):
                    retirement = self._retire_locked(entry)
                else:
                    self._schedule_idle_locked(entry)
        if retirement is not None:
            await asyncio.shield(retirement)

    async def _call_finished(self, entry: _PoolEntry) -> None:
        async with self._lock:
            if _is_idle(entry) and self._connections.get(entry.key) is entry:
                self._schedule_idle_locked(entry)

    def _schedule_idle_locked(self, entry: _PoolEntry) -> None:
        if entry.idle_task is None and not self._closed:
            entry.idle_task = asyncio.create_task(
                self._expire_idle(entry), name=f"agentlane-mcp-idle-{entry.server.name}"
            )

    async def _expire_idle(self, entry: _PoolEntry) -> None:
        await asyncio.sleep(self._limits.idle_timeout_seconds)
        async with self._lock:
            entry.idle_task = None
            if self._connections.get(entry.key) is not entry or not _is_idle(entry):
                return
            retirement = self._retire_locked(entry)
        await asyncio.shield(retirement)

    def _cancel_idle_locked(self, entry: _PoolEntry) -> None:
        if entry.idle_task is not None:
            if entry.idle_task is not asyncio.current_task():
                entry.idle_task.cancel()
            entry.idle_task = None

    def _retire_locked(self, entry: _PoolEntry) -> asyncio.Task[None]:
        if entry in self._retiring:
            return self._retiring[entry]
        if self._connections.get(entry.key) is entry:
            del self._connections[entry.key]
        self._cancel_idle_locked(entry)
        task = asyncio.create_task(
            self._dispose_entry(entry), name=f"agentlane-mcp-retire-{entry.server.name}"
        )
        self._retiring[entry] = task
        return task

    async def _dispose_entry(self, entry: _PoolEntry) -> None:
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
                self._retiring.pop(entry, None)


def _is_idle(entry: _PoolEntry) -> bool:
    return entry.leases == 0 and entry.calls == 0 and not entry.discovery_tasks


def _connection_key(
    server: MCPServer, context: MCPAuthorizationContext
) -> MCPConnectionKey:
    return MCPConnectionKey(
        server_name=server.name,
        transport=_transport_key(server.transport),
        authorization_context_key=context.key,
        authorization_provider_id=(
            id(server.authorization) if server.authorization is not None else None
        ),
        policy=(
            server.tools.include,
            server.tools.exclude,
            server.result_policy,
            server.required,
            server.connect_timeout_seconds,
            server.discovery_timeout_seconds,
            server.tool_timeout_seconds,
            server.catalog_ttl_seconds,
        ),
    )


def _transport_key(
    transport: MCPStreamableHTTPTransport | MCPStdioTransport,
) -> tuple[object, ...]:
    if isinstance(transport, MCPStreamableHTTPTransport):
        return (
            "http",
            transport.url,
            transport.allow_insecure_http,
            transport.connect_timeout_seconds,
            transport.read_timeout_seconds,
        )
    return (
        "stdio",
        transport.command,
        transport.args,
        tuple(sorted(transport.env.items())) if transport.env is not None else None,
        transport.cwd,
    )
