"""Discover a complete tool catalog under one authorization generation."""

import asyncio
import time
from dataclasses import replace
from typing import Any, cast

import structlog

from ._connection import MCPConnection
from ._errors import MCPAuthorizationError, MCPDiscoveryError
from ._operation import MCPOperation, mcp_operation
from ._redaction import contains_secret_key, redact_known_secrets
from ._sdk import exception_kind, read_tool_page
from ._tools import reject_normalized_collisions
from ._types import MCPCatalog, MCPRemoteTool

logger = structlog.get_logger(__name__)


async def get_catalog(connection: MCPConnection) -> MCPCatalog:
    """Fetch every page without retaining a previous discovery result."""
    started_at = time.monotonic()
    with mcp_operation() as operation:
        try:
            # Lock contention and all discovery attempts share one deadline.
            async with (
                asyncio.timeout(connection.server.discovery_timeout_seconds),
                connection.catalog_lock,
            ):
                snapshot = await _discover_catalog(connection, operation)
                if (
                    snapshot.authorization_generation
                    != connection.authorization.generation
                ):
                    raise MCPAuthorizationError(
                        "MCP authorization changed during discovery."
                    )
        except Exception as exc:
            failure_kind = operation.failure_kind or exception_kind(exc)
            if failure_kind == "authorization":
                connection.authorization.reject()
                raise MCPAuthorizationError(
                    f"Authorization failed for MCP server {connection.server.name!r}."
                ) from None
            if failure_kind == "transport":
                connection.failed = True
                raise ConnectionError("MCP discovery transport failed.") from None
            raise
        finally:
            connection.secrets.update(operation.secrets)
    logger.info(
        "mcp_catalog_discovered",
        server=connection.server.name,
        tool_count=len(snapshot.tools),
        duration=time.monotonic() - started_at,
        protocol_version=connection.protocol_version,
        status="ok",
    )
    return snapshot


class _AuthorizationChanged(Exception):
    """Discard pages collected across credential generations."""


async def _discover_catalog(
    connection: MCPConnection, operation: MCPOperation
) -> MCPCatalog:
    # Only read-only discovery can restart after authorization changes. Both
    # attempts share the caller's timeout; tools/call is never replayed here.
    for _attempt in range(2):
        try:
            tools, generation = await _list_all_tools(connection, operation)
        except _AuthorizationChanged:
            continue
        filtered = tuple(
            _safe_remote_tool(item, connection.secrets)
            for item in tools
            if connection.server.tools.allows(item.name)
        )
        reject_normalized_collisions(connection.server.name, filtered)
        return MCPCatalog(
            tools=filtered,
            authorization_generation=generation,
        )
    raise MCPAuthorizationError("MCP authorization changed during discovery.")


async def _list_all_tools(
    connection: MCPConnection, operation: MCPOperation
) -> tuple[tuple[MCPRemoteTool, ...], int]:
    tools: list[MCPRemoteTool] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    authorization_generation: int | None = None
    while True:
        request_generation = connection.authorization.generation
        operation.authorization_generation = None
        result = read_tool_page(
            await connection.get_client().list_tools(cursor=cursor, cache_mode="bypass")
        )
        if operation.authorization_generation is not None:
            request_generation = operation.authorization_generation
        # The first page establishes the credentials actually sent. A provider
        # may select a new token between preflight and that first HTTP request.
        if connection.authorization.generation != request_generation or (
            authorization_generation is not None
            and request_generation != authorization_generation
        ):
            raise _AuthorizationChanged
        authorization_generation = request_generation
        tools.extend(result.tools)
        cursor = result.next_cursor
        if cursor is None:
            return tuple(tools), authorization_generation
        if cursor in seen_cursors:
            raise MCPDiscoveryError(
                "MCP tool discovery returned a repeated page cursor."
            )
        seen_cursors.add(cursor)


def _safe_remote_tool(tool: MCPRemoteTool, secrets: set[str]) -> MCPRemoteTool:
    if any(secret in tool.name for secret in secrets) or contains_secret_key(
        tool.input_schema, secrets
    ):
        raise MCPDiscoveryError("MCP tool schema contains a credential in a name.")
    return replace(
        tool,
        description=cast(str | None, redact_known_secrets(tool.description, secrets)),
        input_schema=cast(
            dict[str, Any], redact_known_secrets(tool.input_schema, secrets)
        ),
    )
