"""Scope SDK logging and failure metadata to one MCP operation."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ._errors import MCPFailureKind


@dataclass(slots=True)
class MCPOperation:
    """Non-persisted outcome metadata for one SDK operation."""

    failure_kind: MCPFailureKind | None = None
    http_status: int | None = None
    retry_reason: str | None = None
    authorization_generation: int | None = None
    expected_authorization_generation: int | None = None
    """Authorization required before the first HTTP tool request is sent."""
    secrets: set[str] = field(default_factory=set[str], repr=False)


_operation: ContextVar[MCPOperation | None] = ContextVar(
    "agentlane_mcp_operation", default=None
)


def current_mcp_operation() -> MCPOperation | None:
    """Return metadata for the current request and its SDK worker tasks."""
    return _operation.get()


class _SDKLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        del record
        return current_mcp_operation() is None


_sdk_log_filter = _SDKLogFilter()
_SDK_LOG_NAMES = (
    "mcp.client.client",
    "mcp.client.streamable_http",
    "mcp.client.stdio",
    "mcp.client.caching",
    "mcp.shared.direct_dispatcher",
    "mcp.shared.jsonrpc_dispatcher",
    "mcp.shared.dispatcher",
    "mcp.shared.tool_name_validation",
    "client",
    "httpx2",
    "httpcore2.connection",
    "httpcore2.http11",
    "httpcore2.http2",
    "httpcore2.proxy",
    "httpcore2.socks",
)


@contextmanager
def mcp_operation() -> Iterator[MCPOperation]:
    """Scope SDK logs and error metadata to one Agent Lane operation.

    The SDK carries the sender's context into its HTTP worker tasks. Each call
    therefore has separate failure metadata, including concurrent calls.
    Logger filters affect only this scope and leave application configuration
    and other SDK clients unchanged.
    """
    for name in _SDK_LOG_NAMES:
        logging.getLogger(name).addFilter(_sdk_log_filter)

    operation = MCPOperation()
    token = _operation.set(operation)
    try:
        yield operation
    finally:
        _operation.reset(token)
