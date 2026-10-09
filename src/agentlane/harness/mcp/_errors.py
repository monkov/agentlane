"""Errors raised at MCP lifecycle and authorization boundaries."""

from strenum import LowercaseStrEnum


class MCPFailureKind(LowercaseStrEnum):
    """Failure categories used by MCP connection and discovery policy."""

    AUTHORIZATION = "authorization"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"


class MCPError(RuntimeError):
    """Base error for MCP discovery and lifecycle failures."""


class MCPDependencyError(MCPError):
    """The optional MCP SDK is not installed."""


class MCPShutdownTimeoutError(MCPError, TimeoutError):
    """The close deadline passed while owned cleanup continues in the background."""


class MCPDiscoveryError(MCPError):
    """An MCP server tool catalog could not be discovered."""

    def __init__(
        self, message: str, *, failure_kind: MCPFailureKind = MCPFailureKind.PROTOCOL
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind

    @property
    def retryable(self) -> bool:
        """Return whether a later read-only discovery attempt can recover."""
        return self.failure_kind in {
            MCPFailureKind.TRANSPORT,
            MCPFailureKind.TIMEOUT,
        }


class MCPAuthorizationError(MCPError):
    """The product authorization provider could not authorize a request."""
