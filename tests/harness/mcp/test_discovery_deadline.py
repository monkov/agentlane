"""Catalog lock waits must use the caller's discovery deadline."""

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPDiscoveryError,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import acquire_lease, http_server


@pytest.mark.asyncio
async def test_http_discovery_lock_wait_times_out_without_cancelling_first_request(
    unused_tcp_port: int,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    listed = 0

    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        nonlocal listed
        del context, params
        listed += 1
        if listed == 1:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return types.ListToolsResult(
            tools=[types.Tool(name="read", input_schema={"type": "object"})]
        )

    sdk_server = Server("discovery-deadline", on_list_tools=list_tools)
    app = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    server = MCPServer(
        name="notes",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{unused_tcp_port}/mcp", allow_insecure_http=True
        ),
        discovery_timeout_seconds=3,
    )
    async with (
        http_server(app, unused_tcp_port),
        mcp_client.MCPClientManager() as manager,
    ):
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="identity")
        )
        connection = lease._entry.connection  # pyright: ignore[reportPrivateUsage]
        assert connection is not None
        owner = connection.owner_task
        assert owner is not None

        first = asyncio.create_task(lease.tools())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            # Give the waiter a shorter budget after the first request starts.
            # Its timeout must not depend on the first request's deadline.
            connection.server = replace(server, discovery_timeout_seconds=0.05)
            second = asyncio.create_task(lease.tools())
            try:
                _, pending = await asyncio.wait((second,), timeout=0.5)
                assert not pending, "The catalog lock wait exceeded the deadline."
                with pytest.raises(MCPDiscoveryError) as error:
                    await second
                assert error.value.failure_kind == "timeout"
                assert error.value.retryable
                assert not first.done()
                assert not cancelled.is_set()
                assert listed == 1
                assert connection.catalog_lock.locked()
                assert not owner.done()
                assert not connection.failed
            finally:
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
                connection.server = server
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(first, return_exceptions=True), timeout=1
            )

        assert [tool.name for tool in first.result()] == ["notes__read"]
        assert not connection.catalog_lock.locked()
        assert [tool.name for tool in await lease.tools()] == ["notes__read"]
        assert listed == 2
        assert connection.owner_task is owner
        assert not owner.done()
        assert not cancelled.is_set()

    assert owner.done()
