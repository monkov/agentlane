"""Lost HTTP subscriptions must not leave a live but stale catalog."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.subscriptions import SUBSCRIPTION_ID_META_KEY
from mcp.shared.exceptions import MCPError
from starlette.types import Message, Receive, Scope, Send

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp import (
    _connection as mcp_connection,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import acquire_lease, http_server


@dataclass
class _SubscriptionPeer:
    tool_name: str = "initial"
    list_calls: int = 0
    listen_calls: int = 0
    acknowledged: asyncio.Event = field(default_factory=asyncio.Event)
    drop: asyncio.Event = field(default_factory=asyncio.Event)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    finish: asyncio.Event = field(default_factory=asyncio.Event)


@asynccontextmanager
async def _subscription_peer(
    port: int, peer: _SubscriptionPeer, *, legacy: bool = False
) -> AsyncIterator[MCPServer]:
    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del context, params
        peer.list_calls += 1
        return types.ListToolsResult(
            tools=[types.Tool(name=peer.tool_name, input_schema={"type": "object"})],
            ttl_ms=300_000,
        )

    async def listen(
        context: ServerRequestContext[Any],
        params: types.SubscriptionsListenRequestParams,
    ) -> types.SubscriptionsListenResult:
        peer.listen_calls += 1
        attempt = peer.listen_calls
        meta = {SUBSCRIPTION_ID_META_KEY: context.request_id}
        await context.session.send_notification(
            types.SubscriptionsAcknowledgedNotification(
                params=types.SubscriptionsAcknowledgedNotificationParams(
                    notifications=params.notifications, _meta=meta
                )
            ),
            related_request_id=context.request_id,
        )
        peer.acknowledged.set()

        if attempt == 1:
            await peer.drop.wait()
            raise MCPError(types.CONNECTION_CLOSED, "Subscription transport closed.")

        await peer.changed.wait()
        await context.session.send_notification(
            types.ToolListChangedNotification(
                params=types.NotificationParams(_meta=meta)
            ),
            related_request_id=context.request_id,
        )
        await peer.finish.wait()
        return types.SubscriptionsListenResult(_meta=meta)

    sdk_server = Server(
        "subscription-recovery",
        on_list_tools=list_tools,
        on_subscriptions_listen=listen,
    )
    inner = sdk_server.streamable_http_app(stateless_http=not legacy, host="127.0.0.1")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if not legacy or scope["type"] != "http" or scope["method"] != "POST":
            await inner(scope, receive, send)
            return

        buffered: list[Message] = []
        while True:
            message = await receive()
            buffered.append(message)
            if not message.get("more_body", False):
                break
        payload = json.loads(b"".join(message.get("body", b"") for message in buffered))
        if payload.get("method") == "server/discover":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Legacy server."})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await inner(scope, replay, send)

    async with http_server(app, port):
        try:
            yield MCPServer(
                name="notes",
                transport=MCPStreamableHTTPTransport(
                    url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
                ),
            )
        finally:
            peer.drop.set()
            peer.changed.set()
            peer.finish.set()


def _connection(lease: mcp_client.MCPClientLease) -> mcp_connection.MCPConnection:
    connection = lease._entry.connection  # pyright: ignore[reportPrivateUsage]
    assert connection is not None
    return connection


@pytest.mark.asyncio
async def test_lost_http_subscription_reconnects_refetches_and_resubscribes(
    unused_tcp_port: int,
) -> None:
    peer = _SubscriptionPeer()
    async with (
        _subscription_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
        asyncio.timeout(5),
    ):
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="user")
        )
        assert [tool.name for tool in await lease.tools()] == ["notes__initial"]
        await peer.acknowledged.wait()
        original = _connection(lease)
        listener = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "agentlane-mcp-listen-notes"
        )

        peer.tool_name = "replacement"
        peer.acknowledged.clear()
        peer.drop.set()
        await asyncio.shield(listener)
        assert original.failed

        assert [tool.name for tool in await lease.tools()] == ["notes__replacement"]
        await peer.acknowledged.wait()
        replacement = _connection(lease)
        assert replacement is not original
        assert original.owner_task is not None and original.owner_task.done()
        assert peer.list_calls == 2
        assert peer.listen_calls == 2

        peer.tool_name = "notified"
        revision = replacement.catalog_revision
        peer.changed.set()
        while replacement.catalog_revision == revision:
            await asyncio.sleep(0)
        assert [tool.name for tool in await lease.tools()] == ["notes__notified"]
        assert peer.list_calls == 3
        assert peer.listen_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_legacy_http_without_listen_keeps_its_connection_and_catalog(
    unused_tcp_port: int,
) -> None:
    peer = _SubscriptionPeer()
    async with (
        _subscription_peer(unused_tcp_port, peer, legacy=True) as server,
        MCPClientManager() as manager,
        asyncio.timeout(5),
    ):
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="user")
        )
        assert [tool.name for tool in await lease.tools()] == ["notes__initial"]
        original = _connection(lease)
        assert original.protocol_version == "2025-11-25"
        assert not original.failed
        assert [tool.name for tool in await lease.tools()] == ["notes__initial"]
        assert _connection(lease) is original
        assert peer.list_calls == 1
        assert peer.listen_calls == 0
        await lease.release()
