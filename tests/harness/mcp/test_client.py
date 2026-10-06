"""MCP transport, authorization, and native tool integration."""

import asyncio
import inspect
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from mcp import types
from mcp.server import MCPServer as SDKServer

from agentlane.harness import RunState, Task
from agentlane.harness import mcp as mcp_api
from agentlane.harness.mcp import (
    MCPAccessToken,
    MCPAuthorizationContext,
    MCPResultPolicy,
    MCPServer,
    MCPStdioTransport,
    MCPStreamableHTTPTransport,
    MCPToolFilter,
    MCPToolsShim,
)
from agentlane.harness.mcp._auth import product_bearer_auth
from agentlane.harness.mcp._client import MCPClientLease, MCPClientManager
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.models import Tool, ToolFailure
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from .helpers import acquire_lease, http_server


class _TokenProvider:
    def __init__(self) -> None:
        self.get_count = 0
        self.invalidated: list[MCPAccessToken] = []

    async def get_access_token(
        self,
        server: MCPServer,
        context: MCPAuthorizationContext,
    ) -> MCPAccessToken:
        del server, context
        self.get_count += 1
        return MCPAccessToken(
            token=f"token-{self.get_count}",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    async def invalidate_access_token(
        self,
        server: MCPServer,
        context: MCPAuthorizationContext,
        token: MCPAccessToken,
    ) -> None:
        del server, context
        self.invalidated.append(token)


def test_http_transport_rejects_insecure_and_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="requires HTTPS"):
        MCPStreamableHTTPTransport(url="http://example.test/mcp")
    with pytest.raises(ValueError, match="must not contain credentials"):
        MCPStreamableHTTPTransport(url="https://secret@example.test/mcp")


def test_access_token_repr_does_not_expose_token() -> None:
    token_value = "super" + "-secret"
    token = MCPAccessToken(token=token_value)  # noqa: S106

    assert "super-secret" not in repr(token)


def test_tool_filter_applies_include_then_exclude() -> None:
    tool_filter = MCPToolFilter(include=("meeting_*",), exclude=("*_delete",))

    assert tool_filter.allows("meeting_list") is True
    assert tool_filter.allows("meeting_delete") is False
    assert tool_filter.allows("account_info") is False


def test_public_mcp_api_exposes_run_owned_connections_only() -> None:
    assert "client_manager" not in inspect.signature(MCPToolsShim).parameters
    assert "catalog_ttl_seconds" not in inspect.signature(MCPServer).parameters
    assert tuple(inspect.signature(MCPServer).parameters)[:5] == (
        "name",
        "transport",
        "authorization",
        "tools",
        "result_policy",
    )
    for name in ("MCPClientManager", "MCPClientLimits", "MCPPoolCapacityError"):
        assert not hasattr(mcp_api, name)


def test_result_policy_omits_binary_and_marks_server_errors() -> None:
    result = types.CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "text",
                    "text": "hello",
                    "_meta": {"authorization": "Bearer hidden"},
                },
                {
                    "type": "image",
                    "data": "encoded-secret",
                    "mimeType": "image/png",
                },
            ],
            "structuredContent": {"count": 1, "access_token": "hidden-token"},
            "isError": True,
        }
    )

    rendered = render_mcp_result(result, MCPResultPolicy())

    assert isinstance(rendered, ToolFailure)
    assert "encoded-secret" not in rendered
    assert "Bearer hidden" not in rendered
    assert "hidden-token" not in rendered
    assert rendered.count("[redacted]") == 2
    assert "encodedBytesOmitted" in rendered
    assert rendered.error.kind == "mcp_server"


def test_result_policy_enforces_hard_character_limit() -> None:
    result = types.CallToolResult(
        content=[types.TextContent(text="x" * 10_000)],
        structured_content={"large": "y" * 10_000},
    )

    rendered = render_mcp_result(result, MCPResultPolicy(max_text_chars=180))

    assert len(rendered) <= 180
    assert "truncated" in rendered
    assert "structuredContent" in rendered


@pytest.mark.asyncio
async def test_product_bearer_auth_retries_one_unauthorized_response() -> None:
    provider = _TokenProvider()
    server = MCPServer(
        name="notes",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=provider,
    )
    seen_headers: list[str | None] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        seen_headers.append(request.headers.get("authorization"))
        status = 401 if len(seen_headers) == 1 else 200
        return httpx2.Response(status, request=request)

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(
            httpx2,
            server,
            MCPAuthorizationContext(key="practitioner-1"),
        ),
        transport=httpx2.MockTransport(handler),
    ) as client:
        response = await client.get("https://example.test/mcp")

    assert response.status_code == 200
    assert seen_headers == ["Bearer token-1", "Bearer token-2"]
    assert provider.get_count == 2
    assert len(provider.invalidated) == 1


@pytest.mark.asyncio
async def test_product_bearer_auth_does_not_retry_a_second_unauthorized() -> None:
    provider = _TokenProvider()
    server = MCPServer(
        name="notes",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=provider,
    )
    request_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        return httpx2.Response(401, request=request)

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(
            httpx2,
            server,
            MCPAuthorizationContext(key="practitioner-1"),
        ),
        transport=httpx2.MockTransport(handler),
    ) as client:
        response = await client.get("https://example.test/mcp")

    assert response.status_code == 401
    assert request_count == 2
    assert provider.get_count == 2
    assert len(provider.invalidated) == 2


@pytest.mark.asyncio
async def test_stdio_discovery_and_tool_call_use_native_agentlane_tool() -> None:
    fixture = Path(__file__).parent / "fixtures/math_server.py"
    manager = MCPClientManager()
    server = MCPServer(
        name="Local Math",
        transport=MCPStdioTransport(command=sys.executable, args=(str(fixture),)),
    )
    lease = await acquire_lease(
        manager, server, MCPAuthorizationContext(key="anonymous")
    )
    try:
        tools = await lease.tools()
        assert [tool.name for tool in tools] == ["local_math__add"]
        tool = tools[0]
        args = tool.args_type().model_validate({"left": 2, "right": 5})

        result = await tool.run(args, CancellationToken())

        assert json.loads(str(result))["structuredContent"]["result"] == 7
        assert tool.schema["parameters"]["properties"].keys() == {"left", "right"}
    finally:
        await lease.release()
        await manager.aclose()


@pytest.mark.asyncio
async def test_local_shim_manager_can_run_twice_without_persisting_runtime_state() -> (
    None
):
    fixture = Path(__file__).parent / "fixtures/math_server.py"
    shim = MCPToolsShim(
        servers=(
            MCPServer(
                name="math",
                transport=MCPStdioTransport(
                    command=sys.executable,
                    args=(str(fixture),),
                ),
            ),
        )
    )
    bound = await shim.bind(
        ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))
    )
    state = RunState(instructions=None, history=[], responses=[])

    for _ in range(2):
        transient = DefaultRunContext()
        await bound.on_run_start(state, transient)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)
        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["math__add"]
        await bound.on_run_end(None, transient)

    assert dict(state.shim_state) == {}


@pytest.mark.asyncio
async def test_bound_runs_own_independent_stdio_processes_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures/math_server.py"
    shim = MCPToolsShim(
        servers=(
            MCPServer(
                name="math",
                transport=MCPStdioTransport(
                    command=sys.executable, args=(str(fixture),)
                ),
            ),
        )
    )
    context = ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))
    first, second = await shim.bind(context), await shim.bind(context)
    state = RunState(instructions=None, history=[], responses=[])
    transient = DefaultRunContext()
    release_called = False

    async def hanging_release(self: MCPClientLease) -> None:
        nonlocal release_called
        del self
        release_called = True
        await asyncio.Event().wait()

    try:
        await asyncio.gather(
            first.on_run_start(state, transient),
            second.on_run_start(state, transient),
        )
        monkeypatch.setattr(MCPClientLease, "release", hanging_release)
        await asyncio.wait_for(first.on_run_end(None, transient), 3)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await second.prepare_turn(turn)
        assert turn.tools is not None
        tool = turn.tools.normalized_tools[0]
        assert isinstance(tool, Tool)
        tool = cast(Tool[Any, object], tool)
        result = await tool.run(
            tool.args_type().model_validate({"left": 2, "right": 5}),
            CancellationToken(),
        )
        assert json.loads(str(result))["structuredContent"]["result"] == 7
        assert not release_called
    finally:
        await asyncio.gather(
            first.on_run_end(None, transient),
            second.on_run_end(None, transient),
        )


@pytest.mark.asyncio
async def test_streamable_http_discovery_and_call(unused_tcp_port: int) -> None:
    sdk_server = SDKServer("agentlane-http-test")

    @sdk_server.tool()
    def echo(value: str) -> str:
        """Echo one value."""
        return value

    _ = echo

    app = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    server = MCPServer(
        name="HTTP Notes",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{unused_tcp_port}/mcp",
            allow_insecure_http=True,
        ),
    )
    async with http_server(app, unused_tcp_port), MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="anonymous")
        )
        tools = await lease.tools()
        assert [tool.name for tool in tools] == ["http_notes__echo"]
        tool = tools[0]
        result = await tool.run(
            tool.args_type().model_validate({"value": "hello"}),
            CancellationToken(),
        )
        assert "hello" in str(result)
        await lease.release()
