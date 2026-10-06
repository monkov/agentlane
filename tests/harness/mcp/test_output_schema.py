"""SDK output validation uses local schema references without extra requests."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp._client import MCPClientManager
from agentlane.models import ToolFailure
from agentlane.runtime import CancellationToken

from .helpers import acquire_lease, http_server


@dataclass
class _SchemaTarget:
    url: str = ""
    requests: list[str] = field(default_factory=list[str])


@pytest.fixture(name="schema_target")
def fixture_schema_target() -> Iterator[_SchemaTarget]:
    target = _SchemaTarget()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target.requests.append(self.path)
            body = b'{"type":"object"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    # The old validator uses synchronous urllib. This separate thread lets a
    # forbidden request complete, so the regression fails instead of hanging.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    target.url = f"http://127.0.0.1:{server.server_port}/unapproved-schema.json"
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield target
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


async def _call_with_output_schema(
    port: int, schema: dict[str, Any], structured_content: dict[str, Any]
) -> tuple[object, int]:
    calls = 0

    async def list_tools(
        context: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del context, params
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="probe",
                    input_schema={"type": "object"},
                    output_schema=schema,
                )
            ]
        )

    async def call_tool(
        context: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        nonlocal calls
        del context
        assert params.name == "probe"
        calls += 1
        return types.CallToolResult(content=[], structured_content=structured_content)

    peer = Server(
        "output-schema-fixture", on_list_tools=list_tools, on_call_tool=call_tool
    )
    app = peer.streamable_http_app(stateless_http=True, host="127.0.0.1")
    server = MCPServer(
        name="schema",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
        ),
    )
    async with http_server(app, port), MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="schema-test")
        )
        tools = await lease.tools()
        assert len(tools) == 1
        result = await tools[0].run(
            tools[0].args_type().model_validate({}), CancellationToken()
        )
        await lease.release()
    return result, calls


@pytest.mark.asyncio
async def test_external_output_schema_reference_makes_no_request_or_replay(
    unused_tcp_port: int,
    schema_target: _SchemaTarget,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "test-only-schema-credential-7c4eb2"
    uri = f"{schema_target.url}?access_token={canary}"
    result, calls = await _call_with_output_schema(unused_tcp_port, {"$ref": uri}, {})

    assert schema_target.requests == []
    assert calls == 1
    assert isinstance(result, ToolFailure)
    assert result.error.kind == "mcp_protocol"
    assert uri not in str(result)
    assert canary not in str(result)
    assert uri not in caplog.text
    assert canary not in caplog.text
    captured = capsys.readouterr()
    assert uri not in captured.out + captured.err
    assert canary not in captured.out + captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize("has_absolute_id", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_local_output_schema_reference_still_validates_results(
    unused_tcp_port: int,
    schema_target: _SchemaTarget,
    has_absolute_id: bool,
    valid: bool,
) -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/count"}},
        "required": ["count"],
        "$defs": {"count": {"type": "integer"}},
    }
    if has_absolute_id:
        schema["$id"] = schema_target.url
    canary = "test-only-invalid-structured-value-7c4eb2"
    result, calls = await _call_with_output_schema(
        unused_tcp_port, schema, {"count": 7 if valid else canary}
    )

    assert schema_target.requests == []
    assert calls == 1
    if valid:
        assert not isinstance(result, ToolFailure)
        assert '"count": 7' in str(result)
    else:
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "mcp_protocol"
        assert canary not in str(result)
