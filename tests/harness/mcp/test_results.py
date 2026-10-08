"""Bounded MCP results retain redaction and model-facing failure status."""

import json
from typing import Any

import pytest
from mcp import types

from agentlane.harness import tool_outcome
from agentlane.harness.mcp import (
    MCPResultPolicy,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.harness.mcp._tools import native_tool
from agentlane.harness.mcp._types import MCPRemoteTool
from agentlane.models import ToolCall, ToolExecutor, ToolFailure, Tools
from agentlane.runtime import CancellationToken

from ..tools_test_utils import make_tool_call


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [128, 50 * 1024])
async def test_server_error_is_visible_in_actual_model_message(limit: int) -> None:
    server = MCPServer(
        name="results",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
    )
    remote_tool = MCPRemoteTool(
        name="read", description="Read a result.", input_schema={"type": "object"}
    )
    call = make_tool_call(tool_id="read-call", name="results__read", arguments="{}")
    messages: list[list[dict[str, Any]]] = []
    outcomes: list[bool] = []
    result: types.CallToolResult

    def on_tool_end(call: ToolCall, result: object) -> None:
        del call
        outcomes.append(tool_outcome(result).ok)

    async def dispatch(
        name: str, arguments: dict[str, Any], token: CancellationToken
    ) -> str | ToolFailure:
        del name, arguments, token
        return render_mcp_result(result, MCPResultPolicy(max_text_chars=limit))

    for is_error in (False, True):
        result = types.CallToolResult(
            content=[types.TextContent(text="The same result. " * 1000)],
            is_error=is_error,
        )

        messages.append(
            await ToolExecutor().execute(
                tool_calls=[call],
                tools=Tools(tools=[native_tool(server, remote_tool, dispatch)]),
                on_tool_end=on_tool_end,
            )
        )

    assert outcomes == [True, False]
    assert messages[0] != messages[1]
    for is_error, message in zip((False, True), messages, strict=True):
        content = message[0]["content"]
        assert len(content) <= limit
        payload = json.loads(content)
        assert payload["isError"] is is_error
        assert payload.get("truncated", False) is (limit == 128)
