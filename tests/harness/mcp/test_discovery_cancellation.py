"""Public run cancellation must stop MCP setup and turn discovery."""

import asyncio
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext

from agentlane.harness import AgentDescriptor, RunResult
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPServer,
    MCPStdioTransport,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)
from agentlane.models import Tools
from agentlane.runtime import CancellationToken

from ..tools_test_utils import (
    SequenceModel,
    StreamingSequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
)
from .helpers import http_server

type _RunMode = Literal["run", "stream", "events"]


async def _run(
    agent: DefaultAgent, token: CancellationToken, mode: _RunMode
) -> RunResult:
    if mode == "run":
        return await agent.run("test", cancellation_token=token)
    stream = (
        await agent.run_stream("test", cancellation_token=token)
        if mode == "stream"
        else await agent.run_events("test", cancellation_token=token)
    )
    try:
        async for _ in stream:
            pass
        return await stream.result()
    finally:
        await stream.aclose()
        with suppress(BaseException):
            await stream.result()


async def _assert_cancelled(run: asyncio.Task[RunResult], mode: _RunMode) -> None:
    _, pending = await asyncio.wait((run,), timeout=1)
    assert not pending, "Run cancellation did not complete."
    if mode == "run":
        with pytest.raises(RuntimeError, match="delivery status `canceled`"):
            await run
    else:
        with pytest.raises(asyncio.CancelledError):
            await run


async def _finish_run(run: asyncio.Task[RunResult]) -> None:
    completed, pending = await asyncio.wait((run,), timeout=4)
    for task in pending:
        task.cancel()
    await asyncio.gather(*completed, return_exceptions=True)
    assert not pending, "The public run did not finish test cleanup."


async def _assert_connection_tasks_closed() -> None:
    async with asyncio.timeout(2):
        while any(
            task.get_name().startswith("agentlane-mcp-") and not task.done()
            for task in asyncio.all_tasks()
        ):
            await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("stage", ["startup", "prepare"])
async def test_public_run_cancellation_stops_http_catalog_before_next_model_call(
    unused_tcp_port: int, mode: _RunMode, stage: str
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    listed = 0
    blocked_page = 1 if stage == "startup" else 3

    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        nonlocal listed
        del context, params
        listed += 1
        if listed == blocked_page:
            started.set()
            await release.wait()
        return types.ListToolsResult(
            tools=[types.Tool(name="read", input_schema={"type": "object"})]
        )

    sdk_server = Server("cancel-discovery", on_list_tools=list_tools)
    app = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    responses = [
        make_assistant_response(
            None,
            tool_calls=[
                make_tool_call(
                    tool_id="local", name="local", arguments='{"text":"next"}'
                )
            ],
        ),
        make_assistant_response("must not run after cancellation"),
    ]
    model = (
        SequenceModel(responses) if mode == "run" else StreamingSequenceModel(responses)
    )
    server = MCPServer(
        name="remote",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{unused_tcp_port}/mcp", allow_insecure_http=True
        ),
        discovery_timeout_seconds=3,
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="cancel-discovery",
            model=model,
            tools=Tools(tools=(echo_tool("local"),)),
            shims=(MCPToolsShim(servers=(server,)),),
        )
    )
    token = CancellationToken()
    async with http_server(app, unused_tcp_port):
        run = asyncio.create_task(_run(agent, token, mode))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            expected_calls = 0 if stage == "startup" else 1
            assert len(model.calls) == expected_calls
            token.cancel()
            await _assert_cancelled(run, mode)
            assert not release.is_set()
            assert len(model.calls) == expected_calls
            await _assert_connection_tasks_closed()
        finally:
            release.set()
            await _finish_run(run)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_public_run_cancellation_during_stdio_handshake_reaps_process(
    tmp_path: Path, mode: _RunMode
) -> None:
    pid_file = tmp_path / "server.pid"
    server = MCPServer(
        name="pending",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/hanging_server.py"),),
            env={"MCP_TEST_PID": str(pid_file)},
        ),
        connect_timeout_seconds=3,
    )
    responses = [make_assistant_response("must not run")]
    model = (
        SequenceModel(responses) if mode == "run" else StreamingSequenceModel(responses)
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="cancel-handshake",
            model=model,
            shims=(MCPToolsShim(servers=(server,)),),
        )
    )
    token = CancellationToken()
    run = asyncio.create_task(_run(agent, token, mode))
    try:
        async with asyncio.timeout(3):
            while not pid_file.exists():
                await asyncio.sleep(0.01)
        token.cancel()
        await _assert_cancelled(run, mode)
        assert not model.calls
        await _assert_connection_tasks_closed()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    finally:
        await _finish_run(run)
