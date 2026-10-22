"""Saved MCP output remains readable through the normal agent lifecycle."""

import asyncio
import json
import re
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pytest
from mcp import types

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPResultPolicy,
    MCPServer,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.tools import HarnessToolsShim, read_tool
from agentlane.io import Writer
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ToolCall,
    ToolFailure,
    Tools,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span

from ..tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
)
from .helpers import ConnectionInstaller, acquire_lease

_SERVER_NAME = "Clinical Notes"
_TOOL_NAME = "get/document"
_VISIBLE_NAME = "clinical_notes__get_document"
_CONTINUATION = re.compile(
    r"\n\n\[Showing lines [^\n]+ Use offset=(\d+) to continue\.\]$"
)


@dataclass
class _Peer:
    result: types.CallToolResult
    opened: list[MCPConnection] = field(default_factory=list[MCPConnection])
    calls: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list[tuple[str, dict[str, Any]]]
    )
    discoveries: int = 0

    async def list_tools(
        self, *, cursor: str | None, cache_mode: Literal["bypass"]
    ) -> types.ListToolsResult:
        del cursor, cache_mode
        self.discoveries += 1
        return types.ListToolsResult(
            tools=[types.Tool(name=_TOOL_NAME, input_schema={"type": "object"})]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> types.CallToolResult:
        del read_timeout_seconds
        self.calls.append((name, arguments))
        # Simulate a catalog-change notification before the next model turn.
        for connection in self.opened:
            connection.catalog_revision += 1
        return self.result

    @asynccontextmanager
    async def listen(
        self, *, tools_list_changed: bool
    ) -> AsyncGenerator[AsyncIterator[object], None]:
        del tools_list_changed

        async def events() -> AsyncIterator[object]:
            yield object()

        yield events()


@pytest.fixture(name="output_peer")
def fixture_output_peer(install_connection: ConnectionInstaller) -> _Peer:
    peer = _Peer(types.CallToolResult(content=[types.TextContent(text="Complete")]))

    async def prepare(connection: MCPConnection) -> None:
        peer.opened.append(connection)
        connection.client = peer

    install_connection(prepare)
    return peer


class _BlockedFileSystem(LocalFileSystem):
    """Pause a real file write before its context manager commits it."""

    def __init__(self, root: Path, gate: threading.Event) -> None:
        super().__init__(root)
        self.gate = gate
        self.started = threading.Event()
        self.finished = threading.Event()
        self.completed: list[str] = []

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        with super().open_write(path) as stream:
            yield stream

            # Hold context exit so cancellation must wait for the actual commit.
            self.started.set()
            if not self.gate.wait(timeout=10):
                raise TimeoutError("The test did not release the file write.")

        self.completed.append(path)
        self.finished.set()


def _server(*, max_chars: int = 512) -> MCPServer:
    return MCPServer(
        name=_SERVER_NAME,
        transport=MCPStdioTransport(command="fixture"),
        result_policy=MCPResultPolicy(max_text_chars=max_chars),
    )


def _call(
    name: str = _VISIBLE_NAME, *, call_id: str | None = None, **arguments: object
) -> ToolCall:
    return make_tool_call(
        tool_id=call_id or f"call-{name}", name=name, arguments=json.dumps(arguments)
    )


class _ReadSavedResultModel(Model[ModelResponse]):
    """Follow only the reference and continuation offsets in tool responses."""

    def __init__(self) -> None:
        self.preview: str | None = None
        self.read_path: str | None = None
        self.pages: list[str] = []
        self.offsets: list[int] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args, schema, tools, cancellation_token, parent_span, kwargs
        tool_messages = [message for message in messages if message["role"] == "tool"]
        if not tool_messages:
            return make_assistant_response(
                None, tool_calls=[_call(meeting_id="meeting-7")]
            )

        content = tool_messages[-1]["content"]
        assert isinstance(content, str)
        if self.preview is None:
            self.preview = content
            self.read_path = json.loads(content)["output"]["read_path"]
            return self._read(1)

        # Treat the tool's continuation as the only source of the next offset.
        continuation = _CONTINUATION.search(content)
        self.pages.append(content[: continuation.start()] if continuation else content)
        if continuation is not None:
            return self._read(int(continuation.group(1)))

        return make_assistant_response("Read the complete saved result.")

    def _read(self, offset: int) -> ModelResponse:
        self.offsets.append(offset)
        return make_assistant_response(
            None,
            tool_calls=[
                _call(
                    "read",
                    call_id=f"read-{offset}",
                    path=self.read_path,
                    offset=offset,
                    limit=7,
                )
            ],
        )


@pytest.mark.asyncio
async def test_runner_reads_complete_saved_output_through_native_pagination(
    output_peer: _Peer, tmp_path: Path
) -> None:
    transcript = "BEGIN_TRANSCRIPT\n" + "Я🙂界" * 11_000 + "\x00\x07\nEND_TRANSCRIPT"
    output_peer.result = types.CallToolResult(
        content=[types.TextContent(text=transcript)],
        structured_content={"separate_fact": "STRUCTURED_ONLY"},
    )
    filesystem = MountedFileSystem({"saved": LocalFileSystem(tmp_path)})
    model = _ReadSavedResultModel()

    async with MCPClientManager() as manager:
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="reader",
                model=model,
                tools=Tools(tools=[], max_tool_round_trips=30),
                shims=(
                    MCPToolsShim(
                        servers=(_server(),),
                        client_manager=manager,
                        result_writer=filesystem,
                        result_directory="saved/./sessions/../tool-results",
                    ),
                    HarnessToolsShim(definitions=(read_tool(reader=filesystem),)),
                ),
            )
        )
        result = await agent.run("Read all of this meeting.")

    assert result.final_output == "Read the complete saved result."
    assert model.preview is not None
    assert len(model.preview) <= 512
    assert len(model.pages) > 1
    assert model.offsets == sorted(set(model.offsets))

    # Resolve only paths published by the MCP tool, in the configured namespace.
    reference = json.loads(model.preview)["output"]
    read_path = tmp_path / PurePosixPath(reference["read_path"]).relative_to("/saved")
    readable_text = read_path.read_text(encoding="utf-8")
    assert "\n".join(model.pages) == readable_text.rstrip("\n")
    assert "END_TRANSCRIPT" in "\n".join(model.pages)
    assert "STRUCTURED_ONLY" in "\n".join(model.pages)
    assert "\n".join(model.pages).count("🙂") == 11_000
    assert "\x00" not in readable_text
    assert "\x07" not in readable_text
    assert max(len(line.encode("utf-8")) for line in readable_text.splitlines()) <= 4096
    assert output_peer.calls == [(_TOOL_NAME, {"meeting_id": "meeting-7"})]
    assert len(list((tmp_path / "tool-results").iterdir())) == 1
    assert reference == {"read_path": f"/saved/tool-results/{read_path.name}"}


@pytest.mark.asyncio
async def test_child_inherits_small_result_selection_with_original_names(
    output_peer: _Peer, tmp_path: Path
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    child_model = SequenceModel(
        [
            make_assistant_response(
                None, tool_calls=[_call(meeting_id="child-meeting")]
            ),
            make_assistant_response("child complete"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model)
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent complete"),
        ]
    )

    async with MCPClientManager() as manager:
        await DefaultAgent(
            descriptor=AgentDescriptor(
                name="parent",
                model=parent_model,
                tools=Tools(tools=[child.as_tool()]),
                shims=(
                    MCPToolsShim(
                        servers=(_server(),),
                        client_manager=manager,
                        result_writer=filesystem,
                        result_directory="./sessions/../tool-results",
                        preserve_tools=frozenset({(_SERVER_NAME, _TOOL_NAME)}),
                    ),
                ),
            )
        ).run("Ask the child to read the meeting.")

    assert output_peer.calls == [(_TOOL_NAME, {"meeting_id": "child-meeting"})]
    (read_path,) = (tmp_path / "tool-results").glob("*.txt")
    assert f"tool-results/{read_path.name}" in json.dumps(child_model.calls[-1])
    assert list((tmp_path / "tool-results").iterdir()) == [read_path]
    assert len(output_peer.opened) == 1
    assert output_peer.discoveries > 1


@pytest.mark.asyncio
async def test_concurrent_shims_keep_writers_separate_after_catalog_refresh(
    output_peer: _Peer, tmp_path: Path
) -> None:
    gate = threading.Event()
    labels = ("first", "second")
    backends = [_BlockedFileSystem(tmp_path / label, gate) for label in labels]
    filesystems = [MountedFileSystem({"saved": backend}) for backend in backends]
    models = [
        SequenceModel(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        _call(
                            call_id=f"{label}-{index}",
                            meeting_id=f"{label}-{index}",
                        )
                    ],
                )
                for index in range(2)
            ]
            + [make_assistant_response("complete")]
        )
        for label in labels
    ]

    async with MCPClientManager() as manager:
        agents = [
            DefaultAgent(
                descriptor=AgentDescriptor(
                    name=label,
                    model=model,
                    shims=(
                        MCPToolsShim(
                            servers=(_server(),),
                            client_manager=manager,
                            result_writer=filesystem,
                            result_directory=f"/saved/{label}/tool-results",
                            preserve_tools=frozenset({(_SERVER_NAME, _TOOL_NAME)}),
                        ),
                    ),
                )
            )
            for label, filesystem, model in zip(
                labels, filesystems, models, strict=True
            )
        ]
        runs = [
            asyncio.create_task(agent.run("Read two meetings.")) for agent in agents
        ]
        try:
            # Both writers must start before either session can finish its save.
            started = await asyncio.gather(
                *(asyncio.to_thread(backend.started.wait, 5) for backend in backends)
            )
            assert all(started)
            assert len(output_peer.opened) == 1
            gate.set()
            await asyncio.gather(*runs)
        finally:
            gate.set()
            for run in runs:
                if not run.done():
                    run.cancel()
            await asyncio.gather(*runs, return_exceptions=True)

    for label, backend, model in zip(labels, backends, models, strict=True):
        assert len(backend.completed) == 2
        saved = tmp_path / label / label / "tool-results"
        read_paths = list(saved.glob("*.txt"))
        assert len(read_paths) == 2
        history = json.dumps(model.calls[-1])
        for path in read_paths:
            assert f"/saved/{label}/tool-results/{path.name}" in history
        other = "second" if label == "first" else "first"
        assert f"/saved/{other}" not in history
        assert not (tmp_path / label / other).exists()

    assert sorted(arguments["meeting_id"] for _, arguments in output_peer.calls) == [
        f"{label}-{index}" for label in labels for index in range(2)
    ]
    assert output_peer.discoveries > 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["shutdown", "token", "task"])
async def test_tool_cancellation_waits_for_active_file_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cancel: Literal["shutdown", "token", "task"],
) -> None:
    gate = threading.Event()
    backend = _BlockedFileSystem(tmp_path, gate)
    filesystem = MountedFileSystem({"saved": backend})
    provider_calls: list[str] = []

    # Use the real owner: ConnectionInstaller does not implement its cleanup.
    server = MCPServer(
        name="math",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/math_server.py"),),
        ),
    )
    manager = MCPClientManager()
    lease = await acquire_lease(
        manager, server, MCPAuthorizationContext(key="shutdown")
    )
    entry = lease._entry  # pyright: ignore[reportPrivateUsage]
    assert entry.connection is not None
    connection = entry.connection
    client = connection.get_client()
    original_call = client.call_tool

    async def counted_call(
        name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> object:
        provider_calls.append(name)
        return await original_call(
            name, arguments, read_timeout_seconds=read_timeout_seconds
        )

    monkeypatch.setattr(client, "call_tool", counted_call)
    (tool,) = await lease.tools(
        result_writer=filesystem,
        result_directory="/saved/tool-results",
        preserve_tools=frozenset({("math", "add")}),
    )
    token = CancellationToken()
    call_task = asyncio.create_task(
        tool.run(
            tool.args_type().model_validate({"left": 2, "right": 3}),
            token,
        )
    )
    shutdown: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(backend.started.wait, 5)
        if cancel == "shutdown":
            shutdown = asyncio.create_task(manager.aclose())
        elif cancel == "token":
            token.cancel()
        else:
            call_task.cancel()

        # Each cancellation path must reach the same active result preparation.
        async with asyncio.timeout(5):
            while not any(task.cancelling() for task in connection.active_calls):
                await asyncio.sleep(0.01)
        assert not backend.finished.is_set()
        assert shutdown is None or not shutdown.done()
        assert not call_task.done()

        gate.set()
        if shutdown is not None:
            await asyncio.wait_for(shutdown, timeout=5)
        if cancel == "task":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call_task, timeout=5)
        else:
            result = await asyncio.wait_for(call_task, timeout=5)
            assert isinstance(result, ToolFailure)
            assert result.error.kind == (
                "cancelled" if cancel == "token" else "mcp_transport"
            )
            assert "read_path" not in result

        assert backend.finished.is_set()
        assert len(backend.completed) == 1
        assert len(list((tmp_path / "tool-results").iterdir())) == 1
        assert provider_calls == ["add"]
    finally:
        gate.set()
        if not call_task.done():
            call_task.cancel()
        await asyncio.gather(call_task, return_exceptions=True)
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await lease.release()
        await manager.aclose()
