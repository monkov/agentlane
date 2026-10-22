"""Save readable MCP results with the existing file writer."""

import json
from collections.abc import Collection, Generator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest
from mcp import types

from agentlane.harness.filesystem import FileWriter, LocalFileSystem
from agentlane.harness.mcp import MCPResultPolicy, _output
from agentlane.harness.mcp._output import prepare_mcp_response
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.harness.mcp._source import prepare_source
from agentlane.io import Writer
from agentlane.models import ToolFailure


class _RecordingFileSystem(LocalFileSystem):
    """Record real writes and inject a failure before one file opens."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.root = root
        self.paths: list[str] = []
        self.fail_write = False

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        self.paths.append(path)
        if self.fail_write:
            raise TimeoutError("private-host-detail")
        with super().open_write(path) as stream:
            yield stream

    def texts(self) -> list[str]:
        return [
            (self.root / path).read_text()
            for path in self.paths
            if (self.root / path).exists()
        ]


@pytest.fixture(name="filesystem")
def fixture_filesystem(tmp_path: Path) -> _RecordingFileSystem:
    return _RecordingFileSystem(tmp_path)


async def _prepare(
    result: object,
    writer: FileWriter | None,
    *,
    limit: int = 512,
    selected: bool = False,
    include_structured_content: bool = True,
    directory: str = "results",
    secrets: tuple[str, ...] = (),
) -> str | ToolFailure:
    return await prepare_mcp_response(
        result,
        policy=MCPResultPolicy(
            max_text_chars=limit,
            include_structured_content=include_structured_content,
        ),
        secrets=secrets,
        server_name="notes",
        tool_name="get/document",
        writer=writer,
        directory=directory if writer is not None else None,
        preserve_tools=(
            frozenset({("notes", "get/document")}) if selected else frozenset()
        ),
    )


@pytest.mark.asyncio
async def test_no_writer_keeps_existing_response(
    filesystem: _RecordingFileSystem,
) -> None:
    for size in (10, 100_000):
        result = types.CallToolResult(content=[types.TextContent(text="x" * size)])
        expected = render_mcp_result(result, MCPResultPolicy(max_text_chars=512))
        assert await _prepare(result, None) == expected

    assert filesystem.paths == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["inline", "saved", "write_failure"])
async def test_enabled_storage_prepares_text_once_without_the_preview_renderer(
    outcome: str,
    filesystem: _RecordingFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared: list[str] = []

    def prepare_once(
        result: object,
        *,
        server_name: str,
        tool_name: str,
        secrets: Collection[str],
    ) -> str:
        text = prepare_source(
            result, server_name=server_name, tool_name=tool_name, secrets=secrets
        )
        prepared.append(text)
        return text

    def reject_preview(*args: object, **kwargs: object) -> str:
        raise AssertionError("Enabled storage must not traverse the preview renderer.")

    monkeypatch.setattr(_output, "prepare_source", prepare_once)
    monkeypatch.setattr(_output, "render_mcp_result", reject_preview)
    filesystem.fail_write = outcome == "write_failure"
    secret = "credential-canary-" + "s" * 5000
    transcript = secret + " " + ("x" * (10 if outcome == "inline" else 5000))
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text=transcript)]),
        filesystem,
        secrets=(secret,),
    )
    assert len(prepared) == 1
    assert "credential-canary" not in response
    assert "[redacted]" in response
    assert len(response) <= 512
    if outcome == "inline":
        assert response == prepared[0]
        assert filesystem.paths == []
    elif outcome == "saved":
        assert filesystem.texts() == prepared
        notice = f"\n\n[Full MCP result: {json.dumps(filesystem.paths[0])}]"
        assert response.endswith(notice)
        assert prepared[0].startswith(response.removesuffix(notice).removesuffix("…"))
    else:
        notice = "\n\n[Full MCP result unavailable.]"
        assert response.endswith(notice)
        assert prepared[0].startswith(response.removesuffix(notice).removesuffix("…"))
        assert filesystem.texts() == []


@pytest.mark.asyncio
async def test_only_text_above_the_character_budget_is_saved(
    filesystem: _RecordingFileSystem,
) -> None:
    result = types.CallToolResult(content=[types.TextContent(text="x" * 200)])
    inline = await _prepare(result, filesystem, limit=1000)
    assert await _prepare(result, filesystem, limit=len(inline)) == inline
    assert filesystem.paths == []

    response = await _prepare(result, filesystem, limit=len(inline) - 1)
    assert len(response) <= len(inline) - 1
    assert filesystem.texts() == [inline]
    assert response.endswith(
        f"\n\n[Full MCP result: {json.dumps(filesystem.paths[0])}]"
    )


@pytest.mark.asyncio
async def test_large_text_is_saved_before_preview_processing_can_omit_it(
    filesystem: _RecordingFileSystem,
) -> None:
    transcript = "An exact transcript line.\n" * 5000 + "final words"
    result = types.CallToolResult(content=[types.TextContent(text=transcript)])
    response = await _prepare(result, filesystem)
    assert len(response) <= 512
    (read_path,) = map(Path, filesystem.paths)
    notice = f"\n\n[Full MCP result: {json.dumps(str(read_path))}]"
    assert response.endswith(notice)
    assert read_path.parent == Path("results")
    assert read_path.stem == str(UUID(read_path.stem))
    assert read_path.suffix == ".txt"
    readable = filesystem.texts()[0]
    assert readable.startswith(response.removesuffix(notice).removesuffix("…"))
    assert "".join(transcript.splitlines()) in "".join(readable.splitlines())
    assert "final words" in readable
    assert all(len(line.encode("utf-8")) <= 4096 for line in readable.splitlines())


@pytest.mark.asyncio
@pytest.mark.parametrize("include_structured_content", [True, False])
async def test_small_structured_content_returns_inline_regardless_of_preview_policy(
    filesystem: _RecordingFileSystem, include_structured_content: bool
) -> None:
    result = types.CallToolResult(
        content=(
            [types.TextContent(text="Transcript details")]
            if include_structured_content
            else []
        ),
        structured_content={"meeting_id": "meeting-123"},
    )
    readable = await _prepare(
        result, filesystem, include_structured_content=include_structured_content
    )
    assert filesystem.paths == []
    if include_structured_content:
        assert "Transcript details" in readable
    assert "meeting_id" in readable
    assert "meeting-123" in readable


@pytest.mark.asyncio
async def test_small_resource_is_returned_inline_and_selected_binary_is_saved(
    filesystem: _RecordingFileSystem,
) -> None:
    binary = types.CallToolResult(
        content=[types.ImageContent(data="binary-canary", mime_type="image/png")]
    )
    await _prepare(binary, filesystem)
    assert filesystem.paths == []

    await _prepare(binary, filesystem, selected=True)
    readable = filesystem.texts()[0]
    assert "binary-canary" not in readable
    assert "binary" in readable

    resource = types.CallToolResult(
        content=[
            types.EmbeddedResource(
                resource=types.TextResourceContents(
                    uri="resource://notes/1", text="Complete embedded text"
                )
            )
        ]
    )
    response = await _prepare(resource, filesystem)
    assert "Complete embedded text" in response
    assert len(filesystem.paths) == 1


@pytest.mark.asyncio
async def test_files_remove_credentials_without_clipping_source_ids(
    filesystem: _RecordingFileSystem,
) -> None:
    canary = "credential-canary"
    transcript = "a" * 4090 + canary + "z" * 5000
    meeting_id = "meeting-" + "x" * 20_000
    result = types.CallToolResult(
        content=[
            types.TextContent(
                text=json.dumps(
                    {
                        "meeting_id": meeting_id,
                        "transcript": transcript,
                        "api_key": canary,
                        "escaped": '{"pass\\u0077ord": "unknown-secret-canary"}',
                    }
                )
            )
        ],
        structured_content={"authorization": canary, "transcript": transcript},
    )
    response = await _prepare(result, filesystem, secrets=(canary,))
    readable = filesystem.texts()[0]
    combined = response + readable
    assert canary not in combined
    assert "unknown-secret-canary" not in combined
    assert meeting_id in "".join(readable.splitlines())
    assert "[redacted]" in readable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '{"patient":"credential-canary","patient":"Bob"}',
        '{"pass\\u0077ord":"unknown-secret-canary"',
        '[0,{"pass\\u0077ord":"unknown-secret-canary"',
        '[null,{"pass\\u0077ord":"unknown-secret-canary"',
        '[false,{"pass\\u0077ord":"unknown-secret-canary"',
        '[[],{"pass\\u0077ord":"unknown-secret-canary"',
    ],
)
async def test_ambiguous_json_is_not_saved_with_unexamined_credentials(
    text: str, filesystem: _RecordingFileSystem
) -> None:
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text=text)]),
        filesystem,
        selected=True,
        secrets=("credential-canary",),
    )
    assert filesystem.paths == []
    assert response == "[MCP result unavailable: could not prepare readable text.]"
    assert "credential-canary" not in response
    assert "unknown-secret-canary" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ['[00:31] Speaker 1: "Hello, doctor."', "[Speaker 1] Hello.", "{draft} reply"],
)
async def test_bracketed_transcript_prose_is_saved_unchanged(
    text: str, filesystem: _RecordingFileSystem
) -> None:
    await _prepare(
        types.CallToolResult(content=[types.TextContent(text=text)]),
        filesystem,
        selected=True,
    )
    assert text in filesystem.texts()[0]


@pytest.mark.asyncio
async def test_small_server_error_keeps_text_and_structure_without_a_write(
    filesystem: _RecordingFileSystem,
) -> None:
    response = await _prepare(
        types.CallToolResult(
            content=[types.TextContent(text="Not found")],
            structured_content={"code": 404},
            is_error=True,
        ),
        filesystem,
    )
    assert isinstance(response, ToolFailure)
    assert response.error.kind == "mcp_server"
    assert "Not found" in response
    assert "404" in response
    assert filesystem.paths == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [128, 160, 512])
@pytest.mark.parametrize("is_error", [False, True])
async def test_paths_share_preview_budget_and_preserve_server_errors(
    limit: int, is_error: bool, filesystem: _RecordingFileSystem
) -> None:
    result = types.CallToolResult(
        content=[types.TextContent(text="x" * 5000)],
        structured_content={"details": "y" * 5000},
        is_error=is_error,
    )
    response = await _prepare(result, filesystem, limit=limit)
    assert len(response) <= limit
    assert response.endswith(
        f"\n\n[Full MCP result: {json.dumps(filesystem.paths[0])}]"
    )
    assert isinstance(response, ToolFailure) is is_error
    if isinstance(response, ToolFailure):
        assert response.error.kind == "mcp_server"


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_chars", [-1, 0, 1])
async def test_complete_quoted_path_notice_at_the_response_budget_boundary(
    extra_chars: int, filesystem: _RecordingFileSystem
) -> None:
    limit = 128
    prefix = 'results/notes\n"quoted" '
    sample_path = f"{prefix}/{'x' * 36}.txt"
    sample_notice = f"\n\n[Full MCP result: {json.dumps(sample_path)}]"
    directory = prefix + "x" * (limit + extra_chars - len(sample_notice))
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text="x" * 5000)]),
        filesystem,
        directory=directory,
        limit=limit,
    )
    (path,) = filesystem.paths
    notice = f"\n\n[Full MCP result: {json.dumps(path)}]"
    assert len(notice) == limit + extra_chars
    assert len(response) <= limit
    assert len(filesystem.texts()) == 1
    if extra_chars > 0:
        assert response.endswith("\n\n[Full MCP result unavailable.]")
    else:
        assert response.endswith(notice)
        assert json.loads(response.split("[Full MCP result: ", 1)[1][:-1]) == path
        if extra_chars == 0:
            assert response == notice


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
async def test_write_failure_keeps_bounded_result_without_publishing_paths(
    is_error: bool, filesystem: _RecordingFileSystem
) -> None:
    filesystem.fail_write = True
    result = types.CallToolResult(
        content=[types.TextContent(text="x" * 5000)], is_error=is_error
    )
    response = await _prepare(result, filesystem)
    assert len(response) <= 512
    assert "[Full MCP result unavailable.]" in response
    assert "private-host-detail" not in response
    assert len(filesystem.paths) == 1
    assert filesystem.texts() == []
    assert isinstance(response, ToolFailure) is is_error
    if isinstance(response, ToolFailure):
        assert response.error.kind == "mcp_server"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directory",
    ["results/credential-canary", 'results/"token":"hidden"'],
)
async def test_credential_output_paths_are_rejected_before_writing(
    directory: str, filesystem: _RecordingFileSystem
) -> None:
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text="x" * 5000)]),
        filesystem,
        directory=directory,
        secrets=("credential-canary",),
    )
    assert filesystem.paths == []
    assert "[Full MCP result unavailable.]" in response
    assert directory not in response


@pytest.mark.asyncio
async def test_long_output_paths_save_files_without_exceeding_preview_budget(
    filesystem: _RecordingFileSystem,
) -> None:
    transcript = "Complete transcript. " * 500
    directory = "/".join(["results"] + ["x" * 100] * 6)
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text=transcript)]),
        filesystem,
        directory=directory,
    )
    assert len(filesystem.paths) == 1
    assert all(len(path) > 512 for path in filesystem.paths)
    assert transcript in "".join(filesystem.texts()[0].splitlines())
    assert "[Full MCP result unavailable.]" in response
    assert len(response) <= 512


@pytest.mark.asyncio
async def test_text_beyond_eight_mib_is_saved_in_full_with_bounded_preview(
    filesystem: _RecordingFileSystem,
) -> None:
    transcript = "x" * (8 * 1024 * 1024) + "final words beyond eight MiB"
    result = types.CallToolResult(content=[types.TextContent(text=transcript)])
    assert await _prepare(result, None) == render_mcp_result(
        result, MCPResultPolicy(max_text_chars=512)
    )

    response = await _prepare(result, filesystem)
    assert len(response) <= 512
    assert response.endswith(
        f"\n\n[Full MCP result: {json.dumps(filesystem.paths[0])}]"
    )
    assert transcript in "".join(filesystem.texts()[0].splitlines())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nodes", "depth", "blocks", "integer"])
async def test_preview_structure_limits_do_not_drop_enabled_result_content(
    kind: str, filesystem: _RecordingFileSystem
) -> None:
    content: list[object] = []
    structured: object = None
    if kind == "nodes":
        structured = [f"node-{index}" for index in range(5000)]
        final_value = "node-4999"
    elif kind == "depth":
        final_value = "deep text"
        structured = final_value
        for _ in range(40):
            structured = {"nested": structured}
    elif kind == "blocks":
        content = [{"type": "text", "text": f"block-{index}"} for index in range(4100)]
        final_value = "block-4099"
    else:
        structured = 1 << 5000
        final_value = str(structured)

    result = {"content": content, "structuredContent": structured}
    expected = render_mcp_result(result, MCPResultPolicy(max_text_chars=512))
    assert await _prepare(result, None) == expected
    assert len(expected) <= 512

    response = await _prepare(result, filesystem)
    assert len(response) <= 512
    if kind == "depth":
        assert filesystem.paths == []
        assert final_value in response
    else:
        assert response.endswith(
            f"\n\n[Full MCP result: {json.dumps(filesystem.paths[0])}]"
        )
        assert final_value in filesystem.texts()[0]
