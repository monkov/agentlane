"""Preserve complete MCP sources with the existing file writer."""

import asyncio
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from mcp import types

from agentlane.harness.filesystem import FileWriter, LocalFileSystem
from agentlane.harness.mcp import MCPResultPolicy
from agentlane.harness.mcp import _source as source_module
from agentlane.harness.mcp._output import prepare_mcp_response
from agentlane.harness.mcp._redaction import redact_known_secrets
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.io import Writer
from agentlane.models import ToolFailure
from agentlane.runtime import CancellationToken


class _RecordingFileSystem(LocalFileSystem):
    """Record real writes and inject a failure before one file opens."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.root = root
        self.paths: list[str] = []
        self.fail_suffix: str | None = None

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        self.paths.append(path)
        if self.fail_suffix is not None and path.endswith(self.fail_suffix):
            raise TimeoutError("private-host-detail")
        with super().open_write(path) as stream:
            yield stream

    def texts(self, suffix: str) -> list[str]:
        return [
            (self.root / path).read_text()
            for path in self.paths
            if path.endswith(suffix) and (self.root / path).exists()
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
    directory: str = "results",
    secrets: tuple[str, ...] = (),
    token: CancellationToken | None = None,
) -> str | ToolFailure:
    return await prepare_mcp_response(
        result,
        policy=MCPResultPolicy(max_text_chars=limit),
        secrets=secrets,
        server_name="notes",
        tool_name="get/document",
        writer=writer,
        directory=directory if writer is not None else None,
        preserve_tools=(
            frozenset({("notes", "get/document")}) if selected else frozenset()
        ),
        cancellation_token=token if token is not None else CancellationToken(),
    )


@pytest.mark.asyncio
async def test_no_writer_and_small_results_keep_existing_response(
    filesystem: _RecordingFileSystem,
) -> None:
    for size in (10, 100_000):
        result = types.CallToolResult(content=[types.TextContent(text="x" * size)])
        expected = render_mcp_result(result, MCPResultPolicy(max_text_chars=512))
        assert await _prepare(result, None) == expected

    result = types.CallToolResult(content=[types.TextContent(text="Small result")])
    assert await _prepare(result, filesystem) == render_mcp_result(
        result, MCPResultPolicy(max_text_chars=512)
    )
    assert filesystem.paths == []


@pytest.mark.asyncio
async def test_large_text_is_saved_before_preview_processing_can_omit_it(
    filesystem: _RecordingFileSystem,
) -> None:
    transcript = "An exact transcript line.\n" * 5000 + "final words"
    result = types.CallToolResult(content=[types.TextContent(text=transcript)])
    response = await _prepare(result, filesystem)
    payload = json.loads(response)
    assert len(response) <= 512
    assert payload["output"] == dict(
        zip(("source_path", "read_path"), filesystem.paths, strict=True)
    )
    source_path, read_path = map(Path, filesystem.paths)
    assert source_path.parent == read_path.parent == Path("results")
    assert source_path.stem == read_path.stem == str(UUID(source_path.stem))
    assert source_path.suffix == ".json"
    assert read_path.suffix == ".txt"
    assert json.loads(filesystem.texts(".json")[0])["content"][0]["text"] == transcript
    readable = filesystem.texts(".txt")[0]
    assert "final words" in readable
    assert all(len(line.encode("utf-8")) <= 4096 for line in readable.splitlines())


@pytest.mark.asyncio
async def test_distinct_text_and_structured_content_are_both_preserved(
    filesystem: _RecordingFileSystem,
) -> None:
    result = types.CallToolResult(
        content=[types.TextContent(text="Transcript details")],
        structured_content={"meeting_id": "meeting-123"},
    )
    await _prepare(result, filesystem)
    source = json.loads(filesystem.texts(".json")[0])
    assert source["content"][0]["text"] == "Transcript details"
    assert source["structuredContent"] == {"meeting_id": "meeting-123"}


@pytest.mark.asyncio
async def test_embedded_text_is_saved_but_binary_only_omission_does_not_trigger_save(
    filesystem: _RecordingFileSystem,
) -> None:
    binary = types.CallToolResult(
        content=[types.ImageContent(data="binary-canary", mime_type="image/png")]
    )
    await _prepare(binary, filesystem)
    assert filesystem.paths == []

    await _prepare(binary, filesystem, selected=True)
    source = filesystem.texts(".json")[0]
    assert "binary-canary" not in source + filesystem.texts(".txt")[0]
    assert json.loads(source)["omissions"] == ["binary"]

    resource = types.CallToolResult(
        content=[
            types.EmbeddedResource(
                resource=types.TextResourceContents(
                    uri="resource://notes/1", text="Complete embedded text"
                )
            )
        ]
    )
    await _prepare(resource, filesystem)
    assert (
        json.loads(filesystem.texts(".json")[1])["content"][0]["resource"]["text"]
        == "Complete embedded text"
    )
    assert "Complete embedded text" in filesystem.texts(".txt")[1]


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
    source = filesystem.texts(".json")[0]
    combined = response + source + filesystem.texts(".txt")[0]
    assert canary not in combined
    assert "unknown-secret-canary" not in combined
    document = json.loads(source)
    assert json.loads(document["content"][0]["text"])["meeting_id"] == meeting_id
    assert "arguments" not in document
    assert "[redacted]" in source


@pytest.mark.asyncio
async def test_json_text_with_no_redactions_preserves_original_formatting(
    filesystem: _RecordingFileSystem,
) -> None:
    original = '{\n  "id": "meeting-1", "value": 1.00\n}'
    await _prepare(
        types.CallToolResult(content=[types.TextContent(text=original)]),
        filesystem,
        selected=True,
    )
    assert json.loads(filesystem.texts(".json")[0])["content"][0]["text"] == original


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
    assert json.loads(response)["output"] == {"status": "unavailable"}


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
    assert json.loads(filesystem.texts(".json")[0])["content"][0]["text"] == text


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
    payload = json.loads(response)
    assert payload["isError"] is is_error
    if limit == 512:
        assert payload["output"] == dict(
            zip(("source_path", "read_path"), filesystem.paths, strict=True)
        )
    else:
        assert payload["output"] == {"status": "unavailable"}
    assert isinstance(response, ToolFailure) is is_error
    if isinstance(response, ToolFailure):
        assert response.error.kind == "mcp_server"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_suffix", [".json", ".txt"])
@pytest.mark.parametrize("is_error", [False, True])
async def test_write_failure_keeps_bounded_result_without_publishing_paths(
    fail_suffix: str, is_error: bool, filesystem: _RecordingFileSystem
) -> None:
    filesystem.fail_suffix = fail_suffix
    result = types.CallToolResult(
        content=[types.TextContent(text="x" * 5000)], is_error=is_error
    )
    response = await _prepare(result, filesystem)
    assert len(response) <= 512
    assert json.loads(response)["output"] == {"status": "unavailable"}
    assert "private-host-detail" not in response
    assert len(filesystem.paths) == (1 if fail_suffix == ".json" else 2)
    # FileWriter has no multi-file transaction: the first commit can remain.
    assert len(filesystem.texts(".json")) == (0 if fail_suffix == ".json" else 1)
    assert filesystem.texts(".txt") == []
    assert isinstance(response, ToolFailure) is is_error
    if isinstance(response, ToolFailure):
        assert response.error.kind == "mcp_server"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directory",
    ["results/" + "x" * 1000, "results/credential-canary", 'results/"token":"hidden"'],
)
async def test_invalid_output_paths_are_rejected_before_writing(
    directory: str, filesystem: _RecordingFileSystem
) -> None:
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text="x" * 5000)]),
        filesystem,
        directory=directory,
        secrets=("credential-canary",),
    )
    assert filesystem.paths == []
    assert json.loads(response)["output"] == {"status": "unavailable"}
    assert directory not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["bytes", "nodes", "depth"])
async def test_preservation_limits_never_publish_partial_sources(
    kind: str, filesystem: _RecordingFileSystem
) -> None:
    value: object
    if kind == "bytes":
        value = "x" * (8 * 1024 * 1024 + 1)
    elif kind == "nodes":
        value = [0] * 5000
    else:
        value = "deep text"
        for _ in range(40):
            value = {"nested": value}
    response = await _prepare({"content": [], "structuredContent": value}, filesystem)
    assert filesystem.paths == []
    assert json.loads(response)["output"] == {"status": "unavailable"}
    assert len(response) <= 512


@pytest.mark.asyncio
async def test_oversized_input_is_rejected_before_source_parsing_or_redaction(
    monkeypatch: pytest.MonkeyPatch, filesystem: _RecordingFileSystem
) -> None:
    parse = source_module.parse_source_json
    redact = redact_known_secrets

    def bounded_parse(value: str) -> object:
        assert len(value) <= 8 * 1024 * 1024
        return parse(value)

    def bounded_redact(value: object, secrets: Any) -> object:
        assert not isinstance(value, str) or len(value) <= 8 * 1024 * 1024
        return redact(value, secrets)

    monkeypatch.setattr(source_module, "parse_source_json", bounded_parse)
    monkeypatch.setattr(source_module, "redact_known_secrets", bounded_redact)
    response = await _prepare(
        {
            "content": [
                {"type": "text", "text": ' {"password":"' + "x" * (9 * 1024 * 1024)}
            ]
        },
        filesystem,
    )
    assert filesystem.paths == []
    assert json.loads(response)["output"] == {"status": "unavailable"}


class _BlockedFileSystem(LocalFileSystem):
    """Hold a real write at context exit until the test permits its commit."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.paths: list[str] = []

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        self.paths.append(path)
        with super().open_write(path) as stream:
            yield stream
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Test did not release the blocked writer.")
        self.finished.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_token", [True, False])
async def test_cancellation_waits_for_active_write_and_does_not_return_paths(
    use_token: bool, tmp_path: Path
) -> None:
    token = CancellationToken()
    filesystem = _BlockedFileSystem(tmp_path)
    task = asyncio.create_task(
        _prepare(
            {"content": [{"type": "text", "text": "x" * 5000}]},
            filesystem,
            token=token,
        )
    )
    try:
        assert await asyncio.to_thread(filesystem.started.wait, 2)
        if use_token:
            token.cancel()
        else:
            task.cancel()
        # Let cancellation reach the shared helper, then cancel the owner again.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not filesystem.finished.is_set()
    finally:
        filesystem.release.set()

    if use_token:
        response = await asyncio.wait_for(task, timeout=2)
        assert isinstance(response, ToolFailure)
        assert response.error.kind == "cancelled"
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    assert filesystem.finished.is_set()
    assert len(filesystem.paths) == 1
    assert (tmp_path / filesystem.paths[0]).is_file()
    assert list(tmp_path.rglob("*.txt")) == []


@pytest.mark.asyncio
async def test_cancelled_token_does_not_start_a_write(
    filesystem: _RecordingFileSystem,
) -> None:
    token = CancellationToken()
    token.cancel()
    response = await _prepare(
        {"content": [{"type": "text", "text": "x" * 5000}]},
        filesystem,
        token=token,
    )
    assert isinstance(response, ToolFailure)
    assert response.error.kind == "cancelled"
    assert filesystem.paths == []
