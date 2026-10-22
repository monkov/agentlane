"""Save supported MCP content with the host's existing file writer."""

import asyncio
import json
from collections.abc import Collection
from pathlib import PurePosixPath
from uuid import uuid4

from agentlane.harness.filesystem import FileWriter
from agentlane.models import ToolError, ToolErrorKind, ToolFailure
from agentlane.runtime import CancellationToken

from .._storage_write import write_content
from ._constants import TOOL_CANCELLED
from ._redaction import redact_known_secrets, redact_secret_pairs
from ._result import render_mcp_result
from ._source import prepare_source
from ._types import MCPResultPolicy


def _cancelled() -> ToolFailure:
    return ToolFailure(
        text=TOOL_CANCELLED,
        error=ToolError(message=TOOL_CANCELLED, kind=ToolErrorKind.CANCELLED.value),
    )


async def prepare_mcp_response(
    result: object,
    *,
    policy: MCPResultPolicy,
    secrets: Collection[str],
    server_name: str,
    tool_name: str,
    writer: FileWriter | None,
    directory: str | None,
    preserve_tools: frozenset[tuple[str, str]],
    cancellation_token: CancellationToken,
) -> str | ToolFailure:
    """Write both files before returning their paths in the bounded preview."""
    preview = render_mcp_result(result, policy, secrets=secrets)
    if writer is None:
        return preview
    if cancellation_token.is_cancelled:
        return _cancelled()

    # Preservation failures must not repeat the MCP operation or replace the
    # provider's error kind. File tools use the same write completion rules.
    try:
        source = prepare_source(
            result,
            server_name=server_name,
            tool_name=tool_name,
            policy=policy,
            secrets=secrets,
        )
        preview_text = preview.text if isinstance(preview, ToolFailure) else preview
        preview_truncated = json.loads(preview_text).get("truncated") is True
        if (
            not source.needs_storage
            and not preview_truncated
            and (server_name, tool_name) not in preserve_tools
        ):
            return preview

        # The shim resolves the host-selected directory in the writer's
        # namespace. Provider data never supplies a path or a basename.
        if not directory:
            raise ValueError("MCP result storage requires a directory.")
        basename = str(uuid4())
        source_path = str(PurePosixPath(directory) / f"{basename}.json")
        read_path = str(PurePosixPath(directory) / f"{basename}.txt")
        for path in (source_path, read_path):
            if len(path) > policy.max_text_chars:
                raise ValueError("MCP output path exceeds the preview budget.")
            if (
                redact_known_secrets(path, secrets) != path
                or redact_secret_pairs(path) != path
            ):
                raise ValueError("MCP output path contains credentials.")

        # The existing helper drains an active thread write during cancellation.
        # A completed first file can remain if the second write does not finish.
        for path, text in (
            (source_path, source.source_json),
            (read_path, source.readable_text),
        ):
            if cancellation_token.is_cancelled:
                return _cancelled()
            writing = asyncio.create_task(
                write_content(writer, path, text.encode("utf-8"))
            )
            cancellation_token.link_future(writing)
            await writing

        if cancellation_token.is_cancelled:
            return _cancelled()
    except asyncio.CancelledError:
        if cancellation_token.is_cancelled:
            return _cancelled()
        raise
    except Exception:
        return render_mcp_result(
            result, policy, secrets=secrets, storage_unavailable=True
        )

    return render_mcp_result(
        result, policy, secrets=secrets, source_path=source_path, read_path=read_path
    )
