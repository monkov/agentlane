"""Save supported MCP content with the host's existing file writer."""

import json
from collections.abc import Collection
from pathlib import PurePosixPath
from uuid import uuid4

from agentlane.harness.filesystem import FileWriter
from agentlane.models import ToolFailure

from .._storage_write import write_content
from ._redaction import redact_known_secrets, redact_secret_pairs
from ._result import render_mcp_result
from ._source import has_hidden_content, prepare_source
from ._types import MCPResultPolicy


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
) -> str | ToolFailure:
    """Write one readable file before returning its path in the preview."""
    preview = render_mcp_result(result, policy, secrets=secrets)
    if writer is None:
        return preview

    # Decide before preparing the full document. Failures must not repeat the
    # provider operation or replace its error kind.
    try:
        preview_text = preview.text if isinstance(preview, ToolFailure) else preview
        if (
            (server_name, tool_name) not in preserve_tools
            and json.loads(preview_text).get("truncated") is not True
            and not has_hidden_content(result, policy)
        ):
            return preview

        # The host selects the directory. Provider data never supplies a path.
        if not directory:
            raise ValueError("MCP result storage requires a directory.")
        read_path = str(PurePosixPath(directory) / f"{uuid4()}.txt")
        if (
            redact_known_secrets(read_path, secrets) != read_path
            or redact_secret_pairs(read_path) != read_path
        ):
            raise ValueError("MCP output path contains credentials.")

        text = prepare_source(
            result, server_name=server_name, tool_name=tool_name, secrets=secrets
        )
        # The call handler owns cancellation. The shared helper waits for a
        # started thread write to finish before cancellation propagates.
        await write_content(writer, read_path, text.encode("utf-8"))
    except Exception:
        return render_mcp_result(
            result, policy, secrets=secrets, storage_unavailable=True
        )

    return render_mcp_result(result, policy, secrets=secrets, read_path=read_path)
