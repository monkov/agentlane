"""Save oversized MCP results for later reads."""

import json
from collections.abc import Collection
from pathlib import PurePosixPath
from uuid import uuid4

from agentlane.harness.filesystem import FileWriter
from agentlane.models import ToolFailure

from .._storage_write import write_content
from ._redaction import redact_known_secrets, redact_secret_pairs
from ._result import read_result, render_mcp_result, result_response
from ._types import MCPResultPolicy

_UNAVAILABLE = "\n\n[Full MCP result unavailable.]"


async def render_mcp_result_with_storage(
    result: object,
    *,
    policy: MCPResultPolicy,
    secrets: Collection[str],
    writer: FileWriter | None,
    directory: str | None,
) -> str | ToolFailure:
    """Return text inline or a preview with the saved file path."""
    if writer is None:
        return render_mcp_result(result, policy, secrets=secrets)

    is_error = read_result(result).is_error
    try:
        serialized = render_mcp_result(result, policy, secrets=secrets, complete=True)

        # Short TXT lines let the existing read tool page through long strings.
        text = "\n".join(
            serialized[i : i + 4096] for i in range(0, len(serialized), 4096)
        )
    except Exception:
        return result_response(
            "[MCP result unavailable: could not prepare readable text.]", is_error
        )

    if len(text) <= policy.max_text_chars:
        return result_response(text, is_error)

    try:
        if not directory:
            raise ValueError("MCP result storage requires a directory.")
        read_path = str(PurePosixPath(directory) / f"{uuid4()}.txt")
        if (
            redact_known_secrets(read_path, secrets) != read_path
            or redact_secret_pairs(read_path) != read_path
        ):
            raise ValueError("MCP output path contains credentials.")
        await write_content(writer, read_path, text.encode("utf-8"))
        notice = f"\n\n[Full MCP result: {json.dumps(read_path, ensure_ascii=False)}]"
    except Exception:
        notice = _UNAVAILABLE

    # Truncate the preview, never the file path the model needs to read it.
    if len(notice) > policy.max_text_chars:
        notice = _UNAVAILABLE
    available = policy.max_text_chars - len(notice)
    preview = text[:available]
    if len(text) > available and available:
        preview = preview[:-1] + "…"

    return result_response(preview + notice, is_error)
