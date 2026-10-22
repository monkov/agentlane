"""Prepare MCP text once, then return it inline or save it for later reads."""

from collections.abc import Collection
from pathlib import PurePosixPath
from uuid import uuid4

from agentlane.harness.filesystem import FileWriter
from agentlane.models import ToolFailure

from .._storage_write import write_content
from ._redaction import redact_known_secrets, redact_secret_pairs
from ._result import render_mcp_result, result_response
from ._result_values import json_text, read_result
from ._source import prepare_source
from ._types import MCPResultPolicy

_UNAVAILABLE = "\n\n[Full MCP result unavailable.]"


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
    """Use the original renderer only when complete storage is disabled."""
    if writer is None:
        return render_mcp_result(result, policy, secrets=secrets)

    # Prepare all supported content once. Inline output, saving, and failure
    # previews use this same text without visiting the provider result again.
    is_error = read_result(result).is_error
    try:
        text = prepare_source(
            result, server_name=server_name, tool_name=tool_name, secrets=secrets
        )
    except Exception:
        return result_response(
            "[MCP result unavailable: could not prepare readable text.]", is_error
        )

    if (
        len(text) <= policy.max_text_chars
        and (server_name, tool_name) not in preserve_tools
    ):
        return result_response(text, is_error)

    # The host selects the directory. The call handler owns cancellation, and
    # the shared writer waits for an active write to finish before propagating it.
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
        notice = f"\n\n[Full MCP result: {json_text(read_path)}]"
    except Exception:
        notice = _UNAVAILABLE

    # Keep the complete reference inside the response budget. Only the already
    # sanitized text can be shortened; the saved file retains the full text.
    if len(notice) > policy.max_text_chars:
        notice = _UNAVAILABLE
    available = policy.max_text_chars - len(notice)
    preview = text[:available]
    if len(text) > available and available:
        preview = preview[:-1] + "…"

    return result_response(preview + notice, is_error)
