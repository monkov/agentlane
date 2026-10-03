"""Convert MCP tool results to safe model-facing text."""

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

from agentlane.models import ToolError, ToolFailure

from ._redaction import redact_sensitive_data
from ._types import MCPResultPolicy


def render_mcp_result(
    result: object,
    policy: MCPResultPolicy,
    *,
    secrets: Collection[str] = (),
) -> str | ToolFailure:
    """Render one SDK CallToolResult without exposing binary payloads."""
    payload = _model_payload(result)
    blocks_value = payload.get("content", [])
    blocks: Sequence[object] = (
        cast(Sequence[object], blocks_value)
        if isinstance(blocks_value, Sequence)
        else ()
    )
    rendered_blocks = [_safe_content_block(block) for block in blocks]
    rendered: dict[str, object] = {"content": rendered_blocks}
    structured = payload.get("structuredContent", payload.get("structured_content"))
    if policy.include_structured_content and structured is not None:
        rendered["structuredContent"] = structured
    safe_payload = cast(
        dict[str, object], redact_sensitive_data(rendered, tuple(secrets))
    )
    limited_payload = {
        **safe_payload,
        "content": cast(list[object], safe_payload["content"])[
            : policy.max_content_blocks
        ],
    }
    text = _bounded_json(limited_payload, policy.max_text_chars, source=safe_payload)

    is_error = payload.get("isError", payload.get("is_error", False)) is True
    if is_error:
        return ToolFailure(
            text=text,
            error=ToolError(
                message="MCP server reported a tool failure.", kind="mcp_server"
            ),
        )
    return text


def _bounded_json(
    payload: dict[str, object], limit: int, *, source: dict[str, object]
) -> str:
    """Keep a useful preview and valid JSON within the character limit."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit and payload == source:
        return text

    source_chars = len(_compact_json(source))
    source_blocks = len(cast(list[object], source["content"]))
    accounting = {
        "truncated": True,
        "omittedChars": source_chars,
        "omittedBlocks": source_blocks,
    }
    if len(_compact_json({**payload, **accounting})) <= limit:
        return _with_omission_counts(payload, source_chars, source_blocks)
    structured = payload.get("structuredContent")
    compact: dict[str, object] = {"content": [], **accounting}
    if structured is not None:
        compact["structuredContent"] = {"omitted": True}
    content = payload.get("content", [])
    available = limit - len(_compact_json(compact))
    # Keep room for both text and structured output. The final pass gives the
    # text any unused structured-output budget.
    text_budget = max(2, int(available * (0.6 if structured is not None else 1)) + 2)
    preview = _preview(content, text_budget)
    compact["content"] = [] if preview is _OMITTED else preview
    if structured is not None:
        previous = compact.pop("structuredContent")
        remaining = limit - len(_compact_json(compact)) - len(',"structuredContent":')
        preview = _preview(structured, remaining)
        compact["structuredContent"] = (
            previous if preview is _OMITTED or preview in ({}, []) else preview
        )
    previous_content = compact.pop("content")
    remaining = limit - len(_compact_json(compact)) - len(',"content":')
    preview = _preview(content, remaining)
    compact["content"] = previous_content if preview is _OMITTED else preview
    return _with_omission_counts(compact, source_chars, source_blocks)


def _with_omission_counts(
    payload: dict[str, object], source_chars: int, source_blocks: int
) -> str:
    data = {
        key: value
        for key, value in payload.items()
        if key not in {"truncated", "omittedChars", "omittedBlocks"}
    }
    return _compact_json(
        {
            **data,
            "truncated": True,
            "omittedChars": max(0, source_chars - len(_compact_json(data))),
            "omittedBlocks": source_blocks
            - len(cast(list[object], data.get("content", []))),
        }
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_OMITTED = object()


def _preview(value: object, budget: int) -> object:
    """Return the leading portion of a JSON value that fits the budget."""
    if len(_compact_json(value)) <= budget:
        return value
    if isinstance(value, str):
        low, high = 0, len(value)
        if len(_compact_json("…")) > budget:
            return _OMITTED
        while low < high:
            midpoint = (low + high + 1) // 2
            if len(_compact_json(value[:midpoint] + "…")) <= budget:
                low = midpoint
            else:
                high = midpoint - 1
        return value[:low] + "…"
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        partial: dict[str, object] = {}
        if budget < 2:
            return _OMITTED
        for key, item in mapping.items():
            remaining = (
                budget - len(_compact_json(partial)) - len(_compact_json(key)) - 1
            )
            if partial:
                remaining -= 1
            preview = _preview(item, remaining)
            if preview is _OMITTED:
                break
            partial[key] = preview
            if preview != item:
                break
        return partial
    if isinstance(value, list):
        sequence = cast(list[object], value)
        items: list[object] = []
        if budget < 2:
            return _OMITTED
        for item in sequence:
            remaining = budget - len(_compact_json(items)) - (1 if items else 0)
            preview = _preview(item, remaining)
            if preview is _OMITTED or preview in ({}, []):
                break
            if isinstance(preview, dict):
                block = cast(dict[str, object], preview)
                if block.get("type") == "text" and "text" not in block:
                    break
            items.append(cast(object, preview))  # type: ignore[redundant-cast]
            if preview != item:
                break
        return items
    return _OMITTED


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self, **kwargs: object) -> dict[str, Any]: ...


def _model_payload(value: object) -> dict[str, Any]:
    if isinstance(value, _ModelDumpable):
        dumped = value.model_dump(mode="json", by_alias=True)
        return dumped
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): item for key, item in mapping.items()}
    raise TypeError(f"Unsupported MCP result type: {type(value).__qualname__}.")


def _safe_content_block(value: object) -> object:
    block: dict[str, Any]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        block = {str(key): item for key, item in mapping.items()}
    else:
        block = _model_payload(value)
    block_type = block.get("type")
    if block_type in {"image", "audio"}:
        data = block.pop("data", None)
        block["encodedBytesOmitted"] = len(data) if isinstance(data, str) else 0
    if block_type == "resource":
        resource_value = block.get("resource")
        if isinstance(resource_value, Mapping):
            resource_mapping = cast(Mapping[object, object], resource_value)
            resource = {str(key): item for key, item in resource_mapping.items()}
            blob = resource.pop("blob", None)
            if blob is not None:
                resource["encodedBytesOmitted"] = (
                    len(blob) if isinstance(blob, str) else 0
                )
            resource_text = resource.pop("text", None)
            if resource_text is not None:
                resource["textCharsOmitted"] = (
                    len(resource_text) if isinstance(resource_text, str) else 0
                )
            block["resource"] = resource
    # Put the useful text before optional metadata when a preview is needed.
    if block_type == "text":
        return {"type": block_type, "text": block.get("text", ""), **block}
    return block
