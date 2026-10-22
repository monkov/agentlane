"""Prepare complete MCP source files independently of preview limits."""

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TypedDict, cast

from pydantic import AnyUrl, BaseModel

from ._redaction import is_secret_key, redact_known_secrets, redact_secret_pairs
from ._result_values import (
    NOT_JSON,
    block_type,
    get_field,
    json_text,
    read_result,
    value_fields,
)
from ._types import MCPResultPolicy

_LINE_BYTES = 4096
_CONTROLS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


class SourceUnavailable(ValueError):
    """The source cannot be safely represented in the saved files."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceUnavailable("MCP JSON text contains duplicate keys.")
        result[key] = value

    return result


def parse_source_json(text: str) -> object:
    """Reject ambiguous JSON instead of losing values before redaction."""
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        # Bracketed prose is valid transcript text. An incomplete JSON object
        # cannot be safely decoded for escaped credential keys.
        if re.match(r'\s*(?:\{\s*"|\[\s*["{])', text) or re.search(
            r'"(?:[^"\\]|\\.)*"\s*:', text
        ):
            raise SourceUnavailable("MCP JSON text is incomplete.") from None

        return NOT_JSON


class _SourceDocument(TypedDict):
    schema_version: int
    server_name: str
    tool_name: str
    content: list[object]
    structuredContent: object
    isError: bool
    omissions: list[str]


@dataclass(frozen=True, slots=True)
class PreparedSource:
    source_json: str
    readable_text: str
    needs_storage: bool


class _Sanitizer:
    """Remove credentials and binary bodies from supported provider values."""

    def __init__(self, secrets: Collection[str]) -> None:
        self.secrets = tuple(secrets)
        self.redactions = 0
        self.omitted_binary = False
        self.has_resource_text = False

    def safe_text(self, value: str) -> str:
        safe = redact_secret_pairs(cast(str, redact_known_secrets(value, self.secrets)))
        if safe != value:
            self.redactions += 1

        return safe

    def value(
        self,
        value: object,
        *,
        block: bool = False,
        resource: bool = False,
    ) -> object:
        if isinstance(value, AnyUrl):
            value = str(value)

        if isinstance(value, str):
            is_json = value.lstrip().startswith(("{", "["))

            # JSON text can hide escaped credential keys. Sanitize the parsed
            # structure, then keep it a string in the source document.
            parsed = parse_source_json(value) if is_json else NOT_JSON
            if parsed is not NOT_JSON:
                before = self.redactions
                safe = self.value(parsed)
                text = value if self.redactions == before else json_text(safe)
                return self.safe_text(text)

            return self.safe_text(value)

        if isinstance(value, (Mapping, BaseModel)):
            return self.object(
                cast(Mapping[str, object] | BaseModel, value),
                block=block,
                resource=resource,
            )

        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            sequence = cast(Sequence[object], value)
            return [self.value(item) for item in sequence]

        if value is None or isinstance(value, (bool, int, float)):
            if isinstance(value, float) and not isfinite(value):
                raise SourceUnavailable("MCP source contains a non-finite number.")

            return value

        raise SourceUnavailable("MCP source contains an unsupported value.")

    def object(
        self,
        value: Mapping[str, object] | BaseModel,
        *,
        block: bool,
        resource: bool,
    ) -> dict[str, object]:
        kind = block_type(value) if block else None
        result: dict[str, object] = {}
        for name, alias in value_fields(value):
            if not isinstance(cast(object, alias), str):
                raise SourceUnavailable("MCP source has a non-string key.")

            safe_key = self.safe_text(alias)
            if safe_key in result:
                raise SourceUnavailable("MCP source keys collide after redaction.")

            # Never visit a credential value. Binary bodies are the only
            # intentional content omission in a saved source.
            if is_secret_key(alias):
                result[safe_key] = "[redacted]"
                self.redactions += 1
                continue

            item = get_field(value, name)
            if (block and kind in {"image", "audio"} and alias == "data") or (
                resource and alias == "blob"
            ):
                result[safe_key] = {
                    "omitted": "binary",
                    "encoded_bytes": len(item) if isinstance(item, str) else 0,
                }
                self.omitted_binary = True
                continue

            if resource and alias == "text":
                self.has_resource_text = True
            result[safe_key] = self.value(
                item,
                resource=block and kind == "resource" and alias == "resource",
            )

        return result


class _ReadableView:
    """Render sanitized fields into short UTF-8 lines for native file tools."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, value: str) -> None:
        escaped = _CONTROLS.sub(lambda match: f"\\u{ord(match[0]):04x}", value)
        encoded = escaped.encode("utf-8")

        # Chunk the encoded text so one long scalar does not block pagination.
        start = 0
        while start < len(encoded):
            end = min(start + _LINE_BYTES, len(encoded))
            while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end -= 1
            part = encoded[start:end].decode("utf-8")
            if not part.endswith("\n"):
                part += "\n"
            self.parts.append(part)
            start = end

    def value(self, value: object, path: str) -> None:
        # Expand JSON text for the reader while JSON source retains its type.
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            parsed = parse_source_json(value)
            if parsed is not NOT_JSON:
                self.value(parsed, path)
                return

        scalar = value
        if isinstance(value, dict) and value:
            for key, item in cast(dict[str, object], value).items():
                self.value(item, f"{path}.{key}")
        elif isinstance(value, list) and value:
            for index, item in enumerate(cast(list[object], value)):
                self.value(item, f"{path}[{index}]")
        else:
            self.write(f"{path}:")
            self.write(scalar if isinstance(scalar, str) else json_text(scalar))


def prepare_source(
    result: object,
    *,
    server_name: str,
    tool_name: str,
    policy: MCPResultPolicy,
    secrets: Collection[str],
) -> PreparedSource:
    """Keep supported content independently of the model preview's limits."""
    source = read_result(result)
    sanitizer = _Sanitizer(secrets)
    safe_server = sanitizer.safe_text(server_name)
    safe_tool = sanitizer.safe_text(tool_name)
    if safe_server != server_name or safe_tool != tool_name:
        raise SourceUnavailable("MCP source identity contains credentials.")

    content = [sanitizer.value(block, block=True) for block in source.content]
    structured = sanitizer.value(source.structured_content)
    document = _SourceDocument(
        schema_version=1,
        server_name=safe_server,
        tool_name=safe_tool,
        content=content,
        structuredContent=structured,
        isError=source.is_error,
        omissions=["binary"] if sanitizer.omitted_binary else [],
    )
    source_json = json_text(document)

    # Include every supported field when assessing the full inline size.
    full_preview: dict[str, object] = {"content": content, "isError": source.is_error}
    if structured is not None:
        full_preview["structuredContent"] = structured
    hidden_text = (
        not source.is_error
        and bool(structured)
        and any(
            isinstance(block, dict)
            and cast(dict[str, object], block).get("type") == "text"
            for block in content
        )
    )
    needs_storage = (
        len(json_text(full_preview)) > policy.max_text_chars
        or len(content) > policy.max_content_blocks
        or hidden_text
        or sanitizer.has_resource_text
        or (structured is not None and not policy.include_structured_content)
    )

    view = _ReadableView()
    view.value(document, "result")

    return PreparedSource(source_json, "".join(view.parts), needs_storage)
