"""Render complete MCP values as readable text, without preview limits."""

import json
import re
from collections.abc import Collection, Mapping, Sequence
from math import isfinite
from typing import cast

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

_LINE_BYTES = 4096
_CONTROLS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("MCP JSON text contains duplicate keys.")
        result[key] = value

    return result


def _parse_json(text: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        # Bracketed prose is valid text. Incomplete JSON can hide escaped
        # credential keys, so it must not be written as unexamined text.
        if re.match(r'\s*(?:\{\s*"|\[\s*["{])', text) or re.search(
            r'"(?:[^"\\]|\\.)*"\s*:', text
        ):
            raise ValueError("MCP JSON text is incomplete.") from None

        return NOT_JSON


class _ReadableView:
    """Redact values and emit short UTF-8 lines for native file tools."""

    def __init__(self, secrets: Collection[str]) -> None:
        self.secrets = tuple(secrets)
        self.parts: list[str] = []

    def safe_text(self, value: str) -> str:
        return redact_secret_pairs(cast(str, redact_known_secrets(value, self.secrets)))

    def write(self, value: str) -> None:
        escaped = _CONTROLS.sub(lambda match: f"\\u{ord(match[0]):04x}", value)
        encoded = escaped.encode("utf-8")

        # Wrap complete values so a long scalar cannot block read pagination.
        start = 0
        while start < len(encoded):
            end = min(start + _LINE_BYTES, len(encoded))
            while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end -= 1
            part = encoded[start:end].decode("utf-8")
            self.parts.append(part if part.endswith("\n") else part + "\n")
            start = end

    def value(
        self,
        value: object,
        path: str,
        *,
        block: bool = False,
        resource: bool = False,
    ) -> None:
        if isinstance(value, AnyUrl):
            value = str(value)

        if isinstance(value, str):
            # Parse JSON text before redaction to expose escaped credential keys
            # and make nested transcript values readable without JSON escapes.
            parsed = (
                _parse_json(value)
                if value.lstrip().startswith(("{", "["))
                else NOT_JSON
            )
            if parsed is not NOT_JSON:
                self.value(parsed, path)
                return
            text = self.safe_text(value)
        elif isinstance(value, (Mapping, BaseModel)):
            self.object(
                cast(Mapping[str, object] | BaseModel, value),
                path,
                block=block,
                resource=resource,
            )
            return
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            sequence = cast(Sequence[object], value)
            if sequence:
                for index, item in enumerate(sequence):
                    self.value(item, f"{path}[{index}]")
                return
            text = "[]"
        elif value is None or isinstance(value, (bool, int, float)):
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("MCP source contains a non-finite number.")
            text = json_text(value)
        else:
            raise ValueError("MCP source contains an unsupported value.")

        self.write(f"{path}:")
        self.write(text)

    def object(
        self,
        value: Mapping[str, object] | BaseModel,
        path: str,
        *,
        block: bool,
        resource: bool,
    ) -> None:
        kind = block_type(value) if block else None
        keys: set[str] = set()
        for name, alias in value_fields(value):
            if not isinstance(cast(object, alias), str):
                raise ValueError("MCP source has a non-string key.")
            safe_key = self.safe_text(alias)
            if safe_key in keys:
                raise ValueError("MCP source keys collide after redaction.")
            keys.add(safe_key)
            item_path = f"{path}.{safe_key}"

            # Never read credential values or emit binary bodies.
            if is_secret_key(alias):
                self.write(f"{item_path}:\n[redacted]")
            elif (block and kind in {"image", "audio"} and alias == "data") or (
                resource and alias == "blob"
            ):
                self.write(f"{item_path}:\n[omitted: binary]")
            else:
                self.value(
                    get_field(value, name),
                    item_path,
                    resource=block and kind == "resource" and alias == "resource",
                )

        if not keys:
            self.write(f"{path}:\n{{}}")


def prepare_source(
    result: object,
    *,
    server_name: str,
    tool_name: str,
    secrets: Collection[str],
) -> str:
    """Keep all supported values in one sanitized, readable document."""
    source = read_result(result)
    view = _ReadableView(secrets)
    view.value(server_name, "result.server_name")
    view.value(tool_name, "result.tool_name")
    view.value(source.is_error, "result.isError")
    for index, block in enumerate(source.content):
        view.value(block, f"result.content[{index}]", block=True)
    view.value(source.structured_content, "result.structuredContent")

    return "".join(view.parts)
