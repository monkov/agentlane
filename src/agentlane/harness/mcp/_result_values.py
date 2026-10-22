"""Read SDK fields without serializing unbounded result bodies."""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel

NOT_JSON = object()


class _SDKResult(Protocol):
    content: Sequence[object]
    structured_content: object
    is_error: bool


@dataclass(frozen=True, slots=True)
class _Result:
    content: Sequence[object]
    structured_content: object
    is_error: bool


def read_result(value: object) -> _Result:
    if isinstance(value, Mapping):
        raw = cast(Mapping[str, object], value)
        content = raw.get("content", [])
        return _Result(
            content=sequence(content),
            structured_content=raw.get(
                "structuredContent", raw.get("structured_content")
            ),
            is_error=raw.get("isError", raw.get("is_error", False)) is True,
        )

    source = cast(_SDKResult, value)
    return _Result(source.content, source.structured_content, source.is_error)


def parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return NOT_JSON


def sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)

    return ()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def value_fields(value: Mapping[str, object] | BaseModel) -> Iterator[tuple[str, str]]:
    if isinstance(value, BaseModel):
        for name, field in type(value).model_fields.items():
            yield name, field.serialization_alias or field.alias or name

        for name in value.model_extra or {}:
            yield name, name

        return

    for name in value:
        yield name, name


def get_field(value: Mapping[str, object] | BaseModel, name: str) -> object:
    if isinstance(value, BaseModel):
        return getattr(value, name)

    return value[name]


def block_type(block: object) -> object | None:
    if isinstance(block, (Mapping, BaseModel)):
        return get_field(cast(Mapping[str, object] | BaseModel, block), "type")

    return None
