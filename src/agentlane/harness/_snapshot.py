"""Versioned JSON snapshots for committed harness agent state."""

from datetime import UTC, datetime
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import ModelResponse, ToolCall, render_instruction_text

from ._render import render_request_messages
from ._run import (
    RunContinuation,
    RunHistoryItem,
    RunState,
    ShimState,
    copy_run_continuation,
)
from ._serialization import serialize_value


class _RunStatePayload(BaseModel):
    """JSON representation of one committed harness run state."""

    model_config = ConfigDict(extra="ignore")

    instructions: str | None
    history: list[dict[str, JsonValue]]
    responses: list[ModelResponse]
    shim_state: dict[str, JsonValue]
    turn_count: NonNegativeInt

    @field_validator("history")
    @classmethod
    def _require_message_roles(
        cls,
        history: list[dict[str, JsonValue]],
    ) -> list[dict[str, JsonValue]]:
        """Reject malformed canonical messages."""
        for index, message in enumerate(history):
            if not isinstance(message.get("role"), str):
                raise ValueError(f"history[{index}].role must be a string")
        return history


class AgentSnapshot(BaseModel):
    """Portable JSON value containing one agent's committed harness state."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1]
    """JSON schema version used by this snapshot."""

    agent_id: AgentId
    """Stable logical address that owns the state."""

    revision: NonNegativeInt = 0
    """Committed state generation used to reject stale saves."""

    created_at: AwareDatetime
    """UTC timestamp for when the snapshot value was created."""

    state: _RunStatePayload
    """Canonical model-ready state captured for the agent."""

    @field_validator("agent_id", mode="before")
    @classmethod
    def _decode_agent_id(cls, value: object) -> AgentId:
        """Accept either the value object or its JSON representation."""
        if isinstance(value, AgentId):
            return value
        return AgentId.from_json(value)

    @field_serializer("agent_id")
    def _encode_agent_id(self, agent_id: AgentId) -> dict[str, object]:
        """Encode the logical address with its existing wire contract."""
        return agent_id.to_json()

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, created_at: datetime) -> datetime:
        """Normalize accepted aware timestamps to UTC."""
        return created_at.astimezone(UTC)

    @classmethod
    def capture(cls, *, agent_id: AgentId, run_state: RunState) -> Self:
        """Capture one live run state as a canonical snapshot value."""
        return cls(
            schema_version=1,
            agent_id=agent_id,
            revision=run_state.revision,
            created_at=datetime.now(UTC),
            state=_capture_run_state(run_state),
        )

    def to_run_state(self) -> RunState:
        """Return an isolated live run state restored from this snapshot."""
        return _restore_run_state(self.state, self.revision)

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON-safe snapshot mapping."""
        return cast(dict[str, object], self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode one version-1 snapshot mapping."""
        return cls.model_validate(data)


class _ContinuationStatePayload(_RunStatePayload):
    """Strict state fields inside a continuation, separate from snapshot v1."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _ContinuationFunctionPayload(BaseModel):
    """Original function identity and raw arguments at the waiting boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    """Registered external tool name."""

    arguments: str
    """Original JSON argument text returned by the model."""


class _ContinuationCallPayload(BaseModel):
    """Exact function call that must remain unresolved in saved history."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    """Original call identity used by the canonical tool-result message."""

    type: Literal["function"]
    """External continuation supports function calls only."""

    function: _ContinuationFunctionPayload
    """The function name and original arguments."""


class _RunContinuationPayload(BaseModel):
    """Versioned external-call envelope that cannot decode as AgentSnapshot."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    kind: Literal["external_tool_continuation"]
    """Separate this envelope from completed agent snapshots."""

    format_version: Literal[1]
    """Version of this continuation codec, not a snapshot revision."""

    agent_id: AgentId
    """Stable logical owner of the pending execution."""

    continuation_id: UUID
    """Immutable identity generated at the suspension boundary."""

    base_revision: NonNegativeInt
    """Revision of the successful baseline retained by the agent."""

    state: _ContinuationStatePayload
    """Canonical working history and persisted shim state."""

    tool_call: _ContinuationCallPayload
    """The sole pending call at the end of the canonical history."""

    arguments: dict[str, JsonValue]
    """Validated external tool arguments for the host."""

    args_schema: dict[str, JsonValue]
    """Saved function schema, including parameters and strict-mode selection."""

    tool_call_counts: dict[str, NonNegativeInt]
    """Used calls per tool, including the pending call."""

    tool_round_trips: NonNegativeInt
    """Used tool rounds, including the pending round."""

    max_turns: NonNegativeInt
    """Maximum number of model turns for the logical execution."""

    tool_call_limits: dict[str, NonNegativeInt] | None
    """Saved per-tool call limits when configured."""

    max_tool_round_trips: NonNegativeInt
    """Saved maximum number of tool rounds."""

    @field_validator("format_version", mode="before")
    @classmethod
    def _require_version_integer(cls, value: object) -> int:
        """Reject booleans that otherwise compare equal to version one."""
        if type(value) is not int:
            raise ValueError("Continuation format_version must be an integer.")

        return value

    @field_validator("agent_id", mode="before")
    @classmethod
    def _decode_agent_id(cls, value: object) -> AgentId:
        """Accept only the documented owner identity fields."""
        if isinstance(value, AgentId):
            return value
        if not isinstance(value, dict):
            raise ValueError("Continuation agent_id requires only type and key.")
        identity = cast(dict[object, object], value)
        if set(identity) != {"type", "key"}:
            raise ValueError("Continuation agent_id requires only type and key.")

        return AgentId.from_json(identity)

    @field_serializer("agent_id")
    def _encode_agent_id(self, agent_id: AgentId) -> dict[str, object]:
        """Use the existing public address format."""
        return agent_id.to_json()

    @field_validator("continuation_id", mode="before")
    @classmethod
    def _decode_continuation_id(cls, value: object) -> UUID:
        """Decode UUID text explicitly without coercing other JSON values."""
        if isinstance(value, UUID):
            return value
        if not isinstance(value, str):
            raise ValueError("Continuation identity must be UUID text.")

        return UUID(value)


def continuation_to_json(continuation: RunContinuation) -> dict[str, object]:
    """Encode one valid pending execution without saving or claiming it.

    Raises:
        ValueError: If the continuation has inconsistent history or limits.
        TypeError: If a value cannot be represented as JSON.
    """
    payload = _capture_continuation(continuation)
    _validate_pending_call(payload)

    return cast(dict[str, object], payload.model_dump(mode="json"))


def continuation_from_json(data: object) -> RunContinuation:
    """Decode a strict pending execution without granting permission to run it.

    The host must claim the persisted value before it supplies ToolResultInput.
    Unknown versions, fields, counters, and call-history mismatches fail decoding.
    """
    payload = _RunContinuationPayload.model_validate(data)
    _validate_pending_call(payload)

    return RunContinuation(
        agent_id=payload.agent_id,
        continuation_id=payload.continuation_id,
        base_revision=payload.base_revision,
        run_state=_restore_run_state(payload.state, payload.base_revision),
        tool_call=ToolCall.model_validate(payload.tool_call.model_dump()),
        arguments=payload.arguments,
        args_schema=payload.args_schema,
        tool_call_counts=payload.tool_call_counts,
        tool_round_trips=payload.tool_round_trips,
        max_turns=payload.max_turns,
        tool_call_limits=payload.tool_call_limits,
        max_tool_round_trips=payload.max_tool_round_trips,
    )


def validate_run_continuation(continuation: RunContinuation) -> None:
    """Check a direct native input before opening model or tool resources."""
    _validate_pending_call(_capture_continuation(continuation))


def _capture_run_state(run_state: RunState) -> _RunStatePayload:
    """Use the same canonical conversion for snapshots and continuations."""
    instructions = run_state.instructions
    if instructions is not None and not isinstance(instructions, str):
        instructions = render_instruction_text(instructions)

    return _RunStatePayload.model_validate(
        {
            "instructions": instructions,
            "history": render_request_messages(None, run_state.history),
            "responses": run_state.responses,
            "shim_state": run_state.shim_state,
            "turn_count": run_state.turn_count,
        }
    )


def _restore_run_state(state: _RunStatePayload, revision: int) -> RunState:
    """Copy decoded canonical data into private runtime working state."""
    copied = state.model_copy(deep=True)

    return RunState(
        instructions=copied.instructions,
        history=cast(list[RunHistoryItem], copied.history),
        responses=copied.responses,
        shim_state=ShimState(context=copied.shim_state),
        turn_count=copied.turn_count,
        revision=revision,
    )


def _capture_continuation(continuation: RunContinuation) -> _RunContinuationPayload:
    """Recheck mutable state and serialize it through the strict boundary."""
    copied = copy_run_continuation(continuation)
    state = _capture_run_state(copied.run_state)
    # Strict inspection rejects nonfinite values even inside provider payloads.
    state_json = serialize_value(state)

    return _RunContinuationPayload.model_validate(
        {
            "kind": "external_tool_continuation",
            "format_version": 1,
            "agent_id": copied.agent_id,
            "continuation_id": copied.continuation_id,
            "base_revision": copied.base_revision,
            "state": state_json,
            "tool_call": copied.tool_call.model_dump(mode="json"),
            "arguments": copied.arguments,
            "args_schema": copied.args_schema,
            "tool_call_counts": dict(copied.tool_call_counts),
            "tool_round_trips": copied.tool_round_trips,
            "max_turns": copied.max_turns,
            "tool_call_limits": (
                dict(copied.tool_call_limits)
                if copied.tool_call_limits is not None
                else None
            ),
            "max_tool_round_trips": copied.max_tool_round_trips,
        }
    )


def _validate_pending_call(payload: _RunContinuationPayload) -> None:
    """Require one unresolved final call and fully resolved preceding rounds."""
    history = payload.state.history
    pending: dict[str, _ContinuationCallPayload] = {}
    seen: set[str] = set()

    # Check the full history so resume cannot skip an earlier missing result.
    for message in history:
        raw_calls = message.get("tool_calls")
        role = message["role"]
        if raw_calls:
            if role != "assistant" or pending or not isinstance(raw_calls, list):
                raise ValueError(
                    "Continuation has overlapping or malformed tool calls."
                )
            for raw_call in raw_calls:
                call = _ContinuationCallPayload.model_validate(raw_call)
                if call.id in seen:
                    raise ValueError("Continuation history repeats a tool call ID.")
                seen.add(call.id)
                pending[call.id] = call
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Continuation has a tool result without its call.")
            del pending[call_id]
        elif pending:
            raise ValueError("Continuation has input after an unresolved tool call.")

    # The terminal assistant batch must contain exactly the saved external call.
    if (
        not history
        or history[-1]["role"] != "assistant"
        or pending != {payload.tool_call.id: payload.tool_call}
        or len(cast(list[JsonValue], history[-1].get("tool_calls", []))) != 1
    ):
        raise ValueError(
            "Continuation history must end with the sole pending tool call."
        )
    if payload.args_schema.get("name") != payload.tool_call.function.name:
        raise ValueError(
            "Continuation tool schema name does not match the pending call."
        )

    # Decode cannot retain nonfinite numbers through a permissive nested model.
    serialize_value(payload)
