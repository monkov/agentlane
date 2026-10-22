"""Versioned codecs for completed state and pending external calls."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Self, cast
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import ToolCall, render_instruction_text

from ._render import render_request_messages
from ._run import RunContinuation, RunHistoryItem, RunState, RunStatePayload, ShimState
from ._serialization import serialize_value


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

    state: RunStatePayload
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


def _capture_run_state(run_state: RunState) -> RunStatePayload:
    """Use the existing canonical conversion for both persistence formats."""
    instructions = run_state.instructions
    if instructions is not None and not isinstance(instructions, str):
        instructions = render_instruction_text(instructions)

    return RunStatePayload.model_validate(
        {
            "instructions": instructions,
            "history": render_request_messages(None, run_state.history),
            "responses": run_state.responses,
            "shim_state": run_state.shim_state,
            "turn_count": run_state.turn_count,
        }
    )


def _restore_run_state(state: RunStatePayload, revision: int) -> RunState:
    """Restore a private working copy without marking the state completed."""
    copied = state.model_copy(deep=True)

    return RunState(
        instructions=copied.instructions,
        history=cast(list[RunHistoryItem], copied.history),
        responses=copied.responses,
        shim_state=ShimState(context=copied.shim_state),
        turn_count=copied.turn_count,
        revision=revision,
    )


def capture_run_continuation(
    *,
    agent_id: AgentId,
    run_state: RunState,
    tool_call: ToolCall,
    arguments: dict[str, JsonValue],
    args_schema: dict[str, JsonValue],
    tool_call_counts: Mapping[str, int],
    tool_round_trips: int,
    max_turns: int,
    tool_call_limits: Mapping[str, int] | None,
    max_tool_round_trips: int,
) -> RunContinuation:
    """Capture one waiting call without committing state or granting a claim."""
    if type(run_state.turn_count) is not int:
        raise ValueError("Continuation turn_count must be an integer.")
    continuation = RunContinuation(
        kind="external_tool_continuation",
        format_version=1,
        agent_id=agent_id,
        continuation_id=uuid4(),
        base_revision=run_state.revision,
        state=_capture_run_state(run_state),
        tool_call=tool_call,
        arguments=arguments,
        args_schema=args_schema,
        tool_call_counts=dict(tool_call_counts),
        tool_round_trips=tool_round_trips,
        max_turns=max_turns,
        tool_call_limits=(
            dict(tool_call_limits) if tool_call_limits is not None else None
        ),
        max_tool_round_trips=max_tool_round_trips,
    )

    return continuation_from_json(continuation.model_dump())


def continuation_to_json(continuation: RunContinuation) -> dict[str, object]:
    """Encode checked pending state; the host must store and claim it."""
    validated = continuation_from_json(continuation.model_dump())

    return cast(dict[str, object], validated.model_dump(mode="json"))


def continuation_from_json(data: object) -> RunContinuation:
    """Decode pending state and reject inconsistent history or saved limits."""
    raw = data.model_dump() if isinstance(data, RunContinuation) else data
    continuation = RunContinuation.model_validate(raw)
    _validate_pending_call(continuation)
    serialize_value(continuation)

    return continuation


def validate_run_continuation(continuation: RunContinuation) -> None:
    """Recheck mutable nested data before the runner opens resources."""
    continuation_from_json(continuation.model_dump())


def restore_run_continuation(continuation: RunContinuation) -> RunState:
    """Copy pending state for the existing run entry points on a new Runner."""
    checked = continuation_from_json(continuation.model_dump())

    return _restore_run_state(checked.state, checked.base_revision)


def _validate_pending_call(continuation: RunContinuation) -> None:
    """Require resolved previous rounds and exactly the original final call."""
    pending: dict[str, ToolCall] = {}
    seen: set[str] = set()
    history = continuation.state.history

    # A missing earlier result must not be skipped by a later external answer.
    for message in history:
        raw_calls = message.get("tool_calls")
        role = message["role"]
        if raw_calls:
            if role != "assistant" or pending or not isinstance(raw_calls, list):
                raise ValueError("Continuation has overlapping or malformed calls.")
            for raw_call in raw_calls:
                call = ToolCall.model_validate(raw_call, strict=True)
                if not call.id or call.id in seen:
                    raise ValueError("Continuation has an empty or repeated call ID.")
                seen.add(call.id)
                pending[call.id] = call
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Continuation has a result without its call.")
            del pending[call_id]
        elif pending:
            raise ValueError("Continuation has input after an unresolved call.")

    # The answer completes this call only, with the same name and raw arguments.
    if (
        not history
        or history[-1]["role"] != "assistant"
        or pending != {continuation.tool_call.id: continuation.tool_call}
        or len(cast(list[JsonValue], history[-1].get("tool_calls", []))) != 1
    ):
        raise ValueError("Continuation must end with the sole pending tool call.")
    if continuation.args_schema.get("name") != continuation.tool_call.function.name:
        raise ValueError("Continuation schema does not match the pending call.")
