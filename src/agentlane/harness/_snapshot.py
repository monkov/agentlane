"""Versioned JSON values for committed state and pending external calls."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Self, cast
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
    model_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import ModelResponse, ToolCall, render_instruction_text

from ._render import render_request_messages
from ._run import RunHistoryItem, RunState, ShimState
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
        instructions = run_state.instructions
        if instructions is not None and not isinstance(instructions, str):
            instructions = render_instruction_text(instructions)

        return cls(
            schema_version=1,
            agent_id=agent_id,
            revision=run_state.revision,
            created_at=datetime.now(UTC),
            state=_RunStatePayload.model_validate(
                {
                    "instructions": instructions,
                    "history": render_request_messages(None, run_state.history),
                    "responses": run_state.responses,
                    "shim_state": run_state.shim_state,
                    "turn_count": run_state.turn_count,
                }
            ),
        )

    def to_run_state(self) -> RunState:
        """Return an isolated live run state restored from this snapshot."""
        state = self.state.model_copy(deep=True)
        return RunState(
            instructions=state.instructions,
            history=cast(list[RunHistoryItem], state.history),
            responses=state.responses,
            shim_state=ShimState(context=state.shim_state),
            turn_count=state.turn_count,
            revision=self.revision,
        )

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON-safe snapshot mapping."""
        return cast(dict[str, object], self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode one version-1 snapshot mapping."""
        return cls.model_validate(data)


class RunContinuation(BaseModel):
    """Uncommitted working state for one external call, stored and claimed by the host.

    The checkpoint reuses the snapshot codec; it must not enter the completed
    snapshot store. This envelope grants no permission to resume. Treat nested
    data as read-only; transport and execution boundaries validate and copy it.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["external_tool_continuation"] = "external_tool_continuation"
    """Keep pending execution separate from completed snapshots."""
    format_version: int = Field(default=1, ge=1, le=1)
    """Version of this continuation format, not a state revision."""
    continuation_id: UUID = Field(default_factory=uuid4)
    """Runtime-generated identity for the host's durable claim."""
    checkpoint: AgentSnapshot
    """Uncommitted state, owner, and completed baseline revision."""
    tool_call: ToolCall
    """Original provider call ID, function name, and raw arguments."""
    arguments: dict[str, JsonValue]
    """Validated arguments for the host's external interaction."""
    args_schema: dict[str, JsonValue]
    """Saved function schema used to reject a changed tool contract."""
    tool_call_counts: dict[str, NonNegativeInt]
    """Used calls, including the pending call; stricter caps may be below usage."""
    tool_round_trips: NonNegativeInt
    """Used tool rounds, including the pending round."""
    max_turns: NonNegativeInt
    """Saved model-turn limit that resume must not increase."""
    tool_call_limits: dict[str, NonNegativeInt] | None
    """Saved per-tool limits, or no configured limits."""
    max_tool_round_trips: NonNegativeInt
    """Saved tool-round limit that resume must not increase."""

    @field_validator("continuation_id", mode="before")
    @classmethod
    def _decode_continuation_id(cls, value: object) -> UUID:
        """Accept UUID objects and their JSON text representation."""
        if isinstance(value, str):
            return UUID(value)
        if isinstance(value, UUID):
            return value
        raise ValueError("Continuation identity must be a UUID.")

    @field_validator("checkpoint", mode="before")
    @classmethod
    def _check_checkpoint(cls, value: object) -> object:
        """Validate strict pending state without changing the snapshot v1 contract."""
        raw = value.model_dump() if isinstance(value, AgentSnapshot) else value
        if not isinstance(raw, dict) or set(cast(dict[object, object], raw)) != set(
            AgentSnapshot.model_fields
        ):
            raise ValueError("Continuation checkpoint requires canonical fields.")
        checkpoint = cast(dict[str, object], raw)
        state = checkpoint["state"]
        identity = checkpoint["agent_id"]
        if not isinstance(state, dict) or set(cast(dict[object, object], state)) != set(
            _RunStatePayload.model_fields
        ):
            raise ValueError("Continuation state requires canonical fields.")
        if not isinstance(identity, dict) or set(
            cast(dict[object, object], identity)
        ) != {"type", "key"}:
            raise ValueError("Continuation agent_id requires only type and key.")
        counters = (
            checkpoint["schema_version"],
            checkpoint["revision"],
            cast(dict[str, object], state)["turn_count"],
        )
        if any(type(counter) is not int for counter in counters):
            raise ValueError("Continuation checkpoint counters must be integers.")

        return checkpoint

    @field_validator("tool_call", mode="before")
    @classmethod
    def _copy_tool_call(cls, value: object) -> object:
        """Detach the pending call from the caller's nested model."""
        return value.model_dump() if isinstance(value, ToolCall) else value

    @model_validator(mode="after")
    def _check_pending_run(self) -> Self:
        """Require consistent pending history, counters, schema, and JSON values."""
        call = self.tool_call
        if not call.id or not call.function.name or call.type != "function":
            raise ValueError("Continuation requires a named function and call ID.")
        if not 1 <= self.checkpoint.state.turn_count <= self.max_turns:
            raise ValueError("Continuation turn_count exceeds the saved limit.")
        if not 1 <= self.tool_round_trips <= self.max_tool_round_trips:
            raise ValueError("Continuation tool rounds exceed the saved limit.")
        if self.tool_call_counts.get(call.function.name, 0) < 1:
            raise ValueError("Continuation must count the pending tool call.")
        _validate_pending_call(self)

        # Check raw JSON payloads before Pydantic can encode nonfinite values as null.
        # Snapshot timestamps use the existing snapshot codec instead.
        for value in (self.checkpoint.state, call, self.arguments, self.args_schema):
            serialize_value(value)

        return self

    def to_json(self) -> dict[str, object]:
        """Return checked pending state for the host to store and claim."""
        checked = self.from_json(self.model_dump())

        return cast(dict[str, object], checked.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode and copy pending state, rejecting inconsistent nested values."""
        raw = data.model_dump() if isinstance(data, cls) else data
        if not isinstance(raw, dict) or set(cast(dict[object, object], raw)) != set(
            cls.model_fields
        ):
            raise ValueError("Continuation requires the canonical envelope fields.")

        return cls.model_validate(raw)

    def to_run_state(self) -> RunState:
        """Restore a private working copy for the existing run entry points."""
        return self.from_json(self.model_dump()).checkpoint.to_run_state()


@dataclass(frozen=True, slots=True)
class ToolResultInput:
    """Host-validated text for a claimed continuation, passed to a new Runner."""

    continuation: RunContinuation
    """Exact pending execution; this is not a new user message."""
    result: str
    """Model-readable result already validated by the host."""

    def __post_init__(self) -> None:
        """Detach and revalidate caller-owned data before execution."""
        if not isinstance(cast(object, self.result), str):
            raise TypeError("External tool result must be a string.")
        copied = RunContinuation.from_json(self.continuation.model_dump())
        object.__setattr__(self, "continuation", copied)


class RunPaused(Exception):
    """Signal a durable waiting boundary after the run's resources close."""

    def __init__(self, continuation: RunContinuation) -> None:
        """Own a validated copy for the host to persist and claim."""
        self.continuation = RunContinuation.from_json(continuation.model_dump())
        super().__init__(f"Run waits for external call {continuation.tool_call.id}.")


def _validate_pending_call(continuation: RunContinuation) -> None:
    """Require resolved previous rounds and exactly the original final call."""
    pending: dict[str, ToolCall] = {}
    seen: set[str] = set()
    history = continuation.checkpoint.state.history

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
