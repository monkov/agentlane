"""Minimal run-state contracts and copy utilities for the harness.

This module defines the canonical data shapes that flow between the agent
lifecycle and the runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Self, TypedDict, cast, overload
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
    model_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    ModelResponse,
    PromptSpec,
    RunStateView,
    ToolCall,
)
from agentlane.models.run import DefaultRunContext

from ._serialization import serialize_value

ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX = ":active-skill-names"
"""Documented `shim_state` key suffix that holds active skill names.

A skills shim records the names of the skills active for the current run under
a key ending in this suffix (the shim name is the prefix, so multiple skills
shims never collide). `RunStateView.active_skill_names` reads every such key so
tools can inspect active skills without coupling to a specific shim name or
reaching for a private key. The value must be a list of strings.
"""


class ShimState(DefaultRunContext):
    """Persisted shim-owned state stored in one harness `RunState`.

    This keeps the same mapping-style access as `DefaultRunContext` while
    making the persisted nature of shim-owned state explicit at the harness
    boundary.
    """

    def __eq__(self, other: object) -> bool:
        """Compare shim state by stored mapping contents."""
        if isinstance(other, DefaultRunContext):
            return self.context == other.context
        if isinstance(other, Mapping):
            other_mapping = cast(Mapping[str, object], other)
            return self.context == dict(other_mapping)
        return NotImplemented


def _empty_shim_state() -> ShimState:
    """Return one typed empty persisted shim-state container."""
    return ShimState()


type RunMessageContent = (
    str | int | float | bool | None | BaseModel | dict[str, object] | list[object]
)
"""Supported non-message content values at the harness run boundary."""


type RunInstructions = str | PromptSpec[Any] | None
"""Supported persisted system-instruction source at the harness boundary."""


type RunHistoryItem = MessageDict | ModelResponse | PromptSpec[Any] | RunMessageContent
"""Supported heterogeneous items stored in run input and persisted history.

The harness accepts a small set of structured item kinds:

1. canonical message dicts,
2. prior model responses,
3. prompt specs,
4. user-side content values that can be normalized into message content.
"""


@dataclass(slots=True)
class RunState:
    """Minimal resumable state for one harness agent run.

    The lifecycle creates private working copies before handing state to the
    runner, so failed turns never corrupt the persisted baseline.
    """

    instructions: RunInstructions
    """Single persisted system instruction for this run."""

    history: list[RunHistoryItem]
    """Persisted conversation history for this run.

    Items may be prior `ModelResponse` assistant turns, canonical message
    dicts, prompt specs, or user-side content values. The runner resolves each
    item into canonical `MessageDict` at request time. Normal operation
    appends to this list; whole-list replacement through
    `PreparedTurn.replace_history(...)` is the sanctioned rewrite point for
    compaction and other history-rewrite shims.
    """

    responses: list[ModelResponse]
    """Raw model responses accumulated across turns."""

    shim_state: ShimState = field(default_factory=_empty_shim_state)
    """Persisted shim-owned state that must survive resumed runs."""

    turn_count: int = 0
    """Number of model turns started for this run (1-based during a turn).

    The runner increments this at the start of each turn, before shims prepare
    the turn and before the model is called. While turn N is being prepared or
    executed, `turn_count` is `N`; the first turn therefore observes
    `turn_count == 1`. After the run finishes, it equals the total number of
    turns that ran. Gate first-turn logic on `turn_count == 1` rather than a
    "completed-turns" reading.
    """

    revision: int = 0
    """Committed state generation, incremented after each successful run."""


class RunStatePayload(BaseModel):
    """JSON representation of one canonical harness run state."""

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


class RunContinuation(BaseModel):
    """Portable state for one unresolved external call, owned and claimed by the host.

    This value grants no permission to resume. Treat nested data as read-only;
    execution and codec boundaries validate and copy it before use.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["external_tool_continuation"]
    """Keep unfinished execution separate from completed snapshots."""
    format_version: Literal[1]
    """Version of this continuation format, not a state revision."""
    agent_id: AgentId
    """Logical owner whose registration must match on resume."""
    continuation_id: UUID
    """Runtime-generated identity for the host's durable claim."""
    base_revision: NonNegativeInt
    """Completed baseline revision, unchanged while the run waits."""
    state: RunStatePayload
    """Canonical working state with the unresolved assistant call."""
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

    @field_validator("agent_id", mode="before")
    @classmethod
    def _decode_agent_id(cls, value: object) -> AgentId:
        """Use the existing public address format without extra fields."""
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
        """Encode the stable owner with its public wire format."""
        return agent_id.to_json()

    @field_validator("continuation_id", mode="before")
    @classmethod
    def _decode_continuation_id(cls, value: object) -> UUID:
        """Accept UUID objects and their JSON text representation."""
        if isinstance(value, str):
            return UUID(value)
        if isinstance(value, UUID):
            return value
        raise ValueError("Continuation identity must be a UUID.")

    @field_validator("state", mode="before")
    @classmethod
    def _check_state(cls, value: object) -> object:
        """Keep snapshot v1 tolerant while continuation counters stay strict."""
        raw = value.model_dump() if isinstance(value, RunStatePayload) else value
        if not isinstance(raw, dict):
            raise ValueError("Continuation state requires the canonical fields.")
        state = cast(dict[object, object], raw)
        if set(state) != set(RunStatePayload.model_fields):
            raise ValueError("Continuation state requires the canonical fields.")
        if type(state.get("turn_count")) is not int:
            raise ValueError("Continuation turn_count must be an integer.")

        return state

    @field_validator("format_version", mode="before")
    @classmethod
    def _check_version(cls, value: object) -> object:
        """Do not accept booleans as version one."""
        if type(value) is not int:
            raise ValueError("Continuation format_version must be an integer.")

        return value

    @model_validator(mode="after")
    def _check_limits(self) -> Self:
        """Require counters to include the waiting call and fit saved limits."""
        call = self.tool_call
        if not call.id or not call.function.name or call.type != "function":
            raise ValueError("Continuation requires a named function and call ID.")
        if not 1 <= self.state.turn_count <= self.max_turns:
            raise ValueError("Continuation turn_count exceeds the saved limit.")
        if not 1 <= self.tool_round_trips <= self.max_tool_round_trips:
            raise ValueError("Continuation tool rounds exceed the saved limit.")
        if self.tool_call_counts.get(call.function.name, 0) < 1:
            raise ValueError("Continuation must count the pending tool call.")

        return self


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
        copied = RunContinuation.model_validate(self.continuation.model_dump())
        object.__setattr__(self, "continuation", copied)


class RunPaused(Exception):
    """Signal a durable waiting boundary after the run's resources close."""

    def __init__(self, continuation: RunContinuation) -> None:
        """Own a validated copy for the host to persist and claim."""
        self.continuation = RunContinuation.model_validate(continuation.model_dump())
        super().__init__(f"Run waits for external call {continuation.tool_call.id}.")


type RunInput = str | list[RunHistoryItem] | RunState
"""Public input accepted by the default harness agent.

A plain `str` starts or continues a conversation with a single user
message. A `list[RunHistoryItem]` provides a richer multi-item payload (e.g.
a `PromptSpec` mixed with prior `ModelResponse` objects). A
`RunState` resumes a previously persisted conversation wholesale.
"""


class RunResultRecord(TypedDict):
    """Complete JSON-ready result inspection record, without an envelope."""

    final_output: JsonValue
    responses: list[JsonValue]
    turn_count: int
    run_state: dict[str, JsonValue] | None


@dataclass(slots=True)
class RunResult:
    """Minimal final result returned by the default harness agent."""

    final_output: object
    """Final output extracted from the terminal run turn."""

    responses: list[ModelResponse]
    """Raw model responses accumulated across the run."""

    turn_count: int
    """Total number of model turns that ran for this completed run.

    Equal to the final value of `RunState.turn_count` (which the runner
    increments at the start of each turn), so a single-turn run reports `1`.
    """

    run_state: RunState | None = None
    """Final resumable run state for this completed run when available."""

    def to_dict(self) -> RunResultRecord:
        """Return all result fields with strict native inspection conversion.

        Preserve full responses and state, including rendered stored prompts.
        This is not a snapshot codec. Unsupported values and non-string keys
        raise TypeError; cycles and nonfinite numbers raise ValueError. Prompt
        rendering errors propagate unchanged, even in fields a host does not show.
        """
        return cast(RunResultRecord, serialize_value(self))


@overload
def copy_run_state(run_state: RunState) -> RunState: ...


@overload
def copy_run_state(run_state: None) -> None: ...


def copy_run_state(run_state: RunState | None) -> RunState | None:
    """Return an isolated copy of one run state, or `None` passthrough."""
    if run_state is None:
        return None

    return RunState(
        instructions=copy_instructions(run_state.instructions),
        history=[copy_history_item(item) for item in run_state.history],
        responses=list(run_state.responses),
        shim_state=copy_shim_state(run_state.shim_state),
        turn_count=run_state.turn_count,
        revision=run_state.revision,
    )


def copy_shim_state(shim_state: ShimState) -> ShimState:
    """Return an isolated copy of one persisted shim-state container."""
    return ShimState(
        context={key: copy_generic_value(value) for key, value in shim_state.items()}
    )


def copy_instructions(instructions: RunInstructions) -> RunInstructions:
    """Copy one persisted instruction source for state ownership."""
    if instructions is None or isinstance(instructions, str):
        return instructions

    copied_values = (
        cast(Any, copy_generic_value(instructions.values))
        if instructions.values is not None
        else None
    )
    return PromptSpec(
        template=instructions.template,
        values=copied_values,
    )


def copy_history_item(item: RunHistoryItem) -> RunHistoryItem:
    """Copy one typed run-history item for isolated ownership.

    Mutable containers and structured `BaseModel` payloads are copied.
    Everything else — strings, `ModelResponse`, `PromptSpec` — is treated
    as immutable.
    """
    return cast(RunHistoryItem, copy_generic_value(item))


def copy_generic_value(value: object) -> object:
    """Copy one generic shim-state value for isolated ownership."""
    if isinstance(value, list):
        return [copy_generic_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            key: copy_generic_value(item)
            for key, item in cast(dict[str, object], value).items()
        }
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return value


@dataclass(frozen=True, slots=True)
class LiveRunStateView(RunStateView):
    """Read-only `RunStateView` backed by one live harness `RunState`.

    Cheap to construct: it wraps the live `RunState` and the run's task
    identity by reference, reading through on each access. The runner builds
    one per local tool call and stamps it onto `ToolExecutionContext` so tool
    handlers can observe the run without an app-built side channel.
    """

    _run_state: RunState
    _task_id: AgentId

    @property
    def task_id(self) -> str:
        """Return the stable task identity (`str` of the run's task id)."""
        return str(self._task_id)

    @property
    def shim_state(self) -> Mapping[str, object]:
        """Return the live persisted shim state as a read-only mapping."""
        return MappingProxyType(self._run_state.shim_state)

    @property
    def active_skill_names(self) -> tuple[str, ...]:
        """Return active skill names read from the documented state keys."""
        names: list[str] = []
        for key, value in self._run_state.shim_state.items():
            if not key.endswith(ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX):
                continue
            if not isinstance(value, list):
                continue
            for entry in cast(list[object], value):
                if isinstance(entry, str) and entry not in names:
                    names.append(entry)
        return tuple(names)
