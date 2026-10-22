"""Minimal run-state contracts and copy utilities for the harness.

This module defines the canonical data shapes that flow between the agent
lifecycle and the runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypedDict, cast, overload
from uuid import UUID

from pydantic import BaseModel, JsonValue

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


@dataclass(frozen=True, slots=True)
class RunContinuation:
    """Native state at one external call, with no completed tool result.

    The host must persist this value and claim it before resume. Its identity
    does not provide authorization or prevent replay in another process.
    Mutable input values are copied; consumers must treat this value as read-only.
    """

    agent_id: AgentId
    """Logical agent address that owns the pending call."""

    continuation_id: UUID
    """Fresh immutable identity assigned at this suspension boundary."""

    base_revision: int
    """Successful snapshot revision from which this execution started."""

    run_state: RunState
    """Private working state, including the assistant's unresolved call."""

    tool_call: ToolCall
    """Original provider call whose result the host must supply."""

    arguments: dict[str, JsonValue]
    """Arguments validated against the registered external tool."""

    args_schema: dict[str, JsonValue]
    """Registered function schema used to detect a changed tool contract."""

    tool_call_counts: Mapping[str, int]
    """Used calls per tool, including the pending external call."""

    tool_round_trips: int
    """Used tool rounds, including the pending external round."""

    max_turns: int
    """Effective model-turn limit that resume must not increase."""

    tool_call_limits: Mapping[str, int] | None
    """Effective per-tool call limits, or no per-tool limits."""

    max_tool_round_trips: int
    """Effective tool-round limit that resume must not increase."""

    def __post_init__(self) -> None:
        """Validate native bounds and detach caller-owned mutable values."""
        if not isinstance(cast(object, self.agent_id), AgentId):
            raise TypeError("Continuation agent_id must be an AgentId.")
        if not isinstance(cast(object, self.continuation_id), UUID):
            raise TypeError("Continuation continuation_id must be a UUID.")
        _require_nonnegative_int(self.base_revision, "base_revision")
        _require_nonnegative_int(self.run_state.turn_count, "turn_count")
        _require_nonnegative_int(self.tool_round_trips, "tool_round_trips")
        _require_nonnegative_int(self.max_turns, "max_turns")
        _require_nonnegative_int(self.max_tool_round_trips, "max_tool_round_trips")
        if self.run_state.revision != self.base_revision:
            raise ValueError("Continuation state revision must match base_revision.")
        if not 1 <= self.run_state.turn_count <= self.max_turns:
            raise ValueError("Continuation turn_count must be within max_turns.")
        if not 1 <= self.tool_round_trips <= self.max_tool_round_trips:
            raise ValueError("Continuation tool rounds must be within the saved limit.")

        # Reject missing correlation before storing an external result slot.
        call = self.tool_call.model_copy(deep=True)
        if not call.id or not call.function.name or call.type != "function":
            raise ValueError("Continuation requires a named function call and call ID.")
        counts = _copy_tool_counts(self.tool_call_counts, "tool_call_counts")
        limits = (
            _copy_tool_counts(self.tool_call_limits, "tool_call_limits")
            if self.tool_call_limits is not None
            else None
        )
        if counts.get(call.function.name, 0) < 1:
            raise ValueError("Continuation must count the pending external tool call.")
        if limits is not None and any(
            count > limits[name] for name, count in counts.items() if name in limits
        ):
            raise ValueError("Continuation tool counts exceed the saved limits.")

        # Each stored value belongs to this continuation, not its creator.
        state = copy_run_state(self.run_state)
        state.responses = [
            response.model_copy(deep=True) for response in state.responses
        ]
        object.__setattr__(self, "run_state", state)
        object.__setattr__(self, "tool_call", call)
        object.__setattr__(self, "arguments", _copy_json_object(self.arguments))
        object.__setattr__(self, "args_schema", _copy_json_object(self.args_schema))
        object.__setattr__(self, "tool_call_counts", MappingProxyType(counts))
        object.__setattr__(
            self,
            "tool_call_limits",
            MappingProxyType(limits) if limits is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ToolResultInput:
    """Trusted text that completes the exact call in a saved continuation."""

    continuation: RunContinuation
    """Host-claimed pending execution; no ordinary user input is appended."""

    result: str
    """Model-readable external result, already validated by the host."""

    def __post_init__(self) -> None:
        """Copy the claimed continuation before asynchronous execution starts."""
        if not isinstance(cast(object, self.result), str):
            raise TypeError("External tool result must be a string.")
        object.__setattr__(
            self, "continuation", copy_run_continuation(self.continuation)
        )


type RunInput = str | list[RunHistoryItem] | RunState | ToolResultInput
"""Public input accepted by the default harness agent.

A plain `str` starts or continues a conversation with a single user
message. A `list[RunHistoryItem]` provides a richer multi-item payload (e.g.
a `PromptSpec` mixed with prior `ModelResponse` objects). A
`RunState` resumes a previously persisted conversation wholesale.
`ToolResultInput` completes one pending external call without a new user message.
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


@dataclass(frozen=True, slots=True)
class RunSuspension:
    """Return control without committing a successful run or a tool result."""

    continuation: RunContinuation
    """Portable native execution state for the host to persist."""


type RunOutcome = RunResult | RunSuspension
"""A completed run or an external call that still needs a result."""


def copy_run_continuation(continuation: RunContinuation) -> RunContinuation:
    """Validate and copy one continuation at a native ownership boundary."""
    return RunContinuation(
        agent_id=continuation.agent_id,
        continuation_id=continuation.continuation_id,
        base_revision=continuation.base_revision,
        run_state=continuation.run_state,
        tool_call=continuation.tool_call,
        arguments=continuation.arguments,
        args_schema=continuation.args_schema,
        tool_call_counts=continuation.tool_call_counts,
        tool_round_trips=continuation.tool_round_trips,
        max_turns=continuation.max_turns,
        tool_call_limits=continuation.tool_call_limits,
        max_tool_round_trips=continuation.max_tool_round_trips,
    )


def copy_tool_result_input(run_input: ToolResultInput) -> ToolResultInput:
    """Detach a supplied result from objects that the caller can still mutate."""
    return ToolResultInput(continuation=run_input.continuation, result=run_input.result)


def _require_nonnegative_int(value: int, name: str) -> None:
    """Reject coerced and negative execution counters."""
    if type(value) is not int or value < 0:
        raise ValueError(f"Continuation {name} must be a nonnegative integer.")


def _copy_tool_counts(values: Mapping[str, int], name: str) -> dict[str, int]:
    """Copy strict counters before storing them in a read-only map."""
    copied: dict[str, int] = {}
    for tool_name, value in values.items():
        if not isinstance(cast(object, tool_name), str) or not tool_name:
            raise ValueError(f"Continuation {name} requires nonempty tool names.")
        _require_nonnegative_int(value, name)
        copied[tool_name] = value

    return copied


def _copy_json_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Copy JSON data and reject values that cannot survive persistence."""
    copied = serialize_value(value)
    if not isinstance(copied, dict):
        raise TypeError("Continuation arguments and schema must be JSON objects.")

    return copied


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
