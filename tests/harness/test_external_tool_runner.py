"""External tool boundaries through the existing runner entry points."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Literal, Self
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    ExternalTool,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    RunSuspension,
    Task,
    ToolResultInput,
)
from agentlane.harness._run import copy_run_state
from agentlane.harness._snapshot import continuation_from_json, continuation_to_json
from agentlane.harness.shims import BoundShim, PreparedTurn
from agentlane.harness.shims._manager import BoundShimManager
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    Model,
    ModelBehaviorError,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
    Tools,
)
from agentlane.runtime import SingleThreadedRuntimeEngine


class _ExternalArgs(BaseModel):
    text: str = Field(description="Information requested from the host.")


class _ChangedArgs(BaseModel):
    topic: str = Field(description="A changed tool contract.")


def _external_tool() -> ExternalTool[_ExternalArgs]:
    return ExternalTool(
        name="external", description="Get an external result.", args_model=_ExternalArgs
    )


def _call(name: str, call_id: str, arguments: str = '{"text":"needed"}') -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    )


def _response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "response",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if calls else "done",
                        "tool_calls": list(calls) or None,
                    },
                }
            ],
        }
    )


class _ScriptedModel(Model[ModelResponse]):
    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        requires_streaming: bool = False,
        extra_events: tuple[ModelStreamEvent, ...] = (),
        continuation: Literal["automatic", "explicit"] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[list[MessageDict]] = []
        self.catalogs: list[Tools | None] = []
        self.opened = 0
        self.closed = 0
        self._requires_streaming = requires_streaming
        self.extra_events = extra_events
        self.continuation: Literal["automatic", "explicit"] | None = continuation

    @property
    def requires_streaming(self) -> bool:
        return self._requires_streaming

    @asynccontextmanager
    async def open_run(self) -> AsyncIterator[Self]:
        self.opened += 1
        try:
            yield self
        finally:
            self.closed += 1

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: object = None,
        schema: object = None,
        tools: Tools | None = None,
        cancellation_token: object = None,
        parent_span: object = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.requests.append([dict(message) for message in messages])
        self.catalogs.append(tools)
        if not self.responses:
            raise AssertionError("Unexpected next model call.")

        return self.responses.pop(0)

    async def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: object = None,
        schema: object = None,
        tools: Tools | None = None,
        cancellation_token: object = None,
        parent_span: object = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamEvent]:
        response = await self.get_response(messages, tools=tools)
        for event in self.extra_events:
            yield event
        yield ModelStreamEvent(
            kind=ModelStreamEventKind.COMPLETED,
            response=response,
            continuation=self.continuation,
        )


class _EndHooks(RunnerHooks):
    def __init__(self) -> None:
        self.results: list[RunResult | None] = []

    async def on_agent_end(self, task: Task, result: RunResult | None) -> None:
        self.results.append(result)


class _RewritePendingShim(BoundShim):
    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.replace_history([{"role": "user", "content": "rewritten"}])


class _ExternalCatalogShim(BoundShim):
    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.add_tools((_external_tool(),))


def _agent(model: _ScriptedModel, runner: Runner, tools: Tools) -> Agent:
    return Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=AgentId.from_values("test", "root"),
        descriptor=AgentDescriptor(name="test", model=model, tools=tools),
    )


def _runner(**kwargs: Any) -> Runner:
    return Runner(external_tool_agent_id=AgentId.from_values("test", "root"), **kwargs)


def _state() -> RunState:
    return RunState(instructions=None, history=["start"], responses=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_external_call_returns_suspension_and_cleans_up(mode: str) -> None:
    runner = _runner()
    model = _ScriptedModel([_response(_call("external", "pending"))])
    agent = _agent(model, runner, Tools(tools=[_external_tool()]))
    hooks = _EndHooks()
    state = _state()

    if mode == "run":
        outcome = await runner.run(agent, state, hooks=hooks)
    else:
        stream = (
            runner.run_events(agent, state, hooks=hooks)
            if mode == "events"
            else runner.run_stream(agent, state, hooks=hooks)
        )
        try:
            outcome = await stream.result()
        finally:
            await stream.aclose()

    assert isinstance(outcome, RunSuspension)
    assert outcome.continuation.tool_call.id == "pending"
    assert outcome.continuation.tool_call_counts == {"external": 1}
    assert outcome.continuation.tool_round_trips == 1
    assert outcome.continuation.run_state.turn_count == 1
    assert outcome.continuation.base_revision == 0
    assert hooks.results == [None]
    assert model.opened == model.closed == 1
    assert len(model.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_restart_resume_keeps_prior_effects_and_original_tool_message(
    mode: str,
) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record one side effect."""
        effects.append(text)
        return "written"

    tools = Tools(
        tools=[write, _external_tool()],
        tool_call_limits={"write": 1, "external": 2},
        max_tool_round_trips=3,
    )
    first = _runner(max_turns=3)
    model = _ScriptedModel(
        [
            _response(_call("write", "effect")),
            _response(_call("external", "pending")),
        ]
    )
    paused = await first.run(_agent(model, first, tools), _state())
    assert isinstance(paused, RunSuspension)
    restored = continuation_from_json(continuation_to_json(paused.continuation))

    # Rebuild the runner and model, as a different host process would do.
    second = _runner(max_turns=999)
    resumed_model = _ScriptedModel([_response()])
    resumed_input = ToolResultInput(continuation=restored, result="host answer")
    resumed_agent = _agent(resumed_model, second, tools)
    resumed_state = copy_run_state(restored.run_state)
    if mode == "run":
        outcome = await second.run(
            resumed_agent, resumed_state, external_input=resumed_input
        )
    else:
        stream = (
            second.run_events(
                resumed_agent, resumed_state, external_input=resumed_input
            )
            if mode == "events"
            else second.run_stream(
                resumed_agent, resumed_state, external_input=resumed_input
            )
        )
        try:
            outcome = await stream.result()
        finally:
            await stream.aclose()

    assert isinstance(outcome, RunResult)
    assert outcome.turn_count == 3
    assert effects == ["needed"]
    tool_messages = [
        message for message in resumed_model.requests[0] if message["role"] == "tool"
    ]
    assert tool_messages == [
        {
            "role": "tool",
            "tool_call_id": "effect",
            "name": "write",
            "content": "written",
        },
        {
            "role": "tool",
            "tool_call_id": "pending",
            "name": "external",
            "content": "host answer",
        },
    ]
    catalog = resumed_model.catalogs[0]
    assert catalog is not None
    assert [tool.name for tool in catalog.normalized_tools] == ["external"]
    assert catalog.tool_call_limits == {"write": 1, "external": 2}
    assert restored.run_state.turn_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_mixed_external_batch_rejects_before_sibling_effects(
    parallel: bool,
) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record one side effect."""
        effects.append(text)
        return text

    runner = _runner()
    model = _ScriptedModel(
        [_response(_call("write", "effect"), _call("external", "pending"))]
    )
    tools = Tools(tools=[write, _external_tool()], parallel_tool_calls=parallel)
    with pytest.raises(ModelBehaviorError, match="only tool call"):
        await runner.run(_agent(model, runner, tools), _state())

    assert effects == []
    assert model.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_unsupported_external_root_rejects_before_model_open(
    enabled: bool,
) -> None:
    runner = Runner(
        external_tool_agent_id=AgentId.from_values("test", "other") if enabled else None
    )
    model = _ScriptedModel([])
    with pytest.raises(ValueError, match="explicit opt-in"):
        await runner.run(
            _agent(model, runner, Tools(tools=[_external_tool()])), _state()
        )

    assert model.opened == 0


@pytest.mark.asyncio
async def test_session_model_rejects_before_model_open() -> None:
    runner = _runner()
    model = _ScriptedModel([], requires_streaming=True)
    with pytest.raises(ValueError, match="live provider session"):
        await runner.run(
            _agent(model, runner, Tools(tools=[_external_tool()])), _state()
        )

    assert model.opened == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_raw_pending_state_requires_typed_result_input(registered: bool) -> None:
    runner = _runner()
    tools = Tools(tools=[_external_tool()])
    first_model = _ScriptedModel([_response(_call("external", "pending"))])
    outcome = await runner.run(_agent(first_model, runner, tools), _state())
    assert isinstance(outcome, RunSuspension)
    resumed_model = _ScriptedModel([])

    with pytest.raises(ValueError, match="requires ToolResultInput"):
        await runner.run(
            _agent(resumed_model, runner, tools if registered else Tools(tools=[])),
            copy_run_state(outcome.continuation.run_state),
        )

    assert resumed_model.opened == 0


@pytest.mark.asyncio
async def test_resume_rejects_changed_contract_before_next_model_request() -> None:
    runner = _runner()
    model = _ScriptedModel([_response(_call("external", "pending"))])
    outcome = await runner.run(
        _agent(model, runner, Tools(tools=[_external_tool()])), _state()
    )
    assert isinstance(outcome, RunSuspension)
    replacement = ExternalTool(
        name="external", description="Get an external result.", args_model=_ChangedArgs
    )
    resumed_model = _ScriptedModel([])
    with pytest.raises(ValueError, match="contract has changed"):
        await runner.run(
            _agent(resumed_model, runner, Tools(tools=[replacement])),
            copy_run_state(outcome.continuation.run_state),
            external_input=ToolResultInput(
                continuation=outcome.continuation, result="text"
            ),
        )

    assert resumed_model.requests == []
    assert resumed_model.opened == resumed_model.closed


@pytest.mark.asyncio
async def test_repeated_pause_preserves_turn_and_tool_budgets() -> None:
    runner = _runner(max_turns=2)
    tools = Tools(tools=[_external_tool()], max_tool_round_trips=2)
    first_model = _ScriptedModel([_response(_call("external", "one"))])
    first = await runner.run(_agent(first_model, runner, tools), _state())
    assert isinstance(first, RunSuspension)
    second_runner = _runner(max_turns=999)
    second_model = _ScriptedModel([_response(_call("external", "two"))])
    second = await second_runner.run(
        _agent(second_model, second_runner, tools),
        copy_run_state(first.continuation.run_state),
        external_input=ToolResultInput(continuation=first.continuation, result="one"),
    )
    assert isinstance(second, RunSuspension)
    assert second.continuation.continuation_id != first.continuation.continuation_id
    assert second.continuation.tool_call_counts == {"external": 2}
    assert second.continuation.tool_round_trips == 2
    assert second.continuation.max_turns == 2
    final_model = _ScriptedModel([])

    with pytest.raises(RuntimeError, match="turn limit of 2"):
        await second_runner.run(
            _agent(final_model, second_runner, tools),
            copy_run_state(second.continuation.run_state),
            external_input=ToolResultInput(
                continuation=second.continuation, result="two"
            ),
        )

    assert final_model.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["async", "automatic", "steering"])
async def test_provider_control_modes_reject_without_local_effects(kind: str) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record one side effect."""
        effects.append(text)
        return text

    events: tuple[ModelStreamEvent, ...] = ()
    continuation: Literal["automatic", "explicit"] | None = None
    if kind == "async":
        events = (
            ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                async_tool_call=_call("write", "effect"),
            ),
        )
    elif kind == "steering":
        events = (
            ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                input_messages=[{"role": "user", "content": "live input"}],
            ),
        )
    else:
        continuation = ModelContinuation.AUTOMATIC.value
    model = _ScriptedModel(
        [_response(_call("external", "pending"))],
        extra_events=events,
        continuation=continuation,
    )
    runner = _runner()
    stream = runner.run_stream(
        _agent(model, runner, Tools(tools=[write, _external_tool()])), _state()
    )
    try:
        with pytest.raises(ModelBehaviorError):
            await stream.result()
    finally:
        await stream.aclose()

    assert effects == []
    assert model.closed == 1


@pytest.mark.asyncio
async def test_malformed_pending_transcript_rejects_before_model_open() -> None:
    runner = _runner()
    tools = Tools(tools=[_external_tool()])
    model = _ScriptedModel([_response(_call("external", "pending"))])
    outcome = await runner.run(_agent(model, runner, tools), _state())
    assert isinstance(outcome, RunSuspension)
    changed_state = copy_run_state(outcome.continuation.run_state)
    changed_state.history.append({"role": "user", "content": "wrong boundary"})
    changed = replace(outcome.continuation, run_state=changed_state)
    resumed_model = _ScriptedModel([])

    with pytest.raises(ValueError):
        await runner.run(
            _agent(resumed_model, runner, tools),
            copy_run_state(changed.run_state),
            external_input=ToolResultInput(continuation=changed, result="text"),
        )

    assert resumed_model.opened == 0


@pytest.mark.asyncio
async def test_history_rewrite_cannot_remove_pending_call_before_resume() -> None:
    runner = _runner()
    tools = Tools(tools=[_external_tool()])
    model = _ScriptedModel([_response(_call("external", "pending"))])
    outcome = await runner.run(_agent(model, runner, tools), _state())
    assert isinstance(outcome, RunSuspension)
    resumed_model = _ScriptedModel([])
    manager = BoundShimManager((_RewritePendingShim(),))

    # Shim preparation runs after the initial continuation validation.
    with patch("agentlane.harness._run_loop._shim_manager", return_value=manager):
        with pytest.raises(ValueError):
            await runner.run(
                _agent(resumed_model, runner, tools),
                copy_run_state(outcome.continuation.run_state),
                external_input=ToolResultInput(
                    continuation=outcome.continuation, result="text"
                ),
            )

    assert resumed_model.requests == []
    assert resumed_model.opened == resumed_model.closed


@pytest.mark.asyncio
async def test_dynamic_external_declaration_rejects_unsupported_root() -> None:
    runner = Runner()
    model = _ScriptedModel([])
    manager = BoundShimManager((_ExternalCatalogShim(),))

    with patch("agentlane.harness._run_loop._shim_manager", return_value=manager):
        with pytest.raises(ValueError, match="explicit opt-in"):
            await runner.run(_agent(model, runner, Tools(tools=[])), _state())

    assert model.requests == []
    assert model.opened == model.closed
