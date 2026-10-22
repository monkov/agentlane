"""Host-owned continuation round trips and execution boundaries."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    ExternalTool,
    RunContinuation,
    Runner,
    RunPaused,
    RunResult,
    RunState,
    ToolResultInput,
)
from agentlane.harness._run_loop import AgentDelegationState
from agentlane.messaging import AgentId
from agentlane.models import (
    Model,
    ModelBehaviorError,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from .tools_test_utils import (
    EchoArgs,
    SequenceModel,
    make_assistant_response,
    make_tool_call,
)

ROOT = AgentId.from_values("test", "root")


class ChangedArgs(BaseModel):
    city: str


def external() -> ExternalTool[EchoArgs]:
    return ExternalTool(
        name="external", description="Get host input.", args_model=EchoArgs
    )


def response(*names: str, call_suffix: str = "") -> ModelResponse:
    calls = [
        make_tool_call(
            tool_id=name + call_suffix, name=name, arguments='{"text":"needed"}'
        )
        for name in names
    ]
    return make_assistant_response(None if calls else "done", tool_calls=calls or None)


class ScriptedModel(SequenceModel):
    def __init__(self, *responses: ModelResponse, cleanup_error: bool = False) -> None:
        super().__init__(list(responses))
        self.closed = False
        self.cleanup_error = cleanup_error

    @asynccontextmanager
    async def open_run(self) -> AsyncIterator[Self]:
        try:
            yield self
        finally:
            self.closed = True
            if self.cleanup_error:
                raise RuntimeError("cleanup failed")


def agent(
    model: Model[ModelResponse], runner: Runner, tools: Tools | None = None
) -> Agent:
    return Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=ROOT,
        descriptor=AgentDescriptor(
            name="test", model=model, tools=tools or Tools(tools=[external()])
        ),
    )


def state() -> RunState:
    return RunState(instructions=None, history=["start"], responses=[])


async def execute(
    runner: Runner,
    model: ScriptedModel,
    run_state: RunState,
    tools: Tools | None = None,
    mode: str = "run",
) -> RunResult:
    task = agent(model, runner, tools)
    if mode == "run":
        return await runner.run(task, run_state)
    stream = (
        runner.run_events(task, run_state)
        if mode == "events"
        else runner.run_stream(task, run_state)
    )
    try:
        return await stream.result()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_restart_retains_effects_calls_and_saved_budgets(mode: str) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)
        return "written"

    tools = Tools(
        tools=[write, external()],
        tool_call_limits={"write": 1, "external": 2},
        max_tool_round_trips=3,
    )
    first = Runner(external_tool_agent_id=ROOT, max_turns=3)
    model = ScriptedModel(response("write"), response("external"))
    with pytest.raises(RunPaused) as paused:
        await execute(first, model, state(), tools, mode)
    saved = RunContinuation.from_json(
        json.loads(json.dumps(paused.value.continuation.to_json()))
    )
    assert model.closed and len(model.calls) == 2
    assert saved.checkpoint.state.turn_count == 2 and saved.checkpoint.revision == 0
    assert saved.tool_call_counts == {"write": 1, "external": 1}
    original = saved.to_json()

    # A new execution can lower limits, but cannot expand the saved budget.
    resumed = Runner(
        external_tool_agent_id=ROOT,
        max_turns=999,
        external_input=ToolResultInput(continuation=saved, result="Prague"),
    )
    next_model = ScriptedModel(response("external", call_suffix="_again"))
    expanded_tools = Tools(
        tools=[write, external()],
        tool_call_limits={"write": 0, "external": 999},
        max_tool_round_trips=999,
    )
    with pytest.raises(RunPaused) as paused_again:
        await execute(resumed, next_model, saved.to_run_state(), expanded_tools, mode)
    again = RunContinuation.from_json(paused_again.value.continuation.to_json())
    assert again.tool_call_counts == {"write": 1, "external": 2}
    assert again.tool_call_limits == {"write": 0, "external": 2}
    assert again.max_turns == again.tool_round_trips == 3
    assert again.max_tool_round_trips == 3
    assert effects == ["needed"] and saved.to_json() == original
    assert [m for m in next_model.calls[0] if m["role"] == "user"] == [
        {"role": "user", "content": "start"}
    ]
    assert [
        (m["tool_call_id"], m["content"])
        for m in next_model.calls[0]
        if m["role"] == "tool"
    ] == [("write", "written"), ("external", "Prague")]
    catalog = next_model.call_tools[0]
    assert catalog is not None
    assert [tool.name for tool in catalog.normalized_tools] == ["external"]
    assert catalog.max_tool_round_trips == 3
    assert catalog.tool_call_limits == {"write": 0, "external": 2}

    # A second answer cannot grant another model turn after the saved limit.
    last = Runner(
        external_tool_agent_id=ROOT,
        max_turns=999,
        external_input=ToolResultInput(continuation=again, result="answer"),
    )
    final_model = ScriptedModel()
    with pytest.raises(RuntimeError, match="turn limit"):
        await execute(last, final_model, again.to_run_state(), expanded_tools, mode)
    assert final_model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["state", "schema", "identity", "pending"])
async def test_invalid_resume_rejects_before_model_io(invalid: str) -> None:
    runner = Runner(external_tool_agent_id=ROOT)
    with pytest.raises(RunPaused) as paused:
        await execute(runner, ScriptedModel(response("external")), state())
    saved = paused.value.continuation.model_copy(deep=True)
    resumed_state = saved.to_run_state()
    tools = Tools(tools=[external()])
    if invalid == "state":
        resumed_state.history.append("unexpected user turn")
    elif invalid == "schema":
        tools = Tools(
            tools=[
                ExternalTool(
                    name="external", description="changed", args_model=ChangedArgs
                )
            ]
        )
    elif invalid == "identity":
        saved.checkpoint.agent_id = AgentId.from_values("test", "other")
    else:
        saved.checkpoint.state.history.append(
            {"role": "user", "content": "wrong boundary"}
        )
    model = ScriptedModel()
    with pytest.raises(ValueError):
        resumed = Runner(
            external_tool_agent_id=ROOT,
            external_input=ToolResultInput(continuation=saved, result="answer"),
        )
        await execute(resumed, model, resumed_state, tools)
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mixed", "opt_in", "cleanup"])
async def test_unsafe_boundary_never_publishes_pause(failure: str) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)
        return text

    runner = Runner(external_tool_agent_id=None if failure == "opt_in" else ROOT)
    calls = ("external", "write") if failure == "mixed" else ("external",)
    model = ScriptedModel(response(*calls), cleanup_error=failure == "cleanup")
    expected = {
        "mixed": ModelBehaviorError,
        "opt_in": ValueError,
        "cleanup": ExceptionGroup,
    }
    with pytest.raises(expected[failure]):
        await execute(runner, model, state(), Tools(tools=[write, external()]))
    assert effects == []
    if failure == "opt_in":
        assert model.calls == []
    else:
        assert model.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_cancelled_input_does_not_consume_root_but_execution_does(
    mode: str,
) -> None:
    runner = Runner(external_tool_agent_id=ROOT)
    model = ScriptedModel(response("external"))
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner.run(agent(model, runner), state(), cancellation_token=token)
    with pytest.raises(RunPaused):
        await execute(runner, model, state(), mode=mode)
    with pytest.raises(RuntimeError):
        await execute(runner, model, state())
    assert len(model.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_turns", False),
        ("tool_round_trips", -1),
        ("format_version", True),
        ("continuation_id", None),
        ("checkpoint.revision", -1),
        ("checkpoint.schema_version", True),
        ("checkpoint.state.turn_count", True),
        ("checkpoint.state.shim_state", {"bad": float("nan")}),
        ("unexpected", 1),
    ],
)
async def test_codec_rejects_invalid_counters_and_owns_copies(
    field: str, value: object
) -> None:
    with pytest.raises(RunPaused) as paused:
        await execute(
            Runner(external_tool_agent_id=ROOT),
            ScriptedModel(response("external")),
            state(),
        )
    saved = paused.value.continuation
    data = saved.to_json()
    result_input = ToolResultInput(continuation=saved, result="answer")
    runner = Runner(external_tool_agent_id=ROOT, external_input=result_input)

    # Neither caller can change the payload already owned by the runner.
    saved.arguments["text"] = "first mutation"
    result_input.continuation.arguments["text"] = "second mutation"
    restored = RunContinuation.from_json(data)
    result = await execute(runner, ScriptedModel(response()), restored.to_run_state())
    assert result.final_output == "done"
    target = data
    *parents, name = field.split(".")
    for parent in parents:
        target = cast(dict[str, object], target[parent])
    if value is None:
        target.pop(name)
    else:
        target[name] = value
    with pytest.raises(ValueError):
        RunContinuation.from_json(data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["automatic", "explicit", "steering", "async", "session", "child"]
)
async def test_unsupported_provider_controls_reject_before_effects(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)
        return text

    model = ScriptedModel()
    call = make_tool_call(tool_id="effect", name="write", arguments='{"text":"needed"}')
    event = ModelStreamEvent(
        kind=ModelStreamEventKind.COMPLETED,
        response=response("external" if mode == "child" else "write"),
    )
    if mode in ("automatic", "explicit"):
        event.continuation = "automatic" if mode == "automatic" else "explicit"
    elif mode == "steering":
        event.input_messages = [{"role": "user", "content": "live input"}]
    elif mode == "async":
        event.async_tool_call = call
    elif mode == "session":
        monkeypatch.setattr(
            ScriptedModel, "requires_streaming", property(lambda self: True)
        )
    else:
        monkeypatch.setattr(
            AgentDelegationState, "has_live_children", property(lambda self: True)
        )

    async def stream_response(
        *args: object, **kwargs: object
    ) -> AsyncIterator[ModelStreamEvent]:
        yield event

    monkeypatch.setattr(model, "stream_response", stream_response)
    expected = ValueError if mode == "session" else ModelBehaviorError
    with pytest.raises(expected):
        await execute(
            Runner(external_tool_agent_id=ROOT),
            model,
            state(),
            Tools(tools=[write, external()]),
            "stream",
        )
    assert effects == []
    assert model.closed == (mode != "session")


@pytest.mark.asyncio
async def test_stream_reserves_root_before_concurrent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    model = ScriptedModel()

    async def blocked_response(*args: object, **kwargs: object) -> ModelResponse:
        entered.set()
        await release.wait()
        return response("external")

    monkeypatch.setattr(model, "get_response", blocked_response)
    runner = Runner(external_tool_agent_id=ROOT)
    stream = runner.run_events(agent(model, runner), state())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="fresh Runner"):
            await runner.run(agent(model, runner), state())
        release.set()
        with pytest.raises(RunPaused):
            await stream.result()
    finally:
        release.set()
        await stream.aclose()
