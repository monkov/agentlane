"""Check external tool continuations across runtime and agent lifecycles."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    AgentSnapshot,
    ExternalTool,
    RunInput,
    Runner,
    RunnerHooks,
    RunOutcome,
    RunResult,
    RunState,
    RunSuspension,
    Task,
    ToolResultInput,
    continuation_from_json,
    continuation_to_json,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    ModelResponse,
    Tool,
    ToolExecutionContext,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import (
    EchoArgs,
    EmptyToolArgs,
    SequenceModel,
    StreamingSequenceModel,
    make_assistant_response,
    make_tool_call,
)

_AGENT_ID = AgentId.from_values("external-tool-agent", "main")
"""Stable agent address used when a host restores a saved continuation."""


def _external_response() -> ModelResponse:
    return make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(
                tool_id="call_external",
                name="external_answer",
                arguments='{"text":"Choose the route."}',
            )
        ],
    )


def _descriptor(
    model: SequenceModel,
    effects: list[str] | None = None,
) -> AgentDescriptor:
    async def record_effect(
        args: EmptyToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del args, cancellation_token, context
        assert effects is not None
        effects.append("saved")

        return "Saved once."

    tools: list[ExternalTool[EchoArgs] | Tool[EmptyToolArgs, str]] = [
        ExternalTool(
            name="external_answer",
            description="Get a result from the host.",
            args_model=EchoArgs,
        )
    ]
    if effects is not None:
        tools.append(
            Tool(
                name="record_effect",
                description="Record one external effect.",
                args_model=EmptyToolArgs,
                handler=record_effect,
            )
        )

    return AgentDescriptor(
        name="External tool agent",
        instructions="Continue after the host supplies the result.",
        model=model,
        tools=Tools(tools=tools),
    )


def _agent(
    model: SequenceModel,
    *,
    snapshot: AgentSnapshot | None = None,
    effects: list[str] | None = None,
    hooks: RunnerHooks | None = None,
    runner: Runner | None = None,
) -> DefaultAgent:
    return DefaultAgent(
        descriptor=_descriptor(model, effects),
        runner=runner or Runner(external_tool_agent_id=_AGENT_ID),
        agent_id=_AGENT_ID,
        snapshot=snapshot,
        hooks=hooks,
    )


class _EndHooks(RunnerHooks):
    def __init__(self) -> None:
        self.results: list[RunResult | None] = []

    async def on_agent_end(self, task: Task, result: RunResult | None) -> None:
        del task
        self.results.append(result)


class _ControlledModel(SequenceModel):
    """Hold a scripted request open or fail after the request has started."""

    def __init__(
        self,
        response: ModelResponse | list[ModelResponse],
        *,
        block: bool = False,
        fail: bool = False,
    ) -> None:
        super().__init__(response if isinstance(response, list) else [response])
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._block = block
        self._fail = fail

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        response = await super().get_response(
            messages,
            extra_call_args=extra_call_args,
            schema=schema,
            tools=tools,
            cancellation_token=cancellation_token,
            parent_span=parent_span,
            **kwargs,
        )
        self.started.set()

        # The gate fixes ordering without a timing-dependent model delay.
        if self._block:
            await self.release.wait()
        if self._fail:
            raise RuntimeError("Model failed after the resumed request started.")

        return response


class _QueuedAgent(Agent):
    async def queue(
        self,
        input: RunInput,
        cancellation_token: CancellationToken | None = None,
    ) -> RunOutcome:
        return await self._enqueue_input(input, cancellation_token=cancellation_token)


class _HeldSuspensionRunner(Runner):
    """Hold the native outcome before the enclosing agent records the pause."""

    def __init__(self) -> None:
        super().__init__(external_tool_agent_id=_AGENT_ID)
        self.suspended: asyncio.Future[RunSuspension] = (
            asyncio.get_running_loop().create_future()
        )
        self.release = asyncio.Event()

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
        external_input: ToolResultInput | None = None,
    ) -> RunOutcome:
        result = await super().run(
            agent,
            state,
            hooks=hooks,
            cancellation_token=cancellation_token,
            external_input=external_input,
        )
        if isinstance(result, RunSuspension):
            self.suspended.set_result(result)
            await self.release.wait()

        return result


async def _submit_input(
    agent: DefaultAgent | _QueuedAgent,
    input: RunInput,
    mode: Literal["run", "stream", "events"] = "run",
) -> RunOutcome:
    if mode == "run":
        return (
            await agent.run(input)
            if isinstance(agent, DefaultAgent)
            else await agent.queue(input)
        )
    if isinstance(agent, DefaultAgent):
        stream = (
            await agent.run_stream(input)
            if mode == "stream"
            else await agent.run_events(input)
        )
    else:
        stream = (
            await agent.enqueue_input_stream(input)
            if mode == "stream"
            else await agent.enqueue_input_events(input)
        )
    try:
        async for _ in stream:
            pass

        return await stream.result()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_baseline", [False, True])
async def test_external_tool_restore_completes_original_call_without_replay(
    tmp_path: Path,
    with_baseline: bool,
) -> None:
    effects: list[str] = []
    responses = (
        [make_assistant_response("Earlier completed answer.")] if with_baseline else []
    )
    responses.extend(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="call_effect",
                        name="record_effect",
                        arguments="{}",
                    )
                ],
            ),
            _external_response(),
        ]
    )
    original_model = SequenceModel(responses)
    original = _agent(original_model, effects=effects)
    if with_baseline:
        assert isinstance(await original.run("Earlier question."), RunResult)
    baseline = original.snapshot()

    # Persist only the native envelopes, then discard the original agent.
    suspended = await original.run("Record the effect, then choose the route.")
    assert isinstance(suspended, RunSuspension)
    after_pause = original.snapshot()
    assert (after_pause.to_run_state() if after_pause else None) == (
        baseline.to_run_state() if baseline else None
    )
    assert effects == ["saved"]
    continuation_path = tmp_path / "continuation.json"
    continuation_path.write_text(
        json.dumps(continuation_to_json(suspended.continuation)), encoding="utf-8"
    )
    snapshot_path = tmp_path / "snapshot.json"
    if baseline is not None:
        snapshot_path.write_text(json.dumps(baseline.to_json()), encoding="utf-8")
    del original, original_model, suspended, baseline, after_pause

    # A new model and auto-provisioned runtime need no old process objects.
    saved = continuation_from_json(
        json.loads(continuation_path.read_text(encoding="utf-8"))
    )
    restored_snapshot = (
        AgentSnapshot.from_json(json.loads(snapshot_path.read_text(encoding="utf-8")))
        if snapshot_path.exists()
        else None
    )
    restored_model = SequenceModel(
        [make_assistant_response("The blue route is ready.")]
    )
    restored = _agent(restored_model, snapshot=restored_snapshot, effects=effects)
    result = await restored.run(ToolResultInput(continuation=saved, result="Use blue."))

    assert isinstance(result, RunResult)
    assert result.final_output == "The blue route is ready."
    assert result.run_state is not None
    assert result.run_state.revision == (2 if with_baseline else 1)
    assert result.turn_count == (4 if with_baseline else 3)
    assert effects == ["saved"]
    assert len(restored_model.calls) == 1
    messages = restored_model.calls[0]
    external_results = [
        message
        for message in messages
        if message.get("tool_call_id") == "call_external"
    ]
    assert external_results == [
        {
            "role": "tool",
            "tool_call_id": "call_external",
            "name": "external_answer",
            "content": "Use blue.",
        }
    ]
    assert messages[-1] == external_results[0]
    assert [
        call["id"] for message in messages for call in message.get("tool_calls", ())
    ] == ["call_effect", "call_external"]
    assert not any(
        message.get("role") == "user" and message.get("content") == "Use blue."
        for message in messages
    )


@pytest.mark.asyncio
async def test_external_tool_run_events_suspends_without_completed_snapshot() -> None:
    hooks = _EndHooks()
    model = StreamingSequenceModel([_external_response()])
    agent = _agent(model, hooks=hooks)

    stream = await agent.run_events("Choose the route.")
    async for _ in stream:
        pass
    result = await stream.result()
    await stream.aclose()

    assert isinstance(result, RunSuspension)
    assert agent.snapshot() is None
    assert agent.run_state is None
    assert hooks.results == [None]
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_external_tool_pending_agent_rejects_plain_input_fork_and_raw_state() -> (
    None
):
    model = SequenceModel([_external_response()])
    agent = _agent(model)
    suspended = await agent.run("Choose the route.")
    assert isinstance(suspended, RunSuspension)

    with pytest.raises((RuntimeError, ValueError)):
        await agent.run("Use blue.")
    with pytest.raises((RuntimeError, ValueError)):
        await agent.fork("Use blue.")

    # A restored raw state cannot bypass typed external-result admission.
    fresh = _agent(model)
    with pytest.raises((RuntimeError, ValueError)):
        await fresh.run(suspended.continuation.run_state)

    assert len(model.calls) == 1
    assert agent.snapshot() is None
    assert fresh.snapshot() is None


@pytest.mark.asyncio
async def test_external_tool_failed_resume_cannot_consume_same_continuation_again() -> (
    None
):
    suspended = await _agent(SequenceModel([_external_response()])).run(
        "Choose the route."
    )
    assert isinstance(suspended, RunSuspension)
    model = _ControlledModel(make_assistant_response("Unused."), fail=True)
    restored = _agent(model)
    answer = ToolResultInput(continuation=suspended.continuation, result="Use blue.")

    with pytest.raises(RuntimeError, match="Model failed"):
        await restored.run(answer)
    with pytest.raises((RuntimeError, ValueError)):
        await restored.run(answer)

    assert len(model.calls) == 1
    assert restored.snapshot() is None


@pytest.mark.asyncio
async def test_external_tool_cancelled_resume_cannot_consume_same_continuation_again() -> (
    None
):
    suspended = await _agent(SequenceModel([_external_response()])).run(
        "Choose the route."
    )
    assert isinstance(suspended, RunSuspension)
    model = _ControlledModel(make_assistant_response("Unused."), block=True)
    restored = _agent(model)
    answer = ToolResultInput(continuation=suspended.continuation, result="Use blue.")

    stream = await restored.run_events(answer)
    await asyncio.wait_for(model.started.wait(), timeout=1)
    await stream.aclose()
    with pytest.raises(asyncio.CancelledError):
        await stream.result()
    with pytest.raises((RuntimeError, ValueError)):
        await restored.run(answer)

    assert len(model.calls) == 1
    assert restored.snapshot() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_queued", [False, True])
async def test_external_tool_suspension_settles_queued_ordinary_inputs(
    cancel_queued: bool,
) -> None:
    model = _ControlledModel(_external_response(), block=True)
    runtime = SingleThreadedRuntimeEngine()
    agent = _QueuedAgent.bind(
        runtime,
        _AGENT_ID,
        runner=Runner(external_tool_agent_id=_AGENT_ID),
        descriptor=_descriptor(model),
    )
    first = asyncio.create_task(agent.queue("Choose the route."))
    await asyncio.wait_for(model.started.wait(), timeout=1)

    # Queue a normal turn before suspension so rejection must drain its future.
    token = CancellationToken()
    queued = asyncio.create_task(agent.queue("Use blue.", token))
    await asyncio.sleep(0)
    assert agent.pending_input_count == 1
    if cancel_queued:
        token.cancel()
    model.release.set()

    assert isinstance(await asyncio.wait_for(first, timeout=1), RunSuspension)
    if cancel_queued:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, timeout=1)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            await asyncio.wait_for(queued, timeout=1)

    assert not agent.is_running
    assert agent.pending_input_count == 0
    assert agent.run_state is None
    assert len(model.calls) == 1
    await runtime.stop_when_idle()


@pytest.mark.asyncio
@pytest.mark.parametrize("high_level", [False, True])
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_external_tool_pending_input_is_rejected_before_active_resume_finishes(
    high_level: bool,
    mode: Literal["run", "stream", "events"],
) -> None:
    model = _ControlledModel(
        [_external_response(), make_assistant_response("The route is ready.")],
        block=True,
    )
    runtime = SingleThreadedRuntimeEngine()
    agent = (
        _agent(model)
        if high_level
        else _QueuedAgent.bind(
            runtime,
            _AGENT_ID,
            runner=Runner(external_tool_agent_id=_AGENT_ID),
            descriptor=_descriptor(model),
        )
    )
    model.release.set()
    suspension = await _submit_input(agent, "Choose the route.")
    assert isinstance(suspension, RunSuspension)

    # Hold the resumed model request so an ordinary input cannot wait for success.
    model.release.clear()
    model.started.clear()
    resumed = asyncio.create_task(
        _submit_input(
            agent,
            ToolResultInput(continuation=suspension.continuation, result="Use blue."),
        )
    )
    await asyncio.wait_for(model.started.wait(), timeout=1)
    try:
        with pytest.raises(ValueError, match="still pending"):
            await asyncio.wait_for(
                _submit_input(agent, "Unrelated turn.", mode), timeout=1
            )
        assert not resumed.done()
    finally:
        model.release.set()
        completed = await asyncio.wait_for(resumed, timeout=1)
        await runtime.stop_when_idle()

    assert isinstance(completed, RunResult)
    assert len(model.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("high_level", [False, True])
@pytest.mark.parametrize("resume_first", [False, True])
async def test_external_tool_pause_rejects_old_queued_input_across_resume(
    high_level: bool,
    resume_first: bool,
) -> None:
    model = SequenceModel(
        [
            _external_response(),
            make_assistant_response("The route is ready."),
            make_assistant_response("An old queued input must not reach this turn."),
        ]
    )
    runtime = SingleThreadedRuntimeEngine()
    runner = _HeldSuspensionRunner()
    agent = (
        _agent(model, runner=runner)
        if high_level
        else _QueuedAgent.bind(
            runtime,
            _AGENT_ID,
            runner=runner,
            descriptor=_descriptor(model),
        )
    )
    first = asyncio.create_task(_submit_input(agent, "Choose the route."))
    suspension = await asyncio.wait_for(runner.suspended, timeout=1)
    answer = ToolResultInput(continuation=suspension.continuation, result="Use blue.")

    # Both inputs enter before the pause is recorded. Test both queue orders.
    inputs: list[RunInput] = (
        [answer, "Unrelated turn."] if resume_first else ["Unrelated turn.", answer]
    )
    tasks: list[asyncio.Task[RunOutcome]] = []
    for input in inputs:
        tasks.append(asyncio.create_task(_submit_input(agent, input)))
        await asyncio.sleep(0)
    resumed, ordinary = tasks if resume_first else list(reversed(tasks))
    runner.release.set()

    assert isinstance(await asyncio.wait_for(first, timeout=1), RunSuspension)
    with pytest.raises(ValueError, match="pending|suspend"):
        await asyncio.wait_for(ordinary, timeout=1)
    completed = await asyncio.wait_for(resumed, timeout=1)

    assert isinstance(completed, RunResult)
    assert completed.final_output == "The route is ready."
    assert len(model.calls) == 2
    assert not any(
        message.get("content") == "Unrelated turn." for message in model.calls[-1]
    )
    await runtime.stop_when_idle()
