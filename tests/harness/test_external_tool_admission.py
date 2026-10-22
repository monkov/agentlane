"""Check cancellation before an external result acquires execution ownership."""

import asyncio
from typing import Literal

import pytest

from agentlane.harness import (
    AgentDescriptor,
    ExternalTool,
    Runner,
    RunResult,
    RunSuspension,
    ToolResultInput,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.models import Tools
from agentlane.runtime import CancellationToken

from .tools_test_utils import (
    EchoArgs,
    SequenceModel,
    make_assistant_response,
    make_tool_call,
)

_AGENT_ID = AgentId.from_values("external-admission-agent", "main")
"""Stable address used to restore the paused agent."""


def _agent(model: SequenceModel) -> DefaultAgent:
    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="External admission agent",
            instructions="Continue after the host supplies the result.",
            model=model,
            tools=Tools(
                tools=[
                    ExternalTool(
                        name="external_answer",
                        description="Get a result from the host.",
                        args_model=EchoArgs,
                    )
                ]
            ),
        ),
        runner=Runner(external_tool_agent_id=_AGENT_ID),
        agent_id=_AGENT_ID,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "run_stream", "run_events"])
async def test_cancelled_answer_does_not_consume_continuation(
    mode: Literal["run", "run_stream", "run_events"],
) -> None:
    paused = await _agent(
        SequenceModel(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_external",
                            name="external_answer",
                            arguments='{"text":"Choose the route."}',
                        )
                    ],
                )
            ]
        )
    ).run("Choose")
    assert isinstance(paused, RunSuspension)
    answer = ToolResultInput(continuation=paused.continuation, result="Blue")
    model = SequenceModel([make_assistant_response("Done.")])
    restored = _agent(model)
    token = CancellationToken()
    token.cancel()

    # Cancellation before admission must leave the answer available for retry.
    with pytest.raises(asyncio.CancelledError):
        if mode == "run":
            await restored.run(answer, cancellation_token=token)
        else:
            stream = (
                await restored.run_stream(answer, cancellation_token=token)
                if mode == "run_stream"
                else await restored.run_events(answer, cancellation_token=token)
            )
            try:
                await stream.result()
            finally:
                await stream.aclose()
    assert model.calls == []
    assert restored.snapshot() is None

    result = await restored.run(answer)

    assert isinstance(result, RunResult)
    assert result.final_output == "Done."
    assert len(model.calls) == 1
