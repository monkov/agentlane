"""External results retain Responses call identity after a process restart."""

import json
from typing import Any, Self, cast
from unittest.mock import AsyncMock

import pytest
from agentlane_openai import ResponsesClient
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseCompletedEvent
from pydantic import BaseModel

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    ExternalTool,
    Runner,
    RunResult,
    RunState,
    RunSuspension,
    ToolResultInput,
    continuation_from_json,
    continuation_to_json,
)
from agentlane.harness._run import copy_run_state
from agentlane.messaging import AgentId
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


class _ExternalArgs(BaseModel):
    text: str


class _CompletedStream:
    """Return one completed SDK event without a network connection."""

    def __init__(self, response: OpenAIResponse) -> None:
        self._events = iter(
            [
                ResponseCompletedEvent(
                    type="response.completed", response=response, sequence_number=1
                )
            ]
        )
        self.closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ResponseCompletedEvent:
        try:
            return next(self._events)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def close(self) -> None:
        self.closed = True


def _response(response_id: str, output: list[dict[str, Any]]) -> OpenAIResponse:
    return OpenAIResponse.model_validate(
        {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "model": "test-model",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _agent(client: ResponsesClient, runner: Runner, agent_id: AgentId) -> Agent:
    return Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=agent_id,
        descriptor=AgentDescriptor(
            name="test",
            model=client,
            tools=Tools(
                tools=[
                    ExternalTool(
                        name="external",
                        description="Get a result from the host.",
                        args_model=_ExternalArgs,
                    )
                ]
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_resume_replays_original_call_and_one_result(
    streaming: bool,
) -> None:
    agent_id = AgentId.from_values("test", "root")
    native_output: list[dict[str, Any]] = [
        {
            "id": "rs_original",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "saved-reasoning",
        },
        {
            "id": "fc_original",
            "type": "function_call",
            "call_id": "call_original",
            "name": "external",
            "arguments": '{"text":"Choose a location."}',
            "status": "completed",
        },
    ]
    first_response = _response("resp_pending", native_output)
    final_response = _response(
        "resp_done",
        [
            {
                "id": "msg_done",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ],
    )
    first_client = ResponsesClient(Config(api_key="test-key", model="test-model"))
    resumed_client = ResponsesClient(Config(api_key="test-key", model="test-model"))
    first_stream = _CompletedStream(first_response)
    final_stream = _CompletedStream(final_response)
    first_create = AsyncMock(return_value=first_stream if streaming else first_response)
    resumed_create = AsyncMock(
        return_value=final_stream if streaming else final_response
    )
    cast(Any, first_client)._openai_client.responses.create = first_create
    cast(Any, resumed_client)._openai_client.responses.create = resumed_create

    try:
        # Suspend through the real adapter, then cross a JSON persistence boundary.
        first_runner = Runner(external_tool_agent_id=agent_id)
        first_agent = _agent(first_client, first_runner, agent_id)
        first_state = RunState(instructions=None, history=["Start."], responses=[])
        if streaming:
            events = first_runner.run_events(first_agent, first_state)
            try:
                paused = await events.result()
            finally:
                await events.aclose()
        else:
            paused = await first_runner.run(first_agent, first_state)
        assert isinstance(paused, RunSuspension)
        restored = continuation_from_json(
            json.loads(json.dumps(continuation_to_json(paused.continuation)))
        )

        # A fresh runner and client must rebuild one stateless provider request.
        resumed_runner = Runner(external_tool_agent_id=agent_id)
        resumed_agent = _agent(resumed_client, resumed_runner, agent_id)
        resumed_state = copy_run_state(restored.run_state)
        result_input = ToolResultInput(restored, result="Selected location: Prague.")
        if streaming:
            events = resumed_runner.run_events(
                resumed_agent, resumed_state, external_input=result_input
            )
            try:
                result = await events.result()
            finally:
                await events.aclose()
        else:
            result = await resumed_runner.run(
                resumed_agent, resumed_state, external_input=result_input
            )

        # Native replay includes the original provider call, not a second projection.
        assert isinstance(result, RunResult)
        first_create.assert_awaited_once()
        resumed_create.assert_awaited_once()
        assert resumed_create.await_args is not None
        request = resumed_create.await_args.kwargs
        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Start."}],
            },
            *native_output,
            {
                "type": "function_call_output",
                "call_id": "call_original",
                "output": "Selected location: Prague.",
            },
        ]
        assert "previous_response_id" not in request
        if streaming:
            assert first_stream.closed and final_stream.closed
    finally:
        await cast(Any, first_client)._openai_client.close()
        await cast(Any, resumed_client)._openai_client.close()
