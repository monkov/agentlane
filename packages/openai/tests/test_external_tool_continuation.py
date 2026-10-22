"""A persisted continuation keeps the native Responses call and reasoning."""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agentlane_openai import ResponsesClient
from openai.types.responses import Response, ResponseCompletedEvent
from pydantic import BaseModel

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    ExternalTool,
    Runner,
    RunPaused,
    RunState,
    ToolResultInput,
    continuation_from_json,
    continuation_to_json,
    restore_run_continuation,
)
from agentlane.messaging import AgentId
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


class Args(BaseModel):
    text: str


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_resume_keeps_original_call_and_one_result(
    streaming: bool,
) -> None:
    root = AgentId.from_values("test", "root")
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
    tools = Tools(
        tools=[
            ExternalTool(name="external", description="Ask the host.", args_model=Args)
        ]
    )
    saved = None
    saved_result = None
    for output in (native_output, []):
        native = Response.model_validate(
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "test",
                "status": "completed",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        transport = MagicMock()
        transport.__aiter__.return_value = [
            ResponseCompletedEvent(
                type="response.completed", response=native, sequence_number=1
            )
        ]
        transport.close = AsyncMock()
        create = AsyncMock(return_value=transport if streaming else native)
        client = ResponsesClient(Config(api_key="test-key", model="test"))
        cast(Any, client)._openai_client.responses.create = create
        runner = Runner(
            external_tool_agent_id=root,
            external_input=(
                ToolResultInput(continuation=saved, result="Prague") if saved else None
            ),
        )
        agent = Agent(
            SingleThreadedRuntimeEngine(),
            runner,
            bind_id=root,
            descriptor=AgentDescriptor(name="test", model=client, tools=tools),
        )

        state = (
            restore_run_continuation(saved)
            if saved
            else RunState(instructions=None, history=["Start."], responses=[])
        )
        try:
            # Each iteration has a new runner, model client, and runtime.
            stream = runner.run_events(agent, state) if streaming else None
            try:
                operation = stream.result() if stream else runner.run(agent, state)
                if saved is None:
                    with pytest.raises(RunPaused) as paused:
                        await operation
                    saved_result = paused.value.continuation
                else:
                    await operation
            finally:
                if stream:
                    await stream.aclose()
                    transport.close.assert_awaited_once()
            create.assert_awaited_once()
            assert create.await_args is not None
            if saved is None:
                assert saved_result is not None
                saved = continuation_from_json(
                    json.loads(json.dumps(continuation_to_json(saved_result)))
                )
            else:
                request = create.await_args.kwargs
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
                        "output": "Prague",
                    },
                ]
                assert "previous_response_id" not in request
        finally:
            await cast(Any, client)._openai_client.close()
