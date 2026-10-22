"""Pause and resume an external tool with an application-owned SQLite claim."""

import argparse
import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from agentlane.harness import (
    AgentDescriptor,
    ExternalTool,
    Runner,
    RunResult,
    RunSuspension,
    ToolResultInput,
    continuation_from_json,
    continuation_to_json,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.models import MessageDict, Model, ModelResponse, Tools
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span

AGENT_ID = AgentId.from_values("external-tool-example", "main")


class RequestInputArgs(BaseModel):
    """Application-owned question schema; no model-generated request ID."""

    question: str
    choices: list[str] = Field(default_factory=list)


class ScriptedModel(Model[ModelResponse]):
    """Use a fixed model response so the example needs no provider key."""

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
        del extra_call_args, schema, tools, cancellation_token, parent_span, kwargs

        # The resumed request contains the answer for the original tool call.
        last = messages[-1]
        if last.get("role") == "tool":
            if last.get("tool_call_id") != "call_request_input":
                raise RuntimeError("The result does not match the original call.")
            message = {
                "role": "assistant",
                "content": f"I will use this delivery instruction: {last['content']}",
            }
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_request_input",
                        "type": "function",
                        "function": {
                            "name": "request_input",
                            "arguments": json.dumps(
                                {
                                    "question": "How should the package be delivered?",
                                    "choices": ["Reception", "Front door"],
                                }
                            ),
                        },
                    }
                ],
            }

        return ModelResponse.model_validate(
            {
                "id": "scripted-response",
                "object": "chat.completion",
                "created": 1,
                "model": "scripted",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            }
        )


def make_agent() -> DefaultAgent:
    """Restore the same live configuration and stable root address."""
    return DefaultAgent(
        agent_id=AGENT_ID,
        runner=Runner(external_tool_agent_id=AGENT_ID),
        descriptor=AgentDescriptor(
            name="Delivery assistant",
            model=ScriptedModel(),
            instructions="Get the delivery instruction before you continue.",
            tools=Tools(
                tools=[
                    ExternalTool(
                        name="request_input",
                        description="Ask the user for a delivery instruction.",
                        args_model=RequestInputArgs,
                    )
                ]
            ),
        ),
    )


async def start(connection: sqlite3.Connection) -> None:
    """Save the waiting value before the application displays its question."""
    if connection.execute("SELECT 1 FROM request").fetchone() is not None:
        raise RuntimeError("This example database already contains a request.")

    outcome = await make_agent().run("Arrange delivery of my package.")
    if not isinstance(outcome, RunSuspension):
        raise RuntimeError("Expected one external tool call.")
    continuation = outcome.continuation

    # This is the first run, so the completed baseline is empty at revision 0.
    with connection:
        connection.execute(
            "INSERT INTO request (slot, id, status, continuation) VALUES (1, ?, 'waiting', ?)",
            (
                str(continuation.continuation_id),
                json.dumps(continuation_to_json(continuation)),
            ),
        )

    question = RequestInputArgs.model_validate(continuation.arguments)
    print(f"Request: {continuation.continuation_id}")
    print(question.question)
    print(f"Choices: {', '.join(question.choices)}. Free text is also accepted.")
    print("The question is saved. This process can now exit.")


async def resume(connection: sqlite3.Connection, answer: str) -> None:
    """Save the answer and claim the request before any resumed execution."""
    if not answer.strip():
        raise ValueError("The answer must contain text.")

    # The conditional update admits one process and stores its exact answer.
    with connection:
        row = connection.execute(
            "UPDATE request SET status = 'running', answer = ? "
            "WHERE slot = 1 AND status = 'waiting' RETURNING continuation",
            (answer,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "No waiting request. A claimed attempt cannot be replayed."
            )

    continuation = continuation_from_json(json.loads(row[0]))
    agent = make_agent()
    outcome = await agent.run(ToolResultInput(continuation=continuation, result=answer))
    if not isinstance(outcome, RunResult):
        raise RuntimeError("The scripted model must finish after one answer.")
    snapshot = agent.snapshot()
    if snapshot is None:
        raise RuntimeError("Expected a completed snapshot.")

    # Save completion and its committed baseline together. An earlier failure
    # leaves status 'running'; the host must reconcile that uncertain attempt.
    with connection:
        connection.execute(
            "UPDATE request SET status = 'done', snapshot = ? WHERE slot = 1 AND status = 'running'",
            (json.dumps(snapshot.to_json()),),
        )

    print(outcome.final_output)
    print(f"Completed revision: {snapshot.revision}")


def main() -> None:
    """Run either half in its own process against the same example database."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "resume"))
    parser.add_argument("database", type=Path)
    parser.add_argument("answer", nargs="?")
    args = parser.parse_args()
    if args.action == "resume" and args.answer is None:
        parser.error("resume requires an answer")

    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    connection = sqlite3.connect(args.database)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS request ("
            "slot INTEGER PRIMARY KEY CHECK (slot = 1), id TEXT NOT NULL UNIQUE, "
            "status TEXT NOT NULL, continuation TEXT NOT NULL, "
            "answer TEXT, snapshot TEXT)"
        )
        if args.action == "start":
            asyncio.run(start(connection))
        else:
            asyncio.run(resume(connection, args.answer))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
