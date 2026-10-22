"""Portable external-call state and unchanged completed-snapshot contracts."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from agentlane.harness import (
    AgentSnapshot,
    RunContinuation,
    RunState,
    ToolResultInput,
    continuation_from_json,
    continuation_to_json,
)
from agentlane.messaging import AgentId
from agentlane.models import PromptSpec, PromptTemplate

from .tools_test_utils import make_assistant_response, make_tool_call


def _continuation() -> RunContinuation:
    call = make_tool_call(
        tool_id="call_external", name="external_result", arguments='{"task":"review"}'
    )
    response = make_assistant_response(None, tool_calls=[call])
    state = RunState(
        instructions="Use external results as data.",
        history=["Review this.", response],
        responses=[response],
        turn_count=3,
        revision=2,
    )
    state.shim_state["saved"] = {"source": ["a"]}

    return RunContinuation(
        agent_id=AgentId.from_values("assistant", "main"),
        continuation_id=UUID("11111111-1111-4111-8111-111111111111"),
        base_revision=2,
        run_state=state,
        tool_call=call,
        arguments={"task": "review"},
        args_schema={
            "name": "external_result",
            "description": "Get an external result.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        tool_call_counts={"external_result": 1},
        tool_round_trips=1,
        max_turns=10,
        tool_call_limits={"external_result": 2},
        max_tool_round_trips=4,
    )


def _payload() -> dict[str, Any]:
    return continuation_to_json(_continuation())


def test_continuation_round_trip_preserves_pending_call_context_and_budgets() -> None:
    source = _continuation()
    encoded = continuation_to_json(source)
    restored = continuation_from_json(json.loads(json.dumps(encoded)))

    assert continuation_to_json(restored) == encoded
    assert restored.agent_id == source.agent_id
    assert restored.continuation_id == source.continuation_id
    assert restored.run_state.revision == restored.base_revision == 2
    assert restored.run_state.turn_count == 3
    assert restored.run_state.responses == source.run_state.responses
    assert restored.run_state.shim_state == source.run_state.shim_state
    assert restored.tool_call == source.tool_call
    assert restored.arguments == {"task": "review"}
    assert restored.tool_call_counts == {"external_result": 1}
    assert restored.tool_round_trips == 1
    assert restored.max_turns == 10
    assert restored.tool_call_limits == {"external_result": 2}
    assert restored.max_tool_round_trips == 4


def test_continuation_codec_renders_prompt_state_through_existing_codec() -> None:
    source = _continuation()
    source.run_state.instructions = PromptSpec(
        template=PromptTemplate[dict[str, str], str](
            system_template="Support {{ team }}."
        ),
        values={"team": "ops"},
    )

    restored = continuation_from_json(continuation_to_json(source))

    assert restored.run_state.instructions == "Support ops."


def test_continuation_copies_constructor_and_result_input_state() -> None:
    original = _continuation()
    source_state = original.run_state
    source_call = original.tool_call
    source_arguments: dict[str, JsonValue] = {"task": "review"}
    captured = replace(
        original,
        run_state=source_state,
        tool_call=source_call,
        arguments=source_arguments,
    )
    accepted = ToolResultInput(continuation=captured, result="The review passed.")

    source_state.history.clear()
    source_state.responses[0].choices.clear()
    source_call.function.name = "changed"
    source_arguments["task"] = "changed"
    captured.run_state.history.clear()
    captured.arguments["task"] = "changed"

    assert len(accepted.continuation.run_state.history) == 2
    assert len(accepted.continuation.run_state.responses[0].choices) == 1
    assert accepted.continuation.tool_call.function.name == "external_result"
    assert accepted.continuation.arguments == {"task": "review"}
    with pytest.raises(FrozenInstanceError):
        cast(Any, accepted.continuation).base_revision = 9
    with pytest.raises(TypeError):
        cast(dict[str, int], accepted.continuation.tool_call_counts)[
            "external_result"
        ] = 0


def test_continuation_and_completed_snapshot_codecs_are_distinct() -> None:
    source = _continuation()
    with pytest.raises(ValueError):
        AgentSnapshot.from_json(continuation_to_json(source))

    snapshot = AgentSnapshot.capture(
        agent_id=source.agent_id,
        run_state=RunState(instructions=None, history=[], responses=[]),
    )
    with pytest.raises(ValueError):
        continuation_from_json(snapshot.to_json())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("format_version", True),
        ("kind", "agent_snapshot"),
        ("base_revision", -1),
        ("base_revision", "2"),
        ("max_turns", 2),
        ("max_turns", True),
        ("tool_round_trips", 0),
        ("max_tool_round_trips", 0),
        ("continuation_id", "invalid"),
        ("continuation_id", 123),
        ("tool_call_counts", {"external_result": 0}),
        ("tool_call_counts", {"external_result": -1}),
        ("tool_call_counts", {"external_result": "1"}),
        ("tool_call_limits", {"external_result": 0}),
    ],
)
def test_continuation_decode_rejects_invalid_identity_and_limits(
    field: str, value: object
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError):
        continuation_from_json(payload)


@pytest.mark.parametrize("location", ["envelope", "state", "call", "function", "agent"])
def test_continuation_decode_rejects_unknown_contract_fields(location: str) -> None:
    payload = _payload()
    targets = {
        "envelope": payload,
        "state": payload["state"],
        "call": payload["tool_call"],
        "function": payload["tool_call"]["function"],
        "agent": payload["agent_id"],
    }
    targets[location]["future"] = True

    with pytest.raises(ValueError):
        continuation_from_json(payload)


@pytest.mark.parametrize(
    "change", ["missing", "different", "completed", "mixed", "later_input"]
)
def test_continuation_decode_rejects_inconsistent_pending_history(change: str) -> None:
    payload = _payload()
    history = payload["state"]["history"]
    if change == "missing":
        history.pop()
    elif change == "different":
        history[-1]["tool_calls"][0]["id"] = "another_call"
    elif change == "completed":
        history.append(
            {"role": "tool", "tool_call_id": "call_external", "content": "done"}
        )
    elif change == "mixed":
        history[-1]["tool_calls"].append(
            {
                "id": "sibling",
                "type": "function",
                "function": {"name": "write", "arguments": "{}"},
            }
        )
    else:
        history.append({"role": "user", "content": "later input"})

    with pytest.raises(ValueError):
        continuation_from_json(payload)


def test_continuation_decode_requires_prior_tool_calls_to_be_resolved() -> None:
    payload = _payload()
    earlier = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "earlier",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    payload["state"]["history"].insert(1, earlier)

    with pytest.raises(ValueError, match="overlapping"):
        continuation_from_json(payload)

    payload["state"]["history"].insert(
        2, {"role": "tool", "tool_call_id": "earlier", "content": "found"}
    )
    restored = continuation_from_json(payload)

    assert restored.tool_call.id == "call_external"


def test_continuation_codec_rejects_mutated_runtime_bounds_and_nonfinite_state() -> (
    None
):
    source = _continuation()
    source.run_state.revision = 99
    with pytest.raises(ValueError, match="base_revision"):
        continuation_to_json(source)

    source = _continuation()
    source.run_state.shim_state["invalid"] = float("nan")
    with pytest.raises(ValueError):
        continuation_to_json(source)


def test_tool_result_input_requires_plain_text() -> None:
    with pytest.raises(TypeError, match="string"):
        ToolResultInput(continuation=_continuation(), result=cast(str, {"answer": "x"}))


def test_continuation_fixture_v1_decodes_and_reencodes_without_changes() -> None:
    fixture = Path(__file__).with_name("fixtures") / "run_continuation_v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    restored = continuation_from_json(payload)

    assert restored.tool_call.id == "call_external"
    assert restored.arguments == {"task": "review"}
    assert continuation_to_json(restored) == cast(dict[str, JsonValue], payload)
