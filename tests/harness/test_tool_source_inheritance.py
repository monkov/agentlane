"""Generic dynamic tool inheritance through child agents and wrapper chains."""

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, TypeGuard, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import (
    INHERIT_TOOLS,
    OVERRIDE_TOOLS,
    RESTRICT_TOOLS,
    AgentDescriptor,
    DefaultAgentTool,
    DefaultHandoff,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
    ToolConfig,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import (
    BoundShim,
    DelegatingBoundShim,
    DelegatingShim,
    ExcludeToolsShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
    ToolSourceBinding,
)
from agentlane.models import (
    MessageDict,
    ModelResponse,
    Tool,
    ToolCall,
    ToolError,
    ToolExecutionContext,
    ToolFailure,
    Tools,
    ToolSpec,
)
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken

from .tools_test_utils import (
    EmptyToolArgs,
    SequenceModel,
    StreamingSequenceModel,
    make_assistant_response,
    make_tool_call,
    named_tool,
)


@dataclass
class _SourceRun:
    task_id: str
    state: RunState | None = None
    transient_state: RunContext[Any] | None = None
    started: int = 0
    ended: int = 0


@dataclass
class _ToolCatalog:
    tools: dict[str, tuple[str, ...]] = field(
        default_factory=dict[str, tuple[str, ...]]
    )
    sessions: list[_SourceRun] = field(default_factory=list[_SourceRun])
    calls: list[str] = field(default_factory=list[str])


@dataclass
class _BoundSource(BoundShim):
    source: object
    catalog: _ToolCatalog
    prefixes: tuple[str, ...]
    session: _SourceRun
    allowed_names: frozenset[str] | None = None
    visible_names: frozenset[str] = frozenset()

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        self.session.started += 1
        self.session.state = state
        self.session.transient_state = transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        assert self.session.started == 1
        assert self.session.ended == 0
        assert turn.run_state is self.session.state
        assert turn.transient_state is self.session.transient_state
        names = frozenset(
            f"{prefix}__{name}"
            for prefix in self.prefixes
            for name in self.catalog.tools.get(prefix, ("read", "write"))
        )
        self.visible_names = (
            names if self.allowed_names is None else names & self.allowed_names
        )
        turn.add_tools(
            tuple(self._tool(name) for name in sorted(self.visible_names)),
            require_unique_names=True,
        )

    def _tool(self, name: str) -> Tool[EmptyToolArgs, str]:
        async def handler(
            args: EmptyToolArgs,
            cancellation_token: CancellationToken,
            context: ToolExecutionContext,
        ) -> str:
            del args, cancellation_token
            assert self.session.started == 1
            assert self.session.ended == 0
            assert context.run_id == self.session.task_id
            assert context.run_state is not None
            assert str(context.run_state.task_id) == self.session.task_id
            self.catalog.calls.append(name)
            return f"{name} completed"

        return Tool(
            name=name,
            description="Call one fixture tool.",
            args_model=EmptyToolArgs,
            handler=handler,
        )

    def inherit_tools(self, names: frozenset[str]) -> tuple[ToolSourceBinding, ...]:
        allowed = names & self.visible_names
        if not allowed:
            return ()

        bind = partial(
            _bind_source,
            source=self.source,
            catalog=self.catalog,
            prefixes=self.prefixes,
            names=allowed,
        )
        return (ToolSourceBinding(source=self.source, bind=bind),)

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        del result
        assert transient_state is self.session.transient_state
        self.session.ended += 1


async def _bind_source(
    context: ShimBindingContext,
    *,
    source: object,
    catalog: _ToolCatalog,
    prefixes: tuple[str, ...],
    names: frozenset[str] | None = None,
) -> BoundShim:
    session = _SourceRun(str(context.task.task_id))
    catalog.sessions.append(session)
    return _BoundSource(source, catalog, prefixes, session, names)


class _SourceShim(Shim):
    def __init__(self, catalog: _ToolCatalog, prefixes: tuple[str, ...]) -> None:
        self.catalog = catalog
        self.prefixes = prefixes

    @property
    def name(self) -> str:
        return "tool-source"

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        for binding in context.tool_source_bindings:
            if binding.source is self:
                return await binding.bind(context)

        if context.tool_source_bindings:
            return BoundShim()

        return await _bind_source(
            context, source=self, catalog=self.catalog, prefixes=self.prefixes
        )


def _source(catalog: _ToolCatalog, *, name: str = "remote") -> Shim:
    return _SourceShim(catalog, (name,))


def _call(name: str, arguments: str = "{}") -> ToolCall:
    return make_tool_call(tool_id=f"call_{name}", name=name, arguments=arguments)


def _names(model: SequenceModel | StreamingSequenceModel, turn: int = 0) -> set[str]:
    tools = model.call_tools[turn]
    return (
        {tool.name for tool in tools.normalized_tools} if tools is not None else set()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
@pytest.mark.parametrize(
    "policy", [INHERIT_TOOLS, OVERRIDE_TOOLS, RESTRICT_TOOLS.only("remote__read")]
)
async def test_source_inheritance_policies_through_default_agent(
    kind: str, streaming: bool, policy: ToolConfig
) -> None:
    catalog = _ToolCatalog()
    # Subagents are subroutines and use terminal calls even in a streamed run.
    child_streams = streaming and kind in {"handoff", "default_handoff"}
    child_model = (
        StreamingSequenceModel([make_assistant_response("child done")])
        if child_streams
        else SequenceModel([make_assistant_response("child done")])
    )
    child = AgentDescriptor(name="child", model=child_model, tools=policy)
    parent_tools = None
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = Tools(tools=[DefaultAgentTool(model=child_model, tools=policy)])
        call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = Tools(tools=[child.as_tool()])
        call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model, tools=policy)
        call = _call("handoff", '{"task":"test"}')
    outcomes = [make_assistant_response(None, tool_calls=[call])]
    if kind in {"generic", "predefined"}:
        outcomes.append(make_assistant_response("parent done"))
    parent_model = (
        StreamingSequenceModel(outcomes) if streaming else SequenceModel(list(outcomes))
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_source(catalog),),
        )
    )
    if streaming:
        stream = await agent.run_stream("test")
        async for _ in stream:
            pass
        await stream.result()
    else:
        await agent.run("test")
    expected: set[str] = (
        {"remote__read", "remote__write"}
        if policy is INHERIT_TOOLS
        else set() if policy is OVERRIDE_TOOLS else {"remote__read"}
    )
    assert {
        name for name in _names(child_model) if name.startswith("remote__")
    } == expected
    assert all(session.ended == 1 for session in catalog.sessions)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
@pytest.mark.parametrize("limit", ["round", "tool"])
@pytest.mark.parametrize("child_local", [False, True])
async def test_restricted_source_inherits_parent_execution_settings(
    kind: str, streaming: bool, limit: str, child_local: bool
) -> None:
    """Dynamic-only restrictions retain budgets before child discovery runs."""
    catalog = _ToolCatalog()
    child_outcomes = [
        make_assistant_response(None, tool_calls=[_call("remote__read")]),
        make_assistant_response("child done"),
    ]
    child_streams = streaming and kind in {"handoff", "default_handoff"}
    child_model = (
        StreamingSequenceModel(child_outcomes)
        if child_streams
        else SequenceModel(list(child_outcomes))
    )
    policy = RESTRICT_TOOLS.only(
        "remote__read",
        tools=Tools(tools=[named_tool("local")]) if child_local else None,
    )
    child = AgentDescriptor(name="child", model=child_model, tools=policy)
    parent_tools = Tools(
        tools=[],
        parallel_tool_calls=True,
        tool_call_timeout=17,
        tool_call_max_retries=0,
        tool_call_limits={"remote__read": 1} if limit == "tool" else None,
        max_tool_round_trips=1 if limit == "round" else 10,
    )
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = replace(
            parent_tools,
            tools=[DefaultAgentTool(model=child_model, tools=policy)],
        )
        call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = replace(parent_tools, tools=[child.as_tool()])
        call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model, tools=policy)
        call = _call("handoff", '{"task":"test"}')
    parent_outcomes = [make_assistant_response(None, tool_calls=[call])]
    if kind in {"generic", "predefined"}:
        parent_outcomes.append(make_assistant_response("parent done"))
    parent_model = (
        StreamingSequenceModel(parent_outcomes)
        if streaming
        else SequenceModel(list(parent_outcomes))
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_source(catalog),),
        )
    )
    if streaming:
        stream = await agent.run_stream("test")
        async for _ in stream:
            pass
        await stream.result()
    else:
        await agent.run("test")

    visible = cast(Tools, child_model.call_tools[0])
    assert visible.tool_call_timeout == 17
    assert visible.tool_call_max_retries == 0
    assert visible.parallel_tool_calls is True
    assert visible.tool_call_limits == parent_tools.tool_call_limits
    assert visible.max_tool_round_trips == parent_tools.max_tool_round_trips
    assert _names(child_model) == (
        {"remote__read", "local"} if child_local else {"remote__read"}
    )
    assert _names(child_model, 1) == (
        {"local"} if child_local and limit == "tool" else set()
    )
    assert catalog.calls == ["remote__read"]
    assert len(catalog.sessions) == 2
    assert all(session.ended == 1 for session in catalog.sessions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy", [OVERRIDE_TOOLS, RESTRICT_TOOLS.only("remote__read")]
)
async def test_child_local_additions_survive_inherited_source_restrictions(
    policy: Any,
) -> None:
    catalog = _ToolCatalog()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=policy.with_tools(Tools(tools=[named_tool("local")])),
        shims=(_source(catalog, name="local_source"),),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_source(catalog),),
        )
    ).run("test")
    assert {"local", "local_source__read", "local_source__write"} <= _names(child_model)
    assert "remote__write" not in _names(child_model)


@pytest.mark.asyncio
async def test_inherited_source_cannot_restore_parent_exclusions() -> None:
    catalog = _ToolCatalog()
    child_model = SequenceModel([make_assistant_response("child done")])
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None, tool_calls=[_call("agent", '{"name":"child","task":"test"}')]
            ),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[DefaultAgentTool(model=child_model)]),
            shims=(ExcludeToolsShim(names=("remote__write",)), _source(catalog)),
        )
    ).run("test")
    assert "remote__read" in _names(child_model)
    assert "remote__write" not in _names(child_model)


class _GrowCatalog(Shim):
    def __init__(self, catalog: _ToolCatalog) -> None:
        self.catalog = catalog

    @property
    def name(self) -> str:
        return "grow-catalog"

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del state, transient_state
        self.catalog.tools["remote"] = ("read", "write", "new")


@pytest.mark.asyncio
async def test_child_catalog_refresh_keeps_parent_visible_name_cap() -> None:
    catalog = _ToolCatalog()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child", model=child_model, shims=(_GrowCatalog(catalog),)
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_source(catalog), ExcludeToolsShim(names=("remote__write",))),
        )
    ).run("test")
    assert "remote__read" in _names(child_model)
    assert "remote__write" not in _names(child_model)
    assert "remote__new" not in _names(child_model)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [INHERIT_TOOLS, OVERRIDE_TOOLS])
async def test_child_local_name_collision_requires_explicit_override(
    policy: Any,
) -> None:
    catalog = _ToolCatalog()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=policy.with_tools(Tools(tools=[named_tool("remote__read")])),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_source(catalog),),
        )
    ).run("test")
    if policy is INHERIT_TOOLS:
        assert not child_model.calls
        assert "delegated agent call failed" in str(parent_model.calls[1])
    else:
        assert _names(child_model) == {"remote__read"}


@dataclass
class _WrapperSession:
    label: str
    task_id: str
    state: RunState | None = None
    transient_state: RunContext[Any] | None = None
    started: int = 0
    ended: int = 0
    prepared: int = 0
    transformed: int = 0
    responses: int = 0
    hook_calls: list[str] = field(default_factory=list[str])
    guarded_calls: list[ToolExecutionContext] = field(
        default_factory=list[ToolExecutionContext]
    )


@dataclass
class _WrapperAudit:
    sessions: list[_WrapperSession] = field(default_factory=list[_WrapperSession])
    denied_agents: frozenset[str] = frozenset()


class _WrapperHooks(RunnerHooks):
    def __init__(self, session: _WrapperSession) -> None:
        self.session = session

    async def on_tool_call_start(self, task: Task, tool_call: ToolCall) -> None:
        assert str(task.task_id) == self.session.task_id
        self.session.hook_calls.append(tool_call.function.name)


def _is_source_tool(tool: ToolSpec[Any]) -> TypeGuard[Tool[Any, Any]]:
    return isinstance(tool, Tool) and "__" in tool.name


class _AuditedBoundShim(DelegatingBoundShim):
    def __init__(
        self, inner: BoundShim, session: _WrapperSession, audit: _WrapperAudit
    ) -> None:
        super().__init__(inner)
        self.session = session
        self.audit = audit

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        assert self.session.started == 0
        self.session.started += 1
        self.session.state = state
        self.session.transient_state = transient_state
        await super().on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        assert self.session.started == 1
        assert self.session.ended == 0
        assert turn.run_state is self.session.state
        assert turn.transient_state is self.session.transient_state
        self.session.prepared += 1
        await super().prepare_turn(turn)
        if turn.tools is not None:
            turn.tools = replace(
                turn.tools,
                tools=tuple(
                    self._guard_tool(tool) if _is_source_tool(tool) else tool
                    for tool in turn.tools.normalized_tools
                ),
            )

    def _guard_tool(self, tool: Tool[Any, Any]) -> Tool[Any, Any]:
        async def guarded(
            args: BaseModel,
            token: CancellationToken,
            context: ToolExecutionContext,
        ) -> object:
            assert self.session.started == 1
            assert self.session.ended == 0
            assert self.session.prepared > 0
            assert context.run_id == self.session.task_id
            assert context.run_state is not None
            assert str(context.run_state.task_id) == self.session.task_id
            assert tool.name in self.session.hook_calls
            self.session.guarded_calls.append(context)
            if context.agent_name in self.audit.denied_agents:
                return ToolFailure(
                    text="Wrapper denied this remote operation.",
                    error=ToolError(
                        message="Wrapper denied this remote operation.",
                        kind="wrapper_denied",
                    ),
                )
            return await tool.run(args, token, context)

        return tool.replace(handler=guarded)

    async def transform_messages(
        self, turn: PreparedTurn, messages: list[MessageDict]
    ) -> list[MessageDict] | None:
        self.session.transformed += 1
        return await super().transform_messages(turn, messages)

    async def on_model_response(
        self, turn: PreparedTurn, response: ModelResponse
    ) -> None:
        self.session.responses += 1
        await super().on_model_response(turn, response)

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        assert transient_state is self.session.transient_state
        self.session.ended += 1
        await super().on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return (*super().runner_hooks(), _WrapperHooks(self.session))


class _AuditedShim(DelegatingShim):
    def __init__(self, inner: Shim, label: str, audit: _WrapperAudit) -> None:
        super().__init__(inner)
        self.label = label
        self.audit = audit

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        session = _WrapperSession(self.label, str(context.task.task_id))
        self.audit.sessions.append(session)
        return _AuditedBoundShim(await self.inner.bind(context), session, self.audit)


def _wrapped_source(
    catalog: _ToolCatalog, audit: _WrapperAudit, *, multiple_sources: bool = False
) -> Shim:
    prefixes = ("remote", "unrelated") if multiple_sources else ("remote",)
    source = _SourceShim(catalog, prefixes)
    return _AuditedShim(_AuditedShim(source, "inner", audit), "outer", audit)


def _assert_wrappers_rebound(audit: _WrapperAudit, *, agent_count: int) -> None:
    assert len(audit.sessions) == agent_count * 2
    for label in ("inner", "outer"):
        sessions = [session for session in audit.sessions if session.label == label]
        assert len({session.task_id for session in sessions}) == agent_count
        assert len({id(session.state) for session in sessions}) == agent_count
        assert len({id(session.transient_state) for session in sessions}) == agent_count
        for session in sessions:
            assert session.started == session.ended == 1
            assert session.prepared == session.transformed == session.responses
            assert session.prepared >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
async def test_nested_source_wrappers_rebind_lifecycle_guards_and_hooks_for_child(
    kind: str,
) -> None:
    catalog = _ToolCatalog()
    audit = _WrapperAudit()
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model, tools=INHERIT_TOOLS)
    parent_tools = None
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = Tools(tools=[DefaultAgentTool(model=child_model)])
        child_call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = Tools(tools=[child.as_tool()])
        child_call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        child_call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model)
        child_call = _call("handoff", '{"task":"test"}')
    parent_outcomes = [make_assistant_response(None, tool_calls=[child_call])]
    if kind in {"generic", "predefined"}:
        parent_outcomes.append(
            make_assistant_response(None, tool_calls=[_call("remote__read")])
        )
        parent_outcomes.append(make_assistant_response("parent done"))
    parent_model = SequenceModel(parent_outcomes)
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_wrapped_source(catalog, audit),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=2)
    assert {"remote__read", "remote__write"} <= _names(child_model)
    expected_calls = 2 if kind in {"generic", "predefined"} else 1
    assert catalog.calls == ["remote__read"] * expected_calls
    for label in ("inner", "outer"):
        calls = [
            context
            for session in audit.sessions
            if session.label == label
            for context in session.guarded_calls
        ]
        assert len(calls) == expected_calls
        assert len({context.run_id for context in calls}) == expected_calls
    assert len(catalog.sessions) == 2
    assert all(session.ended == 1 for session in catalog.sessions)


@pytest.mark.asyncio
async def test_wrapped_source_override_keeps_only_child_local_tools() -> None:
    catalog = _ToolCatalog()
    audit = _WrapperAudit()
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=OVERRIDE_TOOLS.with_tools(Tools(tools=[named_tool("remote__read")])),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_source(catalog, audit),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=1)
    assert _names(child_model) == {"remote__read"}
    assert catalog.calls == []
    assert len(catalog.sessions) == 1
    assert catalog.sessions[0].ended == 1
    assert all(not session.guarded_calls for session in audit.sessions)


@pytest.mark.asyncio
async def test_inherited_source_wrapper_can_deny_before_remote_call() -> None:
    catalog = _ToolCatalog()
    audit = _WrapperAudit(denied_agents=frozenset({"child"}))
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model)
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_source(catalog, audit),),
        )
    ).run("test")

    assert catalog.calls == []
    assert "Wrapper denied this remote operation." in str(child_model.calls[1])
    _assert_wrappers_rebound(audit, agent_count=2)
    guards = [session for session in audit.sessions if session.guarded_calls]
    assert len(guards) == 1
    assert guards[0].label == "outer"
    assert guards[0].guarded_calls[0].agent_name == "child"
    assert all(session.ended == 1 for session in catalog.sessions)


@pytest.mark.asyncio
async def test_nested_source_wrappers_preserve_restriction_through_grandchild() -> None:
    catalog = _ToolCatalog()
    audit = _WrapperAudit()
    grandchild_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("grandchild done"),
        ]
    )
    grandchild = AgentDescriptor(
        name="grandchild", model=grandchild_model, tools=INHERIT_TOOLS
    )
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("grandchild")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=RESTRICT_TOOLS.only(
            "remote__read", tools=Tools(tools=[grandchild.as_tool()])
        ),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_source(catalog, audit, multiple_sources=True),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=3)
    assert {name for name in _names(grandchild_model) if "__" in name} == {
        "remote__read"
    }
    assert catalog.calls == ["remote__read"]
    for label in ("inner", "outer"):
        calls = [
            context
            for session in audit.sessions
            if session.label == label
            for context in session.guarded_calls
        ]
        assert len(calls) == 1
        assert calls[0].agent_name == "grandchild"
    assert all(session.ended == 1 for session in catalog.sessions)
