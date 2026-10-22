"""Agent run entry points and stream completion."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING

from agentlane.messaging import AgentId
from agentlane.models import Model, ModelResponse, ModelStreamEvent, ToolExecutor
from agentlane.runtime import CancellationToken

from ._cancellation import cancel_task_callback
from ._events import RunEventEmitter, RunEventStream, forward_tool_approval_events
from ._hooks import RunnerHooks, coerce_runner_hooks
from ._run import RunResult, RunState
from ._run_loop import AgentDelegationState, RunLoop, RunnerTask
from ._snapshot import ToolResultInput
from ._stream import RunStream
from ._task import Task

if TYPE_CHECKING:
    from .tools import ToolApprovalEvent


class Runner:
    """Start agent runs and settle their result or stream.

    One runner can serve multiple agents. Each call creates an isolated loop;
    tool execution and process-local delegation limits are shared.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        max_turns: int = 128,
        agent_max_depth: int = 4,
        agent_max_threads: int = 16,
        is_retryable: Callable[[BaseException], bool] | None = None,
        external_tool_agent_id: AgentId | None = None,
        external_input: ToolResultInput | None = None,
    ) -> None:
        """Initialize retry and loop limits for one reusable runner.

        Args:
            max_attempts: Total attempts per model call (1 = no retry).
            max_turns: Safety cap on conversation turns per run.
            agent_max_depth: Inclusive process-local recursive depth cap for
                generic spawned ``agent`` tool calls. A direct child has depth
                1.
            agent_max_threads: Process-local live-agent cap for generic
                spawned ``agent`` tool calls.
            is_retryable: Optional predicate for retry eligibility. Defaults
                to the standard HTTP-status-code check in ``retry_on_errors``.
            external_tool_agent_id: Root permitted to pause for external tools.
                This runner accepts one invocation of that root. Create a new
                runner for each resume; the host owns durable admission.
            external_input: Claimed continuation and result for the root's next
                invocation. Pass its restored state through the normal run API.
        """
        if max_attempts < 1:
            raise ValueError("Runner.max_attempts must be at least 1.")
        if max_turns < 1:
            raise ValueError("Runner.max_turns must be at least 1.")
        if agent_max_depth < 1:
            raise ValueError("Runner.agent_max_depth must be at least 1.")
        if agent_max_threads < 1:
            raise ValueError("Runner.agent_max_threads must be at least 1.")
        if external_input is not None and external_tool_agent_id is None:
            raise ValueError("external_input requires external_tool_agent_id.")

        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._is_retryable = is_retryable

        # Own the resume payload and permit one root execution on this instance.
        self._external_tool_agent_id = external_tool_agent_id
        self._external_input = (
            ToolResultInput(external_input.continuation, external_input.result)
            if external_input is not None
            else None
        )
        self._external_execution_started = False

        # Share execution services and delegation limits across this runner's loops.
        self._tool_executor = ToolExecutor()
        self._agent_delegation = AgentDelegationState(
            max_depth=agent_max_depth,
            max_threads=agent_max_threads,
        )

    @property
    def external_tool_agent_id(self) -> AgentId | None:
        """Return the root configured for host-managed external tool results."""
        return self._external_tool_agent_id

    def _claim_external_execution(
        self, agent: Task, cancellation_token: CancellationToken | None
    ) -> None:
        """Reserve the configured root before any task or model I/O starts."""
        if (
            self._external_tool_agent_id is None
            or agent.id != self._external_tool_agent_id
        ):
            return
        if cancellation_token is not None and cancellation_token.is_cancelled:
            return
        if self._external_execution_started:
            raise RuntimeError("External tools require a fresh Runner for each run.")

        self._external_execution_started = True

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute the generic harness loop for the provided run state.

        The lifecycle passes a private working copy of ``state``, so the
        runner mutates it freely (incrementing ``turn_count``, appending
        responses) without risking the persisted baseline.
        """
        self._claim_external_execution(agent, cancellation_token)
        runner_task, model = _resolve_run_dependencies(agent)

        return await RunLoop(
            delegate_runner=self,
            agent_delegation=self._agent_delegation,
            tool_executor=self._tool_executor,
            max_attempts=self._max_attempts,
            max_turns=self._max_turns,
            is_retryable=self._is_retryable,
            agent=agent,
            runner_task=runner_task,
            model=model,
            state=state,
            external_tool_agent_id=self._external_tool_agent_id,
            external_input=(
                self._external_input
                if agent.id == self._external_tool_agent_id
                else None
            ),
            hooks=coerce_runner_hooks(hooks),
            cancellation_token=cancellation_token,
        ).run()

    def run_stream(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream:
        """Execute the generic harness loop with live model streaming."""
        self._claim_external_execution(agent, cancellation_token)
        resolved_hooks = coerce_runner_hooks(hooks)
        stream = RunStream(
            on_close=(
                cancellation_token.cancel if cancellation_token is not None else None
            )
        )

        # Tie the background loop to the lifetime of the returned stream.
        stream_task = asyncio.create_task(
            self._run_stream_task(
                agent=agent,
                state=state,
                stream=stream,
                emit=stream.emit,
                hooks=resolved_hooks,
                cancellation_token=cancellation_token,
            )
        )
        stream.add_cleanup(cancel_task_callback(stream_task))

        return stream

    def run_events(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        approval_events: AsyncIterator["ToolApprovalEvent"] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunEventStream:
        """Execute the generic harness loop with high-level run events."""
        self._claim_external_execution(agent, cancellation_token)
        stream = RunEventStream(
            on_close=(
                cancellation_token.cancel if cancellation_token is not None else None
            )
        )

        # Forward approval and model events through the same run event stream.
        event_emitter = RunEventEmitter(stream.emit)
        if approval_events is not None:
            approval_task = asyncio.create_task(
                forward_tool_approval_events(
                    approval_events,
                    event_emitter=event_emitter,
                )
            )
            stream.add_cleanup(cancel_task_callback(approval_task))

        resolved_hooks = coerce_runner_hooks(hooks, event_emitter.hooks)

        # Start the loop with event hooks and cancel it when the stream closes.
        stream_task = asyncio.create_task(
            self._run_stream_task(
                agent=agent,
                state=state,
                stream=stream,
                emit=event_emitter.model_stream_event,
                event_emitter=event_emitter,
                hooks=resolved_hooks,
                cancellation_token=cancellation_token,
            )
        )
        stream.add_cleanup(cancel_task_callback(stream_task))

        return stream

    async def _run_stream_task(
        self,
        *,
        agent: Task,
        state: RunState,
        stream: RunStream | RunEventStream,
        emit: Callable[[ModelStreamEvent], None],
        event_emitter: RunEventEmitter | None = None,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> None:
        """Drive one streamed run and resolve the provided stream handle."""
        try:
            # Validate dependencies here so setup failures settle the returned stream.
            runner_task, model = _resolve_run_dependencies(agent)
            result = await RunLoop(
                delegate_runner=self,
                agent_delegation=self._agent_delegation,
                tool_executor=self._tool_executor,
                max_attempts=self._max_attempts,
                max_turns=self._max_turns,
                is_retryable=self._is_retryable,
                agent=agent,
                runner_task=runner_task,
                model=model,
                state=state,
                external_tool_agent_id=self._external_tool_agent_id,
                external_input=(
                    self._external_input
                    if agent.id == self._external_tool_agent_id
                    else None
                ),
                emit=emit,
                run_events=event_emitter,
                control_stream=stream,
                hooks=hooks,
                cancellation_token=cancellation_token,
            ).run()
        except Exception as exc:
            stream.fail(exc)
        except BaseException as exc:
            stream.fail(exc)
            raise
        else:
            stream.finish(result)


def _resolve_run_dependencies(agent: Task) -> tuple[RunnerTask, Model[ModelResponse]]:
    if not isinstance(agent, RunnerTask):
        raise RuntimeError(
            "Runner requires the task to expose the RunnerTask interface."
        )

    model = agent.model
    if model is None:
        raise RuntimeError(
            "Runner requires the task to expose a configured 'model' client."
        )

    return agent, model
