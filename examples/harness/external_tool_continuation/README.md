# External tool continuation

This offline example runs a question and its answer in separate processes.
The application declares `request_input` through the generic `ExternalTool`
contract. It accepts a displayed choice or free text. A scripted model makes
the example repeatable and requires no provider key.

From the repository root:

```bash
uv run python examples/harness/external_tool_continuation/main.py start /tmp/delivery-demo.sqlite
uv run python examples/harness/external_tool_continuation/main.py resume /tmp/delivery-demo.sqlite "Leave it with reception after 10 AM."
```

Use a new database path for another run. The first command saves the
continuation before it displays the question, then exits. The second command
restores the stable agent address and inserts the answer as the result of the
original tool call. It saves a completed `AgentSnapshot` at revision 1.

SQLite is the application's storage choice. Its conditional update saves the
answer and claims the waiting request in one transaction. A second resume is
rejected. If execution fails after the claim, the request stays `running`.
The example does not retry that uncertain attempt. A production host must
decide how to reconcile it and protect any external side effects.

This example starts at an empty completed baseline. A host that suspends a
later run must also retain and restore its previous completed `AgentSnapshot`.
Authentication, answer authorization, cancellation, expiration, and request
listing belong to the host and are outside this example.

See [External tool continuations](../../../docs/harness/runner.md#external-tool-continuations)
for the runtime checks and supported execution boundary.
