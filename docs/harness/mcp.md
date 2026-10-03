# Harness MCP Tools

`MCPToolsShim` discovers tools from Model Context Protocol (MCP) servers and
adds them to an agent as native AgentLane tools. It supports Streamable HTTP
and stdio. Install the optional dependency:

```bash
uv add 'agentlane[mcp]'
```

## Connect an agent

Attach the shim through `AgentDescriptor.shims`. This example uses an HTTPS
server and a configured AgentLane `model`:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPServer,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
)

mcp = MCPToolsShim(servers=(notes,))

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Notes assistant",
        model=model,
        shims=(mcp,),
    ),
)

result = await agent.run("Find the latest launch decision in the notes.")
print(result.final_output)
```

The shim opens and closes its connections for each run.

For a runnable example with a local server, see the
[MCP Meeting Assistant](../../examples/harness/mcp_meeting_assistant/README.md).

## Tool names and context

The model receives a flat list of tool definitions named `<server>__<tool>`.
For example, `list_meetings` on the `notes` server becomes
`notes__list_meetings`. The catalog is not added to the system prompt.

Names are converted to lowercase, unsupported character runs become `_`, and
leading or trailing underscores are removed. A name that starts with a digit
gets an `mcp_` prefix. Long combined names receive a stable hash suffix to fit
the 64-character limit. Calls to the MCP server use the original names.

Duplicate names in MCP catalogs fail discovery. A collision between an MCP
tool and another built-in tool contribution raises `ToolNameCollisionError`
from `agentlane.harness.shims` before the model call, in either shim order.
Custom shims must contribute tools through `PreparedTurn.add_tools(...)` to
participate in this check.

Use `MCPToolFilter` to limit which tools enter the model context. Its include
and exclude patterns match original MCP tool names:

```python
from agentlane.harness.mcp import MCPToolFilter

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    tools=MCPToolFilter(include=("list_*", "get_*"), exclude=("*_transcript",)),
)
```

MCP tools preserve the server's input schema with `strict=False`. They set
`retry_on_timeout=False` to prevent automatic repeats after a tool timeout.
See [model tool policy](../models/overview.md#schema-and-timeout-policy).

## Authorization

The application owns OAuth consent, credential storage, token refresh, and
revocation. AgentLane receives only access tokens through an
`MCPAuthorizationProvider` with two async methods:

- `get_access_token(server, context)` returns an `MCPAccessToken` containing
  `token`, optional `expires_at`, and optional `scopes`.
- `invalidate_access_token(server, context, token)` invalidates a rejected
  token in the application's credential service.

Attach your provider to the HTTP server and pass the user's identity to the
shim:

```python
from agentlane.harness.mcp import MCPAuthorizationContext

authorized_notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    authorization=provider,
)

shim = MCPToolsShim(
    servers=(authorized_notes,),
    authorization_context=MCPAuthorizationContext(key=user_id, value=user_id),
)
```

The context `key` must be a stable, non-secret string that identifies the
authorized user or connection. The optional `value` is opaque application data
passed to the provider. Each run owns separate connections. Both the provider
and context value stay in memory.

Before each HTTP request and catalog check, AgentLane asks for a current token.
The provider should reuse a valid token until refresh is needed. A changed token
or set of scopes invalidates the discovered tools. On `401`, AgentLane invalidates
that token, requests another, and retries once. Provider errors, a final `401`,
and `403` are authorization failures. A rejected token, `403`, or provider failure
also invalidates existing catalogs, even if a later lookup returns the same
token. AgentLane does not request or store refresh tokens.

## Connection lifecycle and discovery

The shim opens separate connections for each run and closes them when that run
ends. Child runs own their connections independently. Reusing a shim definition
does not retain runtime connections between runs.

The shim discovers tools at startup and before each model turn. Each check
requests all catalog pages. AgentLane and MCP SDK discovery caching are disabled.
The `max_concurrent_discoveries` parameter on
`MCPToolsShim` limits concurrent server checks to 8 by default. This limit
covers connection setup and catalog discovery, and does not limit tool execution.

Each discovery sees additions and removals from the server. If returned pages
use different credentials, discovery restarts once within the same timeout.
Unstable authorization fails discovery without publishing a mixed catalog.
Authorization is checked again before the first HTTP tool request. Credentials
must still match discovery before that request is sent. The single refresh and
retry after an explicit `401` remains supported.
Failed discovery does not reuse the previous tool list.

Servers are required by default: a connection or discovery failure stops the
run before the next model call. Set `required=False` to let the run continue
without tools from an unavailable server. Optional servers retry transient
connection and timeout failures on a later turn preparation, even
if the first connection attempt failed. The retry delays are 1, 2, 4, 8, 16,
then 30 seconds between attempts. A successful check resets the delay.
There are no background retry attempts. Authorization, configuration, and
protocol failures disable retries for that server for the current run.
Failures during tool execution return `ToolFailure`.

A lost connection is reopened at the next catalog check. AgentLane does not
replay a tool call after a transport failure or timeout. The shim cancels active
work and starts connection cleanup concurrently. It waits up to 10 seconds per
cleanup attempt. A timeout produces `MCPShutdownTimeoutError` through the
harness cleanup-error path, while transport cleanup continues in owned
background tasks. If the run already failed, its original error stays primary.

The connection cleanup deadline also covers credential lookup during HTTP
session termination. Stdio cleanup can continue beyond that deadline while
the SDK completes its bounded graceful-exit and forced-kill stages.

## Timeouts

Set separate connection, discovery, and execution limits on `MCPServer`:

```python
server = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    connect_timeout_seconds=30,
    discovery_timeout_seconds=30,
    tool_timeout_seconds=120,
)
```

These are the defaults, in seconds. The connection limit includes the MCP
handshake for both transports. `MCPStreamableHTTPTransport` also has
`connect_timeout_seconds=30` and `read_timeout_seconds=300` for individual HTTP
operations. These configured timeout values must be finite and greater
than zero.

## Inheritance and tool policies

`INHERIT_TOOLS`, `RESTRICT_TOOLS`, `OVERRIDE_TOOLS`, and `ExcludeToolsShim` use
the model-visible names, such as `notes__list_meetings`. Tool-call and
round-trip limits apply after all shims contribute tools.

Subagents and handoffs bind inherited MCP tools independently, limited to the
names allowed by the parent's policy. Child-local tools remain available.
Only servers that supply inherited tool names are connected in the child.
Name collisions between inherited and child-local tools raise an error; use
`OVERRIDE_TOOLS` for intentional replacement. Each child owns separate
connections, so ending one run does not close another run's connection.

When an MCP source is wrapped, the child binds the original wrapper chain
again and gets fresh wrapper state. The restricted source binding selects
the permitted servers and tool names before it creates the child session.
The original MCP configuration cannot broaden that inherited set. Custom
wrappers must pass the complete `ShimBindingContext` to the inner `bind(...)`
method. See [dynamic source inheritance](./shims.md#advanced-bound-sessions)
for the `ToolSourceBinding` contract.

Place `MCPToolsShim` before `SkillsShim` when skill `tools` or
`disallowed-tools` rules must cover MCP tools. `ExcludeToolsShim` works in
either order. See [shims](./shims.md) for preparation and inheritance callbacks.

## Results and data handling

Results contain text and structured content as JSON. `MCPResultPolicy` defaults
to 32 content blocks and 51,200 output characters. Set `MCPServer.result_policy`
to change these limits or disable structured content with
`include_structured_content=False`. `max_text_chars` must be at least 128.
Truncated results remain valid JSON and include omission counts. Image, audio,
and resource bodies are replaced with metadata.

AgentLane redacts known provider token values and credential-shaped fields
from results and framework errors. Managed SDK and transport logs are
suppressed; AgentLane's own MCP logs contain operation metadata. Tool data is
included in tracing only when the tracing policy permits it. Treat server text
as untrusted input.

Tokens, providers, HTTP clients, and MCP sessions are kept out of prompts,
`RunState`, snapshots, and run events.

## Transport configuration

Remote servers require HTTPS. Set `allow_insecure_http=True` for development
HTTP. Redirects may stay on the same origin or upgrade from HTTP to HTTPS on
the same host using default ports. URLs must not contain credentials in
userinfo, known token or OAuth callback query parameters, or fragments.

For a local server process, use `MCPStdioTransport`:

```python
from agentlane.harness.mcp import MCPStdioTransport

local_server = MCPServer(
    name="local-notes",
    transport=MCPStdioTransport(
        command="python",
        args=("-m", "my_notes_mcp"),
        cwd="/srv/my-product",
        env={"NOTES_DATABASE": "/data/notes.db"},
    ),
)
```

Stdio inherits the MCP SDK's safe environment allow-list plus explicit `env`
values. Pass only the environment variables the server needs.

Legacy SSE and MCP resources or prompts as model capabilities are not supported.
