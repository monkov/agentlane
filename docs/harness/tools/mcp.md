# MCP

`MCPToolsShim` adds tools from Model Context Protocol (MCP) servers to an
AgentLane agent. It supports Streamable HTTP and local stdio servers.
Install the optional dependency:

```bash
uv add 'agentlane[mcp]'
```

## Connect an agent

Attach the shim through `AgentDescriptor.shims`. Replace the server URL and
use your configured AgentLane `model`:

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

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Notes assistant",
        model=model,
        shims=(MCPToolsShim(servers=(notes,)),),
    ),
)

result = await agent.run("Find the latest launch decision in the notes.")
print(result.final_output)
```

The shim opens and closes connections for each run. For a runnable local
example, see the
[MCP meeting assistant](../../../examples/harness/mcp_meeting_assistant/README.md).

HTTP servers require HTTPS. For development HTTP, set
`MCPStreamableHTTPTransport(..., allow_insecure_http=True)`. Supply credentials
through an authorization provider, not the URL. Legacy SSE, MCP resources,
and MCP prompts are not supported as model capabilities.

### Use a local server

Replace the HTTP transport with `MCPStdioTransport`. Set the command, working
directory, and environment for your server:

```python
from agentlane.harness.mcp import MCPStdioTransport

notes = MCPServer(
    name="notes",
    transport=MCPStdioTransport(
        command="python",
        args=("-m", "my_notes_mcp"),
        cwd="/srv/my-product",
        env={"NOTES_DATABASE": "/data/notes.db"},
    ),
)
```

Stdio inherits only the MCP SDK's allowed environment variables plus explicit
`env` values. Pass the variables your server needs.

## Select tools

The model sees names such as `notes__list_meetings`. Use these names in
`RESTRICT_TOOLS`, `OVERRIDE_TOOLS`, and `ExcludeToolsShim` policies.
Names are normalized; use simple, unique server names to avoid collisions.

To expose only selected tools, set `MCPServer.tools`. Filter patterns match
the original names reported by the server:

```python
from agentlane.harness.mcp import MCPToolFilter

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    tools=MCPToolFilter(include=("list_*", "get_*"), exclude=("*_transcript",)),
)
```

Place `MCPToolsShim` before `SkillsShim` when skill tool rules must cover MCP
tools. For subagent and handoff policies, see [shims](../shims.md).

## Authorization

Your app handles OAuth consent, credential storage, refresh, and revocation.
Implement `MCPAuthorizationProvider` with two async methods:

- `get_access_token(server, context)` returns a valid `MCPAccessToken` with
  `token` and optional `expires_at` and `scopes`.
- `invalidate_access_token(server, context, token)` invalidates a token that
  the server rejected.

Attach your `provider` to the HTTP server and pass the user's identity:

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

Use a stable, non-secret context `key` for each authorized user or connection.
Different keys keep connections and tool catalogs separate. The optional
`value` passes application data to your provider.

The provider is called repeatedly; reuse valid tokens until refresh is needed.
On HTTP `401`, AgentLane invalidates the token and retries once with a token
from the provider. AgentLane does not store refresh tokens.

## Reuse connections

For multiple agents or runs, share one application-scoped `MCPClientManager`:

```python
from agentlane.harness.mcp import MCPClientManager

async with MCPClientManager() as manager:
    shim = MCPToolsShim(servers=(notes,), client_manager=manager)
    # Attach this shim to agents and run them inside this context.
```

If your app manages shutdown explicitly, call `await manager.aclose()`.
Finish active runs before closing the manager.
For authorized servers, also pass the user's `authorization_context` to each shim.

Set `MCPClientManager(limits=MCPClientLimits(...))` to change the defaults:
64 connections, 300 seconds of idle time, and a 10-second shutdown timeout.
Both classes are exported from `agentlane.harness.mcp`.

## Configure limits

Set these options on `MCPServer`:

| Option | Default | Use |
| --- | --- | --- |
| `required` | `True` | Stop the run if discovery fails. Set `False` to continue without that server's tools. |
| `connect_timeout_seconds` | `30` | Limit connection setup and the MCP handshake. |
| `discovery_timeout_seconds` | `30` | Limit tool discovery. |
| `tool_timeout_seconds` | `120` | Limit each tool call. |
| `catalog_ttl_seconds` | `300` | Limit how long the tool list can stay cached. |

Timeouts and catalog TTL must be finite and greater than zero. Optional
servers retry transient discovery failures on later turns. AgentLane does
not automatically repeat a tool call after a transport failure or timeout.

To change output limits, set `MCPServer.result_policy=MCPResultPolicy(...)`.
The defaults are 32 content blocks and 51,200 characters. Servers repeat
structured content in text blocks for clients that cannot read it. If a
successful result has structured content, the model gets the structured
content and the other content blocks, but not the text blocks. A failed result
keeps its text blocks, because they contain the failure details. Set
`include_structured_content=False` to omit structured results and keep all
content blocks.
Image, audio, and resource bodies are returned as metadata.

## Store full results

Pass an existing `FileWriter` as `result_writer` and set `result_directory`
on `MCPToolsShim`. AgentLane writes the supported result content as JSON and
readable text, waits for both writes, then returns a preview with file paths.
Both options default to `None` and must be configured together. Without a
writer, result handling stays unchanged.

AgentLane saves a result when the preview loses supported content. This includes
size or block limits, text hidden by structured content, embedded resource
text, and structured content excluded by the result policy. To also save
small results from selected tools, set `preserve_tools` to exact pairs of
original server and tool names:

```python
from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.mcp import MCPToolsShim
from agentlane.harness.tools import HarnessToolsShim, read_tool

filesystem = MountedFileSystem({
    "workspace": LocalFileSystem(root="./workspace"),
})
shim = MCPToolsShim(
    servers=(notes,),
    result_writer=filesystem,
    result_directory="/workspace/tool-results",
    preserve_tools=frozenset({("notes", "get_transcript")}),
)
reader = read_tool(reader=filesystem)
shims = (shim, HarnessToolsShim((reader,)))
```

Pass `shims` to `AgentDescriptor`. MCP storage and the reader use the same paths.
AgentLane generates a UUID basename for each result, writes `.json` first,
then writes `.txt`. A plain writer uses relative paths; a writer with
`FilePathResolver`, such as `MountedFileSystem`, uses its own path rules.
See [File I/O interfaces](../filesystem.md) for backend contracts and mounts.

The default `preserve_tools=frozenset()` saves only results whose supported
content does not fit the preview. Selection uses `("notes", "get_transcript")`,
not the model-visible name `notes__get_transcript`. Selection has no effect
without a writer.

The writer and directory belong to the shim. Catalog refreshes and inherited
MCP tools keep them. Shared client connections keep each shim's storage
configuration separate.

### Read saved content

The returned preview includes an `output` object:

```json
{
  "content": [{"type": "text", "text": "Beginning of the result…"}],
  "isError": false,
  "truncated": true,
  "output": {
    "source_path": "/workspace/tool-results/example.json",
    "read_path": "/workspace/tool-results/example.txt"
  }
}
```

Configure a reader that can access these paths. For example, with the native
[`read`](./read.md) tool, the model can request:

```text
read(path="/workspace/tool-results/example.txt", offset=1, limit=200)
```

Use the reader's continuation offset to read more content. Each text line has
at most 4 KiB of UTF-8 content, plus its newline. Control characters are
escaped. The JSON file is the source for further processing; the text file
is a view for reading in parts.

### Stored content and limits

The JSON source contains `schema_version` (`1`), `server_name`, `tool_name`,
`content`, `structuredContent`, `isError`, and `omissions`. It keeps both text
and structured content. It also keeps embedded resource text. Image, audio,
and resource blob bodies are replaced with omission metadata; `omissions`
includes `"binary"` when this occurs.

AgentLane removes known credentials and values under credential keys before
writing. The files contain result content and tool names. Request arguments
are not added to the files.

Source preparation rejects JSON text with duplicate keys or incomplete
objects and arrays that contain quoted fields. This prevents ambiguous JSON
from bypassing credential redaction. Plain bracketed transcript text remains
supported. Rejected source text produces `output.status="unavailable"`.

Source processing has fixed limits of 8 MiB per representation, 4,096 nodes,
and 32 levels of nesting. These bounds apply after the MCP SDK receives the
result; they do not limit the SDK's initial result allocation. If supported
content cannot be kept within these bounds, AgentLane does not save a partial
source.

File paths share the configured preview budget, which defaults to 51,200
characters and 32 content blocks. AgentLane keeps each returned path complete.
If the paths cannot fit, the preview reports `"output":{"status":"unavailable"}`.
Files can already exist when their paths cannot fit the preview.

### Host responsibilities and failures

Supply a writer and reader in the same namespace. The host owns backend
deadlines, access control, and retention. Remote storage uses the application's
existing filesystem adapter.

Writes use the same helper as native write and patch tools. Adapter methods
run in worker threads. Successful `open_write` context exit completes one
file. `FileWriter` has no two-file transaction, create-only option, or deletion
method. If the second write fails, the completed first file can remain. The
host must handle cleanup and protect saved files if later processing requires
immutable input.

The MCP tool timeout covers the provider call. Configure write deadlines in
the backend. Cancellation cannot stop a blocking writer: run cancellation
waits for a started write to settle before returning a cancelled tool result.
Client manager shutdown also cancels active storage and waits for it to settle,
subject to the manager's shutdown timeout.

If source preparation or storage fails, the model still gets the bounded
preview with `"output":{"status":"unavailable"}`. AgentLane does not repeat
the MCP call. A provider error keeps its `ToolFailure` classification even
when storage fails. Storage exceptions, source bodies, and request arguments
are not added to storage failure messages.

## Troubleshooting

Use the symptom or error to choose the next step:

| Symptom or error | Action |
| --- | --- |
| `MCPDependencyError` | Install `agentlane[mcp]` in the environment that runs the agent. |
| Connection or discovery fails | Check the endpoint or local command, server availability, and credentials. Use `MCPDiscoveryError.failure_kind` to distinguish authorization, transport, timeout, capacity, and protocol failures. |
| A tool is missing | Check the server's tool list, `MCPToolFilter`, and agent tool policies. Filters use original names; agent policies use names such as `notes__list_meetings`. |
| `ToolNameCollisionError` | Give servers distinct names and check for duplicate tool names after normalization. |
| HTTP `401` or `403` | Check token validity and required scopes. Ensure invalidation lets the provider return a fresh token. |
| A tool call times out | Check whether the server completed the operation before retrying. Increase `tool_timeout_seconds` only if the operation needs more time. |
| `MCPPoolCapacityError` | Reduce concurrent runs or increase `MCPClientLimits.max_connections`. |
| `MCPShutdownTimeoutError` | Check for a server that does not stop. Increase `shutdown_timeout_seconds` if needed; call `aclose()` again to wait for ongoing cleanup. |
| Results contain `truncated` or `omittedBlocks` | Read `output.read_path` when present. Otherwise, configure `result_writer` and `result_directory`, request less data, or increase the relevant `MCPResultPolicy` limit. `max_text_chars` must be at least 128. |
| `output.status` is `unavailable` | Check source processing limits, write failures, and whether both paths fit the preview budget. The provider call is not repeated. |

Tool execution failures return `ToolFailure`. Treat server results as
untrusted input, and keep secrets out of tool arguments and results.
