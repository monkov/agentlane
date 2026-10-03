"""Finish the MCP handshake, then resist graceful stdio process shutdown."""

import json
import os
import signal
import sys
import time
from pathlib import Path

Path(os.environ["MCP_TEST_PID"]).write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)

payload: dict[str, object]
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "server/discover":
        payload = {"error": {"code": -32601, "message": "Method not found"}}
    elif message.get("method") == "initialize":
        payload = {
            "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stubborn", "version": "1"},
            }
        }
    elif "id" in message:
        payload = {"result": {}}
    else:
        continue

    print(
        json.dumps({"jsonrpc": "2.0", "id": message["id"], **payload}),
        flush=True,
    )

# The SDK must escalate to a process-tree kill after stdin closes.
time.sleep(60)
