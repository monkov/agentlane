"""Reject discovery and leave the legacy handshake pending."""

import json
import os
import sys
from pathlib import Path

Path(os.environ["MCP_TEST_PID"]).write_text(str(os.getpid()))

for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "server/discover":
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            ),
            flush=True,
        )
