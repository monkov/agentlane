"""Serve synthetic meeting records through a local MCP server."""

import json
import re
from pathlib import Path
from typing import cast

from mcp.server import MCPServer

server = MCPServer("meeting-demo")
meetings = cast(
    list[dict[str, object]],
    json.loads(Path(__file__).with_name("meetings.json").read_text(encoding="utf-8")),
)


@server.tool()
def search_meetings(query: str) -> dict[str, object]:
    """Find meetings by words in their title or topics. Return IDs and titles."""
    words = re.findall(r"\w+", query.casefold())
    if not words:
        raise ValueError("Enter at least one search word.")
    matches: list[dict[str, object]] = []
    for meeting in meetings:
        searchable = (
            f"{meeting['title']} {' '.join(cast(list[str], meeting['topics']))}"
        )
        if all(word in searchable.casefold() for word in words):
            matches.append({"id": meeting["id"], "title": meeting["title"]})
    return {"matches": matches}


@server.tool()
def get_meeting(meeting_id: str) -> dict[str, object]:
    """Read one meeting by its ID, including its decision and supporting quote."""
    for meeting in meetings:
        if meeting["id"] == meeting_id:
            return dict(meeting)
    raise ValueError("No meeting has that ID.")


if __name__ == "__main__":
    server.run("stdio")
