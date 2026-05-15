from __future__ import annotations

from functools import lru_cache
from typing import Any

from deepeval.test_case.mcp import MCPServer, MCPToolCall


@lru_cache(maxsize=1)
def survey_mcp() -> dict[str, Any]:
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
    )

    return {
        "server": MCPServer(
            server_name="Survey MCP Server",
            transport="stdio",
            available_tools=[
                Tool(
                    name="survey_tool_a",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="survey_tool_b",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        ),
        "tools": [
            MCPToolCall(
                name="survey_tool",
                args={"query": "Testing"},
                result=CallToolResult(
                    content=[TextContent(type="text", text="Testing")]
                ),
            )
        ],
    }
