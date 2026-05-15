from __future__ import annotations

from functools import lru_cache
from typing import Any

from deepeval.test_case.mcp import MCPPromptCall, MCPResourceCall, MCPServer, MCPToolCall


@lru_cache(maxsize=1)
def survey_mcp() -> dict[str, Any]:
    from mcp.types import (
        CallToolResult,
        GetPromptResult,
        Prompt,
        PromptMessage,
        ReadResourceResult,
        Resource,
        TextContent,
        TextResourceContents,
        Tool,
    )

    turn_uri = "https://example.com/survey-turn-resource"
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
            ],
            available_resources=[
                Resource(
                    uri="https://example.com/survey-resource",
                    name="survey_resource",
                    mimeType="text/plain",
                    description="Testing",
                ),
            ],
            available_prompts=[Prompt(name="survey_prompt", description="Testing")],
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
        "resources": [
            MCPResourceCall(
                uri=turn_uri,
                result=ReadResourceResult(
                    contents=[
                        TextResourceContents(
                            uri=turn_uri,
                            mimeType="text/plain",
                            text="Testing",
                        )
                    ],
                ),
            )
        ],
        "prompts": [
            MCPPromptCall(
                name="survey_prompt",
                result=GetPromptResult(
                    description="Testing",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(type="text", text="Testing"),
                        )
                    ],
                ),
            )
        ],
    }
