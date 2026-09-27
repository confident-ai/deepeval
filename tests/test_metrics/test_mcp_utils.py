"""Regression tests for the MCP tool-result rendering used by
MCPTaskCompletionMetric and MultiTurnMCPUseMetric judge prompts."""

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from deepeval.metrics.mcp.utils import (
    mcp_tool_result_text,
    turn_mcp_interaction_text,
)
from deepeval.test_case import MCPToolCall, Turn


def _text_result(*texts: str, **kwargs) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=t) for t in texts], **kwargs
    )


class TestMcpToolResultText:
    def test_falls_back_to_content_when_structured_content_is_none(self):
        # structuredContent is Optional and defaults to None per the MCP spec;
        # this is the shape most real tool results have.
        result = _text_result("42 degrees")
        assert result.structuredContent is None
        assert mcp_tool_result_text(result) == "42 degrees"

    def test_joins_multiple_text_blocks(self):
        result = _text_result("line one", "line two")
        assert mcp_tool_result_text(result) == "line one\nline two"

    def test_non_text_blocks_are_summarised_not_dropped(self):
        result = CallToolResult(
            content=[
                TextContent(type="text", text="chart"),
                ImageContent(type="image", data="AAAA", mimeType="image/png"),
            ]
        )
        assert mcp_tool_result_text(result) == "chart\n<ImageContent>"

    def test_prefers_structured_content_result_key(self):
        # FastMCP wraps primitive return values as {"result": value}.
        result = _text_result("42", structuredContent={"result": 42})
        assert mcp_tool_result_text(result) == 42

    def test_returns_whole_structured_content_when_no_result_key(self):
        payload = {"temperature": 42, "unit": "F"}
        result = _text_result("ignored", structuredContent=payload)
        assert mcp_tool_result_text(result) == payload

    def test_empty_structured_content_falls_back_to_content(self):
        result = _text_result("fallback", structuredContent={})
        assert mcp_tool_result_text(result) == "fallback"

    def test_plain_values_pass_through(self):
        assert mcp_tool_result_text("already a string") == "already a string"
        assert mcp_tool_result_text({"k": "v"}) == {"k": "v"}


class TestTurnMcpInteractionText:
    def test_does_not_raise_for_content_only_tool_result(self):
        # Before the fix this raised TypeError:
        # 'NoneType' object is not subscriptable
        turn = Turn(
            role="assistant",
            content="It is 42 degrees.",
            mcp_tools_called=[
                MCPToolCall(
                    name="get_weather",
                    args={"city": "Chicago"},
                    result=_text_result("42 degrees"),
                )
            ],
        )
        text = turn_mcp_interaction_text(turn)
        assert "Name: get_weather" in text
        assert "Args: {'city': 'Chicago'}" in text
        assert "42 degrees" in text

    def test_structured_result_still_rendered(self):
        turn = Turn(
            role="assistant",
            content="Done.",
            mcp_tools_called=[
                MCPToolCall(
                    name="add",
                    args={"a": 1, "b": 2},
                    result=_text_result("3", structuredContent={"result": 3}),
                )
            ],
        )
        assert "Result: \n3\n" in turn_mcp_interaction_text(turn)
