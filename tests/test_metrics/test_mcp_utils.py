"""Offline tests for the MCP interaction text helper (no API key needed)."""

from types import SimpleNamespace

import pytest

from deepeval.metrics.mcp.utils import turn_mcp_interaction_text
from deepeval.test_case import MCPToolCall

mcp = pytest.importorskip("mcp")


def _turn_with_call_result(result):
    call = MCPToolCall(
        name="get_weather", args={"city": "Paris"}, result=result
    )
    return SimpleNamespace(
        _mcp_tool_calls=[call],
        mcp_resources_called=None,
        mcp_prompts_called=None,
    )


def test_reads_structured_content_under_mcp_2_snake_case():
    # mcp>=2: the camelCase constructor alias is no longer an attribute, so
    # reading .structuredContent raised AttributeError before any scoring
    # (#3531).
    from mcp.types import CallToolResult

    result = CallToolResult(
        content=[{"type": "text", "text": "sunny"}],
        structuredContent={"result": "sunny"},
    )

    text = turn_mcp_interaction_text(_turn_with_call_result(result))

    assert "get_weather" in text
    assert "sunny" in text


def test_still_reads_camel_case_attribute_for_older_mcp():
    # mcp 1.x exposed the camelCase alias as the attribute itself.
    result = SimpleNamespace(structuredContent={"result": "raining"})

    text = turn_mcp_interaction_text(_turn_with_call_result(result))

    assert "raining" in text


def test_names_the_problem_when_no_structured_content():
    # Neither attribute present: a clear error beats a TypeError from
    # subscripting None.
    result = SimpleNamespace(content=[{"type": "text", "text": "sunny"}])

    with pytest.raises(AttributeError, match="no structured content"):
        turn_mcp_interaction_text(_turn_with_call_result(result))
