from types import SimpleNamespace

from deepeval.metrics.mcp.utils import turn_mcp_interaction_text
from deepeval.test_case import MCPToolCall


def _turn_with_result(result):
    call = MCPToolCall(
        name="get_weather",
        args={"city": "Paris"},
        result=result,
    )
    return SimpleNamespace(
        _mcp_tool_calls=[call],
        mcp_resources_called=None,
        mcp_prompts_called=None,
    )


def test_turn_mcp_interaction_text_reads_structured_content():
    result = SimpleNamespace(
        content=[],
        structured_content={"result": "sunny"},
    )

    text = turn_mcp_interaction_text(_turn_with_result(result))

    assert "sunny" in text


def test_turn_mcp_interaction_text_reads_legacy_structured_content():
    result = SimpleNamespace(
        content=[],
        structuredContent={"result": "sunny"},
    )

    text = turn_mcp_interaction_text(_turn_with_result(result))

    assert "sunny" in text


def test_turn_mcp_interaction_text_falls_back_to_content_text():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="sunny")],
        structured_content=None,
    )

    text = turn_mcp_interaction_text(_turn_with_result(result))

    assert "sunny" in text
