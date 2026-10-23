from types import SimpleNamespace

import pytest
from pydantic import Field, create_model

from deepeval.metrics.mcp.utils import turn_mcp_interaction_text
from deepeval.test_case import MCPToolCall


@pytest.mark.parametrize("field", ["structuredContent", "structured_content"])
def test_tool_result_uses_the_mcp_wire_alias(field):
    result_type = create_model(
        "ToolResult",
        **{field: (dict, Field(alias="structuredContent"))},
    )
    call = MCPToolCall(
        name="weather",
        args={"city": "Paris"},
        result=result_type(structuredContent={"result": "sunny"}),
    )
    turn = SimpleNamespace(
        _mcp_tool_calls=[call],
        mcp_resources_called=None,
        mcp_prompts_called=None,
    )

    assert "Result: \nsunny" in turn_mcp_interaction_text(turn)
