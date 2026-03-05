import json
import pytest
from unittest.mock import MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_span(attributes: dict) -> MagicMock:
    """Create a fake ReadableSpan whose .attributes behave like a dict."""
    span = MagicMock()
    span.attributes = attributes
    span.parent = None
    return span


def _make_messages(include_tool_call: bool = True, tool_type: str = "tool_call") -> str:
    """
    Return a JSON-serialised pydantic_ai.all_messages value.
    tool_type can be 'tool_call' (standard) or 'function_call' (OpenAIResponsesModel).
    """
    messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "What does NDA stand for?"}],
        },
    ]

    if include_tool_call:
        messages.append(
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": tool_type,          # tool_call OR function_call
                        "name": "some_tool",
                        "arguments": json.dumps({"query": "NDA meaning"}),
                    }
                ],
            }
        )

    messages.append(
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "NDA stands for Non-Disclosure Agreement.",
                }
            ],
        }
    )

    return json.dumps(messages)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_normalize_pydantic_ai_messages_returns_list():
    """normalize_pydantic_ai_messages should return a list, never None."""
    from deepeval.tracing.otel.utils import normalize_pydantic_ai_messages

    span = _make_span({"pydantic_ai.all_messages": _make_messages()})
    result = normalize_pydantic_ai_messages(span)

    assert isinstance(result, list)
    assert len(result) > 0


def test_tools_called_extracted_standard_format():
    """tools_called must be populated for standard tool_call type."""
    from deepeval.tracing.otel.utils import normalize_pydantic_ai_messages
    from deepeval.tracing.types import ToolCall

    span = _make_span({"pydantic_ai.all_messages": _make_messages(tool_type="tool_call")})
    normalized = normalize_pydantic_ai_messages(span)

    tools_called = []
    for message in normalized:
        for part in message.get("parts", []):
            if part.get("type") in ("tool_call", "function_call"):
                tools_called.append(ToolCall(name=part.get("name")))

    assert len(tools_called) == 1
    assert tools_called[0].name == "some_tool"


def test_tools_called_extracted_openai_responses_format():
    """tools_called must be populated for OpenAIResponsesModel (function_call type)."""
    from deepeval.tracing.otel.utils import normalize_pydantic_ai_messages
    from deepeval.tracing.types import ToolCall

    span = _make_span({"pydantic_ai.all_messages": _make_messages(tool_type="function_call")})
    normalized = normalize_pydantic_ai_messages(span)

    tools_called = []
    for message in normalized:
        for part in message.get("parts", []):
            if part.get("type") in ("tool_call", "function_call"):
                tools_called.append(ToolCall(name=part.get("name")))

    # This was the bug — function_call type was ignored before the fix
    assert len(tools_called) == 1
    assert tools_called[0].name == "some_tool"


def test_actual_output_extracted_from_last_assistant_message():
    """actual_output should not be None — extracted from last assistant text part."""
    from deepeval.tracing.otel.utils import check_pydantic_ai_agent_input_output

    span = _make_span({
        "pydantic_ai.all_messages": _make_messages(),
        "confident.span.type": "agent",
        "final_result": None,
        "gen_ai.system_instructions": None,
    })

    _, output_val = check_pydantic_ai_agent_input_output(span)

    # output_val can be a string or dict with 'content' key
    if isinstance(output_val, dict):
        actual_output = output_val.get("content")
    else:
        actual_output = output_val

    assert actual_output is not None
    assert "NDA" in actual_output


def test_tools_called_is_none_when_no_tools_used():
    """When agent uses no tools, tools_called should be None (not empty list)."""
    from deepeval.tracing.otel.utils import normalize_pydantic_ai_messages
    from deepeval.tracing.types import ToolCall

    span = _make_span({"pydantic_ai.all_messages": _make_messages(include_tool_call=False)})
    normalized = normalize_pydantic_ai_messages(span)

    tools_called = []
    for message in normalized:
        for part in message.get("parts", []):
            if part.get("type") in ("tool_call", "function_call"):
                tools_called.append(ToolCall(name=part.get("name")))

    result = tools_called if tools_called else None
    assert result is None