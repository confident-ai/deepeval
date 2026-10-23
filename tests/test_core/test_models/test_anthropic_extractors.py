import pytest

pytest.importorskip("anthropic")

from anthropic.types import Message

from deepeval.anthropic.extractors import safe_extract_output_parameters
from deepeval.anthropic.patch import (
    _patch_async_anthropic_client_method,
    _patch_sync_anthropic_client_method,
)
from deepeval.model_integrations.types import InputParameters
from deepeval.test_case import ToolCall
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import LlmSpan, ToolSpan


MODEL = "claude-sonnet-4-5"
TEXT = {"type": "text", "text": "It is sunny."}
THINKING = {
    "type": "thinking",
    "thinking": "Internal reasoning, not the answer.",
    "signature": "test-signature",
}
REDACTED_THINKING = {"type": "redacted_thinking", "data": "redacted"}
WEATHER = {
    "type": "tool_use",
    "id": "tool_weather",
    "name": "get_weather",
    "input": {"city": "Paris"},
}
TIME = {
    "type": "tool_use",
    "id": "tool_time",
    "name": "get_time",
    "input": {"city": "Paris"},
}
WEATHER_CALL = ToolCall(
    name="get_weather",
    input_parameters={"city": "Paris"},
    description="Get the weather for a city.",
)
TIME_CALL = ToolCall(name="get_time", input_parameters={"city": "Paris"})


def make_message(content):
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": (
                "tool_use"
                if any(block["type"] == "tool_use" for block in content)
                else "end_turn"
            ),
            "stop_sequence": None,
            "usage": {"input_tokens": 1234, "output_tokens": 56},
        }
    )


@pytest.mark.parametrize(
    "content, expected_output, expected_tools",
    [
        pytest.param([TEXT], "It is sunny.", None, id="text-only"),
        pytest.param(
            [
                {"type": "text", "text": "It is "},
                {"type": "text", "text": "sunny."},
            ],
            "It is sunny.",
            None,
            id="multiple-text-blocks",
        ),
        pytest.param([WEATHER], [WEATHER_CALL], [WEATHER_CALL], id="tool-only"),
        pytest.param(
            [THINKING, TEXT], "It is sunny.", None, id="thinking-first"
        ),
        pytest.param(
            [REDACTED_THINKING, TEXT],
            "It is sunny.",
            None,
            id="redacted-thinking-first",
        ),
        pytest.param(
            [WEATHER, TEXT], "It is sunny.", [WEATHER_CALL], id="tool-first"
        ),
        pytest.param(
            [TEXT, WEATHER], "It is sunny.", [WEATHER_CALL], id="text-first"
        ),
        pytest.param(
            [
                THINKING,
                WEATHER,
                TEXT,
                TIME,
                {"type": "text", "text": " It is noon."},
            ],
            "It is sunny. It is noon.",
            [WEATHER_CALL, TIME_CALL],
            id="interleaved-text-and-tools",
        ),
        pytest.param([], "", None, id="empty-content"),
        pytest.param([THINKING], "", None, id="thinking-only"),
    ],
)
def test_extract_response_blocks(content, expected_output, expected_tools):
    params = safe_extract_output_parameters(
        make_message(content),
        InputParameters(
            model=MODEL,
            tool_descriptions={"get_weather": WEATHER_CALL.description},
        ),
    )

    assert params.output == expected_output
    assert params.prompt_tokens == 1234
    assert params.completion_tokens == 56
    assert params.tools_called == expected_tools


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "content, expected_output",
    [
        pytest.param([WEATHER], [WEATHER_CALL], id="tool-only"),
        pytest.param(
            [THINKING, TEXT, WEATHER], "It is sunny.", id="thinking-first"
        ),
    ],
)
async def test_response_blocks_reach_llm_and_tool_spans(
    content, expected_output, async_mode, completed_traces, monkeypatch
):
    # Exercise the same wrappers used by messages.create(), with no API calls.
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda *args, **kwargs: None
    )
    message = make_message(content)
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": WEATHER_CALL.description,
                "input_schema": {"type": "object"},
            }
        ],
    }

    if async_mode:

        async def create(**kwargs):
            return message

        response = await _patch_async_anthropic_client_method(create)(**kwargs)
    else:

        def create(**kwargs):
            return message

        response = _patch_sync_anthropic_client_method(create)(**kwargs)

    assert response is message
    assert len(completed_traces) == 1
    trace = completed_traces[0]
    assert trace.output == expected_output
    assert len(trace.root_spans) == 1
    span = trace.root_spans[0]
    assert isinstance(span, LlmSpan)
    assert span.output == expected_output
    assert span.input_token_count == 1234
    assert span.output_token_count == 56
    assert span.tools_called == [WEATHER_CALL]
    assert len(span.children) == 1
    tool_span = span.children[0]
    assert isinstance(tool_span, ToolSpan)
    assert tool_span.name == "get_weather"
    assert tool_span.input == {"city": "Paris"}
    assert tool_span.description == WEATHER_CALL.description
    assert tool_span.parent_uuid == span.uuid
