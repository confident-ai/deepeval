from anthropic.types.message import Message

from deepeval.anthropic.extractors import (
    extract_messages_api_output_parameters,
    safe_extract_output_parameters,
)
from deepeval.model_integrations.types import InputParameters
from deepeval.test_case.llm_test_case import ToolCall


def _message(content, input_tokens=1234, output_tokens=56):
    # Parse through the real Anthropic SDK so the content items are the
    # concrete block types (TextBlock / ToolUseBlock / ThinkingBlock) the
    # extractor branches on, exactly as they arrive from messages.create(...).
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
    )


_INPUT = InputParameters(
    model="claude-sonnet-4-5",
    tool_descriptions={"get_weather": "Get the weather for a city"},
)


def test_tool_use_first_response_preserves_tokens_and_tools_called():
    # A forced single tool call (tool_choice) returns content == [ToolUseBlock],
    # so content[0] has no `.text`. Token counts and the tool call must survive,
    # and output falls back to the tool calls (mirrors the OpenAI extractor).
    message = _message(
        [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ]
    )

    params = extract_messages_api_output_parameters(message, _INPUT)

    assert params.prompt_tokens == 1234
    assert params.completion_tokens == 56
    assert params.tools_called is not None
    assert len(params.tools_called) == 1
    assert params.tools_called[0].name == "get_weather"
    assert params.tools_called[0].input_parameters == {"city": "Paris"}
    assert isinstance(params.output, list)
    assert params.output[0].name == "get_weather"


def test_tool_use_first_response_is_not_swallowed_by_safe_wrapper():
    # Regression: previously content[0].text raised AttributeError, which the
    # bare `except` in safe_extract_output_parameters swallowed, returning an
    # empty OutputParameters (span recorded output='NA', no tokens, no tools).
    message = _message(
        [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ]
    )

    params = safe_extract_output_parameters(message, _INPUT)

    assert params.prompt_tokens == 1234
    assert params.completion_tokens == 56
    assert isinstance(params.tools_called, list) and params.tools_called
    assert params.output  # not None/empty -> not recorded as 'NA'


def test_thinking_first_response_extracts_text_and_preserves_tokens():
    # Extended thinking puts a ThinkingBlock before the answer's TextBlock.
    message = _message(
        [
            {
                "type": "thinking",
                "thinking": "reasoning...",
                "signature": "sig",
            },
            {"type": "text", "text": "The answer is 42."},
        ]
    )

    params = extract_messages_api_output_parameters(message, _INPUT)

    assert params.output == "The answer is 42."
    assert params.prompt_tokens == 1234
    assert params.completion_tokens == 56
    assert params.tools_called is None


def test_text_first_response_unchanged():
    # Control: a plain text answer is unaffected by the fix.
    message = _message([{"type": "text", "text": "Hello!"}])

    params = extract_messages_api_output_parameters(message, _INPUT)

    assert params.output == "Hello!"
    assert params.prompt_tokens == 1234
    assert params.completion_tokens == 56
    assert params.tools_called is None


def test_text_and_tool_use_response_keeps_text_output_and_tools():
    # When both are present, the text stays the output and the tool is captured.
    message = _message(
        [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ]
    )

    params = extract_messages_api_output_parameters(message, _INPUT)

    assert params.output == "Let me check the weather."
    assert params.tools_called is not None
    assert params.tools_called[0].name == "get_weather"
    assert isinstance(
        params.tools_called[0], ToolCall
    )  # ToolCall imported for type assertion
