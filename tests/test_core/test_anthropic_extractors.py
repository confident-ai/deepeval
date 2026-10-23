from anthropic.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

from deepeval.anthropic.extractors import (
    extract_messages_api_output_parameters,
    safe_extract_input_parameters,
    safe_extract_output_parameters,
)
from deepeval.model_integrations.types import InputParameters
from deepeval.test_case.llm_test_case import ToolCall


def test_extract_tool_use_block_only():
    msg = Message(
        id="msg_1",
        content=[
            ToolUseBlock(
                id="tool_1",
                input={"query": "python"},
                name="search",
                type="tool_use",
            )
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=25),
    )
    input_params = InputParameters(
        model="claude-3-5-sonnet",
        tool_descriptions={"search": "Searches documentation"},
    )

    output_params = extract_messages_api_output_parameters(msg, input_params)

    assert output_params.prompt_tokens == 10
    assert output_params.completion_tokens == 25
    assert output_params.tools_called is not None
    assert len(output_params.tools_called) == 1

    tool = output_params.tools_called[0]
    assert isinstance(tool, ToolCall)
    assert tool.name == "search"
    assert tool.input_parameters == {"query": "python"}
    assert tool.description == "Searches documentation"
    # When no text is generated, output falls back to tool_calls
    assert output_params.output == output_params.tools_called


def test_extract_thinking_block_with_text():
    msg = Message(
        id="msg_2",
        content=[
            ThinkingBlock(
                thinking="Reasoning about the query...",
                signature="sig_abc",
                type="thinking",
            ),
            TextBlock(
                text="The answer is 42.",
                type="text",
            ),
        ],
        model="claude-3-7-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=50, output_tokens=100),
    )
    input_params = InputParameters(model="claude-3-7-sonnet")

    output_params = extract_messages_api_output_parameters(msg, input_params)

    assert output_params.output == "The answer is 42."
    assert output_params.prompt_tokens == 50
    assert output_params.completion_tokens == 100
    assert output_params.tools_called is None


def test_extract_multiple_text_blocks():
    msg = Message(
        id="msg_3",
        content=[
            TextBlock(text="Part 1. ", type="text"),
            TextBlock(text="Part 2.", type="text"),
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=12, output_tokens=18),
    )
    input_params = InputParameters(model="claude-3-5-sonnet")

    output_params = extract_messages_api_output_parameters(msg, input_params)

    assert output_params.output == "Part 1. Part 2."
    assert output_params.prompt_tokens == 12
    assert output_params.completion_tokens == 18
    assert output_params.tools_called is None


def test_extract_mixed_text_and_tool():
    msg = Message(
        id="msg_4",
        content=[
            TextBlock(text="Calling weather tool:", type="text"),
            ToolUseBlock(
                id="tool_2",
                input={"location": "San Francisco"},
                name="get_weather",
                type="tool_use",
            ),
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=20, output_tokens=30),
    )
    input_params = InputParameters(
        model="claude-3-5-sonnet",
        tool_descriptions={"get_weather": "Get weather for location"},
    )

    output_params = extract_messages_api_output_parameters(msg, input_params)

    assert output_params.output == "Calling weather tool:"
    assert output_params.prompt_tokens == 20
    assert output_params.completion_tokens == 30
    assert output_params.tools_called is not None
    assert len(output_params.tools_called) == 1
    assert output_params.tools_called[0].name == "get_weather"
    assert output_params.tools_called[0].input_parameters == {
        "location": "San Francisco"
    }


def test_extract_empty_content():
    msg = Message(
        id="msg_5",
        content=[],
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=5, output_tokens=0),
    )
    input_params = InputParameters(model="claude-3-5-sonnet")

    output_params = extract_messages_api_output_parameters(msg, input_params)

    assert output_params.output == ""
    assert output_params.prompt_tokens == 5
    assert output_params.completion_tokens == 0
    assert output_params.tools_called is None


def test_safe_extract_output_parameters():
    # Verify safe extraction with tool call
    msg = Message(
        id="msg_6",
        content=[
            ToolUseBlock(
                id="tool_3",
                input={"q": "test"},
                name="search",
                type="tool_use",
            )
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=8, output_tokens=15),
    )
    input_params = InputParameters(model="claude-3-5-sonnet")

    res = safe_extract_output_parameters(msg, input_params)
    assert res.prompt_tokens == 8
    assert res.completion_tokens == 15
    assert len(res.tools_called) == 1

    # Verify fallback on exception
    class BadMessage:
        @property
        def content(self):
            raise RuntimeError("Corrupted response")

    fallback_res = safe_extract_output_parameters(BadMessage(), input_params)
    assert fallback_res.output is None
    assert fallback_res.prompt_tokens is None


def test_safe_extract_input_parameters():
    kwargs = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"name": "tool1", "description": "desc1"}],
    }
    params = safe_extract_input_parameters(kwargs)
    assert params.model == "claude-3-5-sonnet"
    assert params.input == "Hello"
    assert params.tool_descriptions == {"tool1": "desc1"}

    # Fallback on invalid kwargs
    fallback = safe_extract_input_parameters(None)
    assert fallback.model == "NA"
