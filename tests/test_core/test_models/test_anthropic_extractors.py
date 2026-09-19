from anthropic.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

from deepeval.anthropic.extractors import (
    extract_messages_api_output_parameters,
    safe_extract_output_parameters,
)
from deepeval.model_integrations.types import InputParameters


def make_message(content, stop_reason="end_turn"):
    return Message(
        id="msg_1",
        content=content,
        model="claude-3-5-sonnet",
        role="assistant",
        type="message",
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=20),
    )


def make_input_parameters():
    return InputParameters(model="claude-3-5-sonnet")


def test_extract_output_parameters_reads_a_text_block():
    response = make_message([TextBlock(type="text", text="hello")])

    output_parameters = extract_messages_api_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.output == "hello"
    assert output_parameters.prompt_tokens == 10
    assert output_parameters.completion_tokens == 20


def test_extract_output_parameters_keeps_every_text_block():
    # Claude splits long answers across several blocks; only reading the
    # first one silently truncates the output that gets traced.
    response = make_message(
        [
            TextBlock(type="text", text="part one "),
            TextBlock(type="text", text="part two"),
        ]
    )

    output_parameters = extract_messages_api_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.output == "part one part two"


def test_extract_output_parameters_with_a_tool_use_block():
    # A tool call with no text before it leaves content[0] without a .text
    # attribute. The call itself, and the token counts, still have to be
    # reported.
    response = make_message(
        [
            ToolUseBlock(
                id="toolu_1",
                input={"query": "test"},
                name="search",
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )

    output_parameters = extract_messages_api_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.tools_called is not None
    assert output_parameters.tools_called[0].name == "search"
    assert output_parameters.tools_called[0].input_parameters == {
        "query": "test"
    }
    assert output_parameters.prompt_tokens == 10
    assert output_parameters.completion_tokens == 20
    # Same as the OpenAI extractor: with no text, the calls are the output.
    assert output_parameters.output == output_parameters.tools_called


def test_extract_output_parameters_with_a_thinking_block():
    response = make_message(
        [
            ThinkingBlock(
                type="thinking", thinking="let me think", signature="sig"
            ),
            TextBlock(type="text", text="the answer"),
        ]
    )

    output_parameters = extract_messages_api_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.output == "the answer"


def test_extract_output_parameters_with_empty_content():
    response = make_message([])

    output_parameters = extract_messages_api_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.output == ""
    assert output_parameters.prompt_tokens == 10


def test_safe_extract_output_parameters_keeps_telemetry_for_tool_calls():
    # The bare except in the safe wrapper turns any failure into an empty
    # OutputParameters, so a crash here is indistinguishable from a response
    # that genuinely carried nothing.
    response = make_message(
        [
            ToolUseBlock(
                id="toolu_1",
                input={"query": "test"},
                name="search",
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )

    output_parameters = safe_extract_output_parameters(
        response, make_input_parameters()
    )

    assert output_parameters.prompt_tokens == 10
    assert output_parameters.completion_tokens == 20
    assert output_parameters.tools_called is not None
    assert output_parameters.tools_called[0].name == "search"
