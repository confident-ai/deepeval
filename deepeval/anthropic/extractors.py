from typing import Any

from anthropic.types import ToolUseBlock
from anthropic.types.message import Message

from deepeval.anthropic.utils import (
    render_messages_anthropic,
    stringify_anthropic_content,
)
from deepeval.model_integrations.types import InputParameters, OutputParameters
from deepeval.test_case.llm_test_case import ToolCall


def safe_extract_input_parameters(kwargs: dict[str, Any]) -> InputParameters:
    # guarding against errors to be compatible with legacy APIs
    try:
        return extract_messages_api_input_parameters(kwargs)
    except Exception:
        return InputParameters(model="NA")


def extract_messages_api_input_parameters(
    kwargs: dict[str, Any],
) -> InputParameters:
    model = kwargs.get("model")
    tools = kwargs.get("tools")
    messages = kwargs.get("messages")
    tool_descriptions = (
        {tool["name"]: tool["description"] for tool in tools}
        if tools is not None
        else None
    )

    input_argument = ""
    user_messages = []
    for message in messages:
        role = message["role"]
        if role == "user":
            user_messages.append(message["content"])
    if len(user_messages) > 0:
        input_argument = user_messages[0]

    return InputParameters(
        model=model,
        input=stringify_anthropic_content(input_argument),
        messages=render_messages_anthropic(messages),
        tools=tools,
        tool_descriptions=tool_descriptions,
    )


def safe_extract_output_parameters(
    message_response: Message,
    input_parameters: InputParameters,
) -> OutputParameters:
    # guarding against errors to be compatible with legacy APIs
    try:
        return extract_messages_api_output_parameters(
            message_response, input_parameters
        )
    except Exception:
        return OutputParameters()


def extract_messages_api_output_parameters(
    message_response: Message,
    input_parameters: InputParameters,
) -> OutputParameters:
    content = getattr(message_response, "content", []) or []
    text_blocks = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    ]
    output = "".join(text_blocks)
    prompt_tokens = (
        message_response.usage.input_tokens
        if getattr(message_response, "usage", None)
        else None
    )
    completion_tokens = (
        message_response.usage.output_tokens
        if getattr(message_response, "usage", None)
        else None
    )

    tools_called = None
    anthropic_tool_calls = [
        block
        for block in content
        if getattr(block, "type", None) == "tool_use"
        or isinstance(block, ToolUseBlock)
    ]
    if anthropic_tool_calls:
        tools_called = []
        tool_descriptions = input_parameters.tool_descriptions or {}
        for tool_call in anthropic_tool_calls:
            tools_called.append(
                ToolCall(
                    name=tool_call.name,
                    input_parameters=tool_call.input,
                    description=tool_descriptions.get(tool_call.name),
                )
            )

    if not output and tools_called:
        output = tools_called

    return OutputParameters(
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools_called=tools_called,
    )
