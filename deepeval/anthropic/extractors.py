from typing import Any, Dict

from deepeval.anthropic.types import InputParameters, OutputParameters


def extract_input_parameters(
    kwargs: Dict[str, Any]
) -> InputParameters:
    return extract_messages_api_input_parameters(kwargs)


def extract_messages_api_input_parameters(
    kwargs: Dict[str, Any],
) -> InputParameters:
    model = kwargs.get("model")
    max_tokens = kwargs.get("max_tokens")
    tools = kwargs.get("tools")
    messages = kwargs.get("messages")
    tool_descriptions = (
        {
            tool["name"]: tool["description"]
            for tool in tools
        }
        if tools is not None
        else None
    )
    return InputParameters(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_descriptions=tool_descriptions
    )


