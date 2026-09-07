from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from deepeval.model_integrations.gateways import OPENROUTER
from deepeval.model_integrations.types import InputParameters, OutputParameters
from deepeval.model_integrations.utils import (
    as_plain_dict,
    parse_tool_arguments,
    render_messages,
    stringify_multimodal_content,
)
from deepeval.test_case.llm_test_case import ToolCall


def _tool_descriptions(tools: Any) -> Optional[Dict[str, str]]:
    """Map tool name -> description for the tools declared on the request.

    Tools arrive as dicts or typed SDK models, in either the nested
    `{"function": {...}}` form or a flat one, so normalize before reading.
    """
    if not tools:
        return None
    descriptions: Dict[str, str] = {}
    for tool in tools:
        tool = as_plain_dict(tool)
        spec = (
            tool.get("function")
            if isinstance(tool.get("function"), dict)
            else tool
        )
        name = spec.get("name")
        if name:
            descriptions[name] = spec.get("description")
    return descriptions or None


def _first_user_message(messages: Any) -> str:
    for message in messages or []:
        message = as_plain_dict(message)
        if message.get("role") == "user":
            return stringify_multimodal_content(message.get("content"))
    return ""


def safe_extract_input_parameters(kwargs: Dict[str, Any]) -> InputParameters:
    try:
        return extract_input_parameters(kwargs)
    except Exception:
        return InputParameters(model="NA")


def extract_input_parameters(kwargs: Dict[str, Any]) -> InputParameters:
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")

    # `responses.send` takes `input`/`instructions` instead of `messages`.
    input_payload = kwargs.get("input")
    instructions = kwargs.get("instructions")

    if messages:
        rendered = render_messages(messages)
        summary = _first_user_message(messages)
    else:
        rendered = []
        if isinstance(input_payload, list):
            rendered = render_messages(input_payload)
        elif input_payload is not None:
            rendered = [
                {
                    "role": "user",
                    "content": stringify_multimodal_content(input_payload),
                }
            ]
        if instructions:
            rendered.insert(0, {"role": "system", "content": instructions})
        summary = stringify_multimodal_content(input_payload)

    return InputParameters(
        model=kwargs.get("model"),
        input=summary,
        messages=rendered,
        instructions=instructions,
        tools=tools,
        tool_descriptions=_tool_descriptions(tools),
    )


def safe_extract_output_parameters(
    response: Any, input_parameters: InputParameters
) -> OutputParameters:
    try:
        if hasattr(response, "choices"):
            parameters = extract_output_parameters_from_chat(
                response, input_parameters
            )
        else:
            parameters = extract_output_parameters_from_responses(
                response, input_parameters
            )
    except Exception:
        parameters = OutputParameters()

    # Metadata is independently guarded, so attach it even if the extraction
    # above fell over.
    parameters.metadata = OPENROUTER.extract_metadata(response)
    return parameters


def extract_output_parameters_from_chat(
    response: Any, input_parameters: InputParameters
) -> OutputParameters:
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    tools_called = _tools_called(_chat_tool_calls(message), input_parameters)

    usage = getattr(response, "usage", None)
    return OutputParameters(
        output=stringify_multimodal_content(getattr(message, "content", None))
        or tools_called,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        tools_called=tools_called,
    )


def extract_output_parameters_from_responses(
    response: Any, input_parameters: InputParameters
) -> OutputParameters:
    tools_called = _tools_called(
        _responses_tool_calls(response), input_parameters
    )

    usage = getattr(response, "usage", None)
    return OutputParameters(
        output=getattr(response, "output_text", None) or tools_called,
        prompt_tokens=getattr(usage, "input_tokens", None),
        completion_tokens=getattr(usage, "output_tokens", None),
        tools_called=tools_called,
    )


def _chat_tool_calls(message: Any) -> Iterator[Tuple[Any, Any]]:
    """Chat Completions nests the name and arguments under `function`."""
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        yield (
            getattr(function, "name", None),
            getattr(function, "arguments", None),
        )


def _responses_tool_calls(response: Any) -> Iterator[Tuple[Any, Any]]:
    """The Responses API puts them on the output item itself."""
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            yield getattr(item, "name", None), getattr(item, "arguments", None)


def _tools_called(
    calls: Iterable[Tuple[Any, Any]], input_parameters: InputParameters
) -> Optional[List[ToolCall]]:
    """Build `ToolCall`s from (name, arguments) pairs, or None if there are none."""
    descriptions = input_parameters.tool_descriptions or {}
    tools_called = [
        ToolCall(
            name=name,
            input_parameters=parse_tool_arguments(arguments),
            description=descriptions.get(name),
        )
        for name, arguments in calls
        if name
    ]
    return tools_called or None
