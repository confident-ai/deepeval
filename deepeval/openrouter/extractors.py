import json
from typing import Any, Dict, List, Optional

from deepeval.model_integrations.types import InputParameters, OutputParameters
from deepeval.model_integrations.utils import (
    extract_openrouter_metadata,
    render_messages,
    stringify_multimodal_content,
)
from deepeval.test_case.llm_test_case import ToolCall


def _plain(value: Any) -> Any:
    """Render a pydantic model as a dict, passing everything else through."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            return value
    return value


def _tool_descriptions(tools: Any) -> Optional[Dict[str, str]]:
    """Map tool name -> description for the tools declared on the request.

    Tools may arrive as dicts or as typed SDK models, and in either the nested
    ``{"function": {...}}`` form or a flat one, so normalize before reading.
    """
    if not tools:
        return None
    descriptions: Dict[str, str] = {}
    for tool in tools:
        tool = _plain(tool)
        if not isinstance(tool, dict):
            continue
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
        message = _plain(message)
        if isinstance(message, dict) and message.get("role") == "user":
            return stringify_multimodal_content(message.get("content"))
    return ""


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; never raise on bad JSON."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except (TypeError, ValueError):
        return {"raw": str(arguments)}


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

    # Metadata is additive and independently guarded, so attach it even if the
    # main extraction above fell over.
    parameters.metadata = extract_openrouter_metadata(response)
    return parameters


def extract_output_parameters_from_chat(
    response: Any, input_parameters: InputParameters
) -> OutputParameters:
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None

    output = stringify_multimodal_content(getattr(message, "content", None))

    tools_called: Optional[List[ToolCall]] = None
    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls:
        descriptions = input_parameters.tool_descriptions or {}
        tools_called = []
        for tool_call in raw_tool_calls:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", None)
            if not name:
                continue
            tools_called.append(
                ToolCall(
                    name=name,
                    input_parameters=_parse_arguments(
                        getattr(function, "arguments", None)
                    ),
                    description=descriptions.get(name),
                )
            )

    usage = getattr(response, "usage", None)
    return OutputParameters(
        output=output or tools_called,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        tools_called=tools_called,
    )


def extract_output_parameters_from_responses(
    response: Any, input_parameters: InputParameters
) -> OutputParameters:
    output = getattr(response, "output_text", None) or ""

    tools_called: Optional[List[ToolCall]] = None
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        name = getattr(item, "name", None)
        if not name:
            continue
        descriptions = input_parameters.tool_descriptions or {}
        tools_called = tools_called or []
        tools_called.append(
            ToolCall(
                name=name,
                input_parameters=_parse_arguments(
                    getattr(item, "arguments", None)
                ),
                description=descriptions.get(name),
            )
        )

    usage = getattr(response, "usage", None)
    return OutputParameters(
        output=output or tools_called,
        prompt_tokens=getattr(usage, "input_tokens", None),
        completion_tokens=getattr(usage, "output_tokens", None),
        tools_called=tools_called,
    )
