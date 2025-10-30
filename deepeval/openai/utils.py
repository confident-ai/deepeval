import json
import uuid
from typing import Any, Dict, List, Iterable

from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)

from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.tracing.context import current_span_context
from deepeval.model_integrations.types import OutputParameters
from deepeval.model_integrations.utils import compact_dump, fmt_url


def create_child_tool_spans(output_parameters: OutputParameters):

    if output_parameters.tools_called is None:
        return

    current_span = current_span_context.get()
    for tool_called in output_parameters.tools_called:
        tool_span = ToolSpan(
            **{
                "uuid": str(uuid.uuid4()),
                "trace_uuid": current_span.trace_uuid,
                "parent_uuid": current_span.uuid,
                "start_time": current_span.start_time,
                "end_time": current_span.start_time,
                "status": TraceSpanStatus.SUCCESS,
                "children": [],
                "name": tool_called.name,
                "input": tool_called.input_parameters,
                "output": None,
                "metrics": None,
                "description": tool_called.description,
            }
        )
        current_span.children.append(tool_span)


def stringify_multimodal_content(content: Any) -> str:
    """
    Return a short, human-readable summary string for an OpenAI-style multimodal `content` value.

    This is used to populate span summaries, such as `InputParameters.input`. It never raises and
    never returns huge blobs.

    Notes:
    - Data URIs are redacted to "[data-uri]".
    - Output is capped via `deepeval.utils.shorten` (configurable through settings).
    - Fields that are not explicitly handled are returned as size-capped JSON dumps
    - This string is for display/summary only, not intended to be parsable.

    Args:
        content: The value of an OpenAI message `content`, may be a str or list of typed parts,
                 or any nested structure.

    Returns:
        A short, readable `str` summary.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return f"[bytes:{len(content)}]"

    # list of parts for Chat & Responses
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            s = stringify_multimodal_content(part)
            if s:
                parts.append(s)
        return "\n".join(parts)

    # documented dict shapes (Chat & Responses)
    if isinstance(content, dict):
        t = content.get("type")

        # Chat Completions
        if t == "text":
            return str(content.get("text", ""))
        if t == "image_url":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{fmt_url(url)}]"

        # Responses API variants
        if t == "input_text":
            return str(content.get("text", ""))
        if t == "input_image":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{fmt_url(url)}]"

        # readability for other input_* types we don't currently handle
        if t and t.startswith("input_"):
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return compact_dump(content)


def render_messages(
    messages: Iterable[ChatCompletionMessageParam],
) -> List[Dict[str, Any]]:

    messages_list = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    # Extract type - either "function" or "custom"
                    tool_type = tool_call.get("type", "function")

                    # Extract name and arguments based on type
                    if tool_type == "function":
                        function_data = tool_call.get("function", {})
                        name = function_data.get("name", "")
                        arguments = function_data.get("arguments", "")
                    elif tool_type == "custom":
                        custom_data = tool_call.get("custom", {})
                        name = custom_data.get("name", "")
                        arguments = custom_data.get("input", "")
                    else:
                        name = ""
                        arguments = ""

                    messages_list.append(
                        {
                            "id": tool_call.get("id", ""),
                            "call_id": tool_call.get(
                                "id", ""
                            ),  # OpenAI uses 'id', not 'call_id'
                            "name": name,
                            "type": tool_type,
                            "arguments": json.loads(arguments),
                        }
                    )

        elif role == "tool":
            messages_list.append(
                {
                    "call_id": message.get("tool_call_id", ""),
                    "type": role,  # "tool"
                    "output": message.get("content", {}),
                }
            )
        else:
            messages_list.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return messages_list


def render_response_input(input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    messages_list = []

    for item in input:
        type = item.get("type")
        role = item.get("role")

        if type == "message":
            messages_list.append(
                {
                    "role": role,
                    "content": item.get("content"),
                }
            )
        else:
            messages_list.append(item)

    return messages_list


def _render_content(content: Dict[str, Any], indent: int = 0) -> str:
    """
    Renders a dictionary as a formatted string with indentation for nested structures.
    """
    if not content:
        return ""

    lines = []
    prefix = "  " * indent

    for key, value in content.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_render_content(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}: {compact_dump(value)}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)
