import json
import uuid
from typing import Any, Dict, List, Optional

from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.tracing.context import current_span_context
from deepeval.utils import shorten, len_long
from deepeval.openai.types import OutputParameters


_URL_MAX = 200
_JSON_MAX = max(
    len_long(), 400
)  # <- make this bigger by increasing DEEPEVAL_MAXLEN_LONG above 400


def _compact_dump(value: Any) -> str:
    try:
        dumped = json.dumps(
            value, ensure_ascii=False, default=str, separators=(",", ":")
        )
    except Exception:
        dumped = repr(value)
    return shorten(dumped, max_len=_JSON_MAX)


def _fmt_url(url: Optional[str]) -> str:
    if not url:
        return ""
    if url.startswith("data:"):
        return "[data-uri]"
    return shorten(url, max_len=_URL_MAX)


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
            return f"[image:{_fmt_url(url)}]"

        # Responses API variants
        if t == "input_text":
            return str(content.get("text", ""))
        if t == "input_image":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{_fmt_url(url)}]"

        # readability for other input_* types we don't currently handle
        if t and t.startswith("input_"):
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return _compact_dump(content)

def render_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    messages_list = []

    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls")

            for tool_call in tool_calls:
                messages_list.append({
                    "role": "Assistant (tool call)",
                    "content": str(tool_call),
                })
        
        else:
            messages_list.append(
                {
                    "role": role,
                    "content": str(message.get("content")),
                }
            )
    
    return messages_list