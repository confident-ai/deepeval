import json
import uuid

from typing import Any, List, Optional

from deepeval.openai.types import OutputParameters
from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.tracing.context import current_span_context
from deepeval.utils import shorten, len_long


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


def stringify_anthropic_content(content: Any) -> str:
    """
    Return a short, human-readable summary string for an Anthropic-style multimodal `content` value.

    This is used to populate span summaries, such as `InputParameters.input`. It never raises and
    never returns huge blobs.

    Notes:
    - Data URIs and base64 content are redacted to "[data-uri]" or "[base64:...]".
    - Output is capped via `deepeval.utils.shorten` (configurable through settings).
    - Fields that are not explicitly handled are returned as size-capped JSON dumps
    - This string is for display/summary only, not intended to be parsable.

    Args:
        content: The value of an Anthropic message `content`, may be a str or list of content blocks,
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

    # list of content blocks for Anthropic Messages API
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            s = stringify_anthropic_content(block)
            if s:
                parts.append(s)
        return "\n".join(parts)

    # dict shapes for Anthropic Messages API
    if isinstance(content, dict):
        t = content.get("type")

        # Text block
        if t == "text":
            return str(content.get("text", ""))

        # Image block
        if t == "image":
            source = content.get("source", {})
            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "unknown")
                data = source.get("data", "")
                data_preview = data[:20] if data else ""
                return f"[image:{media_type}:base64:{data_preview}...]"
            elif source_type == "url":
                url = source.get("url", "")
                return f"[image:{_fmt_url(url)}]"
            else:
                return f"[image:{source_type or 'unknown'}]"

        # Tool use block (in assistant messages)
        if t == "tool_use":
            tool_name = content.get("name", "unknown")
            tool_id = content.get("id", "")
            tool_input = content.get("input", {})
            input_str = _compact_dump(tool_input) if tool_input else ""
            return f"[tool_use:{tool_name}:{tool_id}:{input_str}]"

        # Tool result block (in user messages)
        if t == "tool_result":
            tool_id = content.get("tool_use_id", "")
            tool_content = content.get("content")
            content_str = stringify_anthropic_content(tool_content) if tool_content else ""
            is_error = content.get("is_error", False)
            error_flag = ":error" if is_error else ""
            return f"[tool_result:{tool_id}{error_flag}:{content_str}]"

        # Document block (for PDFs and other documents)
        if t == "document":
            source = content.get("source", {})
            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "unknown")
                return f"[document:{media_type}:base64]"
            elif source_type == "url":
                url = source.get("url", "")
                return f"[document:{_fmt_url(url)}]"
            else:
                return f"[document:{source_type or 'unknown'}]"

        # Thinking block (for extended thinking models)
        if t == "thinking":
            thinking_text = content.get("thinking", "")
            return f"[thinking:{shorten(thinking_text, max_len=100)}]"

        # readability for other block types we don't currently handle
        if t:
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return _compact_dump(content)
