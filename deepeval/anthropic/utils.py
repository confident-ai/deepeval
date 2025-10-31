from typing import Any, Iterable, List

from anthropic.types import Message

from deepeval.model_integrations.utils import compact_dump, fmt_url
from deepeval.utils import shorten


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
                return f"[image:{fmt_url(url)}]"
            else:
                return f"[image:{source_type or 'unknown'}]"

        # Tool use block (in assistant messages)
        if t == "tool_use":
            tool_name = content.get("name", "unknown")
            tool_id = content.get("id", "")
            tool_input = content.get("input", {})
            input_str = compact_dump(tool_input) if tool_input else ""
            return f"[tool_use:{tool_name}:{tool_id}:{input_str}]"

        # Tool result block (in user messages)
        if t == "tool_result":
            tool_id = content.get("tool_use_id", "")
            tool_content = content.get("content")
            content_str = (
                stringify_anthropic_content(tool_content)
                if tool_content
                else ""
            )
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
                return f"[document:{fmt_url(url)}]"
            else:
                return f"[document:{source_type or 'unknown'}]"

        # Thinking block (for extended thinking models)
        if t == "thinking":
            thinking_text = content.get("thinking", "")
            return {
                "role": "thinking",
                "content": shorten(thinking_text, max_len=100),
            }

        # readability for other block types we don't currently handle
        if t:
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return compact_dump(content)


def render_messages_anthropic(
    messages: Iterable[Message],
):
    """
    Extracts and normalizes tool calls and tool results from Anthropic API messages
    for observability/logging purposes.

    Args:
        messages: Iterable of message dictionaries in Anthropic API format

    Returns:
        List of normalized message objects suitable for logging/observability
    """
    messages_list = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "assistant":
            if isinstance(content, str):
                messages_list.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            elif isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        messages_list.append(
                            {
                                "role": role,
                                "content": block.get("text", ""),
                            }
                        )
                    elif block_type == "tool_use":
                        messages_list.append(
                            {
                                "id": block.get("id", ""),
                                "call_id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "type": "function",
                                "arguments": block.get("input", {}),
                            }
                        )

        elif role == "user":
            if isinstance(content, str):
                messages_list.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            elif isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        messages_list.append(
                            {
                                "role": role,
                                "content": block.get("text", ""),
                            }
                        )
                    elif block_type == "image":
                        messages_list.append(
                            {
                                "role": role,
                                "content": "[Image content]",
                                "image_source": block.get("source", {}),
                            }
                        )
                    elif block_type == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            output_parts = []
                            for tool_content_block in tool_content:
                                if isinstance(tool_content_block, dict):
                                    if tool_content_block.get("type") == "text":
                                        output_parts.append(
                                            tool_content_block.get("text", "")
                                        )
                                else:
                                    output_parts.append(str(tool_content_block))
                            output = "\n".join(output_parts)
                        else:
                            output = tool_content

                        messages_list.append(
                            {
                                "call_id": block.get("tool_use_id", ""),
                                "type": "tool",
                                "output": output,
                                "is_error": block.get("is_error", False),
                            }
                        )

        elif role == "system":
            messages_list.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return messages_list
