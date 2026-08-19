"""Renderers for the OpenAI Responses API.

The Chat-Completions-shaped helpers (`stringify_multimodal_content`,
`render_messages`) live in `deepeval.model_integrations.utils` because every
OpenAI-compatible provider shares that wire format.
"""

from typing import Any, Dict, List

from deepeval.model_integrations.utils import compact_dump


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
