"""Decode OTel attribute values the way the platform's OTLP receiver does."""

import json
import math
from typing import Any, Dict, List, Optional

from opentelemetry.util.types import Attributes
from pydantic import ValidationError

from deepeval.test_case import ToolCall
from deepeval.test_case.llm_test_case import ToolCallType

_TOOL_CALL_TYPES = frozenset(member.value for member in ToolCallType)

# Distinguishes an absent attribute from one that holds JSON ``null``.
MISSING = object()


def first_defined(*values: Any) -> Any:
    """First value that is not ``None``; an empty list still counts."""
    return next((value for value in values if value is not None), None)


def structured_value(value: Any) -> Any:
    """Parse a JSON string attribute, or ``None`` if it isn't valid JSON."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def attr_json_content(attributes: Attributes, key: str) -> Any:
    """Decode one JSON layer, so a JSON document sent as text stays text.

    confident-trace JSON-encodes every content field, plain strings included.
    """
    value = attributes.get(key, MISSING)
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


# What confident-trace records in place of content it won't serialize.
_UNSUPPORTED = "[unsupported]"


def unsupported(value: Any) -> bool:
    """Whether confident-trace serialized none of ``value``."""
    if isinstance(value, dict):
        return bool(value) and all(map(unsupported, value.values()))
    if isinstance(value, list):
        return bool(value) and all(map(unsupported, value))
    return value == _UNSUPPORTED


def attr_decoded(attributes: Attributes, key: str) -> Any:
    """An attribute decoded once, or ``None``; plain text stays text."""
    value = attr_json_content(attributes, key)
    return None if value is MISSING else value


def attr_string(attributes: Attributes, key: str) -> Optional[str]:
    value = attributes.get(key)
    return value if isinstance(value, str) and value else None


def attr_json_string(attributes: Attributes, key: str) -> Optional[str]:
    value = attr_json_content(attributes, key)
    return value if isinstance(value, str) else None


def attr_number(attributes: Attributes, key: str) -> Optional[float]:
    value = attributes.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip():
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def attr_count(attributes: Attributes, key: str) -> Optional[int]:
    value = attr_number(attributes, key)
    return None if value is None else int(value)


def attr_string_list(attributes: Attributes, key: str) -> Optional[List[str]]:
    value = attributes.get(key)
    if not isinstance(value, (list, tuple)):
        return None
    return [item for item in value if isinstance(item, str)]


def attr_json_string_list(
    attributes: Attributes, key: str
) -> Optional[List[str]]:
    value = attr_json_content(attributes, key)
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def attr_coerced_string_list(
    attributes: Attributes, key: str
) -> Optional[List[str]]:
    value = attributes.get(key)
    if isinstance(value, str) and value:
        parsed = structured_value(value)
        value = parsed if isinstance(parsed, list) else [value]
    if not isinstance(value, (list, tuple)):
        return None
    return [
        item if isinstance(item, str) else json.dumps(item)
        for item in value
        if item is not None
    ]


def attr_json_object(
    attributes: Attributes, key: str
) -> Optional[Dict[str, Any]]:
    value = structured_value(attributes.get(key))
    return value if isinstance(value, dict) else None


def attr_tool_calls(
    attributes: Attributes, key: str
) -> Optional[List[ToolCall]]:
    """An empty list is kept; a list with no valid calls is dropped."""
    items = structured_value(attributes.get(key))
    if not isinstance(items, (list, tuple)):
        return None
    if not items:
        return []
    calls = [tool_call_from(structured_value(item)) for item in items]
    return [call for call in calls if call] or None


def tool_call_from(candidate: Any) -> Optional[ToolCall]:
    if not isinstance(candidate, dict):
        return None
    if candidate.get("type") not in _TOOL_CALL_TYPES:
        # An unknown type is dropped, not the call.
        candidate = {k: v for k, v in candidate.items() if k != "type"}
    try:
        return ToolCall.model_validate(candidate)
    except ValidationError:
        return None
