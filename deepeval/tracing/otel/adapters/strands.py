"""Strands Agents, whose own telemetry confident-trace switches on.

Strands records content in GenAI events as Bedrock-style content blocks:
``[{"text": ...}]``, ``[{"toolUse": {...}}]``. It reports itself as the
provider, and carries the session and user it was given as attributes.
"""

from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.test_case import ToolCall
from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.attribute_values import (
    attr_coerced_string_list,
    attr_string,
    first_defined,
    structured_value,
    tool_call_from,
)
from deepeval.tracing.otel.attributes import ConfidentAttr, GenAIAttr
from deepeval.tracing.otel.messages import block_text
from deepeval.tracing.types import SpanType

SCOPE = "strands.telemetry.tracer"
_INTEGRATION = "Strands"
_SELF_PROVIDER = "strands-agents"


class StrandsAdapter(Adapter):
    integration = _INTEGRATION
    self_providers = frozenset((_SELF_PROVIDER,))

    def detect(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope
        if scope is not None and scope.name == SCOPE:
            return True
        attributes = span.attributes or {}
        return attributes.get(
            ConfidentAttr.SPAN_INTEGRATION
        ) == _INTEGRATION or (
            _SELF_PROVIDER
            in (
                attributes.get(GenAIAttr.PROVIDER_NAME),
                attributes.get(GenAIAttr.LEGACY_SYSTEM),
            )
        )

    def span_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        if span_type == SpanType.AGENT:
            return {
                "available_tools": attr_coerced_string_list(
                    span.attributes or {}, "gen_ai.agent.tools"
                )
            }
        if span_type == SpanType.LLM:
            return {
                "tools_called": _tool_uses(
                    _event_value(span, "gen_ai.choice", "message")
                )
            }
        if span_type == SpanType.TOOL:
            arguments = _event_value(span, "gen_ai.tool.message", "content")
            result = _event_value(span, "gen_ai.choice", "message")
            return {
                "input": first_defined(structured_value(arguments), arguments),
                "output": None if result is None else block_text(result),
            }
        return {}

    def trace_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        attributes = span.attributes or {}
        return {
            "thread_id": attr_string(attributes, "session.id"),
            "user_id": attr_string(attributes, "user.id"),
        }


def _event_value(span: ReadableSpan, name: str, key: str) -> Any:
    """``key`` on the first ``name`` event that has it."""
    for event in span.events:
        if event.name == name:
            value = (event.attributes or {}).get(key)
            if value is not None:
                return value
    return None


def _tool_uses(message: Any) -> Optional[List[ToolCall]]:
    """The calls in ``toolUse`` content blocks."""
    blocks = structured_value(message) if isinstance(message, str) else message
    if not isinstance(blocks, list):
        return None
    calls = []
    for block in blocks:
        use = block.get("toolUse") if isinstance(block, dict) else None
        if isinstance(use, dict):
            call = tool_call_from(
                {"name": use.get("name"), "input_parameters": use.get("input")}
            )
            if call:
                calls.append(call)
    return calls or None
