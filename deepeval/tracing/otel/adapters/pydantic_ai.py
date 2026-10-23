"""Pydantic AI, whose own telemetry confident-trace switches on.

Model and tool spans follow the GenAI conventions. The agent run keeps its
conversation in ``pydantic_ai.all_messages`` and its answer in
``final_result``; older versions name tool I/O ``tool_arguments`` and
``tool_response``.
"""

from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.attribute_values import (
    attr_decoded,
    attr_string,
    structured_value,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.messages import (
    flatten_messages,
    latest_assistant_text,
    latest_user_text,
)
from deepeval.tracing.types import SpanType

SCOPE = "pydantic-ai"
_INTEGRATION = "PydanticAI"


class PydanticAIAdapter(Adapter):
    def detect(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope
        if scope is not None and scope.name == SCOPE:
            return True
        attributes = span.attributes or {}
        return attributes.get(ConfidentAttr.SPAN_INTEGRATION) == _INTEGRATION

    def span_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        attributes = span.attributes or {}
        if span_type == SpanType.AGENT:
            messages = flatten_messages(
                structured_value(attributes.get("pydantic_ai.all_messages"))
            )
            return {
                "input": latest_user_text(messages) if messages else None,
                "output": attr_string(attributes, "final_result")
                or latest_assistant_text(messages),
            }
        if span_type == SpanType.TOOL:
            return {
                "input": attr_decoded(attributes, "tool_arguments"),
                "output": attr_decoded(attributes, "tool_response"),
            }
        return {}
