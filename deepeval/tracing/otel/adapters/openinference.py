"""OpenInference, the conventions of the ``openinference-instrumentation-*``
packages, mapped as on the platform.

A span's kind is ``openinference.span.kind``; content lives in flattened
attributes such as ``llm.input_messages.0.message.content``, with
``input.value`` / ``output.value`` as the fallback.
"""

from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.util.types import Attributes

from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.attribute_values import (
    attr_json_object,
    attr_number,
    attr_string,
    attr_string_list,
    first_defined,
    structured_value,
)
from deepeval.tracing.otel.messages import (
    answer_content,
    flatten_messages,
    message_tool_calls,
)
from deepeval.tracing.types import SpanType
from deepeval.tracing.utils import normalize_span_provider_for_platform

SCOPE = "openinference.instrumentation"
SPAN_KIND = "openinference.span.kind"
_KIND_TYPES = {
    "LLM": "llm",
    "EMBEDDING": "llm",
    "AGENT": "agent",
    "TOOL": "tool",
    "RETRIEVER": "retriever",
    "RERANKER": "retriever",
}


class OpenInferenceAdapter(Adapter):
    integration = "OpenInference"

    def detect(self, span: ReadableSpan) -> bool:
        if SPAN_KIND in (span.attributes or {}):
            return True
        scope = span.instrumentation_scope
        return scope is not None and (
            scope.name == SCOPE or scope.name.startswith(f"{SCOPE}.")
        )

    def start_span_type(self, span: ReadableSpan) -> Optional[str]:
        kind = (span.attributes or {}).get(SPAN_KIND)
        return None if kind is None else _KIND_TYPES.get(kind, "base")

    def span_type(self, span: ReadableSpan) -> Optional[str]:
        span_type = self.start_span_type(span)
        if span_type == "tool" and "tool.name" not in (span.attributes or {}):
            # Nothing ran under a tool's name, as in a handoff between agents.
            return "base"
        return span_type

    def span_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        attributes = span.attributes or {}
        fields: Dict[str, Any] = {
            "input": _value(attributes, "input"),
            "output": _value(attributes, "output"),
            "metadata": attr_json_object(attributes, "metadata"),
        }
        if span_type == SpanType.LLM:
            fields.update(_llm_fields(attributes))
        elif span_type == SpanType.TOOL:
            arguments = attributes.get("tool_call.function.arguments")
            if arguments is not None:
                fields["input"] = first_defined(
                    structured_value(arguments), arguments
                )
            fields["description"] = attr_string(attributes, "tool.description")
        elif span_type == SpanType.RETRIEVER:
            documents = [
                document["document"]
                for document in _indexed(attributes, "retrieval.documents.")
                if isinstance(document.get("document"), dict)
            ]
            if documents:
                fields["output"] = documents
        return fields

    def trace_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        attributes = span.attributes or {}
        return {
            "thread_id": attr_string(attributes, "session.id"),
            "user_id": attr_string(attributes, "user.id"),
            "tags": attr_string_list(attributes, "tag.tags"),
        }


def _llm_fields(attributes: Attributes) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "model": attr_string(attributes, "llm.model_name"),
        "provider": normalize_span_provider_for_platform(
            attr_string(attributes, "llm.provider")
            or attr_string(attributes, "llm.system")
        ),
        "input_token_count": attr_number(attributes, "llm.token_count.prompt"),
        "output_token_count": attr_number(
            attributes, "llm.token_count.completion"
        ),
    }
    input_messages = _messages(attributes, "llm.input_messages.")
    if input_messages:
        fields["input"] = flatten_messages(input_messages)
    output_messages = _messages(attributes, "llm.output_messages.")
    if output_messages:
        fields["output"] = answer_content(flatten_messages(output_messages))
        fields["tools_called"] = message_tool_calls(output_messages)
    return fields


def _value(attributes: Attributes, prefix: str) -> Any:
    """``input.value`` / ``output.value``, decoded when its mime type is JSON."""
    value = attributes.get(f"{prefix}.value")
    if (
        isinstance(value, str)
        and attributes.get(f"{prefix}.mime_type") == "application/json"
    ):
        return first_defined(structured_value(value), value)
    return value


def _messages(
    attributes: Attributes, prefix: str
) -> Optional[List[Dict[str, Any]]]:
    """Flattened OpenInference messages as GenAI ``{role, parts}`` messages."""
    messages = []
    for flattened in _indexed(attributes, prefix):
        message = flattened.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        role = role if isinstance(role, str) else "user"
        if role == "tool":
            parts = [
                {
                    "type": "tool_call_response",
                    "response": message.get("content"),
                    "id": message.get("tool_call_id"),
                }
            ]
        else:
            parts = []
            contents = _listed(message.get("contents"))
            # ``contents`` is the richer form of ``content``, when both come.
            if not contents and isinstance(message.get("content"), str):
                parts.append({"type": "text", "content": message["content"]})
            for content in contents:
                part = (
                    content.get("message_content")
                    if isinstance(content, dict)
                    else None
                )
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "content": part["text"]})
                elif isinstance(part, dict):
                    parts.append(part)
            for call in _listed(message.get("tool_calls")):
                tool_call = (
                    call.get("tool_call") if isinstance(call, dict) else None
                )
                function = (
                    tool_call.get("function")
                    if isinstance(tool_call, dict)
                    else None
                )
                if isinstance(function, dict):
                    parts.append(
                        {
                            "type": "tool_call",
                            "id": tool_call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                        }
                    )
        messages.append({"role": role, "parts": parts})
    return messages or None


def _indexed(attributes: Attributes, prefix: str) -> List[Any]:
    """``{prefix}{i}.{path}`` attributes as one nested value per index.

    ``llm.input_messages.0.message.role`` becomes
    ``[{"message": {"role": ...}}]``; nested indices stay digit-keyed dicts
    that ``_listed`` reads.
    """
    root: Dict[str, Any] = {}
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        node = root
        *parents, leaf = key[len(prefix) :].split(".")
        for parent in parents:
            node = node.setdefault(parent, {})
            if not isinstance(node, dict):
                break
        else:
            node[leaf] = value
    return _listed(root)


def _listed(node: Any) -> List[Any]:
    """A digit-keyed dict's values in index order."""
    if not isinstance(node, dict):
        return []
    return [
        node[key] for key in sorted((k for k in node if k.isdigit()), key=int)
    ]
