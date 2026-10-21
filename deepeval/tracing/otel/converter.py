"""Convert finished confident-trace OTel spans into deepeval span models.

The mapping mirrors the Confident platform's OTLP receiver, so a trace deepeval
assembles in-process matches the one the platform builds from the same spans.
For every field, ``confident.*`` attributes win over GenAI semantic conventions,
which win over legacy GenAI span events. One deliberate difference: a model's
output is read as its text, since that is what evaluation metrics score.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import Attributes

from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.otel.adapters import Adapter, detect_adapters
from deepeval.tracing.otel.attribute_values import (
    MISSING,
    attr_coerced_string_list,
    attr_count,
    attr_decoded,
    attr_json_content,
    attr_json_object,
    attr_json_string,
    attr_json_string_list,
    attr_number,
    attr_string,
    attr_string_list,
    attr_tool_calls,
    first_defined,
    structured_value,
    unsupported,
)
from deepeval.tracing.otel.attributes import ConfidentAttr, GenAIAttr
from deepeval.tracing.otel.messages import (
    latest_user_text,
    llm_io,
    message_tool_calls,
    tool_arguments,
    tool_response,
    trace_io,
)
from deepeval.tracing.otel.utils import (
    GEN_AI_AGENT_OPERATION_NAMES,
    GEN_AI_OPERATION_NAMES,
    to_hex_string,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    SpanType,
    ToolSpan,
    TraceSpanStatus,
)
from deepeval.tracing.utils import (
    Environment,
    infer_provider_from_model,
    make_json_serializable_for_metadata,
    normalize_span_provider_for_platform,
)

logger = logging.getLogger(__name__)

peb.init_clock_bridge()

# ``None`` is a base span. ``custom`` is deliberately absent: like any unknown
# value it falls through to GenAI classification.
_DECLARED_SPAN_TYPES: Dict[str, Optional[SpanType]] = {
    "agent": SpanType.AGENT,
    "base": None,
    "llm": SpanType.LLM,
    "retriever": SpanType.RETRIEVER,
    "tool": SpanType.TOOL,
}
_RETRIEVAL_OPERATION = "retrieval"
_TOOL_OPERATION = "execute_tool"
# Keys under which retrieved documents carry their text.
_DOCUMENT_TEXT_KEYS = ("page_content", "text", "content")

_CONFIDENT_TRACE_SCOPE = "confident_trace"
_AI_ATTRIBUTE_PREFIXES = ("confident.", "gen_ai.")
_AI_EVENT_PREFIX = "gen_ai."

_ENVIRONMENTS = frozenset(member.value for member in Environment)


@dataclass(frozen=True)
class ConvertedSpan:
    span: BaseSpan
    # Keyed by ``Trace`` field name, holding only the fields this span sets.
    trace_fields: Dict[str, Any]
    # The conversation the span belongs to; the trace's thread when it has none.
    conversation_id: Optional[str] = None


def convert_span(
    span: ReadableSpan, span_type: Optional[SpanType] = None
) -> ConvertedSpan:
    """Convert ``span``, as ``span_type`` when given instead of its own."""
    adapters = detect_adapters(span)
    span_type = span_type or resolve_span_type(span, adapters)
    converted = _BUILDERS.get(span_type, _base_span)(span)
    trace_fields = _trace_fields(span)
    if isinstance(converted, LlmSpan) and converted.provider in {
        provider for adapter in adapters for provider in adapter.self_providers
    }:
        converted.provider = None
    for adapter in adapters:
        try:
            _fill_unset(
                converted,
                {
                    "integration": adapter.integration,
                    **adapter.span_fields(span, span_type),
                },
            )
            for key, value in adapter.trace_fields(span, span_type).items():
                if value is not None:
                    trace_fields.setdefault(key, value)
        except Exception:
            logger.debug("%s failed to map a span", type(adapter).__name__)
    if isinstance(converted, LlmSpan) and not converted.provider:
        converted.provider = infer_provider_from_model(converted.model)
    if (
        isinstance(converted, RetrieverSpan)
        and converted.retrieval_context is None
    ):
        converted.retrieval_context = _document_texts(converted.output)
    if isinstance(converted, ToolSpan):
        converted.input = tool_arguments(converted.input)
        converted.output = tool_response(converted.output)
    return ConvertedSpan(
        span=converted,
        trace_fields=trace_fields,
        conversation_id=_conversation_id(span, adapters),
    )


def is_ai_span(span: ReadableSpan) -> bool:
    """confident-trace's export rule, plus any span a framework adapter reads.

    Other spans only travel as ancestors.
    """
    scope = span.instrumentation_scope
    if scope is not None and scope.name == _CONFIDENT_TRACE_SCOPE:
        return True
    if any(
        key.startswith(_AI_ATTRIBUTE_PREFIXES) for key in span.attributes or ()
    ):
        return True
    if any(event.name.startswith(_AI_EVENT_PREFIX) for event in span.events):
        return True
    return bool(detect_adapters(span))


def resolve_span_type(
    span: ReadableSpan,
    adapters: Optional[Sequence[Adapter]] = None,
    at_start: bool = False,
) -> Optional[SpanType]:
    """Classify a span, returning ``None`` for a base span.

    A declared ``confident.span.type`` wins, then a framework adapter, which
    knows its framework's spans, then the GenAI conventions. ``at_start``
    classifies a span that is still opening: frameworks may only set its
    attributes later, so its GenAI-convention name counts too.
    """
    attributes = span.attributes or {}
    declared = attributes.get(ConfidentAttr.SPAN_TYPE)
    if declared in _DECLARED_SPAN_TYPES:
        return _DECLARED_SPAN_TYPES[declared]
    for adapter in detect_adapters(span) if adapters is None else adapters:
        try:
            wire_type = (
                at_start and adapter.start_span_type(span)
            ) or adapter.span_type(span)
        except Exception:
            logger.debug("%s failed to classify a span", type(adapter).__name__)
            continue
        if wire_type is not None:
            return _DECLARED_SPAN_TYPES.get(wire_type)
    wire_type = _declared_type(attributes)
    if wire_type is None and at_start and declared is None:
        wire_type = _OPERATION_TYPES.get(span.name.split(" ", 1)[0])
    return _DECLARED_SPAN_TYPES.get(wire_type)


def _declared_type(attributes: Attributes) -> Optional[str]:
    """The type that ``confident.*`` or GenAI attributes settle, by wire name.

    As on the platform, an unrecognised operation settles a base span.
    """
    declared = attributes.get(ConfidentAttr.SPAN_TYPE)
    if declared in _DECLARED_SPAN_TYPES:
        return declared
    operation = attr_string(attributes, GenAIAttr.OPERATION_NAME)
    if operation is not None:
        return _OPERATION_TYPES.get(operation, "base")
    if attr_string(attributes, GenAIAttr.TOOL_NAME):
        return "tool"
    return None


def _trace_fields(span: ReadableSpan) -> Dict[str, Any]:
    attributes = span.attributes or {}
    metadata = attr_json_object(attributes, ConfidentAttr.TRACE_METADATA)
    fields = {
        "name": attr_string(attributes, ConfidentAttr.TRACE_NAME),
        "tags": attr_string_list(attributes, ConfidentAttr.TRACE_TAGS),
        "metadata": (
            make_json_serializable_for_metadata(metadata) if metadata else None
        ),
        "thread_id": attr_string(attributes, ConfidentAttr.TRACE_THREAD_ID),
        "user_id": attr_string(attributes, ConfidentAttr.TRACE_USER_ID),
        "environment": _environment(span),
        "retrieval_context": attr_json_string_list(
            attributes, ConfidentAttr.TRACE_RETRIEVAL_CONTEXT
        ),
        "context": attr_json_string_list(
            attributes, ConfidentAttr.TRACE_CONTEXT
        ),
        "tools_called": attr_tool_calls(
            attributes, ConfidentAttr.TRACE_TOOLS_CALLED
        ),
        "expected_tools": attr_tool_calls(
            attributes, ConfidentAttr.TRACE_EXPECTED_TOOLS
        ),
        "expected_output": attr_json_string(
            attributes, ConfidentAttr.TRACE_EXPECTED_OUTPUT
        ),
        "test_case_id": attr_string(
            attributes, ConfidentAttr.TRACE_TEST_CASE_ID
        ),
        "test_run_id": attr_string(attributes, ConfidentAttr.TRACE_TEST_RUN_ID),
        "turn_id": attr_string(attributes, ConfidentAttr.TRACE_TURN_ID),
        "metric_collection": attr_string(
            attributes, ConfidentAttr.TRACE_METRIC_COLLECTION
        ),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    # An explicit null counts: it clears input/output that another span set.
    for key, attribute in (
        ("input", ConfidentAttr.TRACE_INPUT),
        ("output", ConfidentAttr.TRACE_OUTPUT),
    ):
        value = _known_content(attributes, attribute)
        if value is not MISSING:
            fields[key] = trace_io(key, value)
    return fields


################ Span builders ################


def _llm_span(span: ReadableSpan) -> LlmSpan:
    attributes = span.attributes or {}
    model = (
        attr_string(attributes, ConfidentAttr.LLM_MODEL)
        or attr_string(attributes, GenAIAttr.REQUEST_MODEL)
        or attr_string(attributes, GenAIAttr.RESPONSE_MODEL)
    )
    provider = next(
        (
            normalized
            for normalized in map(
                normalize_span_provider_for_platform,
                (
                    attributes.get(ConfidentAttr.SPAN_PROVIDER),
                    attributes.get(GenAIAttr.PROVIDER_NAME),
                    attributes.get(GenAIAttr.LEGACY_SYSTEM),
                ),
            )
            if normalized
        ),
        None,
    ) or infer_provider_from_model(model)
    llm_input, llm_output = llm_io(span)
    fields = _with_output_tool_calls(attributes, _common_fields(span))
    return LlmSpan(
        **_with_declared_io(attributes, fields, llm_input, llm_output),
        model=model,
        provider=provider,
        input_token_count=first_defined(
            attr_count(attributes, ConfidentAttr.LLM_INPUT_TOKEN_COUNT),
            attr_count(attributes, GenAIAttr.USAGE_INPUT_TOKENS),
            attr_count(attributes, GenAIAttr.LEGACY_USAGE_PROMPT_TOKENS),
        ),
        output_token_count=first_defined(
            attr_count(attributes, ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT),
            attr_count(attributes, GenAIAttr.USAGE_OUTPUT_TOKENS),
            attr_count(attributes, GenAIAttr.LEGACY_USAGE_COMPLETION_TOKENS),
        ),
        cost_per_input_token=attr_number(
            attributes, ConfidentAttr.LLM_COST_PER_INPUT_TOKEN
        ),
        cost_per_output_token=attr_number(
            attributes, ConfidentAttr.LLM_COST_PER_OUTPUT_TOKEN
        ),
    )


def _agent_span(span: ReadableSpan) -> AgentSpan:
    attributes = span.attributes or {}
    agent_input, agent_output = llm_io(span)
    # An agent serves the latest request in its conversation.
    agent_input = latest_user_text(agent_input)
    fields = _with_output_tool_calls(
        attributes,
        _common_fields(
            span,
            typed_name=attr_string(attributes, ConfidentAttr.AGENT_NAME)
            or attr_string(attributes, GenAIAttr.AGENT_NAME),
        ),
    )
    return AgentSpan(
        **_with_declared_io(attributes, fields, agent_input, agent_output),
        available_tools=attr_coerced_string_list(
            attributes, ConfidentAttr.AGENT_AVAILABLE_TOOLS
        )
        or [],
        agent_handoffs=attr_coerced_string_list(
            attributes, ConfidentAttr.AGENT_AGENT_HANDOFFS
        )
        or [],
    )


def _retriever_span(span: ReadableSpan) -> RetrieverSpan:
    attributes = span.attributes or {}
    fields = _common_fields(span)
    return RetrieverSpan(
        **_with_declared_io(
            attributes,
            fields,
            attributes.get(GenAIAttr.RETRIEVAL_QUERY_TEXT),
            structured_value(attributes.get(GenAIAttr.RETRIEVAL_DOCUMENTS)),
        ),
        embedder=attr_string(attributes, ConfidentAttr.RETRIEVER_EMBEDDER)
        or "",
        top_k=first_defined(
            attr_count(attributes, ConfidentAttr.RETRIEVER_TOP_K),
            attr_count(attributes, GenAIAttr.RETRIEVAL_TOP_K),
        ),
        chunk_size=attr_count(attributes, ConfidentAttr.RETRIEVER_CHUNK_SIZE),
    )


def _tool_span(span: ReadableSpan) -> ToolSpan:
    attributes = span.attributes or {}
    fields = _common_fields(
        span,
        typed_name=attr_string(attributes, ConfidentAttr.TOOL_NAME)
        or attr_string(attributes, GenAIAttr.TOOL_NAME),
    )
    return ToolSpan(
        **_with_declared_io(
            attributes,
            fields,
            attr_decoded(attributes, GenAIAttr.TOOL_CALL_ARGUMENTS),
            attr_decoded(attributes, GenAIAttr.TOOL_CALL_RESULT),
        ),
        description=attr_string(attributes, ConfidentAttr.TOOL_DESCRIPTION)
        or attr_string(attributes, GenAIAttr.TOOL_DESCRIPTION),
    )


def _base_span(span: ReadableSpan) -> BaseSpan:
    attributes = span.attributes or {}
    return BaseSpan(
        **_with_declared_io(attributes, _common_fields(span), None, None)
    )


_OPERATION_TYPES: Dict[str, str] = {
    **dict.fromkeys(GEN_AI_OPERATION_NAMES, "llm"),
    **dict.fromkeys(GEN_AI_AGENT_OPERATION_NAMES, "agent"),
    _RETRIEVAL_OPERATION: "retriever",
    _TOOL_OPERATION: "tool",
}

_BUILDERS: Dict[Optional[SpanType], Callable[[ReadableSpan], BaseSpan]] = {
    SpanType.LLM: _llm_span,
    SpanType.AGENT: _agent_span,
    SpanType.RETRIEVER: _retriever_span,
    SpanType.TOOL: _tool_span,
}


def _common_fields(
    span: ReadableSpan, typed_name: Optional[str] = None
) -> Dict[str, Any]:
    attributes = span.attributes or {}
    errored = span.status.status_code == StatusCode.ERROR
    metadata = attr_json_object(attributes, ConfidentAttr.SPAN_METADATA)
    return {
        "uuid": to_hex_string(span.context.span_id, 16),
        "trace_uuid": to_hex_string(span.context.trace_id, 32),
        "parent_uuid": (
            to_hex_string(span.parent.span_id, 16) if span.parent else None
        ),
        "start_time": peb.epoch_nanos_to_perf_seconds(span.start_time),
        "end_time": peb.epoch_nanos_to_perf_seconds(
            span.end_time or span.start_time
        ),
        "name": attr_string(attributes, ConfidentAttr.SPAN_NAME)
        or typed_name
        or span.name,
        "status": (
            TraceSpanStatus.ERRORED if errored else TraceSpanStatus.SUCCESS
        ),
        "error": _error(span) if errored else None,
        "metadata": (
            make_json_serializable_for_metadata(metadata) if metadata else None
        ),
        "integration": attr_string(attributes, ConfidentAttr.SPAN_INTEGRATION),
        "metric_collection": attr_string(
            attributes, ConfidentAttr.SPAN_METRIC_COLLECTION
        ),
        "retrieval_context": attr_json_string_list(
            attributes, ConfidentAttr.SPAN_RETRIEVAL_CONTEXT
        ),
        "context": attr_json_string_list(
            attributes, ConfidentAttr.SPAN_CONTEXT
        ),
        "expected_output": attr_json_string(
            attributes, ConfidentAttr.SPAN_EXPECTED_OUTPUT
        ),
        "tools_called": attr_tool_calls(
            attributes, ConfidentAttr.SPAN_TOOLS_CALLED
        ),
        "expected_tools": attr_tool_calls(
            attributes, ConfidentAttr.SPAN_EXPECTED_TOOLS
        ),
    }


def _with_output_tool_calls(
    attributes: Attributes, fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Model and agent spans fall back to the tool calls in their output."""
    output_messages = structured_value(
        attributes.get(GenAIAttr.OUTPUT_MESSAGES)
    )
    return {
        **fields,
        "tools_called": first_defined(
            fields["tools_called"], message_tool_calls(output_messages)
        ),
    }


def _with_declared_io(
    attributes: Attributes,
    fields: Dict[str, Any],
    fallback_input: Any,
    fallback_output: Any,
) -> Dict[str, Any]:
    """Declared ``confident.span.*`` I/O beats the type-specific I/O."""
    declared_input = _known_content(attributes, ConfidentAttr.SPAN_INPUT)
    declared_output = _known_content(attributes, ConfidentAttr.SPAN_OUTPUT)
    return {
        **fields,
        "input": (
            fallback_input if declared_input is MISSING else declared_input
        ),
        "output": (
            fallback_output if declared_output is MISSING else declared_output
        ),
    }


def _known_content(attributes: Attributes, key: str) -> Any:
    """A content attribute, ``MISSING`` when confident-trace couldn't
    serialize any of it."""
    value = attr_json_content(attributes, key)
    return MISSING if unsupported(value) else value


def _fill_unset(converted: BaseSpan, fields: Dict[str, Any]) -> None:
    for key, value in fields.items():
        if value is None or not hasattr(converted, key):
            continue
        if getattr(converted, key) in (None, "", []):
            setattr(converted, key, value)


def _document_texts(documents: Any) -> Optional[List[str]]:
    """Retrieved documents' texts, which retrieval metrics read as context."""
    if not isinstance(documents, list) or not documents:
        return None
    texts = []
    for document in documents:
        if isinstance(document, dict):
            document = next(
                (
                    document[key]
                    for key in _DOCUMENT_TEXT_KEYS
                    if isinstance(document.get(key), str)
                ),
                None,
            )
        if not isinstance(document, str):
            return None
        texts.append(document)
    return texts


def _conversation_id(
    span: ReadableSpan, adapters: Sequence[Adapter]
) -> Optional[str]:
    conversation = attr_string(span.attributes or {}, GenAIAttr.CONVERSATION_ID)
    for adapter in adapters:
        if conversation:
            break
        try:
            conversation = adapter.conversation_id(span)
        except Exception:
            logger.debug("%s failed to read a span", type(adapter).__name__)
    return conversation


def _error(span: ReadableSpan) -> Optional[str]:
    return (
        span.status.description
        or attr_string(span.attributes or {}, "error.type")
        or next(
            (
                message
                for message in (
                    attr_string(event.attributes or {}, "exception.message")
                    for event in span.events
                )
                if message
            ),
            None,
        )
    )


def _environment(span: ReadableSpan) -> Optional[str]:
    resource = span.resource.attributes if span.resource else {}
    candidate = attr_string(
        resource, ConfidentAttr.TRACE_ENVIRONMENT
    ) or attr_string(span.attributes or {}, ConfidentAttr.TRACE_ENVIRONMENT)
    return candidate if candidate in _ENVIRONMENTS else None
