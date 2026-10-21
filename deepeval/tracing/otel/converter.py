"""Convert finished confident-trace OTel spans into deepeval span models.

The mapping mirrors the Confident platform's OTLP receiver, so a trace deepeval
assembles in-process matches the one the platform builds from the same spans.
For every field, ``confident.*`` attributes win over GenAI semantic conventions,
which win over legacy GenAI span events. One deliberate difference: a model's
output is read as its text, since that is what evaluation metrics score.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import Attributes
from pydantic import ValidationError

from deepeval.test_case import ToolCall
from deepeval.test_case.llm_test_case import ToolCallType
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.otel.attributes import ConfidentAttr, GenAIAttr
from deepeval.tracing.otel.utils import (
    GEN_AI_AGENT_OPERATION_NAMES,
    GEN_AI_OPERATION_NAMES,
    _flatten_input,
    _flatten_system_instructions,
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

_LEGACY_CHOICE_EVENT = "gen_ai.choice"
_LEGACY_MESSAGE_EVENT = re.compile(
    r"^gen_ai\.(system|user|assistant|tool)\.message$"
)

_CONFIDENT_TRACE_SCOPE = "confident_trace"
_AI_ATTRIBUTE_PREFIXES = ("confident.", "gen_ai.")
_AI_EVENT_PREFIX = "gen_ai."

_TOOL_CALL_TYPES = frozenset(member.value for member in ToolCallType)
_ENVIRONMENTS = frozenset(member.value for member in Environment)

# Distinguishes an absent attribute from one that holds JSON ``null``.
_MISSING = object()


@dataclass(frozen=True)
class ConvertedSpan:
    span: BaseSpan
    # Keyed by ``Trace`` field name, holding only the fields this span sets.
    trace_fields: Dict[str, Any]


def convert_span(span: ReadableSpan) -> ConvertedSpan:
    attributes = span.attributes or {}
    build = _BUILDERS.get(resolve_span_type(attributes), _base_span)
    return ConvertedSpan(span=build(span), trace_fields=_trace_fields(span))


def is_ai_span(span: ReadableSpan) -> bool:
    """confident-trace's export rule: other spans only travel as ancestors."""
    scope = span.instrumentation_scope
    if scope is not None and scope.name == _CONFIDENT_TRACE_SCOPE:
        return True
    if any(
        key.startswith(_AI_ATTRIBUTE_PREFIXES) for key in span.attributes or ()
    ):
        return True
    return any(event.name.startswith(_AI_EVENT_PREFIX) for event in span.events)


def resolve_span_type(attributes: Attributes) -> Optional[SpanType]:
    """Classify a span, returning ``None`` for a base span."""
    declared = attributes.get(ConfidentAttr.SPAN_TYPE)
    if declared in _DECLARED_SPAN_TYPES:
        return _DECLARED_SPAN_TYPES[declared]
    operation = _string(attributes, GenAIAttr.OPERATION_NAME)
    if operation in GEN_AI_OPERATION_NAMES:
        return SpanType.LLM
    if operation in GEN_AI_AGENT_OPERATION_NAMES:
        return SpanType.AGENT
    if operation == _RETRIEVAL_OPERATION:
        return SpanType.RETRIEVER
    if operation == _TOOL_OPERATION:
        return SpanType.TOOL
    if operation is None and _string(attributes, GenAIAttr.TOOL_NAME):
        return SpanType.TOOL
    return None


def _trace_fields(span: ReadableSpan) -> Dict[str, Any]:
    attributes = span.attributes or {}
    metadata = _json_object(attributes, ConfidentAttr.TRACE_METADATA)
    fields = {
        "name": _string(attributes, ConfidentAttr.TRACE_NAME),
        "tags": _string_list(attributes, ConfidentAttr.TRACE_TAGS),
        "metadata": (
            make_json_serializable_for_metadata(metadata) if metadata else None
        ),
        "thread_id": _string(attributes, ConfidentAttr.TRACE_THREAD_ID),
        "user_id": _string(attributes, ConfidentAttr.TRACE_USER_ID),
        "environment": _environment(span),
        "retrieval_context": _json_string_list(
            attributes, ConfidentAttr.TRACE_RETRIEVAL_CONTEXT
        ),
        "context": _json_string_list(attributes, ConfidentAttr.TRACE_CONTEXT),
        "tools_called": _tool_calls(
            attributes, ConfidentAttr.TRACE_TOOLS_CALLED
        ),
        "expected_tools": _tool_calls(
            attributes, ConfidentAttr.TRACE_EXPECTED_TOOLS
        ),
        "expected_output": _json_string(
            attributes, ConfidentAttr.TRACE_EXPECTED_OUTPUT
        ),
        "test_case_id": _string(attributes, ConfidentAttr.TRACE_TEST_CASE_ID),
        "test_run_id": _string(attributes, ConfidentAttr.TRACE_TEST_RUN_ID),
        "turn_id": _string(attributes, ConfidentAttr.TRACE_TURN_ID),
        "metric_collection": _string(
            attributes, ConfidentAttr.TRACE_METRIC_COLLECTION
        ),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    # An explicit null counts: it clears input/output that another span set.
    for key, attribute in (
        ("input", ConfidentAttr.TRACE_INPUT),
        ("output", ConfidentAttr.TRACE_OUTPUT),
    ):
        value = _json_content(attributes, attribute)
        if value is not _MISSING:
            fields[key] = _trace_io(key, value)
    return fields


################ Span builders ################


def _llm_span(span: ReadableSpan) -> LlmSpan:
    attributes = span.attributes or {}
    model = (
        _string(attributes, ConfidentAttr.LLM_MODEL)
        or _string(attributes, GenAIAttr.REQUEST_MODEL)
        or _string(attributes, GenAIAttr.RESPONSE_MODEL)
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
    llm_input, llm_output = _llm_io(span)
    fields = _with_output_tool_calls(attributes, _common_fields(span))
    return LlmSpan(
        **_with_declared_io(attributes, fields, llm_input, llm_output),
        model=model,
        provider=provider,
        input_token_count=_first(
            _count(attributes, ConfidentAttr.LLM_INPUT_TOKEN_COUNT),
            _count(attributes, GenAIAttr.USAGE_INPUT_TOKENS),
            _count(attributes, GenAIAttr.LEGACY_USAGE_PROMPT_TOKENS),
        ),
        output_token_count=_first(
            _count(attributes, ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT),
            _count(attributes, GenAIAttr.USAGE_OUTPUT_TOKENS),
            _count(attributes, GenAIAttr.LEGACY_USAGE_COMPLETION_TOKENS),
        ),
        cost_per_input_token=_number(
            attributes, ConfidentAttr.LLM_COST_PER_INPUT_TOKEN
        ),
        cost_per_output_token=_number(
            attributes, ConfidentAttr.LLM_COST_PER_OUTPUT_TOKEN
        ),
    )


def _agent_span(span: ReadableSpan) -> AgentSpan:
    attributes = span.attributes or {}
    agent_input, agent_output = _llm_io(span)
    fields = _with_output_tool_calls(
        attributes,
        _common_fields(
            span,
            typed_name=_string(attributes, ConfidentAttr.AGENT_NAME)
            or _string(attributes, GenAIAttr.AGENT_NAME),
        ),
    )
    return AgentSpan(
        **_with_declared_io(attributes, fields, agent_input, agent_output),
        available_tools=_coerced_string_list(
            attributes, ConfidentAttr.AGENT_AVAILABLE_TOOLS
        )
        or [],
        agent_handoffs=_coerced_string_list(
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
            _structured(attributes.get(GenAIAttr.RETRIEVAL_DOCUMENTS)),
        ),
        embedder=_string(attributes, ConfidentAttr.RETRIEVER_EMBEDDER) or "",
        top_k=_first(
            _count(attributes, ConfidentAttr.RETRIEVER_TOP_K),
            _count(attributes, GenAIAttr.RETRIEVAL_TOP_K),
        ),
        chunk_size=_count(attributes, ConfidentAttr.RETRIEVER_CHUNK_SIZE),
    )


def _tool_span(span: ReadableSpan) -> ToolSpan:
    attributes = span.attributes or {}
    fields = _common_fields(
        span,
        typed_name=_string(attributes, ConfidentAttr.TOOL_NAME)
        or _string(attributes, GenAIAttr.TOOL_NAME),
    )
    return ToolSpan(
        **_with_declared_io(
            attributes,
            fields,
            _structured(attributes.get(GenAIAttr.TOOL_CALL_ARGUMENTS)),
            _structured(attributes.get(GenAIAttr.TOOL_CALL_RESULT)),
        ),
        description=_string(attributes, ConfidentAttr.TOOL_DESCRIPTION)
        or _string(attributes, GenAIAttr.TOOL_DESCRIPTION),
    )


def _base_span(span: ReadableSpan) -> BaseSpan:
    attributes = span.attributes or {}
    return BaseSpan(
        **_with_declared_io(attributes, _common_fields(span), None, None)
    )


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
    metadata = _json_object(attributes, ConfidentAttr.SPAN_METADATA)
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
        "name": _string(attributes, ConfidentAttr.SPAN_NAME)
        or typed_name
        or span.name,
        "status": (
            TraceSpanStatus.ERRORED if errored else TraceSpanStatus.SUCCESS
        ),
        "error": _error(span) if errored else None,
        "metadata": (
            make_json_serializable_for_metadata(metadata) if metadata else None
        ),
        "integration": _string(attributes, ConfidentAttr.SPAN_INTEGRATION),
        "metric_collection": _string(
            attributes, ConfidentAttr.SPAN_METRIC_COLLECTION
        ),
        "retrieval_context": _json_string_list(
            attributes, ConfidentAttr.SPAN_RETRIEVAL_CONTEXT
        ),
        "context": _json_string_list(attributes, ConfidentAttr.SPAN_CONTEXT),
        "expected_output": _json_string(
            attributes, ConfidentAttr.SPAN_EXPECTED_OUTPUT
        ),
        "tools_called": _tool_calls(
            attributes, ConfidentAttr.SPAN_TOOLS_CALLED
        ),
        "expected_tools": _tool_calls(
            attributes, ConfidentAttr.SPAN_EXPECTED_TOOLS
        ),
    }


def _with_output_tool_calls(
    attributes: Attributes, fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Model and agent spans fall back to the tool calls in their output."""
    output_messages = _structured(attributes.get(GenAIAttr.OUTPUT_MESSAGES))
    return {
        **fields,
        "tools_called": _first(
            fields["tools_called"], _message_tool_calls(output_messages)
        ),
    }


def _with_declared_io(
    attributes: Attributes,
    fields: Dict[str, Any],
    fallback_input: Any,
    fallback_output: Any,
) -> Dict[str, Any]:
    """Declared ``confident.span.*`` I/O beats the type-specific I/O."""
    declared_input = _json_content(attributes, ConfidentAttr.SPAN_INPUT)
    declared_output = _json_content(attributes, ConfidentAttr.SPAN_OUTPUT)
    return {
        **fields,
        "input": (
            fallback_input if declared_input is _MISSING else declared_input
        ),
        "output": (
            fallback_output if declared_output is _MISSING else declared_output
        ),
    }


def _error(span: ReadableSpan) -> Optional[str]:
    return (
        span.status.description
        or _string(span.attributes or {}, "error.type")
        or next(
            (
                message
                for message in (
                    _string(event.attributes or {}, "exception.message")
                    for event in span.events
                )
                if message
            ),
            None,
        )
    )


def _environment(span: ReadableSpan) -> Optional[str]:
    resource = span.resource.attributes if span.resource else {}
    candidate = _string(resource, ConfidentAttr.TRACE_ENVIRONMENT) or _string(
        span.attributes or {}, ConfidentAttr.TRACE_ENVIRONMENT
    )
    return candidate if candidate in _ENVIRONMENTS else None


################ Model and agent messages ################


def _llm_io(span: ReadableSpan) -> Tuple[Any, Any]:
    """Message attributes win, even when empty; legacy events fill gaps."""
    attributes = span.attributes or {}
    instructions = _system_instructions(attributes)
    input_messages = _structured(attributes.get(GenAIAttr.INPUT_MESSAGES))
    llm_input = None
    if isinstance(input_messages, list) or instructions:
        llm_input = instructions + (
            _flatten_input(input_messages)
            if isinstance(input_messages, list)
            else []
        )
    output_messages = _structured(attributes.get(GenAIAttr.OUTPUT_MESSAGES))
    llm_output = (
        _flatten_input(output_messages)
        if isinstance(output_messages, list)
        else None
    )
    if llm_input is None or llm_output is None:
        legacy_input, legacy_output = _legacy_event_io(span.events)
        llm_input = legacy_input if llm_input is None else llm_input
        llm_output = legacy_output if llm_output is None else llm_output
    return llm_input, _answer(llm_output)


def _answer(messages: Any) -> Any:
    """Flattened output messages as their text; output without text stays."""
    if not isinstance(messages, list):
        return messages
    texts = [
        message["content"]
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    ]
    return "\n".join(texts) if texts else messages


def _trace_io(key: str, value: Any) -> Any:
    """Read a model's messages, which confident-trace copies to its trace.

    Like deepeval's own integrations, the trace takes the first user message
    as input and the answer's text as output.
    """
    if not _is_message_array(value):
        return value
    messages = _flatten_input(value)
    if key == "output":
        return _answer(messages)
    return next(
        (
            message["content"]
            for message in messages
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        messages,
    )


def _is_message_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("parts"), list)
            for message in value
        )
    )


def _system_instructions(attributes: Attributes) -> List[Dict[str, Any]]:
    raw = attributes.get(GenAIAttr.SYSTEM_INSTRUCTIONS)
    if raw is None:
        return []
    parsed = _structured(raw)
    return _flatten_system_instructions(raw if parsed is None else parsed)


def _legacy_event_io(
    events: Sequence[Event],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    for event in events:
        attributes = dict(event.attributes or {})
        body = _structured(attributes.get("body"))
        if not isinstance(body, dict):
            body = attributes
        if event.name == _LEGACY_CHOICE_EVENT:
            message = body.get("message")
            if isinstance(message, str):
                parsed = _structured(message)
                message = message if parsed is None else parsed
            if message is not None and not isinstance(message, dict):
                # Plain text or a content-block list.
                outputs.append({"role": "assistant", "content": message})
                continue
            value = message if isinstance(message, dict) else body
            role = value.get("role")
            outputs.append(
                {
                    "role": role if isinstance(role, str) else "assistant",
                    "content": _first(value.get("content"), value),
                }
            )
            continue
        match = _LEGACY_MESSAGE_EVENT.match(event.name)
        if match:
            inputs.append(
                {
                    "role": match.group(1),
                    "content": _first(body.get("content"), body),
                }
            )
    return inputs or None, outputs or None


def _message_tool_calls(messages: Any) -> Optional[List[ToolCall]]:
    if not isinstance(messages, list):
        return None
    calls = []
    for message in messages:
        parts = message.get("parts") if isinstance(message, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and isinstance(part.get("name"), str)
            ):
                call = _tool_call(
                    {
                        "name": part["name"],
                        "input_parameters": part.get("arguments"),
                    }
                )
                if call:
                    calls.append(call)
    return calls or None


################ Attribute decoding ################


def _first(*values: Any) -> Any:
    """First value that is not ``None``; an empty list still counts."""
    return next((value for value in values if value is not None), None)


def _structured(value: Any) -> Any:
    """Parse a JSON string attribute, or ``None`` if it isn't valid JSON."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def _json_content(attributes: Attributes, key: str) -> Any:
    """Decode one JSON layer, so a JSON document sent as text stays text.

    confident-trace JSON-encodes every content field, plain strings included.
    """
    value = attributes.get(key, _MISSING)
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _string(attributes: Attributes, key: str) -> Optional[str]:
    value = attributes.get(key)
    return value if isinstance(value, str) and value else None


def _json_string(attributes: Attributes, key: str) -> Optional[str]:
    value = _json_content(attributes, key)
    return value if isinstance(value, str) else None


def _number(attributes: Attributes, key: str) -> Optional[float]:
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


def _count(attributes: Attributes, key: str) -> Optional[int]:
    value = _number(attributes, key)
    return None if value is None else int(value)


def _string_list(attributes: Attributes, key: str) -> Optional[List[str]]:
    value = attributes.get(key)
    if not isinstance(value, (list, tuple)):
        return None
    return [item for item in value if isinstance(item, str)]


def _json_string_list(attributes: Attributes, key: str) -> Optional[List[str]]:
    value = _json_content(attributes, key)
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _coerced_string_list(
    attributes: Attributes, key: str
) -> Optional[List[str]]:
    value = attributes.get(key)
    if isinstance(value, str) and value:
        parsed = _structured(value)
        value = parsed if isinstance(parsed, list) else [value]
    if not isinstance(value, (list, tuple)):
        return None
    return [
        item if isinstance(item, str) else json.dumps(item)
        for item in value
        if item is not None
    ]


def _json_object(attributes: Attributes, key: str) -> Optional[Dict[str, Any]]:
    value = _structured(attributes.get(key))
    return value if isinstance(value, dict) else None


def _tool_calls(attributes: Attributes, key: str) -> Optional[List[ToolCall]]:
    """An empty list is kept; a list with no valid calls is dropped."""
    items = _structured(attributes.get(key))
    if not isinstance(items, (list, tuple)):
        return None
    if not items:
        return []
    calls = [_tool_call(_structured(item)) for item in items]
    return [call for call in calls if call] or None


def _tool_call(candidate: Any) -> Optional[ToolCall]:
    if not isinstance(candidate, dict):
        return None
    if candidate.get("type") not in _TOOL_CALL_TYPES:
        # An unknown type is dropped, not the call.
        candidate = {k: v for k, v in candidate.items() if k != "type"}
    try:
        return ToolCall.model_validate(candidate)
    except ValidationError:
        return None
