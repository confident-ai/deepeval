"""Google ADK, whose own telemetry confident-trace switches on.

ADK records a model call's Gemini request and response as JSON on its
``call_llm`` span, in ``gcp.vertex.agent.llm_request`` / ``llm_response``, and
a tool's arguments and response in ``tool_call_args`` / ``tool_response``.
Newer versions also record each model call as a GenAI ``generate_content``
span inside ``call_llm``, with content only when that's opted into; older
ones name their agent spans ``agent_run [name]``. ADK reports itself as the
provider.
"""

import re
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.test_case import ToolCall
from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.attribute_values import (
    attr_string,
    structured_value,
    tool_call_from,
)
from deepeval.tracing.otel.attributes import ConfidentAttr, GenAIAttr
from deepeval.tracing.otel.messages import answer_content, flatten_messages
from deepeval.tracing.types import SpanType

SCOPE = "gcp.vertex.agent"
_PREFIX = f"{SCOPE}."
_INTEGRATION = "Google ADK"
_AGENT_RUN = re.compile(r"^agent_run \[.+\]$")
# What ADK records when a tool has nothing to report.
_NO_VALUE = (None, {}, "<not specified>")


class GoogleADKAdapter(Adapter):
    integration = _INTEGRATION
    self_providers = frozenset((SCOPE,))

    def detect(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope
        if scope is not None and scope.name == SCOPE:
            return True
        attributes = span.attributes or {}
        return attributes.get(
            ConfidentAttr.SPAN_INTEGRATION
        ) == _INTEGRATION or any(key.startswith(_PREFIX) for key in attributes)

    def start_span_type(self, span: ReadableSpan) -> Optional[str]:
        return self.span_type(span)

    def span_type(self, span: ReadableSpan) -> Optional[str]:
        if span.name == "call_llm":
            return "llm"
        if _AGENT_RUN.match(span.name):
            return "agent"
        attributes = span.attributes or {}
        if attributes.get(
            GenAIAttr.OPERATION_NAME
        ) == "generate_content" and not _has_content(span):
            # A second record of its ``call_llm``.
            return "base"
        return None

    def span_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        attributes = span.attributes or {}
        if span_type == SpanType.LLM:
            request = structured_value(attributes.get(f"{_PREFIX}llm_request"))
            response = structured_value(
                attributes.get(f"{_PREFIX}llm_response")
            )
            content = (
                response.get("content") if isinstance(response, dict) else None
            )
            parts = content.get("parts") if isinstance(content, dict) else None
            output = (
                flatten_messages(
                    [{"role": "assistant", "parts": _genai_parts(parts)}]
                )
                if isinstance(parts, list)
                else None
            )
            return {
                "input": _request_messages(request),
                "output": answer_content(output),
                "tools_called": _function_calls(parts),
            }
        if span_type == SpanType.TOOL:
            result = _tool_value(attributes.get(f"{_PREFIX}tool_response"))
            if isinstance(result, dict) and list(result) == ["result"]:
                result = result["result"]
            return {
                "input": _tool_value(
                    attributes.get(f"{_PREFIX}tool_call_args")
                ),
                "output": result,
            }
        return {}

    def conversation_id(self, span: ReadableSpan) -> Optional[str]:
        return attr_string(span.attributes or {}, f"{_PREFIX}session_id")


def _has_content(span: ReadableSpan) -> bool:
    attributes = span.attributes or {}
    return any(
        key in attributes
        for key in (GenAIAttr.INPUT_MESSAGES, GenAIAttr.OUTPUT_MESSAGES)
    ) or any(event.name.startswith("gen_ai.") for event in span.events)


def _request_messages(request: Any) -> Optional[List[Dict[str, Any]]]:
    """A Gemini request's system instruction and contents as messages."""
    if not isinstance(request, dict):
        return None
    messages = []
    config = request.get("config")
    instruction = (
        config.get("system_instruction") if isinstance(config, dict) else None
    )
    if isinstance(instruction, str):
        instruction = {"parts": [{"text": instruction}]}
    if isinstance(instruction, dict):
        messages.append(
            {"role": "system", "parts": _genai_parts(instruction.get("parts"))}
        )
    contents = request.get("contents")
    for content in contents if isinstance(contents, list) else []:
        if isinstance(content, dict):
            role = content.get("role")
            messages.append(
                {
                    "role": "assistant" if role == "model" else role or "user",
                    "parts": _genai_parts(content.get("parts")),
                }
            )
    return flatten_messages(messages) if messages else None


def _genai_parts(parts: Any) -> List[Dict[str, Any]]:
    """Gemini parts as GenAI message parts."""
    converted = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        call = _part(part, "function_call", "functionCall")
        reply = _part(part, "function_response", "functionResponse")
        if isinstance(part.get("text"), str):
            converted.append({"type": "text", "content": part["text"]})
        elif isinstance(call, dict):
            converted.append(
                {
                    "type": "tool_call",
                    "name": call.get("name"),
                    "arguments": call.get("args"),
                }
            )
        elif isinstance(reply, dict):
            converted.append(
                {
                    "type": "tool_call_response",
                    "response": reply.get("response"),
                }
            )
        else:
            converted.append(part)
    return converted


def _function_calls(parts: Any) -> Optional[List[ToolCall]]:
    calls = []
    for part in parts if isinstance(parts, list) else []:
        call = (
            _part(part, "function_call", "functionCall")
            if isinstance(part, dict)
            else None
        )
        if isinstance(call, dict):
            tool_call = tool_call_from(
                {"name": call.get("name"), "input_parameters": call.get("args")}
            )
            if tool_call:
                calls.append(tool_call)
    return calls or None


def _part(part: Dict[str, Any], *keys: str) -> Any:
    """A part's value under its snake_case or camelCase key."""
    return next((part[key] for key in keys if key in part), None)


def _tool_value(raw: Any) -> Any:
    value = structured_value(raw)
    return None if value in _NO_VALUE else value
