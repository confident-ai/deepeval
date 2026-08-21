"""OTel GenAI semconv / Traceloop span attribute extraction.

Shared by the integrations whose frameworks emit standard ``gen_ai.*``
attributes and events (AWS Bedrock AgentCore, Strands Agents). Traceloop /
OpenLLMetry fallbacks are kept alongside them — inert when absent, and
useful when one of these agents runs next to a Traceloop-instrumented
framework.

Every function here inspects the raw OTel span only; nothing depends on
instrumentation settings or deepeval context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from deepeval.tracing.types import SpanType, ToolCall

logger = logging.getLogger(__name__)


# Span classification: ``gen_ai.*`` (OTel GenAI semconv), Traceloop attrs,
# and span-name heuristics. Settings-independent; inspects raw OTel span only.

_AGENT_OP_NAMES = {"invoke_agent", "create_agent"}
_LLM_OP_NAMES = {
    "chat",
    "generate_content",
    "invoke_model",
    "text_completion",
    "embeddings",
}
_TOOL_OP_NAMES = {"execute_tool"}

_TRACELOOP_KIND_MAP = {
    "workflow": "agent",
    "agent": "agent",
    "task": "tool",
    "tool": "tool",
    "retriever": "retriever",
    "llm": "llm",
}


def get_attr(span, *keys: str) -> Optional[str]:
    attrs = span.attributes or {}
    for k in keys:
        v = attrs.get(k)
        if v:
            return str(v)
    return None


def classify_span(span) -> Optional[str]:
    attrs = span.attributes or {}
    span_name_lower = (span.name or "").lower()

    op_name = attrs.get("gen_ai.operation.name", "")
    if op_name in _AGENT_OP_NAMES:
        return SpanType.AGENT
    if op_name in _LLM_OP_NAMES:
        return SpanType.LLM
    if op_name in _TOOL_OP_NAMES:
        return SpanType.TOOL

    traceloop_kind = attrs.get("traceloop.span.kind", "")
    if traceloop_kind in _TRACELOOP_KIND_MAP:
        return _TRACELOOP_KIND_MAP[traceloop_kind]

    if attrs.get("gen_ai.tool.name") or attrs.get("gen_ai.tool.call.id"):
        return SpanType.TOOL
    if attrs.get("gen_ai.agent.name") or attrs.get("gen_ai.agent.id"):
        return SpanType.AGENT

    if any(kw in span_name_lower for kw in ("invoke_agent", "agent")):
        return SpanType.AGENT
    if any(kw in span_name_lower for kw in ("execute_tool", ".tool")):
        return SpanType.TOOL
    if any(kw in span_name_lower for kw in ("retriev", "memory", "datastore")):
        return SpanType.RETRIEVER
    if any(
        kw in span_name_lower
        for kw in ("llm", "chat", "invoke_model", "generate")
    ):
        return SpanType.LLM

    return None


def get_agent_name(span) -> Optional[str]:
    return (
        get_attr(
            span,
            "gen_ai.agent.name",
            "traceloop.entity.name",
            "traceloop.workflow.name",
        )
        or span.name
        or None
    )


def get_tool_name(span) -> Optional[str]:
    return (
        get_attr(span, "gen_ai.tool.name", "traceloop.entity.name")
        or span.name
        or None
    )


# Content / I/O extraction. Walks ``gen_ai.*`` events and Traceloop attrs to
# pull framework-written input/output text and tool calls.


def parse_genai_content(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return str(raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("text") or first.get("content") or str(first)
            return str(first)
        if isinstance(data, dict):
            return data.get("text") or data.get("content") or str(data)
        return str(data)
    except (json.JSONDecodeError, TypeError):
        return raw


def extract_messages(span) -> tuple[Optional[str], Optional[str]]:
    input_text: Optional[str] = None
    output_text: Optional[str] = None

    # Events (Strands / strict OTel GenAI)
    for event in getattr(span, "events", []):
        event_name = event.name or ""
        event_attrs = event.attributes or {}

        if event_name == "gen_ai.user.message":
            input_text = parse_genai_content(event_attrs.get("content"))
        elif event_name in ("gen_ai.choice", "gen_ai.assistant.message"):
            output_text = parse_genai_content(
                event_attrs.get("message") or event_attrs.get("content")
            )
        elif event_name == "gen_ai.system.message":
            if not input_text:
                input_text = parse_genai_content(event_attrs.get("content"))
        elif event_name in (
            "gen_ai.client.inference.operation.details",
            "agent.invocation",
            "tool.invocation",
        ):
            body_raw = event_attrs.get("body") or event_attrs.get("event.body")
            if body_raw:
                try:
                    body = (
                        json.loads(body_raw)
                        if isinstance(body_raw, str)
                        else body_raw
                    )
                    if not input_text and "input" in body:
                        msgs = body["input"].get("messages", [])
                        if msgs:
                            input_text = parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                    if not output_text and "output" in body:
                        msgs = body["output"].get("messages", [])
                        if msgs:
                            output_text = parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                except Exception:
                    pass

    # Fallback: attributes (LangChain / CrewAI / Traceloop)
    if not input_text:
        raw = get_attr(
            span,
            "gen_ai.user.message",
            "gen_ai.input.messages",
            "gen_ai.prompt",
            "traceloop.entity.input",
            "crewai.task.description",
        )
        if raw:
            input_text = parse_genai_content(raw)

    if not output_text:
        raw = get_attr(
            span,
            "gen_ai.choice",
            "gen_ai.output.messages",
            "gen_ai.completion",
            "traceloop.entity.output",
        )
        if raw:
            output_text = parse_genai_content(raw)

    return input_text, output_text


def extract_tool_calls(span) -> List[ToolCall]:
    tools: List[ToolCall] = []

    # Events (Strands / strict OTel)
    for event in getattr(span, "events", []):
        event_attrs = event.attributes or {}
        event_name = event.name or ""

        if event_name in ("gen_ai.tool.call", "tool_call", "execute_tool"):
            try:
                name = (
                    event_attrs.get("gen_ai.tool.name")
                    or event_attrs.get("name")
                    or "unknown_tool"
                )
                args_raw = (
                    event_attrs.get("gen_ai.tool.call.arguments")
                    or event_attrs.get("gen_ai.tool.arguments")
                    or event_attrs.get("input")
                    or "{}"
                )
                input_params = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else args_raw
                )
                tools.append(
                    ToolCall(name=str(name), input_parameters=input_params)
                )
            except Exception as exc:
                logger.debug("Failed to parse tool call event: %s", exc)

    # Fallback: attributes (LangChain / CrewAI / Traceloop)
    attrs = span.attributes or {}

    tool_calls_raw = (
        attrs.get("gen_ai.tool.calls")
        or attrs.get("traceloop.tool_calls")
        or attrs.get("llm.tool_calls")
    )

    if tool_calls_raw:
        try:
            calls = (
                json.loads(tool_calls_raw)
                if isinstance(tool_calls_raw, str)
                else tool_calls_raw
            )
            if isinstance(calls, list):
                for call in calls:
                    # Traceloop / OpenLLMetry nest these under "function".
                    name = (
                        call.get("name")
                        or call.get("function", {}).get("name")
                        or "unknown_tool"
                    )
                    args = (
                        call.get("arguments")
                        or call.get("function", {}).get("arguments")
                        or "{}"
                    )

                    input_params = (
                        json.loads(args) if isinstance(args, str) else args
                    )
                    tools.append(
                        ToolCall(name=str(name), input_parameters=input_params)
                    )
        except Exception as exc:
            logger.debug("Failed to parse tool call attributes: %s", exc)

    return tools


def extract_tool_call_from_tool_span(span) -> Optional[ToolCall]:
    tool_name = get_tool_name(span)
    if not tool_name:
        return None

    attrs = span.attributes or {}
    args_raw = (
        attrs.get("gen_ai.tool.call.arguments")
        or attrs.get("traceloop.entity.input")
        or "{}"
    )
    try:
        input_params = (
            json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        )
    except Exception:
        input_params = {}

    return ToolCall(name=tool_name, input_parameters=input_params)
