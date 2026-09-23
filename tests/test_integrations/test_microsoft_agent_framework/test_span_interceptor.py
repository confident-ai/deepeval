from __future__ import annotations

import json
from itertools import count
from unittest.mock import MagicMock

from deepeval.integrations.microsoft_agent_framework.instrumentator import (
    MicrosoftAgentFrameworkInstrumentationSettings,
    MicrosoftAgentFrameworkSpanInterceptor,
)

_span_ids = count(start=1)
_trace_ids = count(start=1)


def _make_span(
    operation_name: str,
    *,
    name: str,
    attributes: dict | None = None,
):
    span = MagicMock()
    backing = {
        "gen_ai.operation.name": operation_name,
        **(attributes or {}),
    }
    span._attributes = backing
    span.attributes = backing
    span.name = name
    span.events = []
    span.start_time = None
    span.parent = None
    span.set_attribute.side_effect = lambda key, value: backing.__setitem__(
        key, value
    )
    span.get_span_context.return_value = MagicMock(
        trace_id=next(_trace_ids),
        span_id=next(_span_ids),
    )
    return span


def _make_interceptor():
    settings = MicrosoftAgentFrameworkInstrumentationSettings(
        environment="testing"
    )
    return MicrosoftAgentFrameworkSpanInterceptor(settings)


def _process(interceptor, span):
    interceptor.on_start(span, None)
    interceptor.on_end(span)
    return span.attributes


def test_maps_agent_messages_and_conversation_id():
    interceptor = _make_interceptor()
    span = _make_span(
        "invoke_agent",
        name="invoke_agent support_agent",
        attributes={
            "gen_ai.agent.name": "support_agent",
            "gen_ai.conversation.id": "conversation-123",
            "gen_ai.input.messages": json.dumps(
                [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "content": "Where is my order?",
                            }
                        ],
                    }
                ]
            ),
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "text",
                                "content": "It ships tomorrow.",
                            }
                        ],
                    }
                ]
            ),
        },
    )

    attrs = _process(interceptor, span)

    assert attrs["confident.span.type"] == "agent"
    assert attrs["confident.span.integration"] == "Microsoft Agent Framework"
    assert attrs["confident.span.name"] == "support_agent"
    assert attrs["confident.trace.thread_id"] == "conversation-123"
    assert attrs["confident.span.input"] == "Where is my order?"
    assert attrs["confident.span.output"] == "It ships tomorrow."
    assert attrs["confident.trace.input"] == "Where is my order?"
    assert attrs["confident.trace.output"] == "It ships tomorrow."


def test_maps_llm_provider_model_and_usage():
    interceptor = _make_interceptor()
    span = _make_span(
        "chat",
        name="chat gpt-4o-mini",
        attributes={
            "gen_ai.provider.name": "openai",
            "gen_ai.response.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 17,
            "gen_ai.usage.output_tokens": 9,
        },
    )

    attrs = _process(interceptor, span)

    assert attrs["confident.span.type"] == "llm"
    assert attrs["confident.span.provider"] == "OpenAI"
    assert attrs["confident.llm.model"] == "gpt-4o-mini"
    assert attrs["confident.llm.input_token_count"] == 17
    assert attrs["confident.llm.output_token_count"] == 9


def test_maps_tool_arguments_result_and_tool_call():
    interceptor = _make_interceptor()
    span = _make_span(
        "execute_tool",
        name="execute_tool lookup_order",
        attributes={
            "gen_ai.tool.name": "lookup_order",
            "gen_ai.tool.call.arguments": json.dumps({"order_id": "A-1001"}),
            "gen_ai.tool.call.result": json.dumps({"status": "shipped"}),
        },
    )

    attrs = _process(interceptor, span)

    assert attrs["confident.span.type"] == "tool"
    assert json.loads(attrs["confident.span.input"]) == {"order_id": "A-1001"}
    assert json.loads(attrs["confident.span.output"]) == {"status": "shipped"}
    tool_call = json.loads(attrs["confident.span.tools_called"][0])
    assert tool_call["name"] == "lookup_order"
    assert tool_call["input_parameters"] == {"order_id": "A-1001"}
    assert tool_call["output"] == {"status": "shipped"}


def test_explicit_thread_id_wins_over_framework_conversation_id():
    settings = MicrosoftAgentFrameworkInstrumentationSettings(
        thread_id="configured-thread",
        environment="testing",
    )
    interceptor = MicrosoftAgentFrameworkSpanInterceptor(settings)
    span = _make_span(
        "invoke_agent",
        name="invoke_agent test_agent",
        attributes={"gen_ai.conversation.id": "framework-thread"},
    )

    attrs = _process(interceptor, span)

    assert attrs["confident.trace.thread_id"] == "configured-thread"
