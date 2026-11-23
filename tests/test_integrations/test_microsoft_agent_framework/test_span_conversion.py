import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from deepeval.tracing.otel.converter import convert_span
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan


@pytest.fixture
def convert_framework_span():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agent_framework")

    def convert(name, operation, attributes):
        span = tracer.start_span(
            name,
            attributes={"gen_ai.operation.name": operation, **attributes},
        )
        span.end()
        return convert_span(exporter.get_finished_spans()[-1])

    yield convert
    provider.shutdown()


def test_maps_agent_messages_and_conversation_id(convert_framework_span):
    result = convert_framework_span(
        "invoke_agent support_agent",
        "invoke_agent",
        {
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

    assert isinstance(result.span, AgentSpan)
    assert result.span.name == "support_agent"
    assert result.conversation_id == "conversation-123"
    assert result.span.input == "Where is my order?"
    assert result.span.output == "It ships tomorrow."


def test_maps_llm_provider_model_and_usage(convert_framework_span):
    result = convert_framework_span(
        "chat gpt-4o-mini",
        "chat",
        {
            "gen_ai.provider.name": "openai",
            "gen_ai.response.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 17,
            "gen_ai.usage.output_tokens": 9,
        },
    )

    assert isinstance(result.span, LlmSpan)
    assert result.span.provider == "OpenAI"
    assert result.span.model == "gpt-4o-mini"
    assert result.span.input_token_count == 17
    assert result.span.output_token_count == 9


def test_maps_tool_arguments_and_result(convert_framework_span):
    result = convert_framework_span(
        "execute_tool lookup_order",
        "execute_tool",
        {
            "gen_ai.tool.name": "lookup_order",
            "gen_ai.tool.call.arguments": json.dumps({"order_id": "A-1001"}),
            "gen_ai.tool.call.result": json.dumps({"status": "shipped"}),
        },
    )

    assert isinstance(result.span, ToolSpan)
    assert result.span.name == "lookup_order"
    assert result.span.input == {"order_id": "A-1001"}
    assert result.span.output == {"status": "shipped"}
