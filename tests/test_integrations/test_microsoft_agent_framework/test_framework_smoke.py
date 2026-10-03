from __future__ import annotations

from unittest.mock import MagicMock

import pytest

agent_framework = pytest.importorskip("agent_framework")

from agent_framework import Agent, BaseChatClient, ChatResponse, Message
from agent_framework.observability import enable_instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from deepeval.integrations.microsoft_agent_framework.instrumentator import (
    MicrosoftAgentFrameworkInstrumentationSettings,
    MicrosoftAgentFrameworkSpanInterceptor,
)
from deepeval.integrations.microsoft_agent_framework import otel


class FakeChatClient(BaseChatClient):
    OTEL_PROVIDER_NAME = "fake"

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        async def response():
            return ChatResponse(
                messages=Message("assistant", ["Hello from fake"]),
                model="fake-model",
            )

        return response()


@pytest.mark.asyncio
async def test_real_agent_run_emits_deepeval_agent_span():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    settings = MicrosoftAgentFrameworkInstrumentationSettings(
        environment="testing"
    )
    provider.add_span_processor(
        MicrosoftAgentFrameworkSpanInterceptor(settings)
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    previous_provider = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider
    try:
        enable_instrumentation(enable_sensitive_data=True, force=True)
        response = await Agent(
            FakeChatClient(),
            name="test_agent",
        ).run("Hello")
    finally:
        trace._TRACER_PROVIDER = previous_provider

    assert str(response) == "Hello from fake"
    spans = exporter.get_finished_spans()
    agent_span = next(
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    )
    assert agent_span.attributes["confident.span.type"] == "agent"
    assert (
        agent_span.attributes["confident.span.integration"]
        == "Microsoft Agent Framework"
    )
    assert agent_span.attributes["confident.span.name"] == "test_agent"
    assert agent_span.attributes["confident.span.input"] == "Hello"
    assert agent_span.attributes["confident.span.output"] == "Hello from fake"


def test_instrumentation_is_idempotent(monkeypatch):
    provider = TracerProvider()
    enable = MagicMock()
    processor = MagicMock()
    monkeypatch.setattr(otel.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        "agent_framework.observability.enable_instrumentation", enable
    )
    monkeypatch.setattr(
        "deepeval.tracing.otel.context_aware_processor.ContextAwareSpanProcessor",
        MagicMock(return_value=processor),
    )
    monkeypatch.setattr(otel, "_attached_processors", {})

    otel.instrument_microsoft_agent_framework(
        api_key="test-key",
        capture_content=True,
    )
    otel.instrument_microsoft_agent_framework(
        api_key="test-key",
        capture_content=False,
    )

    processors = provider._active_span_processor._span_processors
    assert len(processors) == 2
    assert isinstance(processors[0], MicrosoftAgentFrameworkSpanInterceptor)
    assert processors[1] is processor
    assert enable.call_count == 2
    enable.assert_called_with(enable_sensitive_data=False)
