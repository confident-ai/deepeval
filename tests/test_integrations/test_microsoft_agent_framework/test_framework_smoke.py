from unittest.mock import Mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from deepeval.integrations.microsoft_agent_framework import (
    instrument_microsoft_agent_framework,
    otel,
)
from deepeval.tracing.otel.converter import convert_span
from deepeval.tracing.otel.setup import shutdown_confident_trace
from deepeval.tracing.types import AgentSpan

agent_framework = pytest.importorskip("agent_framework")
observability = pytest.importorskip("agent_framework.observability")


class FakeChatClient(agent_framework.BaseChatClient):
    OTEL_PROVIDER_NAME = "fake"

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        async def response():
            return agent_framework.ChatResponse(
                messages=agent_framework.Message(
                    "assistant", ["Hello from fake"]
                ),
                model="fake-model",
            )

        return response()


@pytest.fixture
def exporter(monkeypatch):
    shutdown_confident_trace()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    settings = observability.OBSERVABILITY_SETTINGS
    monkeypatch.setattr(settings, "enable_instrumentation", True)
    monkeypatch.setattr(settings, "enable_sensitive_data", False)
    try:
        yield exporter
    finally:
        shutdown_confident_trace()
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_content", [False, True])
async def test_real_agent_run_emits_deepeval_agent_span(
    exporter, capture_content
):
    if capture_content:
        instrument_microsoft_agent_framework(
            name="agent trace",
            thread_id="configured-thread",
            capture_content=True,
        )
    else:
        instrument_microsoft_agent_framework(
            name="agent trace",
            thread_id="configured-thread",
        )

    agent = agent_framework.Agent(FakeChatClient(), name="test_agent")
    response = await agent.run("Hello")

    assert str(response) == "Hello from fake"
    native_span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    )
    converted = convert_span(native_span)
    assert isinstance(converted.span, AgentSpan)
    assert converted.span.integration == "Microsoft Agent Framework"
    assert converted.span.name == "test_agent"
    assert converted.trace_fields["name"] == "agent trace"
    assert converted.trace_fields["thread_id"] == "configured-thread"
    assert converted.span.input == ("Hello" if capture_content else None)
    assert converted.span.output == (
        "Hello from fake" if capture_content else None
    )


def test_instrumentation_does_not_enable_framework_when_tracing_unavailable(
    monkeypatch,
):
    start_runtime = Mock(return_value=False)
    enable = Mock()
    monkeypatch.setattr(otel, "ensure_confident_trace", start_runtime)
    monkeypatch.setattr(
        "agent_framework.observability.enable_instrumentation", enable
    )

    instrument_microsoft_agent_framework()

    start_runtime.assert_called_once_with(global_provider=True)
    enable.assert_not_called()
