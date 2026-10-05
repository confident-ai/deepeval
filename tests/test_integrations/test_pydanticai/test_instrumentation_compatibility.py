"""Exercise real Pydantic AI instrumentation without LLM or upload credentials."""

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel as PydanticTestModel

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.integrations.pydantic_ai import instrumentator
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_trace,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing.otel.utils import (
    check_pydantic_ai_agent_input_output,
    check_pydantic_ai_tools_called,
    check_tool_input_parameters_from_gen_ai_attributes,
    check_tool_output,
)
from deepeval.tracing.types import ToolSpan


@pytest.fixture
def instrumented_agent(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    monkeypatch.setattr(instrumentator, "TracerProvider", lambda: provider)
    # Keep the real settings and interceptor; replace only outbound transport.
    monkeypatch.setattr(
        instrumentator,
        "ContextAwareSpanProcessor",
        lambda **kwargs: SimpleSpanProcessor(exporter),
    )
    monkeypatch.setattr(instrumentator, "set_tracer_provider", lambda _: None)
    monkeypatch.setattr(instrumentator, "get_confident_api_key", lambda: None)
    settings = DeepEvalInstrumentationSettings(name="compatibility")
    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:
        # Pydantic AI before 1.95 has no instrumentation capability.
        kwargs = {"instrument": settings}
    else:
        kwargs = {"capabilities": [Instrumentation(settings=settings)]}
    agent = Agent(
        PydanticTestModel(custom_output_text="The weather is sunny."),
        name="weather_agent",
        **kwargs,
    )

    @agent.tool_plain
    def weather(city: str) -> str:
        update_current_trace(user_id="compatibility-user")
        return "sunny"

    yield agent, exporter
    provider.shutdown()


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_real_agent_spans(instrumented_agent, mode):
    import asyncio

    agent, exporter = instrumented_agent
    original_trace = current_trace_context.get()
    original_span = current_span_context.get()

    async def run():
        if mode == "async":
            output = (await agent.run("What is the weather?")).output
        else:
            async with agent.run_stream("What is the weather?") as result:
                chunks = [
                    chunk async for chunk in result.stream_text(delta=True)
                ]
                output = "".join(chunks)
        assert current_trace_context.get() is original_trace
        assert current_span_context.get() is original_span
        return output

    if mode == "sync":
        output = agent.run_sync("What is the weather?").output
    else:
        output = asyncio.run(run())
    assert output == "The weather is sunny."
    assert current_trace_context.get() is original_trace
    assert current_span_context.get() is original_span

    spans = exporter.get_finished_spans()
    agent_spans = [
        span
        for span in spans
        if span.attributes.get(ConfidentAttr.SPAN_TYPE) == "agent"
    ]
    llm_spans = [
        span
        for span in spans
        if span.attributes.get(ConfidentAttr.SPAN_TYPE) == "llm"
    ]
    tool_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.tool.name") == "weather"
    ]
    assert len(agent_spans) == 1
    assert len(llm_spans) == 2
    assert len(tool_spans) == 1
    root = agent_spans[0]
    assert root.attributes[ConfidentAttr.TRACE_USER_ID] == "compatibility-user"
    assert all(span.context.trace_id == root.context.trace_id for span in spans)
    inputs, result = check_pydantic_ai_agent_input_output(root)
    assert inputs
    assert result == output
    calls = check_pydantic_ai_tools_called(root)
    assert len(calls) == 1
    assert calls[0].name == "weather"
    assert calls[0].output == "sunny"
    assert (
        check_tool_input_parameters_from_gen_ai_attributes(tool_spans[0])
        == calls[0].input_parameters
    )
    assert check_tool_output(tool_spans[0]) == "sunny"
    # Verify that the exporter produces the tool data consumed by evaluations.
    converted = (
        ConfidentSpanExporter()
        ._convert_readable_span_to_base_span(tool_spans[0])
        .base_span
    )
    assert isinstance(converted, ToolSpan)
    assert converted.input == calls[0].input_parameters
    assert converted.output == "sunny"


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("result", ["sunny", "", False, 0])
def test_tool_attribute_formats(modern, result):
    arguments_key = "gen_ai.tool.call.arguments" if modern else "tool_arguments"
    result_key = "gen_ai.tool.call.result" if modern else "tool_response"
    span = SimpleNamespace(
        attributes={arguments_key: '{"city":"Paris"}', result_key: result}
    )
    assert check_tool_input_parameters_from_gen_ai_attributes(span) == {
        "city": "Paris"
    }
    assert check_tool_output(span) == result
