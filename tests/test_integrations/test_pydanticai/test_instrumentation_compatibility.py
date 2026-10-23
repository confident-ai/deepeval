"""Real Pydantic AI runs, offline, collected the way an evaluation collects them."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel as PydanticTestModel

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_trace,
)
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan
from tests.test_integrations.utils import (
    collecting_traces,
    confident_trace_instrumentation,
    walk_spans,
)

QUESTION = "What is the weather?"
ANSWER = "The weather is sunny."


@pytest.fixture
def traces():
    with confident_trace_instrumentation("pydantic_ai"):
        with collecting_traces() as collected:
            yield collected


@pytest.fixture
def agent(traces):
    settings = DeepEvalInstrumentationSettings(name="compatibility")
    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:
        # Pydantic AI before 1.95 has no instrumentation capability.
        kwargs = {"instrument": settings}
    else:
        kwargs = {"capabilities": [Instrumentation(settings=settings)]}
    agent = Agent(
        PydanticTestModel(custom_output_text=ANSWER),
        name="weather_agent",
        **kwargs,
    )

    @agent.tool_plain
    def weather(city: str) -> str:
        update_current_trace(user_id="compatibility-user")
        return "sunny"

    return agent


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_real_agent_runs(agent, traces, mode):
    original_trace = current_trace_context.get()
    original_span = current_span_context.get()

    async def run():
        if mode == "async":
            output = (await agent.run(QUESTION)).output
        else:
            async with agent.run_stream(QUESTION) as result:
                chunks = [
                    chunk async for chunk in result.stream_text(delta=True)
                ]
                output = "".join(chunks)
        assert current_trace_context.get() is original_trace
        assert current_span_context.get() is original_span
        return output

    if mode == "sync":
        output = agent.run_sync(QUESTION).output
    else:
        output = asyncio.run(run())
    assert output == ANSWER
    assert current_trace_context.get() is original_trace
    assert current_span_context.get() is original_span

    (collected,) = traces
    assert (collected.name, collected.user_id) == (
        "compatibility",
        "compatibility-user",
    )
    (root,) = collected.root_spans
    assert isinstance(root, AgentSpan)
    assert (root.name, root.input, root.output) == (
        "weather_agent",
        QUESTION,
        ANSWER,
    )
    spans = list(walk_spans(collected.root_spans))
    assert len([span for span in spans if isinstance(span, LlmSpan)]) == 2
    (tool,) = [span for span in spans if isinstance(span, ToolSpan)]
    assert (tool.name, tool.output) == ("weather", "sunny")
    (call,) = root.tools_called
    assert (call.name, call.input_parameters, call.output) == (
        "weather",
        tool.input,
        "sunny",
    )
