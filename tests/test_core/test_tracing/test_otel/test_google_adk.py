"""deepeval's Google ADK integration on confident-trace, with an offline model."""

import asyncio
from contextlib import aclosing
from functools import cached_property

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("confident_trace")

import httpx  # noqa: E402
from google import genai  # noqa: E402
from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from deepeval.integrations.google_adk import instrument_google_adk  # noqa: E402
from deepeval.test_case import ToolCall  # noqa: E402
from deepeval.tracing import next_agent_span, next_llm_span  # noqa: E402
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan  # noqa: E402

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402

QUESTION = "What is 7 times 8?"


@pytest.fixture
def provider():
    """ADK traces through the global provider, which a process sets once."""
    current = otel_trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    provider = TracerProvider(shutdown_on_exit=False)
    otel_trace.set_tracer_provider(provider)
    return provider


def calculate(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def answers(*parts):
    """A Gemini endpoint that answers each request with the next part."""
    replies = iter(parts)

    def handler(request):
        reply = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [next(replies)]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
        }
        return httpx.Response(200, json=reply)

    return handler


def run_agent(handler):
    client = genai.Client(
        api_key="offline",
        http_options=types.HttpOptions(
            async_client_args={"transport": httpx.MockTransport(handler)}
        ),
    )

    class OfflineGemini(Gemini):
        @cached_property
        def api_client(self):
            return client

    runner = InMemoryRunner(
        app_name="calculator",
        agent=LlmAgent(
            name="assistant",
            model=OfflineGemini(model="gemini-2.5-flash"),
            instruction="Use the tool.",
            tools=[calculate],
        ),
    )

    async def run():
        await runner.session_service.create_session(
            app_name="calculator", user_id="user", session_id="session"
        )
        message = types.Content(role="user", parts=[types.Part(text=QUESTION)])
        async with aclosing(
            runner.run_async(
                user_id="user", session_id="session", new_message=message
            )
        ) as events:
            return [event async for event in events]

    return asyncio.run(run())


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def tool_turn():
    return answers(
        {"functionCall": {"name": "calculate", "args": {"a": 7, "b": 8}}},
        {"text": "56"},
    )


def test_a_run_converts_for_evaluation(exporter, async_evaluation):
    instrument_google_adk(name="calculator")

    run_agent(tool_turn())

    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.name, collected.thread_id) == ("calculator", "session")
    assert (collected.input, collected.output) == (QUESTION, "56")
    (agent,) = spans_of(collected, AgentSpan)
    # ADK records no content on agents; theirs is their model calls'.
    assert (agent.input, agent.output) == (QUESTION, "56")
    call = ToolCall(name="calculate", input_parameters={"a": 7, "b": 8})
    first, second = spans_of(collected, LlmSpan)
    assert (first.provider, first.tools_called) == ("Gemini", [call])
    assert (second.input[-1]["role"], second.output) == ("user", "56")
    (tool,) = spans_of(collected, ToolSpan)
    assert (tool.input, tool.output) == ({"a": 7, "b": 8}, 56)
    assert [call.name for call in agent.tools_called] == ["calculate"]


def test_staged_fields_reach_their_spans(exporter, async_evaluation):
    agent_metric = RecordingMetric()
    instrument_google_adk()

    with next_agent_span(metrics=[agent_metric]):
        with next_llm_span(metric_collection="answers"):
            run_agent(answers({"text": "56"}))

    (collected,) = async_evaluation.traces_to_evaluate
    (agent,) = spans_of(collected, AgentSpan)
    assert agent.metrics == [agent_metric]
    (llm,) = spans_of(collected, LlmSpan)
    assert llm.metric_collection == "answers"
