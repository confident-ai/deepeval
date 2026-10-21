"""deepeval's OpenAI Agents integration on confident-trace, with an offline
model."""

import asyncio
import json

import pytest

pytest.importorskip("agents")
pytest.importorskip("openinference.instrumentation.openai_agents")
pytest.importorskip("confident_trace")

try:
    # openai 3 moved to httpx2.
    import httpx2 as httpx
except ImportError:
    import httpx

from agents import (  # noqa: E402
    OpenAIChatCompletionsModel,
    Runner,
    add_trace_processor,
    set_trace_processors,
)
from agents import trace as agents_trace  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from deepeval.openai_agents import (  # noqa: E402
    Agent,
    DeepEvalTracingProcessor,
    function_tool,
)
from deepeval.test_case import ToolCall  # noqa: E402
from deepeval.tracing.tracing import trace_manager  # noqa: E402
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan  # noqa: E402

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402

QUESTION = "What is 7 times 8?"


def completion(request):
    """Calls the agent's first tool, then answers with what it returned."""
    body = json.loads(request.content)
    replied = [m for m in body["messages"] if m["role"] == "tool"]
    if body.get("tools") and not replied:
        name = body["tools"][0]["function"]["name"]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": '{"a": 7, "b": 8}'},
                }
            ],
        }
    else:
        message = {
            "role": "assistant",
            "content": replied[-1]["content"] if replied else "hello",
        }
    return httpx.Response(
        200,
        json={
            "id": "chat-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    )


@pytest.fixture(scope="module")
def processor():
    """OpenInference stays on the first provider it traces for, process-wide,
    so the whole module shares one confident-trace."""
    from deepeval.tracing.otel.setup import (
        ensure_confident_trace,
        shutdown_confident_trace,
    )

    # Drop the SDK's own exporter; OpenInference's processor joins below.
    set_trace_processors([])
    shutdown_confident_trace()
    ensure_confident_trace(
        exporter=InMemorySpanExporter(),
        tracer_provider=TracerProvider(shutdown_on_exit=False),
    )
    processor = DeepEvalTracingProcessor()
    add_trace_processor(processor)
    yield processor
    set_trace_processors([])
    shutdown_confident_trace()


@pytest.fixture(autouse=True)
def fresh_traces():
    yield
    trace_manager.clear_traces()


def calculator(**kwargs):
    client = AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(completion)
        ),
    )

    @function_tool(metric_collection="tools")
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    return Agent(
        name="calculator",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini", openai_client=client
        ),
        tools=[multiply],
        **kwargs,
    )


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def test_a_run_converts_for_evaluation(processor, async_evaluation):
    agent = calculator()

    async def run():
        with agents_trace("math", group_id="conversation"):
            return await Runner.run(agent, QUESTION)

    assert asyncio.run(run()).final_output == "56"

    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.name, collected.thread_id) == ("math", "conversation")
    assert (collected.input, collected.output) == (QUESTION, "56")
    workflow, agent_span = spans_of(collected, AgentSpan)
    # The run's root records nothing; it takes the content of the agent it ran.
    assert (workflow.input, workflow.output) == (QUESTION, "56")
    assert agent_span.name == "calculator"
    assert (agent_span.input, agent_span.output) == (QUESTION, "56")
    assert agent_span.available_tools == ["multiply"]
    call = ToolCall(
        name="multiply",
        description="Multiply two numbers.",
        input_parameters={"a": 7, "b": 8},
        output=56,
    )
    assert agent_span.tools_called == [call]
    first, second = spans_of(collected, LlmSpan)
    assert (first.model, first.provider) == ("gpt-4o-mini", "OpenAI")
    assert [call.name for call in first.tools_called] == ["multiply"]
    (tool,) = spans_of(collected, ToolSpan)
    assert (tool.input, tool.output) == ({"a": 7, "b": 8}, 56)
    assert tool.metric_collection == "tools"


def test_agent_settings_reach_their_spans(processor, async_evaluation):
    agent_metric = RecordingMetric()
    agent = calculator(
        agent_metrics=[agent_metric], llm_metric_collection="answers"
    )

    asyncio.run(Runner.run(agent, QUESTION))

    (collected,) = async_evaluation.traces_to_evaluate
    agent_span = next(
        span
        for span in spans_of(collected, AgentSpan)
        if span.name == "calculator"
    )
    assert agent_span.metrics == [agent_metric]
    assert [
        span.metric_collection for span in spans_of(collected, LlmSpan)
    ] == [
        "answers",
        "answers",
    ]
