"""deepeval's CrewAI integration on confident-trace, with an offline model."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("crewai")
pytest.importorskip("confident_trace")

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_STORAGE_DIR", tempfile.mkdtemp())

import httpx  # noqa: E402
from crewai_core.token_manager import TokenManager  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

# CrewAI makes credential storage on import; keep it away from the user's.
with patch.object(
    TokenManager,
    "_get_secure_storage_path",
    return_value=Path(os.environ["CREWAI_STORAGE_DIR"]),
):
    from crewai import Task  # noqa: E402

    from deepeval.integrations.crewai import (  # noqa: E402
        LLM,
        Agent,
        Crew,
        instrument_crewai,
        tool,
    )

from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan  # noqa: E402

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402


def completion(request):
    """Calls the first tool it's offered, then answers with its result."""
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
                    "function": {
                        "name": name,
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    else:
        message = {
            "role": "assistant",
            "content": replied[-1]["content"] if replied else "done",
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
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        },
    )


def offline_llm(**kwargs):
    model = LLM(model="openai/gpt-4o-mini", api_key="offline", **kwargs)
    model._client = OpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(completion)),
    )
    model._async_client = AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(completion)
        ),
    )
    return model


@tool(metric_collection="tools")
def weather(city: str) -> str:
    """Look up the weather in a city."""
    return "sunny"


def weather_crew(crew_kwargs=None, agent_kwargs=None, llm_kwargs=None):
    agent = Agent(
        role="forecaster",
        goal="Report the weather",
        backstory="Knows the weather.",
        llm=offline_llm(**(llm_kwargs or {})),
        tools=[weather],
        **(agent_kwargs or {}),
    )
    task = Task(
        description="What is the weather in Paris?",
        expected_output="The weather.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], **(crew_kwargs or {}))


@pytest.fixture
def crewai(exporter):
    instrument_crewai()


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def test_a_crew_converts_for_evaluation(crewai, async_evaluation):
    result = weather_crew().kickoff()

    assert result.raw == "sunny"
    (collected,) = async_evaluation.traces_to_evaluate
    assert collected.output == "sunny"
    (agent,) = spans_of(collected, AgentSpan)
    assert agent.name == "forecaster"
    assert [call.name for call in agent.tools_called] == ["weather"]
    llms = spans_of(collected, LlmSpan)
    assert {(llm.model, llm.provider) for llm in llms} == {
        ("gpt-4o-mini", "OpenAI")
    }
    (forecast,) = spans_of(collected, ToolSpan)
    assert (forecast.name, forecast.output) == ("weather", "sunny")
    assert forecast.metric_collection == "tools"


def test_object_metrics_reach_their_spans(crewai, async_evaluation):
    crew_metric, agent_metric = RecordingMetric(), RecordingMetric()

    weather_crew(
        crew_kwargs={"metrics": [crew_metric]},
        agent_kwargs={"metrics": [agent_metric], "metric_collection": "agents"},
        llm_kwargs={"metric_collection": "answers"},
    ).kickoff()

    (collected,) = async_evaluation.traces_to_evaluate
    (root,) = collected.root_spans
    assert root.metrics == [crew_metric]
    (agent,) = spans_of(collected, AgentSpan)
    assert (agent.metrics, agent.metric_collection) == (
        [agent_metric],
        "agents",
    )
    assert {llm.metric_collection for llm in spans_of(collected, LlmSpan)} == {
        "answers"
    }


def test_kickoff_for_each_is_one_trace(crewai, async_evaluation):
    crew_metric = RecordingMetric()

    weather_crew(crew_kwargs={"metrics": [crew_metric]}).kickoff_for_each(
        inputs=[{}, {}]
    )

    (collected,) = async_evaluation.traces_to_evaluate
    (root,) = collected.root_spans
    assert (root.name, root.metrics) == ("kickoff_for_each", [crew_metric])
    assert [crew.name for crew in root.children] == ["crew crew", "crew crew"]
