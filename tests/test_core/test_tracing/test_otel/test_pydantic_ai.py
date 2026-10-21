"""deepeval's Pydantic AI integration on confident-trace, with an offline model."""

import json
import warnings

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("confident_trace")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from deepeval.integrations.pydantic_ai import (  # noqa: E402
    ConfidentInstrumentationSettings,
    DeepEvalInstrumentationSettings,
)
from deepeval.tracing import (  # noqa: E402
    next_agent_span,
    next_llm_span,
    observe,
    update_current_span,
)
from deepeval.tracing.tracing import trace_manager  # noqa: E402
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan  # noqa: E402

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402


def weather_agent(settings):
    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:
        kwargs = {"instrument": settings}
    else:
        kwargs = {"capabilities": [Instrumentation(settings=settings)]}
    agent = Agent(
        TestModel(custom_output_text="sunny"), name="weather_agent", **kwargs
    )

    @agent.tool_plain
    def weather(city: str) -> str:
        update_current_span(metadata={"source": "station"})
        return "sunny"

    return agent


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def test_staged_fields_reach_their_spans(exporter, async_evaluation):
    agent_metric = RecordingMetric()
    agent = weather_agent(DeepEvalInstrumentationSettings())

    with next_agent_span(metrics=[agent_metric]):
        with next_llm_span(metric_collection="answers"):
            agent.run_sync("Weather in Paris?")

    (collected,) = async_evaluation.traces_to_evaluate
    (root,) = collected.root_spans
    assert isinstance(root, AgentSpan)
    assert root.metrics == [agent_metric]
    first, second = spans_of(collected, LlmSpan)
    assert (first.metric_collection, second.metric_collection) == (
        "answers",
        None,
    )
    (tool,) = spans_of(collected, ToolSpan)
    assert tool.metadata == {"source": "station"}


def test_an_observed_agent_nests_under_observe(exporter, posted):
    agent = weather_agent(
        DeepEvalInstrumentationSettings(name="weather", thread_id="thread")
    )

    @observe(name="endpoint")
    def endpoint():
        return agent.run_sync("Weather in Paris?").output

    assert endpoint() == "sunny"

    (collected,) = posted
    (root,) = collected.root_spans
    assert root.name == "endpoint"
    (agent_span,) = root.children
    assert (agent_span.name, agent_span.output) == ("weather_agent", "sunny")
    # The settings fill the trace's gaps; only its root names it.
    assert (collected.name, collected.thread_id) == (None, "thread")


def test_settings_reach_an_exported_trace(exporter, posted):
    agent = weather_agent(
        DeepEvalInstrumentationSettings(
            name="weather", thread_id="thread", metadata={"team": "geo"}
        )
    )

    agent.run_sync("Weather in Paris?")

    import confident_trace

    confident_trace.flush()
    (root,) = [
        span for span in exporter.get_finished_spans() if span.parent is None
    ]
    assert root.attributes["confident.trace.name"] == "weather"
    assert root.attributes["confident.trace.thread_id"] == "thread"
    assert json.loads(root.attributes["confident.trace.metadata"]) == {
        "team": "geo"
    }
    assert posted == []


def test_api_key_is_the_export_key(exporter, monkeypatch):
    monkeypatch.setattr(trace_manager, "confident_api_key", None)

    DeepEvalInstrumentationSettings(api_key="key")

    assert trace_manager.confident_api_key == "key"


@pytest.mark.parametrize(
    "kwarg",
    [
        "is_test_mode",
        "confident_prompt",
        "llm_metric_collection",
        "agent_metrics",
    ],
)
def test_span_kwargs_are_not_settings(exporter, kwarg):
    with pytest.raises(TypeError):
        DeepEvalInstrumentationSettings(**{kwarg: object()})


def test_the_confident_alias_warns(exporter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        settings = ConfidentInstrumentationSettings()

    assert isinstance(settings, DeepEvalInstrumentationSettings)
    assert any(
        issubclass(warning.category, DeprecationWarning) for warning in caught
    )
