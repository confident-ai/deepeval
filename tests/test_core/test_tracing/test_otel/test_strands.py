"""deepeval's Strands and AgentCore integrations, with spans shaped like the
ones Strands records."""

import json

import pytest

pytest.importorskip("confident_trace")

from deepeval.integrations.agentcore import instrument_agentcore  # noqa: E402
from deepeval.integrations.strands import instrument_strands  # noqa: E402
from deepeval.tracing.otel.adapters.strands import SCOPE  # noqa: E402
from deepeval.test_case import ToolCall  # noqa: E402
from deepeval.tracing import next_agent_span, next_tool_span  # noqa: E402
from deepeval.tracing.types import AgentSpan, LlmSpan, ToolSpan  # noqa: E402

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402

QUESTION = "What is 7 times 8?"
SESSION = {
    "gen_ai.system": "strands-agents",
    "session.id": "session",
    "user.id": "user",
}


def run_agent(provider):
    """One tool-using turn, recorded the way Strands records it."""
    tracer = provider.get_tracer(SCOPE)
    question = json.dumps([{"text": QUESTION}])
    with tracer.start_as_current_span(
        "invoke_agent Strands Agents",
        attributes={
            **SESSION,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "Strands Agents",
            "gen_ai.agent.tools": '["calculate"]',
        },
    ) as agent:
        agent.add_event("gen_ai.user.message", {"content": question})
        with tracer.start_as_current_span(
            "chat",
            attributes={
                **SESSION,
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4o-mini",
            },
        ) as chat:
            chat.add_event("gen_ai.user.message", {"content": question})
            tool_use = {"name": "calculate", "input": {"a": 7, "b": 8}}
            chat.add_event(
                "gen_ai.choice",
                {"message": json.dumps([{"toolUse": tool_use}])},
            )
        with tracer.start_as_current_span(
            "execute_tool calculate",
            attributes={
                **SESSION,
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "calculate",
            },
        ) as tool:
            tool.add_event(
                "gen_ai.tool.message", {"content": '{"a": 7, "b": 8}'}
            )
            tool.add_event("gen_ai.choice", {"message": '[{"text": "56.0"}]'})
        agent.add_event("gen_ai.choice", {"message": "56\n"})


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def test_a_run_converts_for_evaluation(provider, exporter, async_evaluation):
    instrument_strands(name="calculator", tags=["math"])

    run_agent(provider)

    (collected,) = async_evaluation.traces_to_evaluate
    assert (
        collected.name,
        collected.tags,
        collected.thread_id,
        collected.user_id,
    ) == ("calculator", ["math"], "session", "user")
    assert (collected.input, collected.output) == (QUESTION, "56\n")
    (agent,) = collected.root_spans
    assert isinstance(agent, AgentSpan)
    assert (agent.input, agent.output) == (QUESTION, "56\n")
    assert agent.available_tools == ["calculate"]
    (llm,) = spans_of(collected, LlmSpan)
    call = ToolCall(name="calculate", input_parameters={"a": 7, "b": 8})
    assert (llm.provider, llm.tools_called) == ("OpenAI", [call])
    (tool,) = spans_of(collected, ToolSpan)
    assert (tool.input, tool.output) == ({"a": 7, "b": 8}, "56.0")
    assert agent.tools_called == [
        ToolCall(
            name="calculate", input_parameters={"a": 7, "b": 8}, output="56.0"
        )
    ]


def test_staged_fields_reach_their_spans(provider, exporter, async_evaluation):
    agent_metric = RecordingMetric()
    instrument_strands()

    with next_agent_span(metrics=[agent_metric]):
        with next_tool_span(metric_collection="tools"):
            run_agent(provider)

    (collected,) = async_evaluation.traces_to_evaluate
    (agent,) = collected.root_spans
    assert agent.metrics == [agent_metric]
    (tool,) = spans_of(collected, ToolSpan)
    assert tool.metric_collection == "tools"


@pytest.mark.parametrize(
    "instrument", [instrument_strands, instrument_agentcore]
)
def test_trace_defaults_reach_exported_spans(
    instrument, provider, exporter, ended
):
    instrument(name="calculator", environment="staging")

    run_agent(provider)

    import confident_trace

    confident_trace.flush()
    spans = exporter.get_finished_spans()
    assert {span.attributes["confident.trace.name"] for span in spans} == {
        "calculator"
    }
    assert {
        span.attributes["confident.trace.environment"] for span in spans
    } == {"staging"}
    assert ended == []
