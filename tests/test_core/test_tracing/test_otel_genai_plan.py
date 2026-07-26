from types import SimpleNamespace

from opentelemetry.trace.status import Status, StatusCode

from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing.otel.utils import check_span_type_from_gen_ai_attributes
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.types import AgentSpan

init_clock_bridge()


class _Span:
    def __init__(self, attributes):
        self.attributes = attributes


class _ReadableSpan:
    attributes = {
        "gen_ai.operation.name": "plan",
        "gen_ai.agent.name": "planner",
    }
    context = SimpleNamespace(trace_id=1, span_id=2)
    parent = None
    status = Status(StatusCode.UNSET)
    start_time = 1_000_000_000
    end_time = 2_000_000_000
    name = "plan planner"


class _ReadablePlanSpanWithoutAgentName:
    attributes = {"gen_ai.operation.name": "plan"}
    context = SimpleNamespace(trace_id=1, span_id=3)
    parent = None
    status = Status(StatusCode.UNSET)
    start_time = 1_000_000_000
    end_time = 2_000_000_000
    name = "plan"


def test_gen_ai_plan_operation_classifies_as_agent():
    span = _Span({"gen_ai.operation.name": "plan"})

    assert check_span_type_from_gen_ai_attributes(span) == "agent"


def test_gen_ai_plan_agent_name_is_preserved_in_boilerplate_span():
    span = ConfidentSpanExporter.prepare_boilerplate_base_span(_ReadableSpan())

    assert isinstance(span, AgentSpan)
    assert span.name == "planner"


def test_gen_ai_plan_uses_span_name_when_agent_name_is_unavailable():
    span = ConfidentSpanExporter.prepare_boilerplate_base_span(
        _ReadablePlanSpanWithoutAgentName()
    )

    assert isinstance(span, AgentSpan)
    assert span.name == "plan"
