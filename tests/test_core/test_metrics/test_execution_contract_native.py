import pytest

from deepeval.tracing.types import (
    AgentSpan,
    LlmSpan,
    ToolSpan,
    Trace,
    TraceSpanStatus,
)


def tool(id="t", **kwargs):
    values = dict(
        uuid=id,
        trace_uuid="trace",
        name="verify",
        status=TraceSpanStatus.SUCCESS,
        start_time=1,
        end_time=2,
        input={"account": "alice"},
    )
    values.update(kwargs)
    return ToolSpan(**values)


def test_native_trace_preserves_identifiers_failures_and_payload_without_mutation():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_trace,
    )

    span = tool(status=TraceSpanStatus.ERRORED, error="failed")
    trace = Trace(
        uuid="trace",
        status=TraceSpanStatus.ERRORED,
        root_spans=[span],
        start_time=0,
        end_time=3,
    )
    result = snapshot_from_trace(trace, clock_domain="process:1")
    assert result.complete
    assert result.spans[0].status == "error"
    assert result.spans[0].id == span.uuid
    assert result.spans[0].start == span.start_time
    span.input["account"] = "bob"
    assert result.spans[0].payload["input"]["account"] == "alice"


def test_subtree_snapshot_declares_parent_boundary_and_does_not_include_siblings():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_span,
    )

    result = snapshot_from_span(
        tool(parent_uuid="outside"), clock_domain="process:1"
    )
    assert result.external_parents == ("outside",)
    assert len(result.spans) == 1


def test_native_incomplete_and_dropped_capture_stays_partial():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_trace,
    )

    trace = Trace(
        uuid="trace",
        status=TraceSpanStatus.SUCCESS,
        root_spans=[tool(drop=True)],
        start_time=0,
        end_time=3,
    )
    assert not snapshot_from_trace(trace).complete
    assert len(snapshot_from_trace(trace).spans) == 0
    trace.root_spans = [tool()]
    trace.end_time = None
    assert not snapshot_from_trace(trace).complete


def test_native_parent_contradiction_and_duplicate_membership_are_rejected():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_span,
    )

    root = AgentSpan(
        uuid="a",
        trace_uuid="trace",
        name="agent",
        status=TraceSpanStatus.SUCCESS,
        start_time=0,
        end_time=4,
        children=[tool(parent_uuid="other")],
    )
    with pytest.raises(ValueError, match="parent"):
        snapshot_from_span(root)
    child = tool(parent_uuid="a")
    root.children = [child, child]
    with pytest.raises(ValueError, match="duplicate"):
        snapshot_from_span(root)


def test_native_resource_counters_are_observed_not_invented():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_span,
    )

    model = LlmSpan(
        uuid="m",
        trace_uuid="trace",
        status=TraceSpanStatus.SUCCESS,
        start_time=0,
        end_time=1,
        input_token_count=10,
        output_token_count=2,
        cost_per_input_token=0.1,
        cost_per_output_token=0.2,
    )
    result = snapshot_from_span(model)
    counters = result.spans[0].payload["resources"]
    assert counters["total_tokens"] == 12
    assert counters["cost"] == pytest.approx(1.4)
    model.output_token_count = None
    counters = snapshot_from_span(model).spans[0].payload["resources"]
    assert "total_tokens" not in counters
    assert "cost" not in counters


def test_unnamed_span_keeps_unknown_name_without_fabrication():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_span,
    )
    from deepeval.metrics.community.execution_contract.selectors import (
        Selector,
        Match,
    )

    model = LlmSpan(
        uuid="m",
        trace_uuid="trace",
        status=TraceSpanStatus.SUCCESS,
        start_time=0,
        end_time=1,
    )
    span = snapshot_from_span(model).spans[0]
    assert span.name is None
    assert Selector(name="foo").matches(span) is Match.UNKNOWN


def test_native_resource_overflow_is_an_evidence_validation_error():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_span,
    )

    model = LlmSpan(
        uuid="m",
        trace_uuid="trace",
        status=TraceSpanStatus.SUCCESS,
        start_time=1,
        end_time=2,
        input_token_count=1,
    )
    model.input_token_count = 10**1000
    with pytest.raises(ValueError, match="resource"):
        snapshot_from_span(model)
