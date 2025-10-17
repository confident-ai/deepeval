from typing import Optional, List


from deepeval.tracing.types import TraceSpanStatus

from tests.test_core.stubs import (
    ApiTestCaseLike,
    make_trace_api_like,
    make_span_api_like,
    _DummyMetric,
    _DummyTaskCompletionMetric,
    _FakeSpan,
    _FakeTrace,
)


def test_make_trace_api_like_shape():
    obj = make_trace_api_like(TraceSpanStatus.SUCCESS)

    # Fields on "TraceApi-like" objects
    required_attrs = [
        "name",
        "status",
        "error",
        "input",
        "output",
        "expected_output",
        "context",
        "retrieval_context",
        "agent_spans",
        "llm_spans",
        "retriever_spans",
        "tool_spans",
        "base_spans",
        "metrics_data",
    ]
    for attr in required_attrs:
        assert hasattr(obj, attr), f"missing attribute: {attr}"

    # assert shape of list fields
    assert isinstance(obj.agent_spans, list)
    assert isinstance(obj.llm_spans, list)
    assert isinstance(obj.retriever_spans, list)
    assert isinstance(obj.tool_spans, list)
    assert isinstance(obj.base_spans, list)
    assert isinstance(obj.metrics_data, list)


def test_make_span_api_like_shape():
    span = make_span_api_like()
    for attr in ["status", "error", "metrics_data"]:
        assert hasattr(span, attr), f"missing attribute: {attr}"
    assert isinstance(span.metrics_data, list)


def test_dummy_metric_behaviour_and_surface():
    # default: measure should suceed and not be skipped
    m_ok = _DummyMetric(name="ok")
    assert hasattr(m_ok, "threshold")
    m_ok.measure(test_case=None)
    assert m_ok.is_successful() is True
    assert m_ok.skipped is False
    assert m_ok.error is None

    # if should_skip=True, then measuring marks skipped and success remains False
    m_skip = _DummyMetric(name="skip", should_skip=True)
    m_skip.measure(test_case=None)
    assert m_skip.skipped is True
    assert m_skip.is_successful() is False


def test_dummy_task_completion_metric_behaviour_and_surface():
    m = _DummyTaskCompletionMetric(name="tc")
    # has the same surface that downstream expects
    assert hasattr(m, "threshold")
    m.measure(test_case=None)
    assert m.is_successful() is True
    assert m.skipped is False
    assert m.error is None


def test_fake_span_shape_and_defaults():
    s = _FakeSpan(
        input="in", output="out", metrics=[_DummyMetric()], children=[]
    )
    # fields that execute utilities and conversions expect
    assert s.input == "in"
    assert s.output == "out"
    assert hasattr(s, "expected_output")
    assert hasattr(s, "context")
    assert hasattr(s, "retrieval_context")
    assert hasattr(s, "tools_called")
    assert hasattr(s, "expected_tools")
    assert isinstance(s.metrics, list)
    assert isinstance(s.children, list)
    assert s.status in (TraceSpanStatus.SUCCESS, TraceSpanStatus.ERRORED)
    assert s.error is None


def test_fake_trace_shape_and_defaults():
    root = _FakeSpan(input="in", output="out")
    t = _FakeTrace(
        input="t-in", output="t-out", metrics=[_DummyMetric()], root_span=root
    )

    # shape that execute and on_task_done logic expects
    assert t.input == "t-in"
    assert t.output == "t-out"
    for attr in [
        "expected_output",
        "context",
        "retrieval_context",
        "tools_called",
        "expected_tools",
        "metrics",
        "root_spans",
        "status",
        "error",
        "uuid",
    ]:
        assert hasattr(t, attr), f"missing attribute: {attr}"

    assert (
        isinstance(t.root_spans, list) and t.root_spans
    ), "root_spans should be non-empty list"
    assert t.root_spans[0] is root
    assert t.status in (TraceSpanStatus.SUCCESS, TraceSpanStatus.ERRORED)
    assert isinstance(t.uuid, str) and t.uuid


def test_api_test_case_like_protocol_conformance():
    """A minimal object with the expected fields/methods should satisfy ApiTestCaseLike."""

    class MinimalCase:
        name: Optional[str] = None
        success: Optional[bool] = None
        metrics_data: List = []
        input: Optional[str] = None
        actual_output: Optional[str] = None
        expected_output: Optional[str] = None
        context: Optional[List[str]] = None
        retrieval_context: Optional[List[str]] = None

        def update_metric_data(self, *args, **kwargs) -> None:
            pass

        def update_status(self, *args, **kwargs) -> None:
            pass

        def update_run_duration(self, *args, **kwargs) -> None:
            pass

    mc = MinimalCase()
    assert isinstance(mc, ApiTestCaseLike)

    # Negative case: missing required methods should not satisfy the protocol
    class NotCase:
        name = None
        metrics_data = []
        input = None
        actual_output = None
        expected_output = None
        context = None
        retrieval_context = None
        # missing update_* methods on purpose

    assert not isinstance(NotCase(), ApiTestCaseLike)
