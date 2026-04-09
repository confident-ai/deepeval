import time
from importlib import import_module

import pytest

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_run import TestRunManager
from deepeval.tracing.tracing import EVAL_DUMMY_SPAN_NAME, TraceManager
from deepeval.tracing.types import LlmSpan, Trace, TraceSpanStatus

exec_mod = import_module("deepeval.evaluate.execute")


class RecordingAsyncMetric(BaseMetric):
    def __init__(
        self,
        name: str,
        *,
        requires_trace: bool = False,
        threshold: float = 0.5,
    ):
        self.name = name
        self.requires_trace = requires_trace
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.strict_mode = False
        self.evaluation_model = None
        self.evaluation_cost = None
        self.verbose_logs = None
        self.skipped = False
        self.captured_trace = None

    @property
    def __name__(self):
        return self.name

    async def a_measure(self, test_case, *args, **kwargs):
        if self.requires_trace:
            self.captured_trace = test_case._trace_dict
        self.score = 1.0
        self.reason = "ok"
        self.success = True
        return self.score

    def measure(self, test_case, *args, **kwargs):
        raise NotImplementedError

    def is_successful(self):
        return bool(self.success)


def _make_llm_span(
    trace_uuid: str,
    name: str,
    *,
    parent_uuid: str | None = None,
    children=None,
    metrics=None,
):
    now = time.perf_counter()
    return LlmSpan(
        uuid=f"{trace_uuid}-{name}",
        trace_uuid=trace_uuid,
        parent_uuid=parent_uuid,
        start_time=now,
        end_time=now,
        status=TraceSpanStatus.SUCCESS,
        children=children or [],
        name=name,
        input=f"{name}-input",
        output=f"{name}-output",
        metrics=metrics or [],
    )


def _make_multi_root_trace():
    trace_uuid = "trace-multi-root"
    child_one = _make_llm_span(
        trace_uuid,
        "child-1",
        parent_uuid=f"{trace_uuid}-root-1",
        metrics=[RecordingAsyncMetric("child-1-metric")],
    )
    child_two = _make_llm_span(
        trace_uuid,
        "child-2",
        parent_uuid=f"{trace_uuid}-root-2",
        metrics=[RecordingAsyncMetric("child-2-metric")],
    )
    root_one = _make_llm_span(
        trace_uuid,
        "root-1",
        children=[child_one],
        metrics=[RecordingAsyncMetric("root-1-metric")],
    )
    root_two = _make_llm_span(
        trace_uuid,
        "root-2",
        children=[child_two],
        metrics=[RecordingAsyncMetric("root-2-metric")],
    )
    now = time.perf_counter()
    trace = Trace(
        uuid=trace_uuid,
        status=TraceSpanStatus.SUCCESS,
        root_spans=[root_one, root_two],
        start_time=now,
        end_time=now,
        input="trace-input",
        output="trace-output",
        metrics=[RecordingAsyncMetric("trace-metric", requires_trace=True)],
    )
    return trace


def test_create_nested_trace_dict_includes_all_root_spans():
    trace = _make_multi_root_trace()

    trace_dict = exec_mod.trace_manager.create_nested_trace_dict(trace)

    assert trace_dict["type"] == "trace"
    assert trace_dict["input"] == "trace-input"
    assert [child["name"] for child in trace_dict["children"]] == [
        "root-1",
        "root-2",
    ]
    assert trace_dict["children"][0]["children"][0]["name"] == "child-1"
    assert trace_dict["children"][1]["children"][0]["name"] == "child-2"


def test_end_trace_preserves_all_roots_when_first_root_has_children():
    manager = TraceManager()
    manager.evaluating = True
    manager.evaluation_loop = False

    trace = manager.start_new_trace(trace_uuid="trace-preserve-roots")
    trace.status = TraceSpanStatus.SUCCESS
    first_child = _make_llm_span(
        trace.uuid,
        "child-1",
        parent_uuid=f"{trace.uuid}-root-1",
    )
    second_child = _make_llm_span(
        trace.uuid,
        "child-2",
        parent_uuid=f"{trace.uuid}-root-2",
    )
    root_one = _make_llm_span(
        trace.uuid,
        "root-1",
        children=[first_child],
    )
    root_two = _make_llm_span(
        trace.uuid,
        "root-2",
        children=[second_child],
    )
    trace.root_spans = [root_one, root_two]

    manager.end_trace(trace.uuid)

    assert [root.name for root in trace.root_spans] == ["root-1", "root-2"]
    assert all(root.parent_uuid is None for root in trace.root_spans)


def test_end_trace_unwraps_dummy_root_into_all_children():
    manager = TraceManager()
    manager.evaluating = True
    manager.evaluation_loop = False

    trace = manager.start_new_trace(trace_uuid="trace-dummy-root")
    trace.status = TraceSpanStatus.SUCCESS
    child_one = _make_llm_span(trace.uuid, "child-1", parent_uuid="dummy-root")
    child_two = _make_llm_span(trace.uuid, "child-2", parent_uuid="dummy-root")
    dummy_root = _make_llm_span(
        trace.uuid,
        EVAL_DUMMY_SPAN_NAME,
        children=[child_one, child_two],
    )
    trace.root_spans = [dummy_root]

    manager.end_trace(trace.uuid)

    assert [root.name for root in trace.root_spans] == ["child-1", "child-2"]
    assert all(root.parent_uuid is None for root in trace.root_spans)


@pytest.mark.asyncio
async def test_async_agentic_evaluation_uses_all_root_spans_for_trace_and_dfs(
    monkeypatch,
):
    trace = _make_multi_root_trace()
    trace_metric = trace.metrics[0]
    captured = {}
    test_run_manager = TestRunManager()

    monkeypatch.setattr(
        test_run_manager,
        "update_test_run",
        lambda api_test_case, _test_case: captured.setdefault(
            "trace_api", api_test_case.trace
        ),
        raising=False,
    )
    monkeypatch.setattr(
        exec_mod, "extract_trace_test_results", lambda _api: [], raising=True
    )

    test_results = []
    await exec_mod._a_execute_agentic_test_case(
        golden=Golden(input="golden-input"),
        test_run_manager=test_run_manager,
        test_results=test_results,
        count=1,
        verbose_mode=False,
        ignore_errors=False,
        skip_on_missing_params=False,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        observed_callback=None,
        trace=trace,
        trace_metrics=None,
        progress=None,
        pbar_id=None,
    )

    assert trace_metric.captured_trace is not None
    assert [
        child["name"] for child in trace_metric.captured_trace["children"]
    ] == [
        "root-1",
        "root-2",
    ]

    trace_api = captured["trace_api"]
    llm_span_names = {span.name for span in trace_api.llm_spans}
    assert llm_span_names == {"root-1", "child-1", "root-2", "child-2"}
    assert all(len(span.metrics_data) == 1 for span in trace_api.llm_spans)
