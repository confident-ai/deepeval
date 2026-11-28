import asyncio
from importlib import import_module
from time import perf_counter

import pytest

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_run import TestRunManager
from deepeval.tracing.types import LlmSpan, Trace, TraceSpanStatus
from tests.test_core.stubs import make_span_api_like

exec_mod = import_module("deepeval.evaluate.execute")


class SharedSpanBarrierMetric(BaseMetric):
    """Deterministic stand-in for a metric shared across @observe spans.

    Two spans measure this metric concurrently. The barrier makes both calls
    enter ``a_measure`` before either finishes, which is exactly the window
    in which a shared object has its ``score`` and ``reason`` overwritten by
    the other span. With per-span copies every span reads its own values.
    """

    _started = 0
    _event = None

    def __init__(self, threshold: float = 1.0):
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

    @property
    def __name__(self):
        return "SharedSpanBarrierMetric"

    @classmethod
    def reset_barrier(cls):
        cls._started = 0
        cls._event = asyncio.Event()

    async def a_measure(self, test_case, *args, **kwargs):
        type(self)._started += 1
        self.reason = test_case.input
        if type(self)._started == 2:
            type(self)._event.set()

        await type(self)._event.wait()
        await asyncio.sleep(0)

        self.score = 1.0 if self.reason == test_case.input else 0.0
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case, *args, **kwargs):
        raise NotImplementedError

    def is_successful(self):
        return bool(self.success)


def _make_trace(trace_uuid: str, span_input: str, metric: BaseMetric) -> Trace:
    now = perf_counter()
    span = LlmSpan(
        uuid=f"{trace_uuid}-root",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        trace_uuid=trace_uuid,
        parent_uuid=None,
        start_time=now,
        end_time=now,
        name="component",
        input=span_input,
        output=span_input,
        expected_output=span_input,
        metrics=[metric],
    )
    return Trace(
        uuid=trace_uuid,
        status=TraceSpanStatus.SUCCESS,
        root_spans=[span],
        start_time=now,
        end_time=now,
        input=span_input,
        output=span_input,
    )


@pytest.mark.asyncio
async def test_async_span_metrics_are_copied_per_span(monkeypatch):
    SharedSpanBarrierMetric.reset_barrier()

    api_spans = []

    def capture_api_span(span, *_):
        api_span = make_span_api_like()
        # Fields the trace-result extraction reads once a span has metrics.
        api_span.name = span.name
        api_span.input = span.input
        api_span.output = span.output
        api_span.expected_output = span.expected_output
        api_span.context = None
        api_span.retrieval_context = None
        api_spans.append(api_span)
        return api_span

    monkeypatch.setattr(
        exec_mod.trace_manager,
        "_convert_span_to_api_span",
        capture_api_span,
        raising=True,
    )

    exec_mod.trace_manager.eval_session.trace_uuid_to_golden.clear()

    # One metric object, as an @observe(metrics=[...]) decorator gives every
    # call of the decorated function.
    shared_metric = SharedSpanBarrierMetric()

    golden_one = Golden(input="golden-1")
    golden_two = Golden(input="golden-2")
    trace_one = _make_trace("trace-1", "span-input-1", shared_metric)
    trace_two = _make_trace("trace-2", "span-input-2", shared_metric)

    monkeypatch.setitem(
        exec_mod.trace_manager.eval_session.trace_uuid_to_golden,
        trace_one.uuid,
        golden_one,
    )
    monkeypatch.setitem(
        exec_mod.trace_manager.eval_session.trace_uuid_to_golden,
        trace_two.uuid,
        golden_two,
    )

    test_results = []
    test_run_manager = TestRunManager()

    await exec_mod._a_evaluate_traces(
        traces_to_evaluate=[trace_one, trace_two],
        goldens=[golden_one, golden_two],
        test_run_manager=test_run_manager,
        test_results=test_results,
        verbose_mode=False,
        ignore_errors=False,
        skip_on_missing_params=False,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        progress=None,
        pbar_id=None,
        throttle_value=0,
        max_concurrent=2,
        trace_metrics=None,
    )

    assert len(api_spans) == 2
    scores_by_span = {
        api_span.input: api_span.metrics_data[0].score for api_span in api_spans
    }
    assert scores_by_span == {"span-input-1": 1.0, "span-input-2": 1.0}
    # The decorator's own object never measured; each span used a copy.
    assert shared_metric.score is None
