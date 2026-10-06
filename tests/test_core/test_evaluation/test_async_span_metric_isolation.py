import asyncio
from importlib import import_module
from time import perf_counter
from types import SimpleNamespace

import pytest

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_run import TestRunManager
from deepeval.tracing.types import LlmSpan, Trace, TraceSpanStatus

exec_mod = import_module("deepeval.evaluate.execute")


class BarrierIsolationMetric(BaseMetric):
    """A metric that only scores correctly if its state is private to one span.

    ``a_measure`` writes the test case's input onto ``self`` and then waits until the other
    span has started, so both measurements overlap. If two spans share the metric object,
    the second write clobbers the first before it computes its score.
    """

    _started = 0
    _event = None
    _observations = None

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
        return "BarrierIsolationMetric"

    @classmethod
    def reset(cls):
        cls._started = 0
        cls._event = asyncio.Event()
        cls._observations = []

    async def a_measure(self, test_case, *args, **kwargs):
        type(self)._started += 1
        self.reason = test_case.input
        if type(self)._started == 2:
            type(self)._event.set()

        await type(self)._event.wait()
        await asyncio.sleep(0)

        self.score = 1.0 if self.reason == test_case.input else 0.0
        self.success = self.score >= self.threshold
        type(self)._observations.append(
            (self.reason, test_case.input, self.score)
        )
        return self.score

    def measure(self, test_case, *args, **kwargs):
        raise NotImplementedError

    def is_successful(self):
        return bool(self.success)


def _make_trace_with_span(trace_uuid: str, span_input: str, metrics) -> Trace:
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
        metrics=metrics,
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
    """Each span's metric result must reflect that span's own test case.

    Regression test for component-level evaluation: ``@observe(metrics=[...])`` holds the
    same metric objects for every call of the component, so evaluating them in place let
    concurrent spans overwrite each other's scores.
    """
    BarrierIsolationMetric.reset()

    # Capture the api span each real span maps to, so the recorded results can be read
    # back after the walk. The attributes are the ones the span path touches.
    api_spans = {}

    def _span_api_for(span):
        api_span = SimpleNamespace(
            status=None,
            error=None,
            metrics_data=[],
            name=span.name,
            input=span.input,
            output=span.output,
            expected_output=span.expected_output,
            context=span.context,
            retrieval_context=span.retrieval_context,
            tools_called=span.tools_called,
            expected_tools=span.expected_tools,
        )
        api_spans[span.uuid] = api_span
        return api_span

    monkeypatch.setattr(
        exec_mod.trace_manager,
        "_convert_span_to_api_span",
        _span_api_for,
        raising=True,
    )
    exec_mod.trace_manager.eval_session.trace_uuid_to_golden.clear()

    golden_one = Golden(input="golden-1")
    golden_two = Golden(input="golden-2")

    # One metric instance shared by both spans, exactly as the decorator holds it.
    shared_metric = BarrierIsolationMetric()
    trace_one = _make_trace_with_span(
        "trace-1", "span-input-1", [shared_metric]
    )
    trace_two = _make_trace_with_span(
        "trace-2", "span-input-2", [shared_metric]
    )

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

    await exec_mod._a_evaluate_traces(
        traces_to_evaluate=[trace_one, trace_two],
        goldens=[golden_one, golden_two],
        test_run_manager=TestRunManager(),
        test_results=[],
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

    assert (
        len(BarrierIsolationMetric._observations) == 2
    ), BarrierIsolationMetric._observations

    # Every measurement must have scored against its own span's input. Sorted because the
    # two spans interleave and finish in either order.
    assert sorted(
        (reason, value)
        for reason, _, value in BarrierIsolationMetric._observations
    ) == [
        ("span-input-1", 1.0),
        ("span-input-2", 1.0),
    ], BarrierIsolationMetric._observations

    scores_by_span = {
        span_uuid: [metric_data.score for metric_data in api_span.metrics_data]
        for span_uuid, api_span in api_spans.items()
    }
    assert scores_by_span == {"trace-1-root": [1.0], "trace-2-root": [1.0]}

    # The instance handed to the decorator must not have been evaluated in place.
    assert shared_metric.score is None
