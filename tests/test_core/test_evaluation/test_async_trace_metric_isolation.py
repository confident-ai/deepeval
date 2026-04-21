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


class BarrierIsolationMetric(BaseMetric):
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
        return "BarrierIsolationMetric"

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


def _make_trace(trace_uuid: str, trace_input: str, trace_output: str) -> Trace:
    now = perf_counter()
    span = LlmSpan(
        uuid=f"{trace_uuid}-root",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        trace_uuid=trace_uuid,
        parent_uuid=None,
        start_time=now,
        end_time=now,
        name="root",
    )
    return Trace(
        uuid=trace_uuid,
        status=TraceSpanStatus.SUCCESS,
        root_spans=[span],
        start_time=now,
        end_time=now,
        input=trace_input,
        output=trace_output,
    )


@pytest.mark.asyncio
async def test_async_trace_metrics_are_copied_per_trace(monkeypatch):
    BarrierIsolationMetric.reset_barrier()

    monkeypatch.setattr(
        exec_mod.trace_manager,
        "_convert_span_to_api_span",
        lambda *_: make_span_api_like(),
        raising=True,
    )

    exec_mod.trace_manager.eval_session.trace_uuid_to_golden.clear()

    golden_one = Golden(input="golden-1")
    golden_two = Golden(input="golden-2")
    trace_one = _make_trace("trace-1", "trace-input-1", "trace-output-1")
    trace_two = _make_trace("trace-2", "trace-input-2", "trace-output-2")

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
        trace_metrics=[BarrierIsolationMetric()],
    )

    top_level_results = [
        result
        for result in test_results
        if result.input in {"golden-1", "golden-2"}
    ]

    assert len(top_level_results) == 2
    assert trace_one.metrics is not None
    assert trace_two.metrics is not None
    assert trace_one.metrics[0] is not trace_two.metrics[0]

    scores_by_golden = {
        result.input: result.metrics_data[0].score
        for result in top_level_results
    }
    assert scores_by_golden == {"golden-1": 1.0, "golden-2": 1.0}
