import pytest
import time
from importlib import import_module
from deepeval.dataset import Golden
from deepeval.evaluate.types import TestResult
from deepeval.test_run import MetricData
from deepeval.tracing.api import TraceApi, TraceSpanApiStatus
from tests.test_core.stubs import make_span_api_like
from tests.test_core.helpers import ts_iso8601_utc


exec_mod = import_module("deepeval.evaluate.execute")


@pytest.mark.asyncio
async def test_trace_metric_produces_additional_test_result(monkeypatch):
    monkeypatch.setattr(
        exec_mod.trace_manager,
        "_convert_span_to_api_span",
        lambda *_: make_span_api_like(),
        raising=True,
    )
    monkeypatch.setattr(
        exec_mod.global_test_run_manager,
        "update_test_run",
        lambda *_a, **_k: None,
        raising=True,
    )

    now = time.time()

    # Build a TraceApi with one metric row
    trace_api = TraceApi(
        uuid="t",
        name="trace",
        status=TraceSpanApiStatus.SUCCESS,
        error=None,
        input=None,
        output=None,
        expectedOutput=None,
        context=None,
        retrievalContext=None,
        agentSpans=[],
        llmSpans=[],
        retrieverSpans=[],
        toolSpans=[],
        baseSpans=[],
        metricsData=[
            MetricData(
                name="trace-metric",
                score=1.0,
                threshold=0.5,
                reason=None,
                success=True,
                strictMode=False,
                evaluationModel=None,
                error=None,
                evaluationCost=None,
                verboseLogs=None,
            )
        ],
        startTime=ts_iso8601_utc(now),
        endTime=ts_iso8601_utc(now),
    )

    # Monkeypatch create_api_trace to return our injected object
    monkeypatch.setattr(
        exec_mod, "create_api_trace", lambda *a, **k: trace_api, raising=True
    )

    # execute just enough to append results
    from time import perf_counter
    from deepeval.tracing.types import Trace, LlmSpan, TraceSpanStatus

    now = perf_counter()
    span = LlmSpan(
        uuid="s",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        trace_uuid="t",
        parent_uuid=None,
        start_time=now,
        end_time=now,
        name="root",
    )
    trace = Trace(
        uuid="t",
        status=TraceSpanStatus.SUCCESS,
        root_spans=[span],
        start_time=now,
        end_time=now,
    )

    results: list[TestResult] = []
    await exec_mod._a_execute_agentic_test_case(
        golden=Golden(input="x"),
        test_run_manager=exec_mod.global_test_run_manager,
        test_results=results,
        count=1,
        verbose_mode=False,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        trace=trace,
        observed_callback=None,
        trace_metrics=[],
        progress=None,
        pbar_id=None,
    )
    # We should have one top level case result and one extracted trace result
    assert len(results) == 2
    assert any(
        any(md.name == "trace-metric" for md in r.metrics_data or [])
        for r in results
    )
