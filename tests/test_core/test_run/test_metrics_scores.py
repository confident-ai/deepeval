"""Tests for the run-level metric aggregate (`TestRun.metrics_scores`).

A metric with `threshold=None` (score-only mode) has a score but no verdict:
its scores are aggregated like any other metric's, and it counts as neither a
pass, a fail nor an error.
"""

import pytest

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.api import ConversationalApiTestCase, LLMApiTestCase
from deepeval.test_run.test_run import TestRun
from deepeval.tracing.api import (
    BaseApiSpan,
    MetricData,
    SpanApiType,
    TraceApi,
    TraceSpanApiStatus,
)

TIME = "2026-01-01T00:00:00.000Z"


def make_run(*metrics_data_per_case, trace=None) -> TestRun:
    run = TestRun()
    for order, metrics_data in enumerate(metrics_data_per_case):
        case = LLMApiTestCase(
            name=f"case {order}",
            input="input",
            order=order,
            metricsData=[],
            trace=trace,
        )
        for metric_data in metrics_data:
            case.update_metric_data(metric_data)
        run.test_cases.append(case)
    return run


def counts(metric_scores):
    return metric_scores.passes, metric_scores.fails, metric_scores.errors


def scores_for(run: TestRun, name: str):
    [metric_scores] = [m for m in run.metrics_scores if m.metric == name]
    return metric_scores


def test_score_only_metric_keeps_its_scores():
    run = make_run(
        [MetricData(name="Score Only", score=0.8, success=None)],
        [MetricData(name="Score Only", score=0.6, success=None)],
    )
    valid_scores = run.construct_metrics_scores()

    metric_scores = scores_for(run, "Score Only")
    assert metric_scores.scores == [0.8, 0.6]
    assert counts(metric_scores) == (0, 0, 0)
    assert valid_scores == 2


def test_score_only_conversational_metric_keeps_its_scores():
    case = ConversationalApiTestCase(
        name="conversation", success=True, metricsData=[], order=0
    )
    case.update_metric_data(
        MetricData(name="Score Only", score=0.5, success=None)
    )
    run = TestRun()
    run.conversational_test_cases = [case]
    run.construct_metrics_scores()

    metric_scores = scores_for(run, "Score Only")
    assert metric_scores.scores == [0.5]
    assert counts(metric_scores) == (0, 0, 0)


def test_verdicts_still_count_as_passes_and_fails():
    run = make_run(
        [MetricData(name="Gate", score=0.9, success=True)],
        [MetricData(name="Gate", score=0.1, success=False)],
    )
    run.construct_metrics_scores()

    metric_scores = scores_for(run, "Gate")
    assert metric_scores.scores == [0.9, 0.1]
    assert counts(metric_scores) == (1, 1, 0)


def test_metric_without_a_score_is_still_an_error():
    run = make_run(
        [MetricData(name="Score Only", success=None, error="boom")],
        [MetricData(name="Gate", success=False, error="boom")],
    )
    valid_scores = run.construct_metrics_scores()

    for name in ("Score Only", "Gate"):
        metric_scores = scores_for(run, name)
        assert metric_scores.scores == []
        assert counts(metric_scores) == (0, 0, 1)
    assert valid_scores == 0


def test_score_only_span_metric_keeps_its_scores():
    span = BaseApiSpan(
        uuid="span",
        name="retrieve",
        status=TraceSpanApiStatus.SUCCESS,
        type=SpanApiType.RETRIEVER,
        startTime=TIME,
        endTime=TIME,
        metricsData=[MetricData(name="Score Only", score=0.7, success=None)],
    )
    trace = TraceApi(
        uuid="trace",
        baseSpans=[],
        agentSpans=[],
        llmSpans=[],
        retrieverSpans=[span],
        toolSpans=[],
        startTime=TIME,
        endTime=TIME,
    )
    run = make_run([], trace=trace)
    run.construct_metrics_scores()

    span_scores = run.trace_metrics_scores.retriever["retrieve"]["Score Only"]
    assert span_scores.scores == [0.7]
    assert counts(span_scores) == (0, 0, 0)
    assert scores_for(run, "Score Only").scores == [0.7]


@pytest.mark.parametrize("run_async", [True, False])
def test_evaluate_saves_score_only_metric_scores(run_async):
    evaluate(
        [
            LLMTestCase(input="a", actual_output="yes", expected_output="yes"),
            LLMTestCase(input="b", actual_output="no", expected_output="no"),
        ],
        [ExactMatchMetric(), PatternMatchMetric(pattern="y.*", threshold=None)],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    run = global_test_run_manager.get_latest_test_run_data()
    pattern_scores = scores_for(run, "Pattern Match")
    assert sorted(pattern_scores.scores) == [0.0, 1.0]
    assert counts(pattern_scores) == (0, 0, 0)
    assert counts(scores_for(run, "Exact Match")) == (2, 0, 0)
