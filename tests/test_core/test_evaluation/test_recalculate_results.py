"""Tests for ``EvaluationResult.recalculate_results``.

The recalculation replays the scores already produced by ``evaluate`` against
new thresholds, so these tests run fully offline (no models involved).
"""

from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.test_run.api import MetricData


def _metric(
    name: str, score, threshold: float, error: str = None
) -> MetricData:
    return MetricData(
        name=name,
        threshold=threshold,
        success=(error is None and score is not None and score >= threshold),
        score=score,
        reason="because",
        error=error,
    )


def _evaluation_result() -> EvaluationResult:
    # Faithfulness fails at 0.9 (score 0.6); Relevancy passes at 0.5.
    test_result_1 = TestResult(
        name="tc_1",
        success=False,
        metrics_data=[
            _metric("Faithfulness", 0.6, 0.9),
            _metric("Relevancy", 0.8, 0.5),
        ],
        conversational=False,
    )
    # Faithfulness passes at 0.9 (score 0.95); Relevancy passes at 0.5.
    test_result_2 = TestResult(
        name="tc_2",
        success=True,
        metrics_data=[
            _metric("Faithfulness", 0.95, 0.9),
            _metric("Relevancy", 0.7, 0.5),
        ],
        conversational=False,
    )
    return EvaluationResult(
        test_results=[test_result_1, test_result_2],
        confident_link=None,
        test_run_id="run-123",
    )


def test_lowering_threshold_flips_metric_and_case_to_pass():
    result = _evaluation_result()
    recalculated = result.recalculate_results(
        {"Faithfulness": 0.5}, print_results=False
    )

    # tc_1's Faithfulness (0.6) now passes against the lowered 0.5 threshold,
    # so the whole test case passes too.
    tc_1 = recalculated.test_results[0]
    assert tc_1.metrics_data[0].name == "Faithfulness"
    assert tc_1.metrics_data[0].threshold == 0.5
    assert tc_1.metrics_data[0].success is True
    assert tc_1.success is True


def test_raising_threshold_flips_metric_and_case_to_fail():
    result = _evaluation_result()
    recalculated = result.recalculate_results(
        {"Relevancy": 0.85}, print_results=False
    )

    # tc_2's Relevancy (0.7) now fails against the raised 0.85 threshold.
    tc_2 = recalculated.test_results[1]
    relevancy = next(m for m in tc_2.metrics_data if m.name == "Relevancy")
    assert relevancy.threshold == 0.85
    assert relevancy.success is False
    assert tc_2.success is False


def test_original_result_is_not_mutated():
    result = _evaluation_result()
    result.recalculate_results({"Faithfulness": 0.5}, print_results=False)

    original_faithfulness = result.test_results[0].metrics_data[0]
    assert original_faithfulness.threshold == 0.9
    assert original_faithfulness.success is False
    assert result.test_results[0].success is False
    # Metadata is carried over onto the returned result, not lost.
    assert result.test_run_id == "run-123"


def test_metrics_absent_from_thresholds_are_unchanged():
    result = _evaluation_result()
    recalculated = result.recalculate_results(
        {"Faithfulness": 0.5}, print_results=False
    )

    relevancy = recalculated.test_results[0].metrics_data[1]
    assert relevancy.name == "Relevancy"
    assert relevancy.threshold == 0.5
    assert relevancy.success is True


def test_errored_metric_stays_failed_regardless_of_threshold():
    test_result = TestResult(
        name="tc_error",
        success=False,
        metrics_data=[_metric("Faithfulness", None, 0.9, error="boom")],
        conversational=False,
    )
    result = EvaluationResult(
        test_results=[test_result], confident_link=None, test_run_id=None
    )

    recalculated = result.recalculate_results(
        {"Faithfulness": 0.0}, print_results=False
    )
    metric = recalculated.test_results[0].metrics_data[0]
    assert metric.success is False
    assert recalculated.test_results[0].success is False


def test_test_case_without_metrics_data_is_preserved():
    test_result = TestResult(
        name="tc_none",
        success=True,
        metrics_data=None,
        conversational=False,
    )
    result = EvaluationResult(
        test_results=[test_result], confident_link=None, test_run_id=None
    )

    recalculated = result.recalculate_results(
        {"Faithfulness": 0.5}, print_results=False
    )
    assert recalculated.test_results[0].success is True
    assert recalculated.test_results[0].metrics_data is None
