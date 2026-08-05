"""Tests for how a test case's success is aggregated from per-metric results.

A test case's `success` is folded together in `LLMApiTestCase.update_metric_data`
and `update_status`: the first metric's success initializes it, `None` never
changes an existing verdict, and any explicit `False` permanently fails the
test case.
"""

import pytest

from deepeval.evaluate.utils import create_metric_data
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.api import LLMApiTestCase
from deepeval.tracing.api import MetricData


def make_api_test_case() -> LLMApiTestCase:
    return LLMApiTestCase(name="test case", input="input", order=0)


def make_metric_data(success, flaky: bool = False) -> MetricData:
    return MetricData(name="Metric", success=success, flaky=flaky)


def test_first_metric_data_initializes_success():
    for first_success in (True, False, None):
        api_test_case = make_api_test_case()
        api_test_case.update_metric_data(make_metric_data(first_success))
        assert api_test_case.success is first_success


def test_none_success_does_not_change_existing_verdict():
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(True))
    api_test_case.update_metric_data(make_metric_data(None))
    assert api_test_case.success is True

    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(False))
    api_test_case.update_metric_data(make_metric_data(None))
    assert api_test_case.success is False


def test_false_success_permanently_fails_test_case():
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(True))
    api_test_case.update_metric_data(make_metric_data(False))
    api_test_case.update_metric_data(make_metric_data(True))
    api_test_case.update_metric_data(make_metric_data(None))
    assert api_test_case.success is False


def test_true_success_after_none_sets_verdict():
    # A no-threshold metric processed first leaves success as None; the
    # first metric with a real verdict then decides it.
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(None))
    api_test_case.update_metric_data(make_metric_data(True))
    assert api_test_case.success is True


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ([True], True),
        ([None], None),
        ([None, True], True),
        ([True, None], True),
        ([True, False, True], False),
        ([None, False, None], False),
    ],
)
def test_update_status_has_same_aggregation_semantics(statuses, expected):
    api_test_case = make_api_test_case()
    for status in statuses:
        api_test_case.update_status(status)
    assert api_test_case.success is expected


def test_flaky_metric_verdict_never_decides_test_case():
    # A failing flaky metric doesn't fail the test case
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(True))
    api_test_case.update_metric_data(make_metric_data(False, flaky=True))
    assert api_test_case.success is True

    # A flaky metric processed first doesn't initialize success either
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(make_metric_data(False, flaky=True))
    assert api_test_case.success is None
    api_test_case.update_metric_data(make_metric_data(True))
    assert api_test_case.success is True

    # But its data is still recorded
    assert len(api_test_case.metrics_data) == 2
    assert api_test_case.metrics_data[0].success is False
    assert api_test_case.metrics_data[0].flaky is True


def test_update_status_skips_flaky_verdicts():
    api_test_case = make_api_test_case()
    api_test_case.update_status(True)
    api_test_case.update_status(False, flaky=True)
    assert api_test_case.success is True

    api_test_case = make_api_test_case()
    api_test_case.update_status(False, flaky=True)
    assert api_test_case.success is None


def test_no_threshold_metric_does_not_contribute_to_test_case_success():
    # End to end: is_successful() -> create_metric_data -> update_metric_data.
    test_case = LLMTestCase(
        input="input", actual_output="output", expected_output="output"
    )

    no_threshold_metric = ExactMatchMetric(threshold=None)
    no_threshold_metric.measure(test_case, _show_indicator=False)
    failing_metric = ExactMatchMetric()
    failing_metric.measure(
        LLMTestCase(
            input="input", actual_output="output", expected_output="different"
        ),
        _show_indicator=False,
    )
    passing_metric = ExactMatchMetric()
    passing_metric.measure(test_case, _show_indicator=False)

    # No-threshold metric alone leaves the test case undecided
    api_test_case = make_api_test_case()
    api_test_case.update_metric_data(create_metric_data(no_threshold_metric))
    assert api_test_case.success is None

    # A passing threshold metric decides it, unaffected by the None
    api_test_case.update_metric_data(create_metric_data(passing_metric))
    assert api_test_case.success is True

    # A failing threshold metric fails it
    api_test_case.update_metric_data(create_metric_data(failing_metric))
    assert api_test_case.success is False
