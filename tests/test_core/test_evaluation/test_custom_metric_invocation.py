import asyncio

import pytest

from deepeval import evaluate
from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.evaluate.execute._common import _execute_metric
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import measure_metrics_with_indicator
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager


class RecordingMetric(BaseMetric):
    threshold = 0.5

    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def record(self, test_case, **kwargs):
        self.calls.append((test_case, kwargs))
        if self.failure is not None and test_case.input == "bad":
            raise self.failure
        self.score = 1.0
        self.success = True
        return self.score

    @property
    def __name__(self):
        return "Recording metric"


class LegacyMetric(RecordingMetric):
    def measure(self, test_case):
        return self.record(test_case)

    async def a_measure(self, test_case):
        return self.measure(test_case)


class IndicatorMetric(RecordingMetric):
    def measure(self, test_case, _show_indicator=True):
        return self.record(test_case, _show_indicator=_show_indicator)

    async def a_measure(self, test_case, _show_indicator=True):
        return self.measure(test_case, _show_indicator=_show_indicator)


class ComponentMetric(RecordingMetric):
    def measure(self, test_case, *, _in_component=False):
        return self.record(test_case, _in_component=_in_component)

    async def a_measure(self, test_case, *, _in_component=False):
        return self.measure(test_case, _in_component=_in_component)


class KwargsMetric(RecordingMetric):
    def measure(self, test_case, **kwargs):
        return self.record(test_case, **kwargs)

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case, **kwargs)


class ModernMetric(RecordingMetric):
    def measure(
        self, test_case, /, *, _show_indicator=True, _in_component=False
    ):
        return self.record(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    async def a_measure(
        self, test_case, /, *, _show_indicator=True, _in_component=False
    ):
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )


@pytest.fixture(params=["sync", "async", "async_indicator"])
def invoke(request):
    async def run(metric, **error_options):
        case = LLMTestCase(input="bad", actual_output="answer")
        config = ErrorConfig(**error_options)
        if request.param == "sync":
            _execute_metric(metric, case, False, True, config)
        else:
            await measure_metrics_with_indicator(
                metrics=[metric],
                test_case=case,
                cached_test_case=None,
                ignore_errors=config.ignore_errors,
                skip_on_missing_params=config.skip_on_missing_params,
                show_indicator=request.param == "async_indicator",
                _in_component=True,
            )

    return run


@pytest.mark.parametrize(
    "metric_class, expected_kwargs",
    [
        (LegacyMetric, {}),
        (IndicatorMetric, {"_show_indicator": False}),
        (ComponentMetric, {"_in_component": True}),
        (KwargsMetric, {"_show_indicator": False, "_in_component": True}),
        (ModernMetric, {"_show_indicator": False, "_in_component": True}),
    ],
)
async def test_supported_internal_parameters(
    invoke, metric_class, expected_kwargs
):
    metric = metric_class()
    await invoke(metric)
    assert len(metric.calls) == 1
    assert metric.calls[0][1] == expected_kwargs
    assert metric.score == 1.0


@pytest.mark.parametrize("metric_class", [LegacyMetric, KwargsMetric])
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_errors_are_handled_without_reinvoking(
    invoke, metric_class, error_type, ignore_errors
):
    failure = error_type("judge failed inside measure")
    metric = metric_class(failure)
    if ignore_errors:
        await invoke(metric, ignore_errors=True)
        assert "judge failed inside measure" in metric.error
        assert metric.success is False
    else:
        with pytest.raises(error_type) as caught:
            await invoke(metric)
        assert caught.value is failure
    assert len(metric.calls) == 1


@pytest.mark.parametrize("metric_class", [LegacyMetric, KwargsMetric])
@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_missing_parameters_still_skip(
    invoke, metric_class, ignore_errors
):
    metric = metric_class(MissingTestCaseParamsError("missing context"))
    await invoke(
        metric,
        ignore_errors=ignore_errors,
        skip_on_missing_params=True,
    )
    assert metric.skipped is True
    assert len(metric.calls) == 1


@pytest.mark.parametrize("metric_class", [LegacyMetric, KwargsMetric])
async def test_cancellation_is_not_retried(invoke, metric_class):
    cancellation = asyncio.CancelledError()
    metric = metric_class(cancellation)
    with pytest.raises(asyncio.CancelledError) as caught:
        await invoke(metric)
    assert caught.value is cancellation
    assert len(metric.calls) == 1


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_evaluate_continues_after_legacy_metric_error(
    monkeypatch, run_async, error_type
):
    monkeypatch.setattr(
        global_test_run_manager, "wrap_up_test_run", lambda *a, **k: None
    )
    cases = [
        LLMTestCase(input="bad", actual_output="answer"),
        LLMTestCase(input="good", actual_output="answer"),
    ]
    try:
        result = evaluate(
            cases,
            metrics=[LegacyMetric(error_type("judge failed"))],
            async_config=AsyncConfig(run_async=run_async),
            error_config=ErrorConfig(ignore_errors=True),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=False,
                inspect_after_run=False,
            ),
        )
        by_input = {row.input: row for row in result.test_results}
        assert len(by_input) == 2
        assert by_input["bad"].success is False
        assert "judge failed" in by_input["bad"].metrics_data[0].error
        assert by_input["good"].success is True
        assert by_input["good"].metrics_data[0].error is None
    finally:
        global_test_run_manager.reset()
