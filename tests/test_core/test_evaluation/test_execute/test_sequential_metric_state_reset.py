import time
import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute import execute_test_cases
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRun, TestRunManager


class _ContextRequiredMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.strict_mode = False

    @property
    def __name__(self):
        return "ContextRequiredMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if test_case.retrieval_context is None:
            self.error = (
                "'retrieval_context' cannot be None for the "
                "'ContextRequiredMetric' metric"
            )
            raise MissingTestCaseParamsError(self.error)
        self.score = 1.0
        self.reason = "Context present"
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


class _FailWithoutSettingScoreMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.strict_mode = False

    @property
    def __name__(self):
        return "FailWithoutSettingScoreMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if test_case.actual_output == "boom":
            raise RuntimeError("judge crashed before assigning score")
        self.score = 0.9
        self.reason = "Valid output"
        self.success = self.is_successful()
        self.verbose_logs = "step-1: passed"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


class _SlowOnSecondCaseMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.strict_mode = False

    @property
    def __name__(self):
        return "SlowOnSecondCaseMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if test_case.input == "slow":
            time.sleep(10)
        self.score = 1.0
        self.reason = "Fast pass"
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


def _make_manager(identifier: str) -> TestRunManager:
    trm = TestRunManager()
    trm.save_to_disk = False
    trm.set_test_run(TestRun(identifier=identifier))
    return trm


def test_sequential_skip_on_missing_params_does_not_leak_skipped_flag():
    """If case 1 skips a metric due to missing params and case 2 fails or
    times out, case 2 must not remain marked as skipped=True from case 1."""
    metric = _ContextRequiredMetric()
    case_missing = LLMTestCase(
        input="q1", actual_output="a1", retrieval_context=None
    )
    case_present = LLMTestCase(
        input="q2", actual_output="a2", retrieval_context=["ctx"]
    )

    results = execute_test_cases(
        test_cases=[case_missing, case_present],
        metrics=[metric],
        error_config=ErrorConfig(
            skip_on_missing_params=True, ignore_errors=True
        ),
        display_config=DisplayConfig(show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        test_run_manager=_make_manager("skip-reset"),
    )

    assert len(results) == 2
    assert results[0].metrics_data == []
    assert len(results[1].metrics_data) == 1
    assert results[1].metrics_data[0].score == 1.0
    assert results[1].metrics_data[0].success is True
    assert metric.skipped is False


def test_sequential_error_does_not_leak_previous_case_verbose_logs_or_score():
    """When case 1 succeeds and case 2 errors under ignore_errors=True,
    case 2's MetricData must not inherit case 1's verbose_logs or stale state."""
    metric = _FailWithoutSettingScoreMetric()
    case_ok = LLMTestCase(input="q1", actual_output="good")
    case_err = LLMTestCase(input="q2", actual_output="boom")

    results = execute_test_cases(
        test_cases=[case_ok, case_err],
        metrics=[metric],
        error_config=ErrorConfig(ignore_errors=True),
        display_config=DisplayConfig(show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        test_run_manager=_make_manager("error-state-leak"),
    )

    assert len(results) == 2
    first_md = results[0].metrics_data[0]
    second_md = results[1].metrics_data[0]

    assert first_md.score == 0.9
    assert first_md.success is True
    assert first_md.verbose_logs == "step-1: passed"

    assert second_md.score is None
    assert second_md.success is False
    assert "judge crashed" in second_md.error
    assert second_md.verbose_logs is None


def test_sequential_timeout_on_second_case_does_not_reuse_first_case_pass(
    settings,
):
    """When case 1 passes (setting metric.success = True and metric.score = 1.0)
    and case 2 times out before finishing, case 2 must be marked as failed with
    a timeout error rather than inheriting case 1's passing score."""
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.05

    metric = _SlowOnSecondCaseMetric()
    case_fast = LLMTestCase(input="fast", actual_output="ok")
    case_slow = LLMTestCase(input="slow", actual_output="ok")

    results = execute_test_cases(
        test_cases=[case_fast, case_slow],
        metrics=[metric],
        error_config=ErrorConfig(ignore_errors=True),
        display_config=DisplayConfig(show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        test_run_manager=_make_manager("timeout-second-case"),
    )

    assert len(results) == 2
    assert results[0].metrics_data[0].score == 1.0
    assert results[0].metrics_data[0].success is True
    assert results[0].metrics_data[0].error is None

    second_md = results[1].metrics_data[0]
    assert second_md.success is False
    assert second_md.score is None
    assert second_md.error is not None
    assert "Timed out" in second_md.error
