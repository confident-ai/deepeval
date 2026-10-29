import time
import asyncio
import threading
import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate import execute as execute_module
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.test_run.test_run import TestRun, TestRunManager
from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
    CacheConfig,
    AsyncConfig,
)
from tests.test_core.stubs import _SleepyMetric, _PerAttemptTimeoutMetric


@pytest.mark.asyncio
async def test_per_task_timeout_async_path(settings):
    """
    Outer, per-task, timeout budget enforced by the async executor via _await_with_outer_deadline.
    Disable inner per-attempt timeout so the outer timeout exceeds first.
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 2
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = None
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _SleepyMetric(sleep_s=10)

    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    with pytest.raises(asyncio.TimeoutError):
        await execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        )


def test_per_task_timeout_sync_path(settings):
    """
    Same outer per-task semantics via the sync executor.
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 2
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = None
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _SleepyMetric(sleep_s=10)

    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        execute_module.execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
        )


@pytest.mark.asyncio
async def test_per_attempt_timeout_async_path(settings):
    """
    Per-attempt timeout enforced inside retry decorator via asyncio.wait_for.
    A larger outer timeout, and a smaller inner timeout ensures Tenacity retries.
    After exhausting attempts, the last attempt's asyncio.TimeoutError is re-raised.
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 20
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 2

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _PerAttemptTimeoutMetric(sleep_s=10)

    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    t0 = time.perf_counter()
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        )
    dur = time.perf_counter() - t0

    # Ballpark duration: ~ 1s (first attempt) + backoff (~1.x s) + 1s (second attempt)
    assert 2.0 <= dur <= 6.0


def test_per_attempt_timeout_sync_path(settings):
    """
    Same per-attempt semantics, but through the sync code path that uses
    run_sync_with_timeout inside the retry decorator.
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 20
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 2

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _PerAttemptTimeoutMetric(sleep_s=10)

    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)

    def run_sync():
        execute_module.execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
        )

    t0 = time.perf_counter()
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        run_sync()
    dur = time.perf_counter() - t0

    assert 2.0 <= dur <= 6.0


@pytest.mark.asyncio
async def test_disable_timeouts_disables_per_task_async(settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_DISABLE_TIMEOUTS = True
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = (
            0.1  # would normally trip
        )
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _SleepyMetric(sleep_s=0.2)

    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    # the test itself must not hang
    await asyncio.wait_for(
        execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        ),
        timeout=2.0,
    )


def test_disable_timeouts_disables_per_task_sync(settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_DISABLE_TIMEOUTS = True
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.1
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _SleepyMetric(sleep_s=0.2)

    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    execute_module.execute_test_cases(
        test_cases=[tc],
        metrics=[metric],
        error_config=error_config,
        display_config=display_config,
        cache_config=cache_config,
    )


@pytest.mark.asyncio
async def test_disable_timeouts_disables_per_attempt_async(settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_DISABLE_TIMEOUTS = True
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 5
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = 0.05
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _PerAttemptTimeoutMetric(sleep_s=0.2)

    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    await asyncio.wait_for(
        execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        ),
        timeout=2.0,
    )


def test_disable_timeouts_disables_per_attempt_sync(settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_DISABLE_TIMEOUTS = True
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 5
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = 0.05
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _PerAttemptTimeoutMetric(sleep_s=0.2)

    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    execute_module.execute_test_cases(
        test_cases=[tc],
        metrics=[metric],
        error_config=error_config,
        display_config=display_config,
        cache_config=cache_config,
    )


class _Gated:
    """Scores 1.0 right away. Cases keyed in `missing` raise
    MissingTestCaseParamsError, and the case keyed "stuck" waits on `gate`
    like a judge call that outlives the per-task timeout."""

    threshold = 0.5

    def __init__(self, gate: threading.Event, missing=()):
        self.gate = gate
        self.missing = missing

    def measure(self, test_case, *_args, **_kwargs):
        if isinstance(test_case, LLMTestCase):
            key = test_case.input
        else:
            key = test_case.turns[0].content
        if key in self.missing:
            raise MissingTestCaseParamsError("missing params")
        if key == "stuck":
            self.gate.wait(30)
        self.score = 1.0
        self.reason = f"scored {key}"
        self.verbose_logs = f"logs for {key}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "gated"


class _GatedMetric(_Gated, BaseMetric):
    pass


class _GatedConversationalMetric(_Gated, BaseConversationalMetric):
    pass


@pytest.fixture
def gate():
    gate = threading.Event()
    yield gate
    # let the abandoned worker thread finish, after the assertions
    gate.set()


def _make_test_run_manager() -> TestRunManager:
    trm = TestRunManager()
    trm.save_to_disk = False
    trm.set_test_run(TestRun(identifier="sync-timeout-metric-state"))
    return trm


def _run_sync_with_stuck_second_case(
    settings, trm, gate, kind="llm", missing=(), ignore_errors=True
):
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = None
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1

    if kind == "llm":
        test_cases = [
            LLMTestCase(input=key, actual_output="a")
            for key in ("first", "stuck")
        ]
        metric = _GatedMetric(gate, missing=missing)
    else:
        test_cases = [
            ConversationalTestCase(turns=[Turn(role="user", content=key)])
            for key in ("first", "stuck")
        ]
        metric = _GatedConversationalMetric(gate, missing=missing)

    return execute_module.execute_test_cases(
        test_cases=test_cases,
        metrics=[metric],
        error_config=ErrorConfig(
            ignore_errors=ignore_errors, skip_on_missing_params=True
        ),
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        test_run_manager=trm,
    )


def _recorded_stuck_case(trm, kind="llm"):
    test_run = trm.get_test_run()
    if kind == "llm":
        return test_run.test_cases[-1]
    return test_run.conversational_test_cases[-1]


def _assert_timed_out(metric_data):
    assert metric_data.error is not None
    assert "Timed out" in metric_data.error
    assert metric_data.score is None
    assert metric_data.success is False
    assert metric_data.verbose_logs is None


@pytest.mark.parametrize("kind", ["llm", "conversational"])
@pytest.mark.parametrize(
    "missing", [(), ("first",)], ids=["first_scored", "first_skipped"]
)
def test_sync_timeout_does_not_reuse_previous_case_result(
    settings, gate, kind, missing
):
    """
    The sync executor reuses metric instances across test cases. A case that
    times out must report the timeout, not the previous case's result.
    """
    trm = _make_test_run_manager()
    results = _run_sync_with_stuck_second_case(
        settings, trm, gate, kind, missing
    )

    stuck = results[1]
    assert stuck.success is False
    [metric_data] = stuck.metrics_data
    _assert_timed_out(metric_data)

    recorded = _recorded_stuck_case(trm, kind)
    assert recorded.success is False
    [recorded_metric_data] = recorded.metrics_data
    _assert_timed_out(recorded_metric_data)


def test_sync_timeout_records_failure_before_raising(settings, gate):
    """
    With ignore_errors=False the timeout raises, but the case is already in
    the test run that `deepeval test run` reports at the end. It must be
    recorded as a timeout, not as the previous case's pass.
    """
    trm = _make_test_run_manager()
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        _run_sync_with_stuck_second_case(
            settings, trm, gate, ignore_errors=False
        )

    recorded = _recorded_stuck_case(trm)
    assert recorded.success is False
    [metric_data] = recorded.metrics_data
    _assert_timed_out(metric_data)
