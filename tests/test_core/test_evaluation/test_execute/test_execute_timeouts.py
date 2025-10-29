import time
import asyncio
import pytest
import tenacity

from deepeval.evaluate import execute as execute_module
from deepeval.test_case import LLMTestCase
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
    A larger outer timeout, and a smaller inner timeout ensures Tenacity retries and raises RetryError.
    After exhausting attempts, the last exception is asyncio.TimeoutError.
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
    with pytest.raises(tenacity.RetryError) as ei:
        await execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        )
    dur = time.perf_counter() - t0

    last_exc = ei.value.last_attempt.exception()
    assert isinstance(last_exc, (asyncio.TimeoutError, TimeoutError))
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
    with pytest.raises(tenacity.RetryError) as err:
        run_sync()
    dur = time.perf_counter() - t0

    last_exc = err.value.last_attempt.exception()
    assert isinstance(last_exc, (asyncio.TimeoutError, TimeoutError))
    assert 2.0 <= dur <= 6.0
