import time
import asyncio
import pytest
import tenacity

from deepeval.evaluate import evaluate, execute as execute_module
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
    CacheConfig,
    AsyncConfig,
)
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe
from deepeval.test_run import global_test_run_manager
from deepeval.utils import get_gather_timeout
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


def test_gather_timeout_budgets_one_per_task_timeout_per_round(settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 10
        settings.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE = 2

    # omitted: a single task's budget, as before
    assert get_gather_timeout() == 12
    # 20 tasks through 4 slots is 5 rounds, 21 tasks is 6
    assert get_gather_timeout(n_tasks=20, max_concurrent=4) == 52
    assert get_gather_timeout(n_tasks=21, max_concurrent=4) == 62
    # all tasks fit in one round
    assert get_gather_timeout(n_tasks=3, max_concurrent=20) == 12
    # never below a single task's budget
    assert get_gather_timeout(n_tasks=0, max_concurrent=4) == 12

    with settings.edit(persist=False):
        settings.DEEPEVAL_DISABLE_TIMEOUTS = True
    assert get_gather_timeout(n_tasks=20, max_concurrent=4) is None


@pytest.mark.parametrize("show_indicator", [False, True])
def test_evaluate_runs_rounds_of_max_concurrent_past_one_task_budget(
    settings, show_indicator
):
    """
    8 cases x 0.5s through 2 slots is 4 rounds (~2s). Each case fits its 1s
    budget, so the batch must not be cut off at one task's budget (1.1s).
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE = 0.1

    global_test_run_manager.reset()
    result = evaluate(
        test_cases=[
            LLMTestCase(input=f"q{i}", actual_output="a") for i in range(8)
        ],
        metrics=[_SleepyMetric(sleep_s=0.5, succeed=True)],
        async_config=AsyncConfig(max_concurrent=2),
        display_config=DisplayConfig(
            show_indicator=show_indicator, print_results=False
        ),
        error_config=ErrorConfig(ignore_errors=False),
    )

    assert len(result.test_results) == 8
    assert all(r.success for r in result.test_results)


def test_evals_iterator_runs_rounds_of_max_concurrent_past_one_task_budget(
    settings,
):
    """
    Same as above for the async evals_iterator: its app phase (8 goldens x
    0.5s) and its eval phase (8 traces x 0.5s) each take 4 rounds through
    2 slots.
    """
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE = 0.1

    @observe()
    async def app(q):
        await asyncio.sleep(0.5)
        return "a"

    global_test_run_manager.reset()
    dataset = EvaluationDataset(
        goldens=[Golden(input=f"q{i}") for i in range(8)]
    )
    for golden in dataset.evals_iterator(
        metrics=[_SleepyMetric(sleep_s=0.5, succeed=True)],
        async_config=AsyncConfig(max_concurrent=2),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=False),
    ):
        dataset.evaluate(asyncio.create_task(app(golden.input)))

    test_cases = global_test_run_manager.get_test_run().test_cases
    assert len(test_cases) == 8
    assert all(tc.success for tc in test_cases)


class _IgnoresPerTaskDeadlineMetric(_SleepyMetric):
    """Swallows its per-task cancellation and keeps hanging."""

    async def a_measure(self, test_case, *args, **kwargs):
        try:
            await super().a_measure(test_case, *args, **kwargs)
        except asyncio.CancelledError:
            await super().a_measure(test_case, *args, **kwargs)


@pytest.mark.asyncio
async def test_gather_timeout_stops_tasks_that_ignore_per_task_deadline(
    settings,
):
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.5
        settings.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE = 0.2

    t0 = time.perf_counter()
    try:
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await execute_module.a_execute_test_cases(
                test_cases=[
                    LLMTestCase(input=f"q{i}", actual_output="a")
                    for i in range(4)
                ],
                metrics=[_IgnoresPerTaskDeadlineMetric(sleep_s=30)],
                async_config=AsyncConfig(max_concurrent=2),
                display_config=DisplayConfig(
                    show_indicator=False, verbose_mode=False
                ),
                cache_config=CacheConfig(write_cache=False, use_cache=False),
                error_config=ErrorConfig(ignore_errors=False),
            )
        # 2 rounds x 0.5s + 0.2s, not the 60s the metric hangs for
        assert 1.1 <= time.perf_counter() - t0 < 2
    finally:
        # the hung metrics outlive the gather; stop them leaking into later tests
        leftover = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        for t in leftover:
            t.cancel()
        await asyncio.gather(*leftover, return_exceptions=True)
