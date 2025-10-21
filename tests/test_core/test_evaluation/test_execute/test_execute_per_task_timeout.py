# tests/test_execute_per_task_timeout_integration.py
import asyncio
import pytest
from deepeval.evaluate import execute as execute_module
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
    CacheConfig,
    AsyncConfig,
)
from tests.test_core.stubs import _SleepyMetric


@pytest.mark.asyncio
async def test_per_task_timeout_via_a_execute_test_cases(settings):
    """Test that per-task timeout works in a_execute_test_cases"""
    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS = 2

    tc = LLMTestCase(input="hello", actual_output="test")
    metric = _SleepyMetric(async_sleep=10)

    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(
        ignore_errors=False, skip_on_missing_params=False
    )

    # The timeout should be raised from within execute_with_semaphore
    with pytest.raises(asyncio.TimeoutError):
        await execute_module.a_execute_test_cases(
            test_cases=[tc],
            metrics=[metric],
            error_config=error_config,
            display_config=display_config,
            cache_config=cache_config,
            async_config=async_config,
        )
