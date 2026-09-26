import asyncio
import time

import pytest

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import execute as execute_module
from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
    CacheConfig,
    AsyncConfig,
)


class _DelayByInputMetric(BaseMetric):
    """
    Stub metric whose async delay (in seconds) is driven by the test case's
    input value. With concurrent execution, tasks that ask for a shorter delay
    finish before longer ones, so completion order differs from input order.
    """

    threshold = 0.0
    name = "_DelayByInputMetric"

    def __init__(self):
        self.skipped = False
        self.error = None
        self.success = False
        self.score = None
        self.reason = None

    def measure(self, test_case, *_args, **_kwargs):
        time.sleep(float(test_case.input))
        self.success = True

    async def a_measure(self, test_case, *_args, **_kwargs):
        await asyncio.sleep(float(test_case.input))
        self.success = True

    def is_successful(self) -> bool:
        return bool(self.success)


@pytest.mark.asyncio
async def test_async_results_preserve_input_order():
    """
    Regression test for #1034: a_execute_test_cases must return results in the
    same order as the input test cases, not in completion order. Here the 0.1s
    case finishes well before the 0.3s one, so an executor that collects results
    as tasks complete would return a reversed list.
    """
    inputs = ["0.3", "0.2", "0.1"]
    test_cases = [LLMTestCase(input=s, actual_output="out" + s) for s in inputs]

    metric = _DelayByInputMetric()

    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(ignore_errors=True, skip_on_missing_params=False)
    async_config = AsyncConfig(max_concurrent=3, throttle_value=0)

    results = await execute_module.a_execute_test_cases(
        test_cases=test_cases,
        metrics=[metric],
        error_config=error_config,
        display_config=display_config,
        cache_config=cache_config,
        async_config=async_config,
    )

    assert [r.input for r in results] == inputs
