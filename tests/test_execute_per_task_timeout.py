# tests/test_execute_per_task_timeout_integration.py
import asyncio
import pytest
from deepeval.evaluate import execute as execute_module
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.evaluate.configs import ErrorConfig, DisplayConfig, CacheConfig, AsyncConfig


class StalledMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.5
        self.score = 0
        self.success = False
        self.reason = ""
        self.error = None
        self.skipped = False
    
    async def a_measure(self, test_case, _show_indicator=False, _in_component=False):
        """Async measure that stalls"""
        await asyncio.sleep(9999)
        self.score = 1.0
        self.success = True
    
    def measure(self, test_case, _show_indicator=False, _in_component=False):
        """Sync measure - not used in async tests"""
        pass
    
    def is_successful(self):
        return self.success
    
    @property
    def __name__(self):
        return "stalled_metric"


@pytest.mark.asyncio
async def test_per_task_timeout_via_a_execute_test_cases(monkeypatch):
    """Test that per-task timeout works in a_execute_test_cases"""
    # Import settings to patch it
    from deepeval.config.settings import get_settings
    settings = get_settings()
    
    # Set small timeout for the test
    monkeypatch.setattr(settings, "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS", 2)
    
    tc = LLMTestCase(input="hello", actual_output="test")
    metric = StalledMetric()
    
    async_config = AsyncConfig(max_concurrent=1, throttle_value=0)
    display_config = DisplayConfig(show_indicator=False, verbose_mode=False)
    cache_config = CacheConfig(write_cache=False, use_cache=False)
    error_config = ErrorConfig(ignore_errors=False, skip_on_missing_params=False)
    
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
