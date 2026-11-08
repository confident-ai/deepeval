"""Regression coverage for hyperparameters used by evaluation cache keys."""

import importlib
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.cache import global_test_run_cache_manager


class _PassingMetric(BaseMetric):
    threshold = 1

    @property
    def __name__(self):
        return "PassingMetric"

    def measure(self, test_case, **kwargs):
        self.score = 1
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)


@pytest.mark.parametrize("run_async", [False, True])
def test_evaluate_sets_hyperparameters_before_execution(monkeypatch, run_async):
    evaluation_module = importlib.import_module("deepeval.evaluate.evaluate")
    global_test_run_manager.reset()
    observed = []

    def capture_execution(*args, **kwargs):
        observed.append(global_test_run_manager.get_test_run().hyperparameters)
        return []

    async def capture_async_execution(*args, **kwargs):
        return capture_execution(*args, **kwargs)

    monkeypatch.setattr(
        evaluation_module, "execute_test_cases", capture_execution
    )
    monkeypatch.setattr(
        evaluation_module, "a_execute_test_cases", capture_async_execution
    )
    monkeypatch.setattr(
        evaluation_module,
        "capture_evaluation_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        global_test_run_manager,
        "wrap_up_test_run",
        Mock(return_value=(None, None)),
    )
    monkeypatch.setattr(global_test_run_manager, "save_test_run", Mock())
    monkeypatch.setattr(evaluation_module, "get_is_running_deepeval", lambda: True)

    try:
        # An accumulated test run should use the configuration for each
        # invocation, not the previous call's values during cache lookup.
        # Omitting hyperparameters should retain the accumulated run's last
        # explicit configuration (the existing _skip_reset behavior).
        for hyperparameters in (
            {"model": "first", "temperature": 0.4},
            {"model": "second", "temperature": 0.9},
            None,
        ):
            evaluation_module.evaluate(
                [LLMTestCase(input="question", actual_output="answer")],
                metrics=[_PassingMetric()],
                hyperparameters=hyperparameters,
                _skip_reset=True,
                async_config=AsyncConfig(run_async=run_async),
                display_config=DisplayConfig(
                    show_indicator=False, print_results=False
                ),
            )

        assert observed == [
            {"model": "first", "temperature": "0.4"},
            {"model": "second", "temperature": "0.9"},
            {"model": "second", "temperature": "0.9"},
        ]
    finally:
        global_test_run_manager.reset()


class _CacheProbeMetric(BaseMetric):
    """Deterministic metric for exercising the real executor without an LLM."""

    def __init__(self):
        self.threshold = 0.5
        self.strict_mode = False

    @property
    def __name__(self):
        return "CacheProbe"

    def measure(self, test_case):
        self.score = 1.0
        self.success = True
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    def is_successful(self):
        return self.success


@pytest.mark.parametrize("run_async", [False, True])
def test_evaluate_cache_reads_and_writes_use_current_hyperparameters(
    monkeypatch, run_async
):
    """Cache lookups and both writes must use each invocation's settings."""
    evaluation_module = importlib.import_module("deepeval.evaluate.evaluate")
    global_test_run_manager.reset()
    observed = []
    original_disabled = global_test_run_cache_manager.disable_write_cache

    def record_lookup(test_case, hyperparameters):
        observed.append(("lookup", dict(hyperparameters), False))
        return None

    def record_write(test_case, cached_test_case, hyperparameters, to_temp=False):
        observed.append(("write", dict(hyperparameters), to_temp))

    monkeypatch.setattr(
        global_test_run_cache_manager, "get_cached_test_case", record_lookup
    )
    monkeypatch.setattr(global_test_run_cache_manager, "cache_test_case", record_write)
    monkeypatch.setattr(
        evaluation_module,
        "capture_evaluation_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        evaluation_module, "get_is_running_deepeval", lambda: True
    )
    monkeypatch.setattr(global_test_run_manager, "save_test_run", Mock())

    try:
        for model in ("first", "second"):
            evaluation_module.evaluate(
                [LLMTestCase(input="question", actual_output="answer")],
                metrics=[_CacheProbeMetric()],
                hyperparameters={"model": model},
                _skip_reset=True,
                async_config=AsyncConfig(run_async=run_async),
                display_config=DisplayConfig(
                    show_indicator=False, print_results=False
                ),
                cache_config=CacheConfig(use_cache=True),
            )

        assert observed == [
            ("lookup", {"model": "first"}, False),
            ("write", {"model": "first"}, False),
            ("write", {"model": "first"}, True),
            ("lookup", {"model": "second"}, False),
            ("write", {"model": "second"}, False),
            ("write", {"model": "second"}, True),
        ]
    finally:
        global_test_run_cache_manager.disable_write_cache = original_disabled
        global_test_run_manager.reset()
