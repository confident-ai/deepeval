"""Regression coverage for hyperparameters used by evaluation cache keys."""

import importlib
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager


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

    monkeypatch.setattr(evaluation_module, "execute_test_cases", capture_execution)
    monkeypatch.setattr(evaluation_module, "a_execute_test_cases", capture_async_execution)
    monkeypatch.setattr(
        evaluation_module,
        "capture_evaluation_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        global_test_run_manager, "wrap_up_test_run", Mock(return_value=(None, None))
    )
    monkeypatch.setattr(global_test_run_manager, "save_test_run", Mock())
    monkeypatch.setattr(evaluation_module, "get_is_running_deepeval", lambda: True)

    try:
        evaluation_module.evaluate(
            [LLMTestCase(input="question", actual_output="answer")],
            metrics=[_PassingMetric()],
            hyperparameters={"model": "first", "temperature": 0.4},
            async_config=AsyncConfig(run_async=run_async),
            display_config=DisplayConfig(show_indicator=False, print_results=False),
        )
        assert observed == [{"model": "first", "temperature": "0.4"}]
    finally:
        global_test_run_manager.reset()
