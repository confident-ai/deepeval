"""Tests for the skip_reset parameter of evaluate()."""

import pytest
from unittest.mock import patch

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.test_run import global_test_run_manager


class _AlwaysPassMetric(BaseMetric):
    """Deterministic metric that always scores 1.0. No LLM calls."""

    def __init__(self):
        self.threshold = 0.5
        self.strict_mode = False

    @property
    def __name__(self):
        return "AlwaysPass"

    def measure(self, test_case):
        self.success = True
        self.score = 1.0
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    def is_successful(self):
        return self.success


_QUIET_DISPLAY = DisplayConfig(show_indicator=False, print_results=False)
_QUIET_ASYNC = AsyncConfig(run_async=False)


def _make_case(label: str) -> LLMTestCase:
    return LLMTestCase(input=f"input-{label}", actual_output=f"output-{label}")


@pytest.fixture(autouse=True)
def _reset_test_run_manager():
    """Ensure every test starts and ends with a clean test run manager."""
    global_test_run_manager.reset()
    yield
    global_test_run_manager.reset()


class TestSkipResetDefault:
    """skip_reset=False (default) -- each call resets state."""

    def test_second_call_does_not_accumulate(self):
        evaluate(
            test_cases=[_make_case("a")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        result = evaluate(
            test_cases=[_make_case("b")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert len(result.test_results) == 1
        assert result.test_results[0].input == "input-b"

    def test_returns_evaluation_result(self):
        result = evaluate(
            test_cases=[_make_case("x")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert isinstance(result, EvaluationResult)
        assert len(result.test_results) == 1


class TestSkipResetTrue:
    """skip_reset=True -- results accumulate across calls."""

    def test_accumulates_test_cases(self):
        result1 = evaluate(
            test_cases=[_make_case("1")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        result2 = evaluate(
            test_cases=[_make_case("2")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert len(result1.test_results) == 1
        assert len(result2.test_results) == 1
        # The underlying test run has accumulated both
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 2

    def test_three_calls_accumulate(self):
        for i in range(3):
            evaluate(
                test_cases=[_make_case(str(i))],
                metrics=[_AlwaysPassMetric()],
                skip_reset=(i > 0),
                display_config=_QUIET_DISPLAY,
                async_config=_QUIET_ASYNC,
            )
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 3

    def test_skip_reset_true_skips_wrap_up(self):
        with patch.object(
            global_test_run_manager, "wrap_up_test_run"
        ) as mock_wrap_up:
            evaluate(
                test_cases=[_make_case("a")],
                metrics=[_AlwaysPassMetric()],
                skip_reset=True,
                display_config=_QUIET_DISPLAY,
                async_config=_QUIET_ASYNC,
            )
            mock_wrap_up.assert_not_called()

    def test_skip_reset_false_calls_wrap_up(self):
        with patch.object(
            global_test_run_manager,
            "wrap_up_test_run",
            return_value=None,
        ) as mock_wrap_up:
            evaluate(
                test_cases=[_make_case("a")],
                metrics=[_AlwaysPassMetric()],
                display_config=_QUIET_DISPLAY,
                async_config=_QUIET_ASYNC,
            )
            mock_wrap_up.assert_called_once()

    def test_skip_reset_true_returns_no_confident_link(self):
        result = evaluate(
            test_cases=[_make_case("a")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert result.confident_link is None
        assert result.test_run_id is None

    def test_hyperparameters_not_erased_by_subsequent_none(self):
        evaluate(
            test_cases=[_make_case("1")],
            metrics=[_AlwaysPassMetric()],
            hyperparameters={"model": "gpt-4"},
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        evaluate(
            test_cases=[_make_case("2")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        test_run = global_test_run_manager.get_test_run()
        assert test_run.hyperparameters is not None

    def test_run_duration_accumulates(self):
        evaluate(
            test_cases=[_make_case("1")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        duration_after_first = (
            global_test_run_manager.get_test_run().run_duration
        )
        evaluate(
            test_cases=[_make_case("2")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        duration_after_second = (
            global_test_run_manager.get_test_run().run_duration
        )
        assert duration_after_second > duration_after_first
        assert duration_after_first > 0

    def test_skip_reset_true_as_very_first_call(self):
        result = evaluate(
            test_cases=[_make_case("first")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert len(result.test_results) == 1
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 1


class TestAccumulatedOrdersAreUnique:
    """Accumulated test cases must have unique sequential orders after sort."""

    def test_sort_assigns_unique_orders_after_accumulation(self):
        """Multiple evaluate() calls start their order counters from 0.
        sort_test_cases() must re-number so Confident AI sees no duplicates."""
        evaluate(
            test_cases=[_make_case("a1"), _make_case("a2")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        evaluate(
            test_cases=[_make_case("b1"), _make_case("b2")],
            metrics=[_AlwaysPassMetric()],
            skip_reset=True,
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 4

        test_run.sort_test_cases()
        orders = [tc.order for tc in test_run.test_cases]
        assert orders == list(range(4)), f"Expected unique [0,1,2,3], got {orders}"

    @patch(
        "deepeval.evaluate.evaluate.get_is_running_deepeval", return_value=True
    )
    def test_cli_mode_orders_unique_across_files(self, _mock):
        """Simulates two test files run via 'deepeval test run'."""
        evaluate(
            test_cases=[_make_case("file1_a"), _make_case("file1_b"), _make_case("file1_c")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        evaluate(
            test_cases=[_make_case("file2_a"), _make_case("file2_b"), _make_case("file2_c")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 6

        test_run.sort_test_cases()
        orders = [tc.order for tc in test_run.test_cases]
        assert orders == list(range(6)), f"Expected unique [0..5], got {orders}"
        assert len(set(orders)) == len(orders), "Orders must be unique"


class TestCLIModeAutoSkipsReset:
    """When running under `deepeval test run`, evaluate() should auto-skip reset."""

    @patch(
        "deepeval.evaluate.evaluate.get_is_running_deepeval", return_value=True
    )
    def test_cli_mode_accumulates_without_explicit_skip_reset(self, _mock):
        evaluate(
            test_cases=[_make_case("a")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        evaluate(
            test_cases=[_make_case("b")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        test_run = global_test_run_manager.get_test_run()
        assert len(test_run.test_cases) == 2

    @patch(
        "deepeval.evaluate.evaluate.get_is_running_deepeval", return_value=True
    )
    def test_cli_mode_does_not_call_wrap_up(self, _mock):
        with patch.object(
            global_test_run_manager, "wrap_up_test_run"
        ) as mock_wrap_up:
            evaluate(
                test_cases=[_make_case("a")],
                metrics=[_AlwaysPassMetric()],
                display_config=_QUIET_DISPLAY,
                async_config=_QUIET_ASYNC,
            )
            mock_wrap_up.assert_not_called()

    @patch(
        "deepeval.evaluate.evaluate.get_is_running_deepeval", return_value=True
    )
    def test_cli_mode_returns_no_confident_link(self, _mock):
        result = evaluate(
            test_cases=[_make_case("a")],
            metrics=[_AlwaysPassMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=_QUIET_ASYNC,
        )
        assert result.confident_link is None
        assert result.test_run_id is None
