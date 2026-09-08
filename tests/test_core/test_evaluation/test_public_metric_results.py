import pytest

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager


@pytest.fixture(autouse=True)
def _reset_test_run_manager():
    global_test_run_manager.reset()
    yield
    global_test_run_manager.reset()


def test_evaluate_exposes_per_case_metric_results_offline():
    result = evaluate(
        test_cases=[
            LLMTestCase(
                input="matching",
                actual_output="same",
                expected_output="same",
            ),
            LLMTestCase(
                input="different",
                actual_output="actual",
                expected_output="expected",
            ),
        ],
        metrics=[ExactMatchMetric()],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 2

    matching_result, different_result = result.test_results
    assert matching_result.input == "matching"
    assert different_result.input == "different"
    assert matching_result.metrics_data is not None
    assert different_result.metrics_data is not None

    matching_metric = matching_result.metrics_data[0]
    different_metric = different_result.metrics_data[0]
    assert matching_metric.name == "Exact Match"
    assert matching_metric.score == 1.0
    assert matching_metric.success is True
    assert different_metric.name == "Exact Match"
    assert different_metric.score == 0.0
    assert different_metric.success is False
    assert isinstance(matching_metric.reason, str)
    assert isinstance(different_metric.reason, str)
