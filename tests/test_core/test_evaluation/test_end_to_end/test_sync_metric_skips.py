from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import (
    AgentResponsivenessMetric,
    ExactMatchMetric,
    PatternMatchMetric,
    TurnTakingNaturalnessMetric,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn


def _evaluate_sync(test_cases, metrics, **kwargs):
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
        **kwargs,
    ).test_results


def test_metric_that_skips_measurement_does_not_fail_the_case():
    case = ConversationalTestCase(turns=[Turn(role="user", content="Goodbye")])
    [result] = _evaluate_sync(
        [case], [AgentResponsivenessMetric(), TurnTakingNaturalnessMetric()]
    )

    assert result.success is True
    assert [row.name for row in result.metrics_data] == ["Agent Responsiveness"]


def test_skipping_a_metric_does_not_hide_its_next_measurement():
    first, second = _evaluate_sync(
        [
            LLMTestCase(input="First", actual_output="yes"),
            LLMTestCase(
                input="Second", actual_output="yes", expected_output="no"
            ),
        ],
        [PatternMatchMetric("yes"), ExactMatchMetric()],
        error_config=ErrorConfig(skip_on_missing_params=True),
    )

    assert [row.name for row in first.metrics_data] == ["Pattern Match"]
    assert second.success is False
    assert [row.name for row in second.metrics_data] == [
        "Pattern Match",
        "Exact Match",
    ]
