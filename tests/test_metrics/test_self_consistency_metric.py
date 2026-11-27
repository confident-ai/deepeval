import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import SelfConsistencyMetric
from deepeval.test_case import LLMTestCase


def _test_case(actual_output="Paris is the capital of France.", **kwargs):
    return LLMTestCase(
        input="What is the capital of France?",
        actual_output=actual_output,
        **kwargs,
    )


class TestSelfConsistencyMetric:
    """SelfConsistencyMetric is deterministic, no API key needed."""

    def test_identical_samples_score_one(self):
        metric = SelfConsistencyMetric()
        metric.measure(
            _test_case(),
            samples=[
                "Paris is the capital of France.",
                "Paris is the capital of France.",
            ],
        )
        assert metric.score == pytest.approx(1.0)
        assert metric.score_breakdown["n_samples"] == 3.0
        assert metric.is_successful() is True

    def test_divergent_samples_score_low(self):
        metric = SelfConsistencyMetric()
        metric.measure(
            _test_case("Paris is the capital of France."),
            samples=["Quantum field theory Lagrangians."],
        )
        assert metric.score < 0.3
        assert metric.is_successful() is False

    def test_metadata_samples_are_used(self):
        metric = SelfConsistencyMetric()
        case = _test_case(
            metadata={"samples": ["Paris is the capital of France."]}
        )
        metric.measure(case)
        assert metric.score == 1.0

    def test_single_output_rejected(self):
        metric = SelfConsistencyMetric()
        with pytest.raises(ValueError):
            metric.measure(_test_case())

    def test_missing_actual_output_raises(self):
        metric = SelfConsistencyMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(input="What is the capital?"), samples=["x"]
            )

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = SelfConsistencyMetric()
        score = await metric.a_measure(
            _test_case(), samples=["Paris is the capital of France."]
        )
        assert score == pytest.approx(1.0)
