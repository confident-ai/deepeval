import importlib.util

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import (
    BleuMetric,
    PassAtKMetric,
    RougeMetric,
)
from deepeval.test_case import LLMTestCase

needs_nltk = pytest.mark.skipif(
    importlib.util.find_spec("nltk") is None, reason="nltk not installed"
)
needs_rouge = pytest.mark.skipif(
    importlib.util.find_spec("rouge_score") is None,
    reason="rouge-score not installed",
)
needs_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="numpy not installed"
)


def _test_case():
    return LLMTestCase(
        input="Summarize.",
        actual_output="The cat sat on the mat.",
        expected_output="The cat sat on the mat.",
    )


class TestClassicTextMetrics:
    """Overlap metrics are deterministic, no API key needed."""

    @needs_nltk
    def test_bleu_identical_is_one(self):
        metric = BleuMetric()
        metric.measure(_test_case())
        assert metric.score == pytest.approx(1.0)
        assert metric.is_successful() is True

    @needs_nltk
    def test_bleu_divergent_is_low(self):
        metric = BleuMetric()
        case = _test_case()
        case.actual_output = "Quantum chromodynamics confines quarks."
        metric.measure(case)
        assert metric.score < 0.3

    def test_bleu_bad_type_rejected(self):
        with pytest.raises(ValueError):
            BleuMetric(bleu_type="bleu9")

    @needs_rouge
    def test_rouge_identical_is_one(self):
        metric = RougeMetric()
        metric.measure(_test_case())
        assert metric.score == pytest.approx(1.0)

    def test_rouge_bad_type_rejected(self):
        with pytest.raises(ValueError):
            RougeMetric(score_type="rouge3")

    @needs_numpy
    def test_pass_at_k_all_correct(self):
        metric = PassAtKMetric(k=1)
        case = _test_case()
        metric.measure(case, n=5, c=5)
        assert metric.score == 1.0

    @needs_numpy
    def test_pass_at_k_partial(self):
        metric = PassAtKMetric(k=2)
        case = _test_case()
        metric.measure(case, n=5, c=1)
        assert metric.score == pytest.approx(0.4)

    @needs_numpy
    def test_pass_at_k_from_metadata(self):
        metric = PassAtKMetric(k=1)
        case = _test_case()
        case.metadata = {"pass_at_k_n": 4, "pass_at_k_c": 2}
        metric.measure(case)
        assert metric.score == pytest.approx(0.5)

    def test_pass_at_k_invalid_rejected(self):
        metric = PassAtKMetric(k=1)
        with pytest.raises(ValueError):
            metric.measure(_test_case(), n=3, c=4)
        with pytest.raises(ValueError):
            PassAtKMetric(k=0)

    def test_missing_expected_output_raises(self):
        metric = BleuMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="Summarize.", actual_output="Hi."))

    @needs_rouge
    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = RougeMetric()
        score = await metric.a_measure(_test_case())
        assert score == pytest.approx(1.0)
