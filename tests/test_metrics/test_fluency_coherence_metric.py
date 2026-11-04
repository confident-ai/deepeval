import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import FluencyCoherenceMetric
from deepeval.metrics.community.fluency_coherence.schema import (
    FluencyJudgment,
)
from deepeval.test_case import LLMTestCase


def _test_case():
    return LLMTestCase(
        input="Tell me about Paris.",
        actual_output="Paris is lovely in spring.",
    )


class TestFluencyCoherenceMetric:
    """LLM-judged; only key-free paths run without calling the model."""

    def test_prompt_contains_answer(self):
        prompt = FluencyCoherenceMetric._build_prompt(_test_case())
        assert "Paris is lovely in spring." in prompt
        assert '"score"' in prompt

    def test_fill_normalizes_five_point_scale(self):
        metric = FluencyCoherenceMetric(threshold=0.5, model="gpt-3.5-turbo")
        metric._fill(FluencyJudgment(score=5.0, reason="polished"))
        assert metric.score == 1.0
        assert "5.0/5" in metric.reason
        metric._fill(FluencyJudgment(score=1.0, reason="broken"))
        assert metric.score == 0.0
        assert metric.is_successful() is False

    def test_fill_clamps_out_of_range(self):
        metric = FluencyCoherenceMetric(model="gpt-3.5-turbo")
        metric._fill(FluencyJudgment(score=9.0, reason="x"))
        assert metric.score == 1.0

    def test_missing_actual_output_raises(self):
        metric = FluencyCoherenceMetric(model="gpt-3.5-turbo")
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="Tell me about Paris."))
