import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import ContextSufficiencyMetric
from deepeval.metrics.community.context_sufficiency.schema import (
    SufficiencyJudgment,
)
from deepeval.test_case import LLMTestCase


def _test_case():
    return LLMTestCase(
        input="How tall is the Eiffel Tower?",
        actual_output="It is 330 metres tall.",
        retrieval_context=["The Eiffel Tower is 330 metres tall."],
    )


class TestContextSufficiencyMetric:
    """LLM-judged; only key-free paths run without calling the model."""

    def test_prompt_contains_question_and_passages(self):
        prompt = ContextSufficiencyMetric._build_prompt(_test_case())
        assert "How tall is the Eiffel Tower?" in prompt
        assert "330 metres" in prompt
        assert '"score"' in prompt

    def test_fill_clamps_and_sets_success(self):
        metric = ContextSufficiencyMetric(threshold=0.5, model="gpt-3.5-turbo")
        metric._fill(_test_case(), SufficiencyJudgment(score=2.0, reason="ok"))
        assert metric.score == 1.0
        assert metric.is_successful() is True
        assert metric.reason == "ok"

    def test_fill_low_score_fails(self):
        metric = ContextSufficiencyMetric(threshold=0.5, model="gpt-3.5-turbo")
        metric._fill(
            _test_case(), SufficiencyJudgment(score=0.1, reason="thin")
        )
        assert metric.score == pytest.approx(0.1)
        assert metric.is_successful() is False

    def test_missing_retrieval_context_raises(self):
        metric = ContextSufficiencyMetric(model="gpt-3.5-turbo")
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(
                    input="How tall?",
                    actual_output="Very.",
                    retrieval_context=None,
                )
            )
