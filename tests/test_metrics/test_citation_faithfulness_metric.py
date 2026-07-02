"""Tests for CitationFaithfulnessMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They prove the metric FAILS the misattribution case (a claim cited to a passage
that does not support it, even though another passage would) and PASSES the
correctly-cited case.
"""

from deepeval.metrics.community import CitationFaithfulnessMetric
from deepeval.metrics.community.citation_faithfulness.schema import (
    CitationFaithfulnessVerdict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class FakeJudge(DeepEvalBaseLLM):
    """Returns a preset verdict, capturing the prompt it was given."""

    def __init__(self, verdict: CitationFaithfulnessVerdict):
        self._verdict = verdict
        self.last_prompt = None
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


QUERY = "How tall is the Eiffel Tower and when was it completed?"
# Passage [1] supports the height claim; passage [2] supports the year claim.
RETRIEVAL_CONTEXT = [
    "The Eiffel Tower stands 330 metres tall in Paris.",
    "The Eiffel Tower was completed in 1889 for the World Fair.",
]


def test_fails_misattribution_case():
    # The completion-year claim is cited to passage [1], which only covers height.
    # The claim is supported elsewhere (passage [2]), so plain faithfulness would
    # pass, but attribution-aware checking must fail.
    judge = FakeJudge(
        CitationFaithfulnessVerdict(
            verdict="unfaithful",
            reasoning="The year claim is cited to [1], which only covers height.",
        )
    )
    metric = CitationFaithfulnessMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower was completed in 1889 [1].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert metric.reason is not None
    # The prompt must number the passages so [N] markers resolve.
    assert "[1] The Eiffel Tower stands 330 metres tall" in judge.last_prompt
    assert "[2] The Eiffel Tower was completed in 1889" in judge.last_prompt


def test_passes_correctly_cited_case():
    judge = FakeJudge(
        CitationFaithfulnessVerdict(
            verdict="faithful",
            reasoning="Each citation marker points to a passage that supports its claim.",
        )
    )
    metric = CitationFaithfulnessMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower is 330 metres tall [1] and was completed in 1889 [2].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert metric.reason is not None


def test_async_measure_matches_sync():
    judge = FakeJudge(
        CitationFaithfulnessVerdict(verdict="unfaithful", reasoning="bad cite")
    )
    metric = CitationFaithfulnessMetric(model=judge, async_mode=True)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower was completed in 1889 [1].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    metric.measure(test_case)

    assert metric.score == 0.0
    assert metric.is_successful() is False
