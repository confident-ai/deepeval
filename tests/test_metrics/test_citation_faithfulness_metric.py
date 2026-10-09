"""Tests for CitationFaithfulnessMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They prove the metric FAILS the misattribution case (a claim cited to a passage
that does not support it, even though another passage would) and PASSES the
correctly-cited case.
"""

import asyncio
from unittest.mock import patch

import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.community import CitationFaithfulnessMetric
from deepeval.metrics.community.citation_faithfulness.schema import (
    CitationFaithfulnessVerdict,
)
from deepeval.metrics.utils.system_one_batch import measure_system_one_batch
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from tests.test_metrics.system_one_fakes import (
    FakeSystemOneModel,
    answer_everything,
)


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
    assert metric._system_one_eval_spec(test_case) is not None


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


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("citation", ["[0]", "[-1]", "[3]"])
def test_invalid_citation_fails_without_judge(async_mode, citation):
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(model=judge, async_mode=async_mode)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output=f"The Eiffel Tower was completed in 1889 {citation}.",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    with patch(
        "deepeval.metrics.community.citation_faithfulness.citation_faithfulness.run_system_one_eval"
    ) as system_one, patch(
        "deepeval.metrics.community.citation_faithfulness.citation_faithfulness.a_run_system_one_eval"
    ) as a_system_one:
        assert metric.measure(test_case, _show_indicator=False) == 0.0

    assert metric.is_successful() is False
    assert citation in metric.reason
    assert judge.last_prompt is None
    assert metric._system_one_eval_spec(test_case) is None
    system_one.assert_not_called()
    a_system_one.assert_not_called()


def test_invalid_citation_direct_async_respects_include_reason():
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(
        model=judge, include_reason=False, async_mode=True
    )
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower was completed in 1889 [2] and [7].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    score = asyncio.run(metric.a_measure(test_case, _show_indicator=False))
    assert score == 0.0
    assert metric.reason is None
    assert metric.is_successful() is False
    assert judge.last_prompt is None


def test_reason_lists_each_invalid_citation_once():
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The answer cites [3], [0], [3], and valid [1].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    assert metric.measure(test_case, _show_indicator=False) == 0.0
    assert "[0], [3]" in metric.reason
    assert judge.last_prompt is None


def test_invalid_citation_skips_system_one_batch():
    system_one = FakeSystemOneModel(answer_fn=answer_everything())
    citation = CitationFaithfulnessMetric(
        system_one_model=system_one, eval_mode="system_one", async_mode=False
    )
    relevancy = AnswerRelevancyMetric(
        system_one_model=system_one, eval_mode="system_one"
    )
    test_case = LLMTestCase(
        input=QUERY,
        actual_output="The Eiffel Tower was completed in 1889 [3].",
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    handled = measure_system_one_batch(
        [citation, relevancy],
        test_case,
        ignore_errors=False,
        skip_on_missing_params=False,
    )

    assert handled == [relevancy]
    assert len(system_one.calls) == 1
    assert citation.measure(test_case, _show_indicator=False) == 0.0
    assert len(system_one.calls) == 1
