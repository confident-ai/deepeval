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


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("verdict", ["faithful", "unfaithful"])
@pytest.mark.parametrize(
    "case_citation",
    [
        pytest.param("[2026] KEHC 11082 (KLR)", id="kenya"),
        pytest.param("[2019] UKSC 5", id="uk-supreme-court"),
        pytest.param("[2024] EWCA Civ 355", id="uk-civil-division"),
    ],
)
def test_case_citation_year_is_left_to_judge(
    async_mode, verdict, case_citation
):
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict=verdict))
    metric = CitationFaithfulnessMetric(model=judge, async_mode=async_mode)
    test_case = LLMTestCase(
        input="Which case supports the answer?",
        actual_output=f"The answer relies on {case_citation} [1].",
        retrieval_context=[f"The relevant case is {case_citation}."],
    )

    assert metric.measure(test_case, _show_indicator=False) == (
        1.0 if verdict == "faithful" else 0.0
    )
    assert test_case.actual_output in judge.last_prompt
    assert metric._system_one_eval_spec(test_case) is not None


def test_case_citation_direct_async_is_left_to_judge():
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(model=judge)
    test_case = LLMTestCase(
        input="Which case supports the answer?",
        actual_output="The answer relies on [2019] UKSC 5 [1].",
        retrieval_context=["The relevant case is [2019] UKSC 5."],
    )

    assert (
        asyncio.run(metric.a_measure(test_case, _show_indicator=False)) == 1.0
    )
    assert test_case.actual_output in judge.last_prompt


@pytest.mark.parametrize("citation", ["[7]", "[0]", "[2019]"])
def test_case_citation_does_not_hide_another_invalid_marker(citation):
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input="Which case supports the answer?",
        actual_output=f"The answer relies on [2019] UKSC 5 {citation}.",
        retrieval_context=["The relevant case is [2019] UKSC 5."],
    )

    assert metric.measure(test_case, _show_indicator=False) == 0.0
    assert citation in metric.reason
    assert judge.last_prompt is None
    assert metric._system_one_eval_spec(test_case) is None


@pytest.mark.parametrize(
    "output",
    [
        pytest.param("The answer cites [2019].", id="bare-year"),
        pytest.param("The answer cites [2019] UKSC.", id="no-case-number"),
        pytest.param("The answer cites [7] UKSC 5.", id="short-number"),
        pytest.param("The answer cites [-2019] UKSC 5.", id="negative-number"),
        pytest.param("The answer cites [2019] ordinary text 5.", id="prose"),
        pytest.param("The answer cites [2019]\nUKSC 5.", id="next-line"),
    ],
)
def test_non_case_citations_still_fail_without_judge(output):
    judge = FakeJudge(CitationFaithfulnessVerdict(verdict="faithful"))
    metric = CitationFaithfulnessMetric(model=judge, async_mode=False)
    test_case = LLMTestCase(
        input=QUERY,
        actual_output=output,
        retrieval_context=RETRIEVAL_CONTEXT,
    )

    assert metric.measure(test_case, _show_indicator=False) == 0.0
    assert judge.last_prompt is None


@pytest.mark.parametrize("probability", [0.1, 0.9])
def test_case_citation_reaches_system_one_batch(probability):
    system_one = FakeSystemOneModel(
        answer_fn=answer_everything(probability=probability)
    )
    citation = CitationFaithfulnessMetric(
        system_one_model=system_one, eval_mode="system_one", async_mode=False
    )
    relevancy = AnswerRelevancyMetric(
        system_one_model=system_one, eval_mode="system_one"
    )
    test_case = LLMTestCase(
        input="Which case supports the answer?",
        actual_output="The answer relies on [2026] KEHC 11082 (KLR) [1].",
        retrieval_context=["The relevant case is [2026] KEHC 11082 (KLR)."],
    )

    handled = measure_system_one_batch(
        [citation, relevancy],
        test_case,
        ignore_errors=False,
        skip_on_missing_params=False,
    )

    assert handled == [citation, relevancy]
    assert len(system_one.calls) == 1
    assert (citation.score >= 0.5) == (probability >= 0.5)
