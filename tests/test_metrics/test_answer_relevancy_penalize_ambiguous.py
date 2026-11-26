"""Unit tests for AnswerRelevancyMetric.penalize_ambiguous_claims scoring.

No LLM calls — verdicts are injected and _calculate_score is exercised directly.
"""

from unittest.mock import patch

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy.schema import AnswerRelevancyVerdict
from deepeval.metrics.base_metric import Verdict
from tests.test_core.stubs import DummyModel


def _metric(*, penalize_ambiguous_claims: bool) -> AnswerRelevancyMetric:
    with patch(
        "deepeval.metrics.answer_relevancy.answer_relevancy.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return AnswerRelevancyMetric(
            async_mode=False,
            include_reason=False,
            penalize_ambiguous_claims=penalize_ambiguous_claims,
        )


def _verdicts() -> list[AnswerRelevancyVerdict]:
    return [
        AnswerRelevancyVerdict(verdict=Verdict.YES),
        AnswerRelevancyVerdict(
            verdict=Verdict.BORDERLINE, reason="hedging / unclear"
        ),
        AnswerRelevancyVerdict(verdict=Verdict.NO, reason="off-topic"),
    ]


def test_borderline_counts_as_relevant_by_default():
    metric = _metric(penalize_ambiguous_claims=False)
    metric.verdicts = _verdicts()
    # yes + borderline / 3
    assert metric._calculate_score() == 2 / 3


def test_borderline_is_penalized_when_flag_enabled():
    metric = _metric(penalize_ambiguous_claims=True)
    metric.verdicts = _verdicts()
    # only yes / 3
    assert metric._calculate_score() == 1 / 3


def test_empty_verdicts_score_perfect_regardless_of_flag():
    for penalize in (False, True):
        metric = _metric(penalize_ambiguous_claims=penalize)
        metric.verdicts = []
        assert metric._calculate_score() == 1
