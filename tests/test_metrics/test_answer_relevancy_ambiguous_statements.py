from unittest.mock import patch

import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy.schema import (
    AnswerRelevancyVerdict,
)
from tests.test_core.stubs import DummyModel


def make_metric(
    *,
    penalize_ambiguous_statements: bool = False,
    strict_mode: bool = False,
) -> AnswerRelevancyMetric:
    with patch(
        "deepeval.metrics.answer_relevancy.answer_relevancy.initialize_model"
    ) as mock_initialize_model:
        mock_initialize_model.return_value = (DummyModel(), True)
        return AnswerRelevancyMetric(
            async_mode=False,
            include_reason=False,
            strict_mode=strict_mode,
            penalize_ambiguous_statements=penalize_ambiguous_statements,
        )


def make_verdicts(*values: str) -> list[AnswerRelevancyVerdict]:
    return [AnswerRelevancyVerdict(verdict=value) for value in values]


def test_default_behavior_counts_idk_as_relevant():
    metric = make_metric()
    metric.verdicts = make_verdicts("yes", "no", "idk")

    assert metric.penalize_ambiguous_statements is False
    assert metric._calculate_score() == pytest.approx(2 / 3)


def test_penalize_ambiguous_statements_counts_idk_as_irrelevant():
    metric = make_metric(penalize_ambiguous_statements=True)
    metric.verdicts = make_verdicts("yes", "no", "idk")

    assert metric._calculate_score() == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("penalize_ambiguous_statements", "expected_score"),
    [
        (False, 1.0),
        (True, 0.0),
    ],
)
def test_only_idk_verdict(
    penalize_ambiguous_statements: bool,
    expected_score: float,
):
    metric = make_metric(
        penalize_ambiguous_statements=penalize_ambiguous_statements
    )
    metric.verdicts = make_verdicts("idk")

    assert metric._calculate_score() == expected_score


def test_penalized_idk_respects_strict_mode():
    metric = make_metric(
        penalize_ambiguous_statements=True,
        strict_mode=True,
    )
    metric.verdicts = make_verdicts("yes", "idk")

    assert metric._calculate_score() == 0
