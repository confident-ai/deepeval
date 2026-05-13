import pytest

from deepeval.metrics.contextual_precision.contextual_precision import (
    ContextualPrecisionMetric,
)
from deepeval.metrics.contextual_precision.schema import (
    ContextualPrecisionVerdict,
)


def make_metric(verdicts, group_by=None):
    metric = object.__new__(ContextualPrecisionMetric)
    metric.threshold = 0.5
    metric.strict_mode = False
    metric.group_by = group_by
    metric.verdicts = [
        ContextualPrecisionVerdict(verdict=verdict, reason="")
        for verdict in verdicts
    ]
    return metric


def test_contextual_precision_score_defaults_to_per_chunk_verdicts():
    metric = make_metric(["yes", "no", "yes"])

    assert metric._calculate_score() == pytest.approx(0.8333333333)


def test_contextual_precision_group_by_scores_retrieval_units():
    retrieval_context = [
        "doc=10k section=revenue chunk=1",
        "doc=10k section=revenue chunk=2",
        "doc=10k section=mda chunk=3",
    ]
    metric = make_metric(
        ["yes", "no", "yes"],
        group_by=lambda context: context.split(" chunk=")[0],
    )
    metric._score_verdicts = metric._get_score_verdicts(retrieval_context)

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "yes",
        "yes",
    ]
    assert metric._calculate_score() == 1.0


def test_contextual_precision_group_by_promotes_relevant_overlapping_chunk():
    retrieval_context = [
        "doc=10k section=revenue chunk=1",
        "doc=10k section=revenue chunk=2",
        "doc=10k section=risk-factors chunk=3",
    ]
    metric = make_metric(
        ["no", "yes", "no"],
        group_by=lambda context: context.split(" chunk=")[0],
    )
    metric._score_verdicts = metric._get_score_verdicts(retrieval_context)

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "yes",
        "no",
    ]
    assert metric._calculate_score() == 1.0
