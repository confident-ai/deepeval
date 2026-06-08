import pytest

import deepeval.metrics.contextual_precision.contextual_precision as cp_module
from deepeval.metrics.contextual_precision.contextual_precision import (
    ContextualPrecisionMetric,
)
from deepeval.metrics.contextual_precision.schema import (
    ContextualPrecisionVerdict,
)
from deepeval.test_case import RetrievedContextData


class _FakeModel:
    def get_model_name(self):
        return "fake-model"


def _make_metric(verdicts, group_by=None):
    metric = object.__new__(ContextualPrecisionMetric)
    metric.threshold = 0.5
    metric.strict_mode = False
    metric.group_by = group_by
    metric.verdicts = [
        ContextualPrecisionVerdict(verdict=verdict, reason="")
        for verdict in verdicts
    ]
    return metric


def test_contextual_precision_accepts_group_by(monkeypatch):
    def group_by(context):
        return context

    monkeypatch.setattr(
        cp_module,
        "initialize_model",
        lambda model: (_FakeModel(), False),
    )

    metric = ContextualPrecisionMetric(group_by=group_by)

    assert metric.group_by is group_by


def test_contextual_precision_group_by_scores_retrieval_units():
    retrieval_context = [
        "doc=10k section=revenue chunk=1",
        "doc=10k section=revenue chunk=2",
        "doc=10k section=risk chunk=3",
    ]
    metric = _make_metric(
        ["no", "yes", "no"],
        group_by=lambda context: context.split(" chunk=")[0],
    )

    metric._score_verdicts = metric._get_score_verdicts(retrieval_context)

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "yes",
        "no",
    ]
    assert metric._calculate_score() == 1.0


def test_contextual_precision_keeps_ungrouped_contexts_as_singletons():
    retrieval_context = [
        "unrelated singleton",
        "doc=10k section=revenue chunk=1",
        "doc=10k section=revenue chunk=2",
    ]

    def group_by(context):
        if "section=revenue" in context:
            return "revenue"
        return None

    metric = _make_metric(["no", "yes", "no"], group_by=group_by)

    metric._score_verdicts = metric._get_score_verdicts(retrieval_context)

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "no",
        "yes",
    ]
    assert metric._calculate_score() == pytest.approx(0.5)


def test_contextual_precision_does_not_merge_none_group_keys():
    metric = _make_metric(["no", "yes"], group_by=lambda context: None)

    metric._score_verdicts = metric._get_score_verdicts(["chunk 1", "chunk 2"])

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "no",
        "yes",
    ]
    assert metric._calculate_score() == pytest.approx(0.5)


def test_contextual_precision_group_by_uses_retrieved_context_metadata():
    retrieval_context = [
        RetrievedContextData(
            source="revenue",
            context="Revenue table first overlapping window",
        ),
        RetrievedContextData(
            source="revenue",
            context="Revenue table second overlapping window",
        ),
        RetrievedContextData(
            source="risk",
            context="Risk factor section",
        ),
    ]

    metric = _make_metric(
        ["no", "yes", "no"], group_by=lambda item: item.source
    )

    assert metric._get_raw_retrieval_contexts(retrieval_context) == [
        "Revenue table first overlapping window",
        "Revenue table second overlapping window",
        "Risk factor section",
    ]

    metric._score_verdicts = metric._get_score_verdicts(retrieval_context)

    assert [verdict.verdict for verdict in metric._score_verdicts] == [
        "yes",
        "no",
    ]
    assert metric._calculate_score() == 1.0
