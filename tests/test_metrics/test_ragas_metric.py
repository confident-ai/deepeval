"""Offline regression tests for deepeval's ragas>=0.2.x compatibility contract.

Issue #1089 reported RAGASFaithfulnessMetric.measure() crashing against
ragas 0.2.1 with:

    TypeError: '>=' not supported between instances of 'list' and 'float'

because ragas 0.2.x returns an EvaluationResult whose ``result[name]`` is a
list of per-row scores, while deepeval compared that value directly against
``self.threshold``. Commit 5c13b10 fixed this by extracting
``scores[name][0]``, but tests/test_ragas.py was later deleted (commit
5f9c21689), so the contract lost all coverage. These tests restore it.

The fakes below emulate the empirically verified ragas 0.2.1 return shape
(``{metric_name: [row_score]}``) because upstream CI installs neither ragas
nor datasets. Everything runs offline: no network, no API keys.
"""

import math
import sys
import types

import pytest

from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASContextualEntitiesRecall,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASFaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

METRICS = [
    (
        RAGASContextualPrecisionMetric,
        "context_precision",
        "contextual-precision",
    ),
    (RAGASContextualRecallMetric, "context_recall", "contextual-recall"),
    (
        RAGASContextualEntitiesRecall,
        "context_entity_recall",
        "context-entities-recall",
    ),
    (RAGASAnswerRelevancyMetric, "answer_relevancy", "answer-relevancy"),
    (RAGASFaithfulnessMetric, "faithfulness", "faithfulness"),
]


class FakeEvaluate:
    """Stands in for ragas.evaluate; returns the configured single-row result."""

    score = float("nan")

    def __call__(self, dataset, metrics=None, llm=None, **kwargs):
        return {score_key: [self.score] for _, score_key, _ in METRICS}


@pytest.fixture
def ragas_env(monkeypatch):
    """Inject fake ragas/datasets modules and force offline code paths."""
    evaluate = FakeEvaluate()

    class ContextEntityRecall:
        pass

    class ResponseRelevancy:
        def __init__(self, embeddings=None):
            self.embeddings = embeddings

    class Dataset:
        @staticmethod
        def from_dict(data):
            return data

    ragas = types.ModuleType("ragas")
    ragas.__version__ = "0.2.1"
    ragas.evaluate = evaluate

    ragas_metrics = types.ModuleType("ragas.metrics")
    ragas_metrics.context_precision = object()
    ragas_metrics.context_recall = object()
    ragas_metrics.ContextEntityRecall = ContextEntityRecall
    ragas_metrics.ResponseRelevancy = ResponseRelevancy
    ragas_metrics.faithfulness = object()

    datasets = types.ModuleType("datasets")
    datasets.Dataset = Dataset

    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr("deepeval.metrics.ragas.langchain_available", True)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    return evaluate


@pytest.fixture
def test_case():
    return LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="They're fine.",
        expected_output="They fit perfectly.",
        retrieval_context=["All shoe sizes run true to fit."],
    )


@pytest.mark.parametrize(
    "metric_cls,score_key",
    [(metric_cls, score_key) for metric_cls, score_key, _ in METRICS],
    ids=[metric_id for _, _, metric_id in METRICS],
)
def test_measure_extracts_first_row_score(
    ragas_env, test_case, metric_cls, score_key
):
    """A list-shaped ragas result must yield the row score as a float."""
    ragas_env.score = 0.8
    metric = metric_cls(model=None, threshold=0.5, _track=False)

    assert metric.measure(test_case) == 0.8
    assert isinstance(metric.score, float)
    assert metric.success is True


@pytest.mark.parametrize(
    "metric_cls,score_key",
    [(metric_cls, score_key) for metric_cls, score_key, _ in METRICS],
    ids=[metric_id for _, _, metric_id in METRICS],
)
def test_measure_below_threshold_is_not_successful(
    ragas_env, test_case, metric_cls, score_key
):
    ragas_env.score = 0.1
    metric = metric_cls(model=None, threshold=0.5, _track=False)

    assert metric.measure(test_case) == 0.1
    assert metric.success is False


@pytest.mark.parametrize(
    "metric_cls,score_key",
    [(metric_cls, score_key) for metric_cls, score_key, _ in METRICS],
    ids=[metric_id for _, _, metric_id in METRICS],
)
def test_measure_nan_row_score_does_not_crash(
    ragas_env, test_case, metric_cls, score_key
):
    """A NaN row score must not raise on the threshold comparison."""
    ragas_env.score = float("nan")
    metric = metric_cls(model=None, threshold=0.5, _track=False)

    metric.measure(test_case)
    assert math.isnan(metric.score)
    assert metric.success is False
