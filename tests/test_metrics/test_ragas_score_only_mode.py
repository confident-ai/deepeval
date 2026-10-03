"""Regression tests for the RAGAS wrappers in score-only mode.

Every RAGAS wrapper computed `self.success = score >= self.threshold`
directly, so `threshold=None` (score-only mode, documented for all metrics)
raised `TypeError: '>=' not supported between instances of 'float' and
'NoneType'` after the evaluation had already run. Routing through
`BaseMetric.is_successful()` gives the documented `success is None`.

The ragas / datasets packages are stubbed in sys.modules, so the tests run
without ragas, an LLM provider, or API keys.
"""

import sys
import types

import pytest

import deepeval.metrics.ragas as ragas_module
from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASContextualEntitiesRecall,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASFaithfulnessMetric,
    RagasMetric,
)
from deepeval.test_case import LLMTestCase

STUB_SCORE = 0.8
SCORE_KEYS = [
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "answer_relevancy",
    "faithfulness",
]


@pytest.fixture
def fake_ragas(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    monkeypatch.setattr(
        ragas_module, "_check_langchain_available", lambda: None
    )

    ragas = types.ModuleType("ragas")
    ragas.__version__ = "0.3.0"
    ragas.evaluate = lambda dataset, metrics, llm, **kwargs: {
        key: [STUB_SCORE] for key in SCORE_KEYS
    }
    metrics = types.ModuleType("ragas.metrics")
    metrics.context_precision = object()
    metrics.context_recall = object()
    metrics.faithfulness = object()
    metrics.ContextEntityRecall = lambda *args, **kwargs: object()
    metrics.ResponseRelevancy = lambda *args, **kwargs: object()
    ragas.metrics = metrics

    datasets = types.ModuleType("datasets")

    class Dataset:
        @staticmethod
        def from_dict(data):
            return data

    datasets.Dataset = Dataset

    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    monkeypatch.setitem(sys.modules, "datasets", datasets)


TEST_CASE = LLMTestCase(
    input="What is the refund window?",
    actual_output="30 days.",
    expected_output="30 days.",
    retrieval_context=["Refunds are accepted within 30 days."],
)

WRAPPERS = [
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASContextualEntitiesRecall,
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
]


@pytest.mark.parametrize("metric_cls", WRAPPERS)
def test_wrapper_score_only_mode_does_not_raise(fake_ragas, metric_cls):
    metric = metric_cls(threshold=None, model=object())

    score = metric.measure(TEST_CASE)

    assert score == STUB_SCORE
    assert metric.score == STUB_SCORE
    assert metric.success is None
    assert metric.is_successful() is None


@pytest.mark.parametrize("metric_cls", WRAPPERS)
def test_wrapper_threshold_still_sets_success(fake_ragas, metric_cls):
    metric = metric_cls(threshold=0.5, model=object())
    metric.measure(TEST_CASE)
    assert metric.success is True

    metric = metric_cls(threshold=0.9, model=object())
    metric.measure(TEST_CASE)
    assert metric.success is False


def test_ragas_metric_score_only_mode_does_not_raise(fake_ragas):
    metric = RagasMetric(threshold=None, model=object())

    score = metric.measure(TEST_CASE)

    assert score == pytest.approx(STUB_SCORE)
    assert metric.success is None
    assert len(metric.score_breakdown) == len(WRAPPERS)
