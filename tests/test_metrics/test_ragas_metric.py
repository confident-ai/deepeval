"""Tests for the ragas metric wrappers.

ragas reports NaN when it has nothing to score, so these cover what the
wrappers do with that. No API key is needed: ragas itself is stubbed, since
what is under test is deepeval's handling of the score it hands back.
"""

import sys
import types
from unittest.mock import patch

import pytest

from deepeval.test_case import LLMTestCase

RAGAS_METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "answer_relevancy",
    "faithfulness",
)

SUB_METRICS = (
    "RAGASContextualPrecisionMetric",
    "RAGASContextualRecallMetric",
    "RAGASContextualEntitiesRecall",
    "RAGASAnswerRelevancyMetric",
    "RAGASFaithfulnessMetric",
)


@pytest.fixture
def ragas_scores(monkeypatch):
    """Stub `ragas.evaluate` and yield the dict its result is read from."""
    scores = {}

    ragas = types.ModuleType("ragas")
    ragas.__version__ = "0.2.1"
    ragas.evaluate = lambda *args, **kwargs: scores
    ragas_metrics = types.ModuleType("ragas.metrics")
    for name in RAGAS_METRIC_NAMES:
        setattr(ragas_metrics, name, object())
    ragas.metrics = ragas_metrics

    datasets = types.ModuleType("datasets")
    datasets.Dataset = type(
        "Dataset", (), {"from_dict": staticmethod(lambda data: data)}
    )

    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    return scores


@pytest.fixture
def test_case():
    return LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers get a 30 day full refund."],
    )


# A non-string model skips the OpenAI client construction, which would
# otherwise need an API key.
MODEL = object()


class TestRagasSubMetrics:
    """Each wrapper reads one score straight out of the ragas result."""

    def test_a_score_is_recorded_as_measured(self, ragas_scores, test_case):
        from deepeval.metrics.ragas import RAGASFaithfulnessMetric

        ragas_scores["faithfulness"] = [0.8]
        metric = RAGASFaithfulnessMetric(threshold=0.5, model=MODEL)

        assert metric.measure(test_case) == 0.8
        assert metric.success is True
        assert metric.error is None

    def test_nan_is_not_reported_as_a_failing_score(
        self, ragas_scores, test_case
    ):
        from deepeval.metrics.ragas import RAGASFaithfulnessMetric

        # ragas returns NaN when no statements could be extracted from the
        # answer. Comparing that to the threshold is False, which would report
        # a test case that was never measured as one that failed.
        ragas_scores["faithfulness"] = [float("nan")]
        metric = RAGASFaithfulnessMetric(threshold=0.5, model=MODEL)

        assert metric.measure(test_case) is None
        assert metric.score is None
        assert metric.error is not None
        assert "faithfulness" in metric.error

    def test_a_missing_score_is_not_reported_as_a_failing_score(
        self, ragas_scores, test_case
    ):
        # Defensive: ragas is a third-party dependency and the version floor is
        # a lower bound, so the shape of its result is not guaranteed.
        from deepeval.metrics.ragas import RAGASFaithfulnessMetric

        ragas_scores["faithfulness"] = [None]
        metric = RAGASFaithfulnessMetric(threshold=0.5, model=MODEL)

        assert metric.measure(test_case) is None
        assert metric.error is not None


class TestRagasCompositeMetric:
    """The composite averages the five sub-metric scores."""

    @staticmethod
    def _run(test_case, sub_scores):
        from deepeval.metrics.ragas import RagasMetric

        patches = [
            patch(f"deepeval.metrics.ragas.{name}.measure", return_value=score)
            for name, score in zip(SUB_METRICS, sub_scores)
        ]
        for p in patches:
            p.start()
        try:
            metric = RagasMetric(threshold=0.5, model=MODEL)
            return metric, metric.measure(test_case)
        finally:
            for p in patches:
                p.stop()

    def test_averages_when_every_sub_metric_was_measured(
        self, ragas_scores, test_case
    ):
        metric, score = self._run(test_case, [0.9, 0.8, 0.7, 0.9, 0.6])

        assert score == pytest.approx(0.78)
        assert metric.success is True
        assert metric.error is None

    def test_no_composite_when_a_sub_metric_was_not_measured(
        self, ragas_scores, test_case
    ):
        # Averaging the four that did compute would report a plausible score
        # for a test case that was only measured in part.
        metric, score = self._run(test_case, [0.9, 0.8, 0.7, 0.9, None])

        assert score is None
        assert metric.error is not None
        assert metric.score_breakdown["Faithfulness (ragas)"] is None
