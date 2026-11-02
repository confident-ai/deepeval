import inspect

import pytest

from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASContextualEntitiesRecall,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASFaithfulnessMetric,
    RagasMetric,
)
from deepeval.models.llms.constants import DEFAULT_GPT_MODEL


@pytest.mark.parametrize(
    "metric_cls",
    [
        RAGASContextualPrecisionMetric,
        RAGASContextualRecallMetric,
        RAGASContextualEntitiesRecall,
        RAGASAnswerRelevancyMetric,
        RAGASFaithfulnessMetric,
        RagasMetric,
    ],
)
def test_ragas_metrics_default_to_default_gpt_model(metric_cls):
    model = inspect.signature(metric_cls.__init__).parameters["model"]
    assert model.default == DEFAULT_GPT_MODEL
