"""
Regression test: the Ragas-backed metrics must not default to a model that
OpenAI has deprecated/shut down (e.g. gpt-3.5-turbo), since users who don't
pass `model` explicitly would otherwise start getting API errors once the
model is retired.
"""

import inspect

from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASContextualEntitiesRecall,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASFaithfulnessMetric,
    RagasMetric,
)
from deepeval.models.llms.constants import DEFAULT_GPT_MODEL

DEPRECATED_DEFAULTS = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
}

RAGAS_METRIC_CLASSES = [
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASContextualEntitiesRecall,
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
    RagasMetric,
]


class TestRagasDefaultModel:
    def test_ragas_metrics_do_not_default_to_a_deprecated_model(self):
        for cls in RAGAS_METRIC_CLASSES:
            default = (
                inspect.signature(cls.__init__).parameters["model"].default
            )
            assert default not in DEPRECATED_DEFAULTS, (
                f"{cls.__name__} defaults to a deprecated OpenAI model "
                f"({default!r}); callers who omit `model` will get API "
                "errors once that model is retired."
            )

    def test_ragas_metrics_default_to_the_project_default_model(self):
        for cls in RAGAS_METRIC_CLASSES:
            default = (
                inspect.signature(cls.__init__).parameters["model"].default
            )
            assert default == DEFAULT_GPT_MODEL, (
                f"{cls.__name__} should default to deepeval's own "
                f"DEFAULT_GPT_MODEL ({DEFAULT_GPT_MODEL!r}) instead of "
                f"{default!r}, matching every other metric in the library."
            )
