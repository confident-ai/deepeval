from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    BaseMetric,
)

import pytest

# List of metric classes to test
metric_classes = [
    AnswerRelevancyMetric,
    # HallucinationMetric,
    # FaithfulnessMetric,
    # ContextualRelevancyMetric,
]


@pytest.mark.parametrize(
    "metric_class,async_mode",
    [(m, mode) for m in metric_classes for mode in [True, False]],
    ids=[
        f'{m.__name__}_{"async" if mode else "sync"}'
        for m in metric_classes
        for mode in [True, False]
    ],
)
def test_metric_no_repeat(metric_class, async_mode):
    test_case = LLMTestCase(
        input="Summarize: 'The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.'",
        actual_output="Paris is cold though. Pineapples are great on pizza. The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
    )
    metric: BaseMetric = metric_class(async_mode=async_mode)
    metric.measure(test_case)
    assert metric.repeat == 1
    assert not hasattr(metric, "standard_deviation")
    assert 0 <= metric.score <= 1
    assert not hasattr(metric, "scores")


@pytest.mark.parametrize(
    "metric_class,async_mode",
    [(m, mode) for m in metric_classes for mode in [True, False]],
    ids=[
        f'{m.__name__}_{"async" if mode else "sync"}'
        for m in metric_classes
        for mode in [True, False]
    ],
)
def test_metric_repeat(metric_class, async_mode):
    test_case = LLMTestCase(
        input="Summarize: 'The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.'",
        actual_output="Paris is cold though. Pineapples are great on pizza. The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
    )
    metric: BaseMetric = metric_class(async_mode=async_mode, repeat=2)
    metric.measure(test_case)
    assert metric.repeat == 2
    assert metric.standard_deviation < max(metric.scores) - min(metric.scores)
    assert 0 <= metric.score <= 1
