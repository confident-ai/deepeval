import pytest

from deepeval.metrics import (
    BaseConversationalMetric,
    BaseMetric,
    BiasMetric,
    DAGMetric,
    ExactMatchMetric,
    HallucinationMetric,
    MisuseMetric,
    TopicAdherenceMetric,
    ToxicityMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)


INHERITED_METRICS = [
    DAGMetric,
    ExactMatchMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
    TopicAdherenceMetric,
]

BASE_METRICS = [BaseMetric, BaseConversationalMetric]

LOWER_IS_BETTER_METRICS = [
    BiasMetric,
    HallucinationMetric,
    MisuseMetric,
    ToxicityMetric,
]

ALL_THRESHOLD_METRICS = (
    BASE_METRICS + INHERITED_METRICS + LOWER_IS_BETTER_METRICS
)


def make_metric(metric_class, score):
    metric = metric_class.__new__(metric_class)
    metric.error = None
    metric.score = score
    metric.threshold = 0.5
    metric.success = None
    return metric


@pytest.mark.parametrize("metric_class", BASE_METRICS + INHERITED_METRICS)
def test_is_successful_assigns_threshold_comparison(metric_class):
    assert make_metric(metric_class, 0.6).is_successful() is True
    assert make_metric(metric_class, 0.4).is_successful() is False


@pytest.mark.parametrize("metric_class", BASE_METRICS + INHERITED_METRICS)
def test_is_successful_handles_type_error(metric_class):
    assert make_metric(metric_class, None).is_successful() is False


@pytest.mark.parametrize("metric_class", ALL_THRESHOLD_METRICS)
def test_is_successful_returns_none_for_none_threshold(metric_class):
    metric = make_metric(metric_class, 0.6)
    metric.threshold = None
    metric.error = "Metric failed"

    assert metric.is_successful() is None
    assert metric.success is None


@pytest.mark.parametrize("metric_class", BASE_METRICS + INHERITED_METRICS)
def test_is_successful_returns_false_for_metric_error(metric_class):
    metric = make_metric(metric_class, 0.6)
    metric.error = "Metric failed"

    assert metric.is_successful() is False


class UnexpectedComparisonError:
    def __ge__(self, other):
        raise ValueError("unexpected comparison failure")


@pytest.mark.parametrize("metric_class", BASE_METRICS + INHERITED_METRICS)
def test_is_successful_does_not_suppress_unexpected_errors(metric_class):
    metric = make_metric(metric_class, UnexpectedComparisonError())

    with pytest.raises(ValueError, match="unexpected comparison failure"):
        metric.is_successful()


@pytest.mark.parametrize("metric_class", LOWER_IS_BETTER_METRICS)
def test_lower_is_better_metrics_keep_maximum_threshold_behavior(metric_class):
    assert make_metric(metric_class, 0.4).is_successful() is True
    assert make_metric(metric_class, 0.6).is_successful() is False
