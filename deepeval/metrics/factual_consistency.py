from .entailment_metric import EntailmentScoreMetric


class FactualConsistencyMetric(EntailmentScoreMetric):
    @property
    def __name__(self):
        return "Factual Consistency"


def assert_factual_consistency(
    output: str, context: str, minimum_score: float = 0.3
):
    """Assert that the output is factually consistent with the context."""

    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    score = metric(context, output)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."
