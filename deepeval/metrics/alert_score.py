"""Alert Score
"""
from .metric import Metric
from .entailment_metric import EntailmentScoreMetric
from .answer_relevancy import AnswerRelevancy


class AlertScore(Metric):
    def __init__(self, success_threshold: float = 0.5):
        self.success_threshold = success_threshold
        self.entailment_metric = EntailmentScoreMetric()
        self.answer_relevancy = AnswerRelevancy()

    def __call__(self, generated_text: str, expected_output: str, context: str):
        score = self.measure(generated_text, expected_output, context)
        return score

    def measure(
        self, query: str, generated_text: str, expected_output: str, context: str
    ) -> float:
        entailment_score = self.entailment_metric.measure(
            context,
            generated_text,
        )
        answer_expected_score = self.entailment_metric.measure(
            expected_output,
            generated_text,
        )
        answer_relevancy_score = self.answer_relevancy.measure(
            query=query, answer=generated_text
        )
        alert_score = min(
            entailment_score, answer_relevancy_score, answer_expected_score
        )
        self.success = alert_score > self.success_threshold
        return alert_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Alert Score"


def assert_alert_score(
    query: str,
    generated_text: str,
    expected_output: str,
    context: str,
    success_threshold: float = 0.5,
):
    """Create alert score."""
    metric = AlertScore(success_threshold=success_threshold)
    score = metric.measure(
        query=query,
        generated_text=generated_text,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), f"Found issue - Alert score: {score}"
