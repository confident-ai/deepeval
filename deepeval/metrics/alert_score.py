"""Alert Score
"""
from .metric import Metric
from .entailment_metric import EntailmentScoreMetric

# from .answer_relevancy import AnswerRelevancy


class AlertScoreMetric(Metric):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score
        self.entailment_metric = EntailmentScoreMetric()
        # self.answer_relevancy = AnswerRelevancy()

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
            generated_text,
            expected_output,
        )

        # This metric is very very bad right now as it requires the answer
        # to re-gurgitate the question.
        # answer_relevancy_score = self.answer_relevancy.measure(
        #     query=query, answer=generated_text
        # )
        alert_score = min(entailment_score, answer_expected_score)
        self.success = alert_score > self.minimum_score
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
    minimum_score: float = 0.5,
):
    """Create alert score."""
    metric = AlertScoreMetric(minimum_score=minimum_score)
    score = metric.measure(
        query=query,
        generated_text=generated_text,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), f"Found issue - Alert score: {score}"
