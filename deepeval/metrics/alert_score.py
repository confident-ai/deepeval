"""Alert Score
"""
from ..test_case import LLMTestCase
from .entailment_metric import EntailmentScoreMetric
from .metric import Metric

# from .answer_relevancy import AnswerRelevancyMetric


class AlertScoreMetric(Metric):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score
        self.entailment_metric = EntailmentScoreMetric()
        # self.answer_relevancy = AnswerRelevancyMetric()

    def __call__(self, test_case: LLMTestCase):
        score = self.measure(test_case)
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        entailment_score = self.entailment_metric.measure(
            test_case.context,
            test_case.output,
        )

        answer_expected_score = self.entailment_metric.measure(
            test_case.output,
            test_case.expected_output,
        )

        # This metric is very very bad right now as it requires the answer
        # to re-gurgitate the question.
        # answer_relevancy_score = self.answer_relevancy.measure(
        #     query=query, answer=output
        # )
        alert_score = min(entailment_score, answer_expected_score)
        self.success = alert_score > self.minimum_score
        self.log(
            success=self.success,
            score=alert_score,
            metric_name=self.__name__,
            query=test_case.query,
            output=test_case.output,
            expected_output=test_case.expected_output,
            context=test_case.context,
        )
        return alert_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Alert Score"


def assert_alert_score(
    query: str,
    output: str,
    expected_output: str,
    context: str,
    minimum_score: float = 0.5,
):
    """Create alert score."""
    metric = AlertScoreMetric(minimum_score=minimum_score)
    test_case = LLMTestCase(
        query=query,
        expected_output=expected_output,
        context=context,
        output=output,
    )
    score = metric.measure(test_case)
    assert metric.is_successful(), f"Found issue - Alert score: {score}"
