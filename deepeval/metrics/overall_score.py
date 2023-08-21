"""Overall Score
"""
from .metric import Metric
from .entailment_metric import EntailmentScoreMetric

# from .answer_relevancy import AnswerRelevancy


class OverallScoreMetric(Metric):
    def __init__(self, success_threshold: float = 0.5):
        self.success_threshold = success_threshold
        self.entailment_metric = EntailmentScoreMetric()

    def __call__(self, generated_output: str, expected_output: str, context: str):
        score = self.measure(generated_output, expected_output, context)
        return score

    def measure(self, generated_text: str, expected_output: str, context: str) -> float:
        entailment_score = self.entailment_metric.measure(
            context,
            generated_text,
        )

        answer_expected_score = self.entailment_metric.measure(
            generated_text,
            expected_output,
        )

        overall_score = 0.5 * entailment_score + 0.5 * answer_expected_score
        self.success = overall_score > self.success_threshold
        return overall_score

    def is_successful(self) -> bool:
        return self.success

    def __name__(self):
        return "Alert Score"


def assert_overall_score(
    generated_output: str, expected_output: str, success_threshold: float = 0.5
):
    metric = OverallScoreMetric(success_threshold=success_threshold)
    score = metric.measure(
        generated_output=generated_output, expected_output=expected_output
    )
    assert metric.is_successful(), f"Metric is not conceptually similar - got {score}"
