"""Overall Score
"""
import asyncio
from .metric import Metric
from .entailment_metric import EntailmentScoreMetric
from .answer_relevancy import AnswerRelevancy
from ..singleton import Singleton


class OverallScoreMetric(Metric, metaclass=Singleton):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score
        self.entailment_metric = EntailmentScoreMetric()
        self.answer_relevancy = AnswerRelevancy()

    def __call__(self, query, generated_text: str, expected_output: str, context: str):
        score = self.measure(
            query=query,
            generated_text=generated_text,
            expected_output=expected_output,
            context=context,
        )
        success = score > self.minimum_score
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=context,
                output=generated_text,
                expected_output=expected_output,
                success=success,
            )
        )
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

        answer_relevancy_score = self.answer_relevancy.measure(
            query=query, output=generated_text
        )

        overall_score = (
            0.33 * entailment_score
            + 0.33 * answer_expected_score
            + 0.33 * answer_relevancy_score
        )
        self.success = overall_score > self.minimum_score
        return overall_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Alert Score"


def assert_overall_score(
    generated_text: str,
    expected_output: str,
    context: str,
    minimum_score: float = 0.5,
):
    metric = OverallScoreMetric(minimum_score=minimum_score)
    score = metric.measure(
        generated_text=generated_text,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), f"Metric is not conceptually similar - got {score}"
