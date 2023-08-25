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

    def __call__(self, query, output: str, expected_output: str, context: str):
        score = self.measure(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context,
        )
        success = score > self.minimum_score
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=context,
                output=output,
                expected_output=expected_output,
                success=success,
            )
        )
        return score

    def measure(
        self, query: str, output: str, expected_output: str, context: str
    ) -> float:
        factual_consistency_score = self.entailment_metric.measure(
            context,
            output,
        )

        answer_relevancy_score = self.answer_relevancy.measure(
            query=query, output=output
        )

        answer_similarity_score = self.entailment_metric.measure(
            expected_output, output
        )

        overall_score = (
            +0.33 * factual_consistency_score
            + 0.33 * answer_relevancy_score
            + 0.33 * answer_similarity_score
        )
        self.success = overall_score > self.minimum_score
        metadata = {
            "factual_consistency": factual_consistency_score,
            "answer_relevancy": answer_relevancy_score,
            "answer_similarity_score": answer_similarity_score,
        }
        self.log(
            success=self.success,
            score=overall_score,
            metric_name=self.__name__,
            query=query,
            output=output,
            expected_output=output,
            metadata=metadata,
        )
        return overall_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Alert Score"


def assert_overall_score(
    query: str,
    output: str,
    expected_output: str,
    context: str,
    minimum_score: float = 0.5,
):
    metric = OverallScoreMetric(minimum_score=minimum_score)
    score = metric.measure(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), f"Metric is not conceptually similar - got {score}"
