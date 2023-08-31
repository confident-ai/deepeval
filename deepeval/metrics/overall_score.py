"""Overall Score
"""
from typing import Optional
from .metric import Metric
from .factual_consistency import FactualConsistencyMetric
from .answer_relevancy import AnswerRelevancy
from .conceptual_similarity import ConceptualSimilarityMetric
from ..singleton import Singleton


class OverallScoreMetric(Metric, metaclass=Singleton):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score
        self.answer_relevancy = AnswerRelevancy()
        self.factual_consistency_metric = FactualConsistencyMetric()
        self.conceptual_similarity_metric = ConceptualSimilarityMetric()

    def __call__(self, query, output: str, expected_output: str, context: str):
        score = self.measure(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context,
        )
        self.success = score > self.minimum_score
        return score

    def measure(
        self,
        query: Optional[str] = None,
        output: Optional[str] = None,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
    ) -> float:
        metadata = {}
        if context is not None:
            factual_consistency_score = self.factual_consistency_metric.measure(
                context=context,
                output=output,
            )
            metadata["factual_consistency"] = float(factual_consistency_score)

        if query is not None:
            answer_relevancy_score = self.answer_relevancy.measure(
                query=query, output=output
            )
            metadata["answer_relevancy"] = float(answer_relevancy_score)

        if expected_output is not None:
            conceptual_similarity_score = self.conceptual_similarity_metric.measure(
                expected_output, output
            )
            metadata["conceptual_similarity"] = float(conceptual_similarity_score)

        overall_score = sum(metadata.values()) / len(metadata)

        self.success = bool(overall_score > self.minimum_score)
        print({"scores": metadata})
        self.log(
            success=self.success,
            score=overall_score,
            metric_name=self.__name__,
            query=query,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        return overall_score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Overall Score"


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
