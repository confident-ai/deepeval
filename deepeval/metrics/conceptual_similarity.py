"""Asserting conceptual similarity
"""
from typing import Optional
from .metric import Metric
from ..utils import cosine_similarity
from ..singleton import Singleton


class ConceptualSimilarityMetric(Metric, metaclass=Singleton):
    """basic implementation of BertScoreMetric"""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        minimum_score: float = 0.7,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name).eval()
        self.minimum_score = minimum_score

    def _vectorize(self, text_a: str, text_b: str):
        vectors = self.model.encode([text_a, text_b])
        return vectors

    def measure(self, output: str, expected_output: str):
        vectors = self._vectorize(output, expected_output)
        self.score = cosine_similarity(vectors[0], vectors[1])
        return float(self.score)

    def is_successful(self) -> bool:
        return self.score >= self.minimum_score


def assert_conceptual_similarity(text_1: str, text_2: str, minimum_score=0.3):
    metric = ConceptualSimilarityMetric(minimum_score=minimum_score)
    score = metric.measure(text_1, text_2)
    assert metric.is_successful(), f"Metric is not conceptually similar - got {score}"
