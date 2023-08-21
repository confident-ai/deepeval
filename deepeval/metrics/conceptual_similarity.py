"""Asserting conceptual similarity
"""
from typing import Optional
from .metric import Metric
from ..utils import cosine_similarity


class ConceptualSimilarityMetric(Metric):
    """basic implementation of BertScoreMetric"""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        success_threshold: float = 0.7,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name).eval()
        self.success_threshold = success_threshold

    def _vectorize(self, text_a: str, text_b: str):
        vectors = self.model.encode([text_a, text_b])
        return vectors

    def measure(self, a: str, b: str):
        vectors = self._vectorize(a, b)
        self.score = cosine_similarity(vectors[0], vectors[1])
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.success_threshold


def assert_conceptual_similarity(text_1: str, text_2: str, success_threshold=0.3):
    metric = ConceptualSimilarityMetric(success_threshold=success_threshold)
    score = metric.measure(text_1, text_2)
    assert metric.is_successful(), f"Metric is not conceptually similar - got {score}"
